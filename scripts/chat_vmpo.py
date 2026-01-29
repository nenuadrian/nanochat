"""
V-MPO for NanoChat (sequence-level, on-policy)

This implements the algorithm from:
Song et al. "V-MPO: On-Policy Maximum a Posteriori Policy Optimization" (2019)

Key properties:
- On-policy
- Sequence-level advantages
- E-step: ψ-weights via temperature η
- M-step: policy update + KL trust region via dual α
- Snapshot reference policy per step
"""

import argparse
import os
import itertools
import math
import copy
import time
import wandb
import torch
import torch.nn as nn
import torch.distributed as dist
from contextlib import nullcontext

from nanochat.common import (
    compute_init,
    compute_cleanup,
    print0,
    get_base_dir,
    DummyWandb,
    autodetect_device_type,
)
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K

# -----------------------------------------------------------------------------
# CLI
parser = argparse.ArgumentParser("NanoChat V-MPO")

# Logging
parser.add_argument("--run", type=str, default="dummy")

# Runtime
parser.add_argument("--device-type", type=str, default="")
parser.add_argument("--dtype", type=str, default="bfloat16")

# Model loading
parser.add_argument("--source", type=str, default="sft")
parser.add_argument("--model-tag", type=str, default=None)
parser.add_argument("--model-step", type=int, default=None)

# Training horizon
parser.add_argument("--num-epochs", type=int, default=1)

# Sampling
parser.add_argument("--device-batch-size", type=int, default=8)
parser.add_argument("--examples-per-step", type=int, default=16)
parser.add_argument("--num-samples", type=int, default=16)

# Generation
parser.add_argument("--max-new-tokens", type=int, default=256)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top-k", type=int, default=50)

# Optimization (policy)
parser.add_argument("--embedding-lr", type=float, default=0.2)
parser.add_argument("--unembedding-lr", type=float, default=0.004)
parser.add_argument("--matrix-lr", type=float, default=0.02)
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("--init-lr-frac", type=float, default=0.05)

# V-MPO duals
parser.add_argument("--eps-eta", type=float, default=2.0)
parser.add_argument("--eps-alpha", type=float, default=0.01)
parser.add_argument("--eta-init", type=float, default=1.0)
parser.add_argument("--alpha-init", type=float, default=1.0)
parser.add_argument("--eta-min", type=float, default=1e-4)
parser.add_argument("--alpha-min", type=float, default=1e-4)
parser.add_argument("--eta-steps", type=int, default=10)

# Eval / checkpoint
parser.add_argument("--eval-every", type=int, default=60)
parser.add_argument("--eval-examples", type=int, default=400)
parser.add_argument("--save-every", type=int, default=60)

args = parser.parse_args()
user_config = vars(args).copy()

# -----------------------------------------------------------------------------
# Init compute
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, rank, local_rank, world_size, device = compute_init(device_type)
master = rank == 0

ptdtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
autocast_ctx = (
    torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    if device_type == "cuda"
    else nullcontext()
)

# -----------------------------------------------------------------------------
# Logging
use_dummy = args.run == "dummy" or not master
wandb_run = (
    DummyWandb()
    if use_dummy
    else wandb.init(
        project="nanochat-vmpo",
        name=args.run,
        config=user_config,
    )
)

# -----------------------------------------------------------------------------
# Model
model, tokenizer, meta = load_model(
    args.source, device, phase="eval", model_tag=args.model_tag, step=args.model_step
)
engine = Engine(model, tokenizer)

# -----------------------------------------------------------------------------
# Tasks
train_task = GSM8K(subset="main", split="train")
val_task = GSM8K(subset="main", split="test")

num_steps = (len(train_task) // args.examples_per_step) * args.num_epochs
print0(f"Total steps: {num_steps}")


# -----------------------------------------------------------------------------
# Dual variables
def inv_softplus(x):
    return x + torch.log1p(-torch.exp(-x))


eta_raw = nn.Parameter(inv_softplus(torch.tensor(args.eta_init, device=device)))
alpha_raw = nn.Parameter(inv_softplus(torch.tensor(args.alpha_init, device=device)))

eta_opt = torch.optim.Adam([eta_raw], lr=1e-3)
alpha_opt = torch.optim.Adam([alpha_raw], lr=1e-3)

# -----------------------------------------------------------------------------
# Optimizers (policy)
optimizers = model.setup_optimizers(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)
for opt in optimizers:
    for g in opt.param_groups:
        g["lr"] *= args.init_lr_frac
        g["initial_lr"] = g["lr"]


def lr_mult(step):
    return 1.0 - step / num_steps


# -----------------------------------------------------------------------------
# Rollout generator
@torch.no_grad()
def get_batch(step):
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    indices = range(rank, len(train_task), world_size)

    for idx in itertools.cycle(indices):
        convo = train_task[idx]
        tokens = tokenizer.render_for_completion(convo)
        prefix_len = len(tokens)

        model.eval()
        seqs, masks = [], []

        for s in range(args.num_samples // args.device_batch_size):
            seed = hash((step, idx, s)) & 0x7FFFFFFF
            with autocast_ctx:
                bseqs, bmasks = engine.generate_batch(
                    tokens,
                    num_samples=args.device_batch_size,
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    seed=seed,
                )
            seqs.extend(bseqs)
            masks.extend(bmasks)

        rewards = []
        for seq in seqs:
            text = tokenizer.decode(seq[prefix_len:])
            rewards.append(train_task.reward(convo, text))

        maxlen = max(len(s) for s in seqs)
        seqs = [s + [assistant_end] * (maxlen - len(s)) for s in seqs]
        masks = [m + [0] * (maxlen - len(m)) for m in masks]

        ids = torch.tensor(seqs, device=device)
        mask_ids = torch.tensor(masks, device=device)

        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone()
        targets[mask_ids[:, 1:] == 0] = -1

        rewards = torch.tensor(rewards, device=device, dtype=torch.float)
        yield inputs, targets, rewards


# -----------------------------------------------------------------------------
# Training loop
batch_iter = get_batch(0)
step_start = time.time()

# Keep a single frozen reference policy on device to avoid per-step deepcopies that
# balloon GPU memory. We refresh its weights each step via load_state_dict.
ref_model = copy.deepcopy(model).eval()
for p in ref_model.parameters():
    p.requires_grad_(False)

for step in range(num_steps):
    model.train()

    # Refresh reference policy weights without allocating a new model
    ref_model.load_state_dict(model.state_dict(), strict=True)

    rewards_all = []
    logp_all = []
    logp_ref_all = []
    seq_mask_all = []
    seq_len_all = []

    # Collect rollouts
    for _ in range(args.examples_per_step // world_size):
        inputs, targets, rewards = next(batch_iter)
        rewards_all.append(rewards)

        with autocast_ctx:
            logp = -model(inputs, targets, loss_reduction="none")
            logp = logp.view_as(inputs)

            with torch.no_grad():
                logp_ref = -ref_model(inputs, targets, loss_reduction="none")
                logp_ref = logp_ref.view_as(inputs)

        mask = targets >= 0
        logp_seq = (logp * mask).sum(dim=1)
        logp_ref_seq = (logp_ref * mask).sum(dim=1)

        logp_all.append(logp_seq)
        logp_ref_all.append(logp_ref_seq)
        seq_mask_all.append(mask.any(dim=1))
        seq_len_all.append(mask.sum(dim=1).float())

    # Group by prompt to avoid cross-prompt leakage when centering
    rewards = torch.stack(rewards_all)  # (num_prompts, num_samples)
    logp = torch.stack(logp_all)
    logp_ref = torch.stack(logp_ref_all)
    valid = torch.stack(seq_mask_all)
    seq_lens = torch.stack(seq_len_all)

    # Advantages
    adv = rewards - rewards.mean(dim=1, keepdim=True)  # per-prompt baseline
    adv_flat = adv.flatten()
    rewards_flat = rewards.flatten()
    valid_mask = valid
    valid = valid.flatten()
    seq_lens = seq_lens.flatten()

    # -------------------------
    # E-step: optimize η
    # -------------------------
    counts = valid_mask.float().sum(dim=1, keepdim=True)
    adv_masked = adv.masked_fill(~valid_mask, float("-inf"))

    for _ in range(args.eta_steps):
        eta = torch.nn.functional.softplus(eta_raw) + args.eta_min
        adv_max = torch.where(
            counts > 0,
            adv_masked.max(dim=1, keepdim=True).values,
            torch.zeros_like(counts),
        )
        adv_centered = adv_masked - adv_max
        log_mean_exp = torch.logsumexp(adv_centered / eta, dim=1) - torch.log(
            counts.clamp_min(1.0)
        )
        loss_eta = eta * (args.eps_eta + log_mean_exp.mean())

        eta_opt.zero_grad()
        loss_eta.backward()
        eta_opt.step()

    with torch.no_grad():
        eta = torch.nn.functional.softplus(eta_raw) + args.eta_min
        adv_max = torch.where(
            counts > 0,
            adv_masked.max(dim=1, keepdim=True).values,
            torch.zeros_like(counts),
        )
        adv_centered = adv_masked - adv_max
        w = torch.exp(adv_centered / eta) * valid_mask.float()
        psi = w / w.sum(dim=1, keepdim=True).clamp_min(1e-8)

    # -------------------------
    # M-step: policy + α
    # -------------------------
    logp_flat = logp.flatten()
    logp_ref_flat = logp_ref.flatten()
    psi_flat = psi.flatten()

    logp_valid = logp_flat[valid]
    logp_ref_valid = logp_ref_flat[valid]
    psi_valid = psi_flat[valid]

    policy_loss = -(psi_valid * logp_valid).sum() / rewards.shape[0]

    kl = logp_ref_valid - logp_valid
    kl_mean = (psi_valid * kl).sum() / rewards.shape[0]

    alpha = torch.nn.functional.softplus(alpha_raw) + args.alpha_min
    total_loss = policy_loss + alpha.detach() * kl_mean

    total_loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=float("inf")
    )

    for opt in optimizers:
        for g in opt.param_groups:
            g["lr"] = g["initial_lr"] * lr_mult(step)
        opt.step()

    model.zero_grad(set_to_none=True)

    # α update
    alpha_opt.zero_grad()
    loss_alpha = -alpha * (kl_mean.detach() - args.eps_alpha)
    loss_alpha.backward()
    alpha_opt.step()

    # Logging
    if master:
        psi_entropy = -(psi * psi.clamp_min(1e-8).log()).sum(dim=1).mean().item()
        lr_values = []
        for opt in optimizers:
            lr_values.extend([g["lr"] for g in opt.param_groups])
        lr_mean = sum(lr_values) / len(lr_values) if lr_values else 0.0
        log_data = {
            "step": step,
            "reward_mean": rewards_flat.mean().item(),
            "reward_std": rewards_flat.std().item(),
            "reward_max": rewards_flat.max().item(),
            "reward_min": rewards_flat.min().item(),
            "valid_frac": valid.float().mean().item(),
            "seq_len_mean": seq_lens.mean().item(),
            "adv_mean": adv_flat.mean().item(),
            "adv_std": adv_flat.std().item(),
            "psi_entropy": psi_entropy,
            "eta": eta.item(),
            "alpha": alpha.item(),
            "kl": kl_mean.item(),
            "kl_max": kl.max().item(),
            "logp_mean": logp_valid.mean().item(),
            "logp_std": logp_valid.std().item(),
            "logp_ref_mean": logp_ref_valid.mean().item(),
            "loss_eta": loss_eta.item(),
            "loss_alpha": loss_alpha.item(),
            "grad_norm": (
                grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
            ),
            "lr_mean": lr_mean,
            "step_time": time.time() - step_start,
        }
        step_start = time.time()
        wandb_run.log(log_data)

    # Checkpoint
    if master and (step % args.save_every == 0 or step == num_steps - 1):
        base = get_base_dir()
        depth = model.config.n_layer
        tag = args.model_tag or f"d{depth}"
        out = os.path.join(base, "chat_vmpo_checkpoints", tag)
        save_checkpoint(
            out,
            step,
            model.state_dict(),
            None,
            {"model_config": model.config.__dict__},
        )
        print0(f"Saved checkpoint to {out}")

# -----------------------------------------------------------------------------
wandb_run.finish()
compute_cleanup()
