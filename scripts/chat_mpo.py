"""
Reinforcement learning on GSM8K via VMPO

1 GPU:
python -m scripts.chat_vmpo -- --run=default

8 GPUs:
torchrun --standalone --nproc_per_node=8 -m scripts.chat_vmpo -- --run=default
"""

import argparse
import os
import itertools
import wandb
import torch
import torch.distributed as dist
from contextlib import nullcontext
import copy
import math

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
# CLI arguments
parser = argparse.ArgumentParser(description="Reinforcement learning on GSM8K")
# Logging
parser.add_argument(
    "--run",
    type=str,
    default="dummy",
    help="wandb run name ('dummy' disables wandb logging)",
)
# Runtime
parser.add_argument(
    "--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)"
)
parser.add_argument("--dtype", type=str, default="bfloat16", help="float32|bfloat16")
# Model loading
parser.add_argument(
    "--source", type=str, default="sft", help="mid|sft - which checkpoint to load from"
)
parser.add_argument(
    "--model-tag", type=str, default=None, help="model tag to load from"
)
parser.add_argument(
    "--model-step", type=int, default=None, help="model step to load from"
)
# Training horizon
parser.add_argument(
    "--num-epochs", type=int, default=1, help="number of epochs over GSM8K"
)
# Batch sizes / sampling
parser.add_argument(
    "--device-batch-size", type=int, default=8, help="max batch size per forward pass"
)
parser.add_argument(
    "--examples-per-step",
    type=int,
    default=16,
    help="total examples per optimization step across all ranks",
)
parser.add_argument(
    "--num-samples", type=int, default=16, help="number of samples per example/question"
)
# Generation
parser.add_argument(
    "--max-new-tokens", type=int, default=256, help="max tokens to generate per sample"
)
parser.add_argument(
    "--temperature", type=float, default=1.0, help="sampling temperature"
)
parser.add_argument(
    "--top-k", type=int, default=50, help="top-k sampling (0 = disabled)"
)
# Optimization
parser.add_argument(
    "--embedding-lr",
    type=float,
    default=0.2,
    help="learning rate for embedding parameters (Adam)",
)
parser.add_argument(
    "--unembedding-lr",
    type=float,
    default=0.004,
    help="learning rate for unembedding parameters (Adam)",
)
parser.add_argument(
    "--matrix-lr",
    type=float,
    default=0.02,
    help="learning rate for matrix parameters (Muon)",
)
parser.add_argument(
    "--weight-decay",
    type=float,
    default=0.0,
    help="weight decay for embedding/unembedding parameters (Adam)",
)
parser.add_argument(
    "--init-lr-frac", type=float, default=0.05, help="initial LR as fraction of base LR"
)
# Advantage normalization (per-prompt, across the N samples for that prompt)
parser.add_argument(
    "--adv-norm",
    type=str,
    default="zscore",
    choices=["none", "zscore", "mad"],
    help="per-prompt advantage normalization across samples (recommended for VMPO E-step)",
)
parser.add_argument(
    "--adv-eps",
    type=float,
    default=1e-6,
    help="epsilon for advantage normalization",
)
parser.add_argument(
    "--adv-clip",
    type=float,
    default=10.0,
    help="clip normalized advantages to [-adv_clip, adv_clip] (0 disables)",
)
# MPO/VMPO
parser.add_argument(
    "--vmpo-eps", type=float, default=0.5, help="VMPO top-fraction for weighting"
)
parser.add_argument(
    "--q-kl-eps",
    type=float,
    default=0.1,
    help="E-step constraint: KL(q || Unif(topK)) <= q_kl_eps (sample-based proxy)",
)
parser.add_argument(
    "--eta-min", type=float, default=1e-4, help="min eta for E-step binary search"
)
parser.add_argument(
    "--eta-max", type=float, default=1e3, help="max eta for E-step binary search"
)
parser.add_argument(
    "--eta-search-iters",
    type=int,
    default=32,
    help="binary search iterations for E-step eta solve",
)
parser.add_argument(
    "--kl-target", type=float, default=0.02, help="target KL to reference"
)
parser.add_argument(
    "--entropy-target", type=float, default=2.0, help="target entropy proxy"
)
parser.add_argument(
    "--dual-lr", type=float, default=0.01, help="dual variable update rate"
)
parser.add_argument(
    "--ref-update-every",
    type=int,
    default=50,
    help="update reference model every N steps (0 = never)",
)
# Evaluation / checkpointing
parser.add_argument(
    "--eval-every", type=int, default=60, help="evaluate pass@k every N steps"
)
parser.add_argument(
    "--eval-examples",
    type=int,
    default=400,
    help="number of examples for pass@k evaluation",
)
parser.add_argument(
    "--save-every", type=int, default=60, help="save checkpoint every N steps"
)
parser.add_argument(
    "--completion-norm",
    action="store_true",
    help="normalize sequence logprobs by completion length (per-token objective/KL)",
)
args = parser.parse_args()
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

# Init compute/precision
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
ptdtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
autocast_ctx = (
    torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    if device_type == "cuda"
    else nullcontext()
)

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = (
    DummyWandb()
    if use_dummy_wandb
    else wandb.init(project="nanochat-rl", name=args.run, config=user_config)
)

# Init model and tokenizer
model, tokenizer, meta = load_model(
    args.source, device, phase="eval", model_tag=args.model_tag, step=args.model_step
)
engine = Engine(model, tokenizer)  # for sampling rollouts
ref_model = copy.deepcopy(model).eval()
for p in ref_model.parameters():
    p.requires_grad_(False)

# -----------------------------------------------------------------------------
# Rollout / sampling generator loop that yields batches of examples for training

train_task = GSM8K(subset="main", split="train")
val_task = GSM8K(subset="main", split="test")
num_steps = (len(train_task) // args.examples_per_step) * args.num_epochs
print0(f"Calculated number of steps: {num_steps}")


@torch.no_grad()
def get_batch():
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    rank_indices = range(
        ddp_rank, len(train_task), ddp_world_size
    )  # each rank is responsible for different examples in the training data
    for example_idx in itertools.cycle(rank_indices):

        # First get the full conversation of both user and assistant messages
        conversation = train_task[example_idx]

        # Tokenize the conversation, deleting the last Assistant message and priming the Assistant for a completion instead
        # (i.e. keep the <|assistant_start|>, but delete everything after it)
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        # Generate num_samples samples using batched generation, use loop to avoid OOMs
        model.eval()  # ensure the model is in eval mode
        generated_token_sequences = []
        masks = []
        num_sampling_steps = (
            args.num_samples // args.device_batch_size
        )  # go sequentially to prevent OOMs
        for sampling_step in range(num_sampling_steps):
            seed = (
                hash((step, example_idx, sampling_step)) & 0x7FFFFFFF
            )  # positive half of int32
            with autocast_ctx:
                generated_token_sequences_batch, masks_batch = engine.generate_batch(
                    tokens,
                    num_samples=args.device_batch_size,
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    seed=seed,  # must make sure to change the seed for each sampling step
                )
            generated_token_sequences.extend(generated_token_sequences_batch)
            masks.extend(masks_batch)

        # Calculate the rewards for each sample
        rewards = []
        for sample_tokens in generated_token_sequences:
            # Get just the generated tokens (after the prompt)
            generated_tokens = sample_tokens[prefix_length:]
            # Decode the generated response
            generated_text = tokenizer.decode(generated_tokens)
            # Calculate the reward
            reward = train_task.reward(conversation, generated_text)
            rewards.append(reward)

        # Pad the sequences so that their lengths (in time) match
        max_length = max(len(seq) for seq in generated_token_sequences)
        padded_generated_token_sequences = [
            seq + [assistant_end] * (max_length - len(seq))
            for seq in generated_token_sequences
        ]
        padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]
        # Stack up the sequences and masks into PyTorch tensors
        ids = torch.tensor(
            padded_generated_token_sequences, dtype=torch.long, device=device
        )
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
        # Generate autoregressive inputs and targets to the Transformer
        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone()  # clone to avoid in-place modification:
        targets[mask_ids[:, 1:] == 0] = (
            -1
        )  # <-- inplace modification right here. -1 is the ignore index
        # NOTE also that the Engine returns mask=0 for BOTH the prompt tokens AND the tool use tokens.
        # So we will (correctly) end up not training on the prompt tokens, or the tool use forced tokens.
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)

        # Baseline + advantage normalization (per prompt, across samples)
        advantages = rewards - rewards.mean()

        if args.adv_norm != "none":
            adv = advantages.float()
            if args.adv_norm == "zscore":
                scale = adv.std(unbiased=False).clamp_min(args.adv_eps)
                adv = (adv - adv.mean()) / scale
            elif args.adv_norm == "mad":
                med = adv.median()
                mad = (adv - med).abs().median().clamp_min(args.adv_eps)
                adv = (adv - med) / mad
            if args.adv_clip and args.adv_clip > 0:
                adv = adv.clamp(min=-args.adv_clip, max=args.adv_clip)
            advantages = adv.to(dtype=rewards.dtype)

        # VMPO E-step: temperature-controlled soft weights on top-fraction
        weights, e_step_eta, q_kl = _vmpo_e_step_q_from_advantages(
            advantages=advantages,
            top_frac=args.vmpo_eps,
            kl_eps=args.q_kl_eps,
            eta_min=args.eta_min,
            eta_max=args.eta_max,
            iters=args.eta_search_iters,
        )
        weights = weights.to(dtype=rewards.dtype)

        yield (
            generated_token_sequences,
            inputs,
            targets,
            rewards,
            advantages,
            weights,
            e_step_eta.to(dtype=rewards.dtype),
            q_kl.to(dtype=rewards.dtype),
        )


# -----------------------------------------------------------------------------
# Simple evaluation loop for GSM8K pass@k
def run_gsm8k_eval(
    task,
    tokenizer,
    engine,
    max_examples=None,
    num_samples=1,
    max_completion_tokens=256,
    temperature=0.0,
    top_k=50,
):
    """
    Evaluates GSM8K task and returns a list of records of evaluation outcomes.
    In a distributed setting, all ranks cooperate but this function will NOT
    do the reduction across ranks. This is the responsibility of the caller.
    Because the evaluation can take a while, this function will yield records one by one.
    """
    max_examples = (
        min(max_examples, len(task)) if max_examples is not None else len(task)
    )
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)
        # Generate k samples using batched generation inside the Engine
        assert (
            num_samples <= args.device_batch_size
        )  # usually this is true. we can add a loop if not...
        generated_token_sequences, masks = engine.generate_batch(
            tokens,
            num_samples=num_samples,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        # Check each sample for correctness
        outcomes = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            is_correct = task.evaluate(conversation, generated_text)
            outcomes.append({"is_correct": is_correct})
        # A bit bloated because I wanted to do more complex logging at one point.
        record = {
            "idx": idx,
            "outcomes": outcomes,
        }
        yield record


# -----------------------------------------------------------------------------
# Training loop

# Init the optimizer
optimizers = model.setup_optimizers(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)

# Set the initial learning rate as a fraction of the base learning rate
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * args.init_lr_frac
        group["initial_lr"] = group[
            "lr"
        ]  # save the initial learning so we can decay easily later


# Learning rate scheduler: simple rampdown to zero over num_steps
def get_lr_multiplier(it):
    lrm = 1.0 - it / num_steps
    return lrm


# Calculate the number of examples each rank handles to achieve the desired examples_per_step
print0(
    f"Total sequences per step: {args.examples_per_step * args.num_samples}"
)  # total batch size in sequences/step
assert (
    args.examples_per_step % ddp_world_size == 0
), "Desired examples per step must be divisible by the number of ranks"
examples_per_rank = args.examples_per_step // ddp_world_size  # per GPU
print0(f"Calculated examples per rank: {examples_per_rank}")

# Kick off the training loop
batch_iterator = get_batch()

# Dual variables for constraints in the M-step (keep semantics: these are NOT E-step eta)
log_kl_coef = torch.tensor(0.0, device=device, requires_grad=True)
log_ent_coef = torch.tensor(0.0, device=device, requires_grad=True)
dual_optimizer = torch.optim.Adam([log_kl_coef, log_ent_coef], lr=args.dual_lr)

for step in range(num_steps):
    if args.ref_update_every > 0 and step % args.ref_update_every == 0:
        ref_model.load_state_dict(model.state_dict())
    # Evaluate the model once in a while and log to wandb
    if step % args.eval_every == 0:
        model.eval()
        passk = torch.zeros(
            args.device_batch_size, device=device
        )  # pass@k for k=1..device_batch_size
        with autocast_ctx:
            records_iter = run_gsm8k_eval(
                val_task,
                tokenizer,
                engine,
                num_samples=args.device_batch_size,
                max_examples=args.eval_examples,
                temperature=1.0,
            )
            records = list(records_iter)  # collect all records
        for k in range(1, args.device_batch_size + 1):
            passk[k - 1] = sum(
                any(o["is_correct"] for o in r["outcomes"][:k]) for r in records
            )
        num_records = torch.tensor(len(records), dtype=torch.long, device=device)
        if ddp:
            dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
            dist.all_reduce(passk, op=dist.ReduceOp.SUM)
        passk = passk / num_records.item()  # normalize by the total number of records
        print_passk = [
            f"Pass@{k}: {passk[k - 1].item():.4f}"
            for k in range(1, args.device_batch_size + 1)
        ]
        print0(f"Step {step} | {', '.join(print_passk)}")
        log_passk = {
            f"pass@{k}": passk[k - 1].item()
            for k in range(1, args.device_batch_size + 1)
        }
        wandb_run.log(
            {
                "step": step,
                **log_passk,
            }
        )

    # Forward/Backward on rollouts over multiple examples in the dataset
    rewards_list = []
    sequence_lengths = []
    e_step_eta_list = []
    q_kl_list = []
    kl_sum = torch.zeros((), device=device)
    entropy_sum = torch.zeros((), device=device)
    kl_count = 0
    for example_step in range(examples_per_rank):
        (
            sequences_all,
            inputs_all,
            targets_all,
            rewards_all,
            advantages_all,
            weights_all,
            e_step_eta,
            q_kl,
        ) = next(batch_iterator)

        e_step_eta_list.append(float(e_step_eta.item()))
        q_kl_list.append(float(q_kl.item()))

        # Evaluate the loss and gradients
        model.train()  # ensure the model is in train mode
        # We need one more loop because we can never exceed the device_batch_size
        assert inputs_all.size(0) % args.device_batch_size == 0
        num_passes = inputs_all.size(0) // args.device_batch_size
        for pass_idx in range(num_passes):
            # Pluck out the batch for this pass
            b0, b1 = (
                pass_idx * args.device_batch_size,
                (pass_idx + 1) * args.device_batch_size,
            )
            inputs = inputs_all[b0:b1]
            targets = targets_all[b0:b1]
            rewards = rewards_all[b0:b1]
            advantages = advantages_all[b0:b1]
            weights = weights_all[b0:b1]
            # Calculate log probabilities and token-level entropy from logits
            with autocast_ctx:
                logits = model(inputs)
                log_probs = torch.log_softmax(logits, dim=-1)
                safe_targets = targets.clamp(min=0)
                token_logp = log_probs.gather(
                    dim=-1, index=safe_targets.unsqueeze(-1)
                ).squeeze(-1)
            with torch.no_grad():
                logits_ref = ref_model(inputs)
                log_probs_ref = torch.log_softmax(logits_ref, dim=-1)
                token_logp_ref = log_probs_ref.gather(
                    dim=-1, index=safe_targets.unsqueeze(-1)
                ).squeeze(-1)
            # MPO/VMPO objective (sequence-level, length-normalized)
            valid_mask = targets >= 0
            seq_logp_sum = (token_logp * valid_mask).sum(dim=1)
            seq_logp_ref_sum = (token_logp_ref * valid_mask).sum(dim=1)
            completion_len = valid_mask.sum(dim=1).clamp(min=1)

            seq_logp_mean = seq_logp_sum / completion_len
            seq_logp_ref_mean = seq_logp_ref_sum / completion_len

            if args.completion_norm:
                kl_est = (seq_logp_mean - seq_logp_ref_mean).mean()
                pg_term = seq_logp_mean
            else:
                kl_est = (seq_logp_sum - seq_logp_ref_sum).mean()
                pg_term = seq_logp_sum

            token_entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
            entropy_est = (token_entropy * valid_mask).sum() / valid_mask.sum().clamp(
                min=1
            )
            pg_obj = (weights * pg_term).sum()
            # normalize by number of passes and examples_per_rank
            pg_obj = pg_obj / (num_passes * examples_per_rank)

            # Dual variables (positive via exp)
            kl_coef = log_kl_coef.exp().detach().clamp(min=1e-6, max=1e6)
            ent_coef = log_ent_coef.exp().detach().clamp(min=1e-6, max=1e6)

            # Loss with constraints
            loss = (
                -pg_obj
                + kl_coef * (kl_est - args.kl_target)
                + ent_coef * (args.entropy_target - entropy_est)
            )
            loss.backward()

            # Dual update (projected via exp)
            dual_optimizer.zero_grad(set_to_none=True)
            kl_coef = log_kl_coef.exp().clamp(min=1e-6, max=1e6)
            ent_coef = log_ent_coef.exp().clamp(min=1e-6, max=1e6)
            dual_obj = kl_coef * (kl_est.detach() - args.kl_target) + ent_coef * (
                args.entropy_target - entropy_est.detach()
            )
            (-dual_obj).backward()
            dual_optimizer.step()

            kl_sum = kl_sum + kl_est.detach()
            entropy_sum = entropy_sum + entropy_est.detach()
            kl_count += 1
            print0(
                f"Step {step}/{num_steps} | Example step {example_step} | Pass {pass_idx} | loss: {loss.item():.6f} | Average reward: {rewards.mean().item()}"
            )
        # For logging
        rewards_list.append(rewards_all.mean().item())
        sequence_lengths.extend(len(seq) for seq in sequences_all)

    # A bunch of logging for how the rollouts went this step
    mean_reward = sum(rewards_list) / len(rewards_list)
    mean_sequence_length = sum(sequence_lengths) / len(sequence_lengths)

    mean_e_step_eta = sum(e_step_eta_list) / max(len(e_step_eta_list), 1)
    mean_q_kl = sum(q_kl_list) / max(len(q_kl_list), 1)

    if ddp:  # aggregate across ranks
        mean_reward_tensor = torch.tensor(mean_reward, dtype=torch.float, device=device)
        mean_sequence_length_tensor = torch.tensor(
            mean_sequence_length, dtype=torch.float, device=device
        )
        mean_e_step_eta_tensor = torch.tensor(
            mean_e_step_eta, dtype=torch.float, device=device
        )
        mean_q_kl_tensor = torch.tensor(mean_q_kl, dtype=torch.float, device=device)

        dist.all_reduce(mean_reward_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(mean_sequence_length_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(mean_e_step_eta_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(mean_q_kl_tensor, op=dist.ReduceOp.AVG)

        mean_reward = mean_reward_tensor.item()
        mean_sequence_length = mean_sequence_length_tensor.item()
        mean_e_step_eta = mean_e_step_eta_tensor.item()
        mean_q_kl = mean_q_kl_tensor.item()

    wandb_run.log(
        {
            "step": step,
            "reward": mean_reward,
            "sequence_length": mean_sequence_length,
            "kl_est": (kl_sum / max(kl_count, 1)).item(),
            "entropy_est": (entropy_sum / max(kl_count, 1)).item(),
            "e_step_eta": mean_e_step_eta,
            "q_kl": mean_q_kl,
            "kl_coef": log_kl_coef.exp().item(),
            "ent_coef": log_ent_coef.exp().item(),
        }
    )

    # Update the model parameters
    lrm = get_lr_multiplier(step)
    for opt in optimizers:  # first set the learning rate
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    for opt in optimizers:  # then step the optimizers
        opt.step()
    model.zero_grad(set_to_none=True)
    wandb_run.log(
        {
            "step": step,
            "lrm": lrm,
        }
    )

    # Master process saves the model once in a while. Skip first step. Save last step.
    if master_process and (
        (step > 0 and step % args.save_every == 0) or step == num_steps - 1
    ):
        base_dir = get_base_dir()
        depth = model.config.n_layer
        output_dirname = (
            args.model_tag if args.model_tag else f"d{depth}"
        )  # base the model tag on the depth of the base model
        checkpoint_dir = os.path.join(base_dir, "chatmpo_checkpoints", output_dirname)
        model_config_kwargs = (
            model.config.__dict__
        )  # slightly naughty, abusing the simplicity of GPTConfig, TODO nicer
        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            None,  # note: we don't bother to save the optimizer state
            {
                "model_config": model_config_kwargs,
            },
        )
        print(f"✅ Saved model checkpoint to {checkpoint_dir}")

# Log to report
from nanochat.report import get_report

get_report().log(
    section="Chat RL",
    data=[
        user_config,  # CLI args
    ],
)

wandb_run.finish()  # wandb run finish
compute_cleanup()
