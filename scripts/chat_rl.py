"""
Reinforcement learning via "GRPO", on GSM8K or ARC (see --task).

I put GRPO in quotes because we actually end up with something a lot
simpler and more similar to just REINFORCE:

1) Delete trust region, so there is no KL regularization to a reference model
2) We are on policy, so there's no need for PPO ratio+clip.
3) We use DAPO style normalization that is token-level, not sequence-level.
4) Instead of z-score normalization (r - mu)/sigma, only use (r - mu) as the advantage.

--objective tpo swaps that policy gradient for Target Policy Optimization
(arXiv:2604.06159): instead of weighting each completion's log-prob by a scalar
advantage, build the distribution you want over the group,
q ∝ p_old * exp(zscore(r)), and fit the policy to it by cross-entropy. See
nanochat/tpo.py. The rollouts, the allocators and everything else are shared, so
--objective is a clean A/B on the loss alone.

1 GPU:
python -m scripts.chat_rl

8 GPUs:
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=default
"""

import argparse
import os
import itertools
import wandb
import numpy as np
import torch
import torch.distributed as dist
from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb, autodetect_device_type
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K
from tasks.arc import ARC
from nanochat.allocation import (
    uniform_allocation, gvm_allocation, vip_allocation,
    allocation_stats, rollout_accounting,
)
from nanochat.vip_gp import PromptSuccessGP
from nanochat.tpo import tpo_weights

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Reinforcement learning on GSM8K or ARC")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Model loading
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
# Training horizon
parser.add_argument("--task", type=str, default="gsm8k",
                    choices=["gsm8k", "arc-easy", "arc-challenge"],
                    help="RL task. GSM8K is what nanochat shipped with, but a small model "
                         "scores ~2%% on it, which leaves every prompt at p=0 and nothing for "
                         "a budget allocator to allocate on. ARC is multiple-choice and lands "
                         "near p=0.5, where the allocators actually differ.")
parser.add_argument("--num-epochs", type=int, default=1, help="number of epochs over the task")
# Batch sizes / sampling
parser.add_argument("--device-batch-size", type=int, default=8, help="max batch size per forward pass")
parser.add_argument("--examples-per-step", type=int, default=16, help="total examples per optimization step across all ranks")
parser.add_argument("--num-samples", type=int, default=16, help="number of samples per example/question")
# Generation
parser.add_argument("--max-new-tokens", type=int, default=None,
                    help="max tokens to generate per sample (default: task-dependent, see below)")
parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
parser.add_argument("--top-k", type=int, default=50, help="top-k sampling (0 = disabled)")
# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.2, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay for embedding/unembedding parameters (Adam)")
parser.add_argument("--init-lr-frac", type=float, default=0.05, help="initial LR as fraction of base LR")
# Evaluation / checkpointing
parser.add_argument("--eval-every", type=int, default=60, help="evaluate pass@k every N steps")
parser.add_argument("--eval-examples", type=int, default=400, help="number of examples for pass@k evaluation")
# --- rollout budget allocation -------------------------------------------
parser.add_argument("--allocator", type=str, default="uniform", choices=["uniform", "gvm", "vip"],
                    help="how to split the rollout budget across prompts in a step")
parser.add_argument("--alloc-min", type=int, default=0,
                    help="minimum rollouts per prompt (VIP's Theorem 5.1 needs >=3)")
parser.add_argument("--alloc-max", type=int, default=0,
                    help="maximum rollouts per prompt; 0 means the whole step budget")
# GVM (arXiv:2505.02391): measures p_i and G_i with a pilot pass
parser.add_argument("--gvm-pilot-samples", type=int, default=4,
                    help="N', pilot rollouts per prompt used to measure p_i and G_i")
parser.add_argument("--gvm-alpha", type=float, default=1e-3)
parser.add_argument("--gvm-beta", type=float, default=2.0)
parser.add_argument("--gvm-grad-norm", type=str, default="embed", choices=["embed", "off"],
                    help="'embed' takes ||grad|| wrt the token embedding as G_i (as the GVM repo does); "
                         "'off' fixes G_i=1, isolating the accept-rate half of Proposition 1")
# VIP (arXiv:2602.01601): predicts p_i with a GP, no pilot pass
parser.add_argument("--vip-bandwidth", type=float, default=0.0, help="RBF bandwidth; 0 = median heuristic")
parser.add_argument("--vip-eps", type=float, default=0.05)
parser.add_argument("--prompt-pool", type=int, default=0,
                    help="cap the prompt set for ALL allocators, so a VIP run and a "
                         "uniform run see the same data; 0 = no cap (auto-set for VIP)")
parser.add_argument("--vip-embed-prompts", type=int, default=1024,
                    help="size of the prompt pool the GP is fitted over")
# Which gradient estimator to form once the rollouts exist
parser.add_argument("--estimator-weight", type=str, default="per_prompt",
                    choices=["per_prompt", "per_token", "inv_np"],
                    help="per_prompt: equal weight per prompt (GVM Alg.1 line 8, nanochat default); "
                         "per_token: global token-mean, so weight grows with n_i (what verl does); "
                         "inv_np: explicit 1/(n_i p_i) weighting from GVM Lemma 1")
# Which loss turns the scored rollouts into a gradient
parser.add_argument("--objective", type=str, default="grpo", choices=["grpo", "tpo"],
                    help="grpo: mean-subtracted policy gradient (what nanochat shipped with); "
                         "tpo: Target Policy Optimization, arXiv:2604.06159, cross-entropy to "
                         "q ∝ p_old * exp(zscore(r)) over the group")
parser.add_argument("--tpo-eta", type=float, default=1.0,
                    help="TPO target temperature. 1.0 is the paper default and robust over ~[0.25, 2]")
parser.add_argument("--tpo-anchor", type=int, default=1,
                    help="keep the p_old anchor in the target; 0 gives the q ∝ exp(u) ablation")
parser.add_argument("--tpo-logp", type=str, default="sum", choices=["sum", "mean"],
                    help="how a completion's sequence log-prob enters the group softmax. "
                         "'sum' is the paper's objective; 'mean' divides by the completion length, "
                         "which stops long completions collapsing the group softmax onto whichever "
                         "sample was shortest (watch tpo/p_max to see whether you need it)")
parser.add_argument("--tpo-epochs", type=int, default=1,
                    help="gradient epochs over each rollout batch. TPO's frozen target is a fixed "
                         "point, so reuse needs no PPO ratio or clip (paper 5.3); the paper reports "
                         "4 epochs converging ~5x earlier. Only valid with --objective tpo")
parser.add_argument("--save-every", type=int, default=60, help="save checkpoint every N steps")
args = parser.parse_args()
if args.objective != "tpo":
    assert args.tpo_epochs == 1, "--tpo-epochs > 1 needs --objective tpo: the grpo path here has no " \
        "PPO ratio or clip, so reusing a rollout batch for it would be uncorrected off-policy"
assert args.tpo_epochs >= 1
# Resolve the task-dependent rollout length. GSM8K answers are chain-of-thought (median 89
# tokens, max 262 in the training set) and need the long budget; ARC's target is a single
# letter plus <|assistant_end|>, i.e. 2 tokens, and reward() parses the first letter out of
# whatever was emitted. The budget is not just a cap on waste: Engine.generate only breaks
# early once EVERY row in the group has emitted a stop token, so one rambling sample holds
# the whole group open to max_new_tokens. 16 leaves room for "The answer is A." while
# cutting the worst case 16x.
MAX_NEW_TOKENS_BY_TASK = {"gsm8k": 256, "arc-easy": 16, "arc-challenge": 16}
if args.max_new_tokens is None:
    args.max_new_tokens = MAX_NEW_TOKENS_BY_TASK[args.task]
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

# Init compute/precision
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-rl", name=args.run, config=user_config)

# Init model and tokenizer
model, tokenizer, meta = load_model("sft", device, phase="eval", model_tag=args.model_tag, step=args.model_step)
engine = Engine(model, tokenizer) # for sampling rollouts

# -----------------------------------------------------------------------------
# Rollout / sampling generator loop that yields batches of examples for training

if args.task == "gsm8k":
    train_task = GSM8K(subset="main", split="train")
    val_task = GSM8K(subset="main", split="test")
else:
    subset = "ARC-Easy" if args.task == "arc-easy" else "ARC-Challenge"
    train_task = ARC(subset=subset, split="train")
    val_task = ARC(subset=subset, split="test")
print0(f"Task: {args.task} | train {len(train_task)} | val {len(val_task)} | max_new_tokens: {args.max_new_tokens}")
num_steps = (len(train_task) // args.examples_per_step) * args.num_epochs
print0(f"Calculated number of steps: {num_steps}")
print0(f"Objective: {args.objective}"
       + (f" (eta={args.tpo_eta}, anchor={bool(args.tpo_anchor)}, logp={args.tpo_logp}, "
          f"epochs={args.tpo_epochs})" if args.objective == "tpo" else ""))
if args.objective == "tpo" and args.estimator_weight != "per_prompt":
    print0(f"NOTE: --estimator-weight={args.estimator_weight} is ignored under --objective tpo, "
           "which uses the paper's normalisation (sum over candidates, mean over prompts).")

# -----------------------------------------------------------------------------
# Allocation machinery: pilot rollouts for GVM, a GP for VIP.

alloc_min = args.alloc_min if args.alloc_min > 0 else (3 if args.allocator == "vip" else 0)

# The prompts this rank will cycle through. VIP needs an embedding per prompt for
# its GP, so its pool is capped -- and the cap is applied to EVERY allocator, or
# uniform and GVM would see a larger prompt set than VIP and the comparison would
# be confounded by data, not allocation.
rank_indices = list(range(ddp_rank, len(train_task), ddp_world_size))
pool_cap = args.prompt_pool if args.prompt_pool > 0 else (
    args.vip_embed_prompts if args.allocator == "vip" else 0)
if pool_cap > 0:
    rank_indices = rank_indices[:pool_cap]
# Exact position of each prompt in the pool. Using `idx % pool_size` instead
# would silently collide unrelated prompts onto the same GP entry.
pool_pos = {int(idx): i for i, idx in enumerate(rank_indices)}
print0(f"Prompt pool for this rank: {len(rank_indices)} prompts"
       + (f" (capped from {len(range(ddp_rank, len(train_task), ddp_world_size))})" if pool_cap else ""))

@torch.no_grad()
def prompt_embeddings(indices):
    """Mean hidden state over the prompt tokens, one vector per prompt.

    VIP needs a representation to put a kernel over; the paper uses the model's
    own embeddings of the prompt.
    """
    model.eval()
    out = []
    wte = model.transformer.wte
    for i in indices:
        toks = tokenizer.render_for_completion(train_task[int(i)])
        ids = torch.tensor(toks[-256:], dtype=torch.long, device=device)[None, :]
        out.append(wte(ids).float().mean(dim=1).squeeze(0).cpu().numpy())
    return np.stack(out)

def gvm_pilot(example_idx, n_pilot, step):
    """GVM's E-step for one prompt: measure the accept rate p_i and gradient norm G_i.

    These are the two quantities Proposition 1 balances. Note the rollouts drawn
    here are thrown away afterwards -- the training rollouts are drawn fresh --
    which is why `cost/rollouts_pilot` is tracked separately.
    """
    conversation = train_task[int(example_idx)]
    toks = tokenizer.render_for_completion(conversation)
    prefix_len = len(toks)
    seqs, _ = engine.generate_batch(
        toks, num_samples=n_pilot, max_tokens=args.max_new_tokens,
        temperature=args.temperature, top_k=args.top_k,
        seed=hash(("pilot", step, int(example_idx))) & 0x7FFFFFFF,
    )
    correct = []
    for seq in seqs:
        text = tokenizer.decode(seq[prefix_len:])
        if train_task.reward(conversation, text) > 0:
            correct.append(seq)
    p_i = len(correct) / max(n_pilot, 1)
    if not correct or args.gvm_grad_norm == "off":
        # G_i is undefined with no accepted sample; Proposition 1 sends n_i to 0.
        return p_i, (1.0 if (correct and args.gvm_grad_norm == "off") else 0.0)

    # ||grad log P(y|x)|| wrt the token embedding, averaged over accepted samples.
    # Mirrors em/stage_2_calc_acceptRates_grads.py, which uses embed_tokens and
    # sums (not averages) the token log-probs before differentiating.
    model.train()
    emb = model.transformer.wte.weight
    prev = emb.requires_grad
    emb.requires_grad_(True)
    norms = []
    # This runs inside get_step_batches, which is @torch.no_grad(); without
    # re-enabling grad the forward produces no graph and autograd.grad raises.
    with torch.enable_grad():
        for seq in correct:
            ids = torch.tensor(seq, dtype=torch.long, device=device)[None, :]
            inp, tgt = ids[:, :-1], ids[:, 1:].clone()
            tgt[:, : prefix_len - 1] = -1                  # score only the completion
            nll = model(inp, tgt, loss_reduction='sum')
            g, = torch.autograd.grad(nll, [emb], retain_graph=False)
            norms.append(float(g.norm(p=2).item()))
            model.zero_grad(set_to_none=True)
    emb.requires_grad_(prev)
    return p_i, float(np.mean(norms)) if norms else 0.0

vip_gp = None
if args.allocator == "vip":
    print0(f"VIP: embedding {len(rank_indices)} prompts for the GP prior...")
    vip_gp = PromptSuccessGP(
        prompt_embeddings(rank_indices),
        bandwidth=args.vip_bandwidth if args.vip_bandwidth > 0 else None,
        eps=args.vip_eps,
        reward_range=(0.0, 1.0),      # both tasks give 0/1 rewards, not -1/+1
    )
    print0(f"VIP: GP ready, bandwidth={vip_gp.h:.4f}")

def allocate(indices, step):
    """Return (n_per_prompt, metrics) for this step's prompts, on this rank."""
    B = len(indices)
    step_budget = B * args.num_samples    # exactly what uniform would spend
    alloc_max = args.alloc_max if args.alloc_max > 0 else step_budget
    if args.allocator == "uniform":
        n = uniform_allocation(B, step_budget, lo=alloc_min, hi=alloc_max)
        return n, {**allocation_stats(n), **rollout_accounting(n, 0)}
    if args.allocator == "gvm":
        p, G, pilot = [], [], []
        for idx in indices:
            pi, Gi = gvm_pilot(idx, args.gvm_pilot_samples, step)
            p.append(pi); G.append(Gi); pilot.append(args.gvm_pilot_samples)
        n = gvm_allocation(p, G, step_budget, alpha=args.gvm_alpha, beta=args.gvm_beta,
                           lo=alloc_min, hi=alloc_max)
        m = {**allocation_stats(n, p), **rollout_accounting(n, pilot),
             "alloc/G_mean": float(np.mean(G)), "alloc/G_max": float(np.max(G))}
        return n, m
    # vip
    local = np.array([pool_pos[int(i)] for i in indices])   # exact position in the GP pool
    p_hat = vip_gp.predict(local)
    n = vip_allocation(p_hat, step_budget, lo=max(alloc_min, 3), hi=alloc_max)
    m = {**allocation_stats(n, p_hat), **rollout_accounting(n, 0)}
    return n, m

# How many rows went through the extra scoring forward TPO needs, tracked for the
# same reason rollout_accounting tracks GVM's pilot rollouts: it is real compute
# the rollout budget does not account for (~1 extra forward per rollout).
tpo_forward_rows = 0

@torch.no_grad()   # also reached from the multi-epoch refresh, which is NOT no_grad
def sequence_logprobs(inputs, targets):
    """Sum of log pi_theta(y_t | ...) over the scored tokens, one scalar per row.

    This is ell in nanochat/tpo.py. It deliberately re-runs the training forward
    rather than reusing the sampler's logits: the Engine samples with a KV cache
    and applies temperature/top-k, so its probabilities are not pi_theta, and any
    drift between the two would show up as a wrong p_old rather than as an error.
    """
    global tpo_forward_rows
    model.train()   # the mode whose gradient we later take; no dropout either way
    out = []
    for b0 in range(0, inputs.size(0), args.device_batch_size):
        inp = inputs[b0:b0 + args.device_batch_size]
        tgt = targets[b0:b0 + args.device_batch_size]
        # cross_entropy(reduction='none') is exactly 0 at ignore_index positions,
        # so summing over T already restricts to the completion.
        nll = model(inp, tgt, loss_reduction='none').view_as(inp)
        out.append(-nll.sum(dim=-1))
    tpo_forward_rows += int(inputs.size(0))
    return torch.cat(out)

def tpo_coefficients(inputs, targets, rewards, target=None):
    """(coef, q, diagnostics) for one group, ready to multiply a row's log-prob sum.

    With --tpo-logp mean the group softmax is taken over length-normalised
    sequence log-probs, so the same 1/T factor has to ride along on the
    coefficient for the surrogate to keep matching the objective.
    """
    ell = sequence_logprobs(inputs, targets)
    if args.tpo_logp == "mean":
        scale = 1.0 / (targets >= 0).sum(dim=-1).clamp(min=1).float()
    else:
        scale = torch.ones_like(ell)
    w, q, diag = tpo_weights(ell * scale, rewards, eta=args.tpo_eta,
                             anchor=bool(args.tpo_anchor), target=target)
    return w * scale, q, diag

@torch.no_grad()
def rollout_one(example_idx, n_rollouts, step, assistant_end):
    """Draw n_rollouts for a single prompt and build its training tensors."""
    conversation = train_task[int(example_idx)]
    # Keep the <|assistant_start|> but drop the reference answer, priming a completion.
    tokens = tokenizer.render_for_completion(conversation)
    prefix_length = len(tokens)

    model.eval()
    seqs, masks = [], []
    # Chunk by device_batch_size to bound memory; n_rollouts need not divide it.
    remaining = int(n_rollouts)
    chunk_idx = 0
    while remaining > 0:
        take = min(remaining, args.device_batch_size)
        seed = hash((step, int(example_idx), chunk_idx)) & 0x7FFFFFFF
        s_batch, m_batch = engine.generate_batch(
            tokens, num_samples=take, max_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k, seed=seed,
        )
        seqs.extend(s_batch); masks.extend(m_batch)
        remaining -= take; chunk_idx += 1

    rewards = [train_task.reward(conversation, tokenizer.decode(s[prefix_length:])) for s in seqs]

    max_length = max(len(s) for s in seqs)
    padded = [s + [assistant_end] * (max_length - len(s)) for s in seqs]
    padded_masks = [m + [0] * (max_length - len(m)) for m in masks]
    ids = torch.tensor(padded, dtype=torch.long, device=device)
    mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
    inputs = ids[:, :-1]
    targets = ids[:, 1:].clone()
    targets[mask_ids[:, 1:] == 0] = -1   # -1 is the ignore index
    rewards_t = torch.tensor(rewards, dtype=torch.float, device=device)
    # Dr. GRPO style: subtract the mean, no std normalisation.
    advantages = rewards_t - rewards_t.mean()
    # TPO replaces that scalar advantage with w = q - p^theta (see nanochat/tpo.py).
    # Both are computed regardless so the reward/advantage logging is identical
    # across objectives; only one of them is read by the training loop.
    tpo_coef, tpo_q, tpo_diag = (None, None, {})
    if args.objective == "tpo":
        tpo_coef, tpo_q, tpo_diag = tpo_coefficients(inputs, targets, rewards_t)
    return {
        "idx": int(example_idx),
        "sequences": seqs,
        "inputs": inputs,
        "targets": targets,
        "rewards": rewards_t,
        "advantages": advantages,
        # Realised accept rate, needed both for the inv_np estimator and to tell
        # whether the allocator had any signal to work with.
        "p_hat": float(rewards_t.mean().item()),
        "n": int(n_rollouts),
        "tpo_coef": tpo_coef,
        "tpo_q": tpo_q,
        "tpo_diag": tpo_diag,
    }

@torch.no_grad()
def get_step_batches():
    """Yield one whole step at a time: allocation is a batch-level decision."""
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    cycler = itertools.cycle(rank_indices)
    step = 0
    while True:
        indices = [next(cycler) for _ in range(examples_per_rank)]
        n_alloc, alloc_metrics = allocate(indices, step)
        records = []
        for example_idx, n_i in zip(indices, n_alloc):
            # GVM legitimately assigns zero to prompts it judges uninformative;
            # such a prompt contributes nothing and is simply skipped.
            if int(n_i) <= 0:
                continue
            records.append(rollout_one(example_idx, n_i, step, assistant_end))
        alloc_metrics["alloc/prompts_used"] = float(len(records))
        alloc_metrics["alloc/prompts_skipped"] = float(len(indices) - len(records))
        # Realised accept rates, measured identically for every allocator. The
        # alloc/p_* series is not comparable across allocators -- it is predicted
        # for VIP, pilot-measured for GVM, absent for uniform -- so the decision
        # of whether a run had any difficulty signal must be made on these.
        if records:
            ps = np.array([r["p_hat"] for r in records], dtype=float)
            alloc_metrics.update({
                "realised/p_mean": float(ps.mean()),
                "realised/p_spread": float(ps.std()),
                "realised/p_frac_zero": float((ps <= 0).mean()),
                "realised/p_frac_one": float((ps >= 1).mean()),
                # p=0 or p=1 means every advantage in the group is zero under
                # mean subtraction, so the prompt contributes no gradient at all.
                "realised/p_frac_degenerate": float(((ps <= 0) | (ps >= 1)).mean()),
                "realised/rollouts_wasted": float(
                    sum(r["n"] for r in records if r["p_hat"] <= 0 or r["p_hat"] >= 1)),
            })
        # TPO health. p_max is the one to watch: the group softmax is taken over
        # SUMS of token log-probs, so on long completions it can collapse onto a
        # single sample, leaving q ~ p_old and nothing to redistribute however the
        # rewards fell. weight_l1 = sum|q - p| is the size of the redistribution
        # actually requested, so tpo/frac_inactive counts the groups TPO no-oped
        # on -- which SHOULD include every degenerate group and ideally little else.
        if records and args.objective == "tpo":
            d = [r["tpo_diag"] for r in records]
            l1 = np.array([x["weight_l1"] for x in d])
            pm = np.array([x["p_max"] for x in d])
            alloc_metrics.update({
                "tpo/p_max_mean": float(pm.mean()),
                "tpo/p_max_max": float(pm.max()),
                "tpo/p_entropy_mean": float(np.mean([x["p_entropy"] for x in d])),
                "tpo/weight_l1_mean": float(l1.mean()),
                "tpo/frac_inactive": float((l1 < 1e-6).mean()),
            })
        yield records, alloc_metrics
        step += 1

# -----------------------------------------------------------------------------
# Simple evaluation loop for pass@k on the chosen task
def run_task_eval(task, tokenizer, engine,
    max_examples=None,
    num_samples=1,
    max_completion_tokens=256,
    temperature=0.0,
    top_k=50
):
    """
    Evaluates the task and returns a list of records of evaluation outcomes.
    In a distributed setting, all ranks cooperate but this function will NOT
    do the reduction across ranks. This is the responsibility of the caller.
    Because the evaluation can take a while, this function will yield records one by one.
    """
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)
        # Generate k samples using batched generation inside the Engine
        assert num_samples <= args.device_batch_size # usually this is true. we can add a loop if not...
        generated_token_sequences, masks = engine.generate_batch(
            tokens,
            num_samples=num_samples,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k
        )
        # Check each sample for correctness
        outcomes = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            is_correct = task.reward(conversation, generated_text) > 0
            outcomes.append({
                "is_correct": is_correct
            })
        # A bit bloated because I wanted to do more complex logging at one point.
        record = {
            "idx": idx,
            "outcomes": outcomes,
        }
        yield record

# -----------------------------------------------------------------------------
# Training loop

# Init the optimizer
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)

# Set the initial learning rate as a fraction of the base learning rate
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]

# Learning rate scheduler: simple rampdown to zero over num_steps
def get_lr_multiplier(it):
    lrm = 1.0 - it / num_steps
    return lrm

# Calculate the number of examples each rank handles to achieve the desired examples_per_step
print0(f"Total sequences per step: {args.examples_per_step * args.num_samples}") # total batch size in sequences/step
assert args.examples_per_step % ddp_world_size == 0, "Desired examples per step must be divisible by the number of ranks"
examples_per_rank = args.examples_per_step // ddp_world_size # per GPU
print0(f"Calculated examples per rank: {examples_per_rank}")

# Kick off the training loop
batch_iterator = get_step_batches()
for step in range(num_steps):

    # Evaluate the model once in a while and log to wandb
    if step % args.eval_every == 0:
        model.eval()
        passk = torch.zeros(args.device_batch_size, device=device) # pass@k for k=1..device_batch_size
        records_iter = run_task_eval(val_task, tokenizer, engine, num_samples=args.device_batch_size, max_examples=args.eval_examples, temperature=1.0, max_completion_tokens=args.max_new_tokens)
        records = list(records_iter) # collect all records
        for k in range(1, args.device_batch_size + 1):
            passk[k - 1] = sum(any(o["is_correct"] for o in r["outcomes"][:k]) for r in records)
        num_records = torch.tensor(len(records), dtype=torch.long, device=device)
        if ddp:
            dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
            dist.all_reduce(passk, op=dist.ReduceOp.SUM)
        passk = passk / num_records.item() # normalize by the total number of records
        print_passk = [f"Pass@{k}: {passk[k - 1].item():.4f}" for k in range(1, args.device_batch_size + 1)]
        print0(f"Step {step} | {', '.join(print_passk)}")
        log_passk = {f"pass@{k}": passk[k - 1].item() for k in range(1, args.device_batch_size + 1)}
        wandb_run.log({
            "step": step,
            **log_passk,
        })

    # Forward/Backward on rollouts. One step = one allocation decision over
    # `examples_per_rank` prompts, then n_i rollouts for each of them.
    rewards_list = []
    sequence_lengths = []
    rows_before = tpo_forward_rows
    records, alloc_metrics = next(batch_iterator)

    # How each prompt's contribution is weighted. This is the knob the GVM
    # analysis turns on: the theory (Alg.1 line 8 / Lemma 1) wants equal weight
    # per prompt, while a global token-mean -- what verl does -- lets a prompt's
    # weight grow with the rollouts it was allocated, which is exactly the set
    # GVM deliberately oversamples.
    total_valid = None
    if args.estimator_weight == "per_token":
        total_valid = sum(int((r["targets"] >= 0).sum().item()) for r in records)
        total_valid = max(total_valid, 1)
    inv_np = {}
    if args.estimator_weight == "inv_np":
        raw, toks = {}, {}
        for pos, r in enumerate(records):
            # E[accepted] = n_i * p_i; guard the p_i = 0 case, which contributes
            # no gradient anyway since every advantage in that group is zero.
            raw[pos] = 1.0 / (r["n"] * max(r["p_hat"], 1e-6))
            toks[pos] = max(int((r["targets"] >= 0).sum().item()), 1)
        # Rescale so the total weight over the step is 1, matching what per_prompt
        # and per_token already sum to. Without this the raw 1/(n_i p_i) weights
        # are ~40x larger and switching estimator would silently change the
        # effective learning rate -- the ablation would measure that, not the
        # estimator. Relative weighting across prompts is untouched.
        mass = sum(raw[k] * toks[k] for k in raw) or 1.0
        for k in raw:
            inv_np[k] = raw[k] / mass

    n_prompts = max(len(records), 1)
    # TPO's target is a fixed point of its own update, so a rollout batch can be
    # reused for more gradient epochs with no PPO ratio and no clip: q stays
    # frozen at what the rollout policy saw and only p^theta is recomputed
    # (paper sections 2 and 5.3). The grpo path has no such correction, which is
    # why --tpo-epochs is rejected for it. With tpo_epochs=1 this loop runs once
    # and the update below is exactly what it was before.
    lrm = get_lr_multiplier(step)
    for inner_epoch in range(args.tpo_epochs):
        if inner_epoch > 0:
            # theta moved on the previous epoch, so p^theta -- and with it the
            # coefficient w = q - p^theta -- is stale and has to be remeasured.
            for rec in records:
                rec["tpo_coef"], _, rec["tpo_diag"] = tpo_coefficients(
                    rec["inputs"], rec["targets"], rec["rewards"], target=rec["tpo_q"])
        for example_step, rec in enumerate(records):
            inputs_all, targets_all = rec["inputs"], rec["targets"]
            rewards_all, advantages_all = rec["rewards"], rec["advantages"]
            model.train()
            # n_i need not divide device_batch_size, so the last pass may be ragged.
            total_rows = inputs_all.size(0)
            num_passes = (total_rows + args.device_batch_size - 1) // args.device_batch_size
            for pass_idx in range(num_passes):
                b0 = pass_idx * args.device_batch_size
                b1 = min(b0 + args.device_batch_size, total_rows)
                inputs, targets = inputs_all[b0:b1], targets_all[b0:b1]
                logp = -model(inputs, targets, loss_reduction='none').view_as(inputs)  # (B, T)
                if args.objective == "tpo":
                    # Surrogate for -sum_i q_i log p_i^theta. Its gradient wrt a
                    # sequence log-prob is exactly p_i^theta - q_i, so the group
                    # can be split across passes even though one softmax couples
                    # it. Normalisation is the paper's -- sum over candidates,
                    # mean over prompts -- so there is no token division here and
                    # --estimator-weight does not apply.
                    obj = (logp.sum(dim=-1) * rec["tpo_coef"][b0:b1]).sum() / n_prompts
                else:
                    advantages = advantages_all[b0:b1]
                    obj = (logp * advantages.unsqueeze(-1)).sum()
                    if args.estimator_weight == "per_token":
                        # One global token-mean across the whole step.
                        obj = obj / total_valid
                    elif args.estimator_weight == "inv_np":
                        # Lemma 1's unbiased estimator, averaged over prompts.
                        obj = obj * inv_np[example_step]
                    else:  # per_prompt (nanochat's original behaviour)
                        num_valid = (targets >= 0).sum().clamp(min=1)
                        obj = obj / (num_valid * num_passes * n_prompts)
                # On-policy, so no PPO ratio/clip is needed.
                loss = -obj
                loss.backward()
            if inner_epoch == 0:
                print0(f"Step {step}/{num_steps} | prompt {example_step} (idx {rec['idx']}) | "
                       f"n={rec['n']} p={rec['p_hat']:.3f} | reward {rewards_all.mean().item():.3f}")
                rewards_list.append(rewards_all.mean().item())
                sequence_lengths.extend(len(seq) for seq in rec["sequences"])
        # Update the model parameters. Every rank runs this the same number of
        # times per step -- the optimizer all-reduces gradients inside step(), so
        # a rank that skipped one would hang the others.
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
        optimizer.step()
        model.zero_grad(set_to_none=True)

    # VIP's GP learns from the rollouts we were drawing anyway -- no extra cost.
    # Calibration is logged BEFORE the update, so it scores a genuine prediction
    # rather than the value just fitted to.
    if vip_gp is not None and records:
        local = np.array([pool_pos[r["idx"]] for r in records])
        realised = np.array([r["p_hat"] for r in records])
        alloc_metrics.update(vip_gp.calibration(local, realised))
        vip_gp.update(local, realised)

    # A bunch of logging for how the rollouts went this step.
    # A rank can legitimately end up with nothing: GVM assigns zero budget to
    # prompts it judges uninformative, and a whole batch can be unsolvable. Do
    # NOT `continue` here -- the collectives below are collective, so one rank
    # bailing would desync the all_reduce, and skipping past optimizer.step()
    # would also skip the zero_grad that follows it. Contributing zero gradient
    # is the correct behaviour and happens naturally.
    empty_rank = not rewards_list
    if empty_rank:
        print0(f"Step {step}/{num_steps} | no prompt received budget on this rank")
    mean_reward = sum(rewards_list) / len(rewards_list) if rewards_list else 0.0
    mean_sequence_length = (sum(sequence_lengths) / len(sequence_lengths)) if sequence_lengths else 0.0
    if ddp: # aggregate across ranks
        mean_reward_tensor = torch.tensor(mean_reward, dtype=torch.float, device=device)
        mean_sequence_length_tensor = torch.tensor(mean_sequence_length, dtype=torch.float, device=device)
        dist.all_reduce(mean_reward_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(mean_sequence_length_tensor, op=dist.ReduceOp.AVG)
        mean_reward = mean_reward_tensor.item()
        mean_sequence_length = mean_sequence_length_tensor.item()
    print0(f"Step {step}/{num_steps} | Average reward: {mean_reward} | Average sequence length: {mean_sequence_length:.2f}")
    wandb_run.log({
        "step": step,
        "reward": mean_reward,
        "sequence_length": mean_sequence_length,
        "allocator": args.allocator,
        "objective": args.objective,
        "empty_rank": float(empty_rank),
        # Rows pushed through the extra no-grad forward TPO needs to read p^old
        # (and p^theta again on every extra gradient epoch). Roughly one extra
        # forward per rollout per epoch, and not charged to the rollout budget --
        # same reason rollout_accounting exists for GVM's pilot pass.
        "tpo/scoring_rows": float(tpo_forward_rows - rows_before),
        **alloc_metrics,
    })

    wandb_run.log({
        "step": step,
        "lrm": lrm,
    })

    # Master process saves the model once in a while. Skip first step. Save last step.
    if master_process and ((step > 0 and step % args.save_every == 0) or step == num_steps - 1):
        base_dir = get_base_dir()
        depth = model.config.n_layer
        output_dirname = args.model_tag if args.model_tag else f"d{depth}" # base the model tag on the depth of the base model
        checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", output_dirname)
        model_config_kwargs = model.config.__dict__ # slightly naughty, abusing the simplicity of GPTConfig, TODO nicer
        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            None, # note: we don't bother to save the optimizer state
            {
                "model_config": model_config_kwargs,
            }
        )
        print(f"✅ Saved model checkpoint to {checkpoint_dir}")

wandb_run.finish() # wandb run finish
compute_cleanup()
