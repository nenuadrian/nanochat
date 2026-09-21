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
import time
import itertools
import json
import wandb
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
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
from nanochat.grow import (
    grow_depth, set_learning_rates, parse_schedule, capture_logits, new_layer_indices,
)

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
parser.add_argument("--max-steps", type=int, default=0,
                    help="cap the horizon at N steps (0 = the full --num-epochs pass). The LR "
                         "schedule and any --grow-at fractions are computed against the cap, so a "
                         "short run is a complete run and not a truncated one")
# Batch sizes / sampling
parser.add_argument("--device-batch-size", type=int, default=8, help="max batch size per forward pass")
parser.add_argument("--examples-per-step", type=int, default=16, help="total examples per optimization step across all ranks")
parser.add_argument("--num-samples", type=int, default=16, help="number of samples per example/question")
# Generation
parser.add_argument("--max-new-tokens", type=int, default=None,
                    help="max tokens to generate per sample (default: task-dependent, see below)")
# Rollout dumping for RAFT
parser.add_argument("--dump-rollouts", action="store_true", help="dump generated rollouts to disk for RAFT selection (JSONL)")
parser.add_argument("--dump-rollouts-dir", type=str, default=None, help="override base dir for dumped rollouts (default: <base>/raft_data/<run>)")
parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
parser.add_argument("--top-k", type=int, default=50, help="top-k sampling (0 = disabled)")
# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.2, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for Muon or AdamW transformer matrices")
parser.add_argument("--matrix-optimizer", type=str, default="muon", choices=["muon", "adamw", "sophia"], help="default optimizer for all transformer blocks")
parser.add_argument("--layer-optimizers", type=str, default="", help="per-block matrix optimizer layout: sophia-middle, or e.g. muon*4,sophia*12,adamw*4")
parser.add_argument("--sophia-lr", type=float, default=1e-4, help="Sophia-G learning rate for Sophia-assigned blocks")
parser.add_argument("--sophia-rho", type=float, default=0.04, help="Sophia-G clipping parameter rho")
parser.add_argument("--sophia-hessian-update-interval", type=int, default=10, help="refresh Sophia sampled-label GNB curvature every N optimizer steps")
parser.add_argument("--sophia-batch-size", type=int, default=-1, help="tokens in the Sophia curvature batch; -1 uses the sampled rollout batch")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay for transformer matrix parameters")
parser.add_argument("--init-lr-frac", type=float, default=0.05, help="initial LR as fraction of base LR")
# Evaluation / checkpointing
parser.add_argument("--eval-every", type=int, default=60, help="evaluate pass@k every N steps (-1 = disable)")
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
parser.add_argument("--save-every", type=int, default=60, help="save checkpoint every N steps (-1 = disable)")
parser.add_argument("--output-tag", type=str, default=None,
                    help="subdirectory to write checkpoints into. Defaults to --model-tag, else "
                         "d<initial depth>. Growth changes the depth mid-run, so without this "
                         "a growing run would scatter checkpoints across d20/, d24/, ...")
parser.add_argument("--seed", type=int, default=0,
                    help="mixes into the rollout sampling seeds and shuffles the prompt order, so "
                         "the same config can be run as several seeds. 0 reproduces the original "
                         "(previously fixed, and therefore unseedable) rollout stream exactly")
# --- capacity growth -----------------------------------------------------
# Insert transformer blocks mid-RL. See nanochat/grow.py; the blocks are identity
# at birth (zeroed c_proj), so reward is continuous across the seam by construction.
parser.add_argument("--grow-at", type=str, default="",
                    help="when to grow, comma separated. Values <1 are fractions of the run "
                         "(e.g. '0.33,0.66'), whole numbers are absolute steps. Empty = never grow")
parser.add_argument("--grow-layers", type=int, default=4, help="blocks inserted per growth event")
parser.add_argument("--grow-position", type=str, default="middle",
                    help="middle|start|end|<int>. middle by default: arXiv:2607.01232 finds RL's "
                         "high-contribution layers concentrate mid-stack")
parser.add_argument("--grow-init", type=str, default="copy", choices=["copy", "fresh", "random"],
                    help="copy: clone the neighbour's features, zero its output projections "
                         "(LLaMA Pro, function preserving); fresh: random features, still zeroed "
                         "projections (also function preserving); random: projections random too, "
                         "NOT function preserving -- the arm that tests reward dip-and-recover")
parser.add_argument("--grow-ve", type=int, default=0,
                    help="give grown layers a value embedding. Off by default: each VE table is "
                         "vocab*kv_dim (16.8M on d20narrow) of sparse-gradient parameters that RL "
                         "is poorly placed to train")
parser.add_argument("--grow-lr-mult", type=float, default=50.0,
                    help="learning rate multiplier for the grown parameters. At the base RL rate a "
                         "Muon step moves a matrix ~1e-3 in spectral norm against a trained c_proj "
                         "of ~5, so grown layers would never leave identity and the run would "
                         "measure nothing. Set 1.0 to measure exactly that null")
parser.add_argument("--grow-warmup", type=int, default=5,
                    help="steps to ramp the grown groups' LR. Muon's step size is independent of "
                         "gradient magnitude, so without this its first update hits a zero matrix "
                         "at full size")
parser.add_argument("--lr-bump-at", type=str, default="",
                    help="raise the learning rate of the EXISTING parameters at these points "
                         "(same format as --grow-at). This is the control arm for growth: it "
                         "separates 'the extra capacity helped' from 'you turned the learning "
                         "rate up a third of the way in'")
parser.add_argument("--lr-bump-mult", type=float, default=1.0,
                    help="multiplier applied at each --lr-bump-at point")
parser.add_argument("--grow-eval-radius", type=int, default=3,
                    help="force evaluations at g-1 .. g+N around each growth step. A transient you "
                         "sample every --eval-every steps is a transient you cannot characterise")
parser.add_argument("--grow-verify", type=int, default=1,
                    help="log max|dlogits| on a fixed probe batch across each growth event. Should "
                         "be ~0 for copy/fresh; this is the runtime form of the identity test")
# --- diagnostics ---------------------------------------------------------
parser.add_argument("--diag-every", type=int, default=0,
                    help="every N steps, measure policy entropy, ReDo dormant-unit fraction, "
                         "residual-stream effective rank and per-layer spectral norms on a fixed "
                         "probe batch. 0 = off, which leaves the step cost exactly as it was")
args = parser.parse_args()
assert args.grow_lr_mult > 0, "--grow-lr-mult must be positive"
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

def warn_if_checkpoint_dir_collides():
    """The default checkpoint dirname is derived from depth alone, so two runs that differ
    only in --objective or --task write the same files and the second silently wins. Check
    the newest meta already there and say so loudly; --output-tag is the fix."""
    import glob, json
    dirname = args.output_tag or args.model_tag or f"d{model.config.n_layer}"
    d = os.path.join(get_base_dir(), "chatrl_checkpoints", dirname)
    metas = sorted(glob.glob(os.path.join(d, "meta_*.json")))
    if not metas:
        return
    try:
        prev = json.load(open(metas[-1])).get("user_config") or {}
    except (OSError, json.JSONDecodeError):
        return
    clashes = {k: (prev.get(k), getattr(args, k)) for k in ("objective", "task")
               if prev.get(k) is not None and prev.get(k) != getattr(args, k)}
    if clashes:
        desc = ", ".join(f"{k}: {was} -> {now}" for k, (was, now) in clashes.items())
        print0(f"WARNING: {d} already holds checkpoints from a different run ({desc}). "
               f"They will be OVERWRITTEN. Pass --output-tag to keep the arms separate.")
    elif prev:
        print0(f"NOTE: reusing checkpoint dir {d} (previous run: {prev.get('run')})")
warn_if_checkpoint_dir_collides()
num_steps = (len(train_task) // args.examples_per_step) * args.num_epochs
if args.max_steps > 0:
    num_steps = min(num_steps, args.max_steps)
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
# Rollout seeds were previously a pure function of (step, prompt, chunk), i.e. fixed
# across runs, so repeating a config gave a bit-identical run and there was no way to
# get a second seed out of it. --seed mixes in here; seed 0 XORs with 0 and therefore
# reproduces the original stream exactly.
SEED_MIX = 0x9E3779B97F4A7C15
def rollout_seed(*parts):
    return (hash(parts) ^ (args.seed * SEED_MIX)) & 0x7FFFFFFF

rank_indices = list(range(ddp_rank, len(train_task), ddp_world_size))
if args.seed != 0:
    np.random.default_rng(args.seed + 9973 * ddp_rank).shuffle(rank_indices)
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
        seed=rollout_seed(-1, step, int(example_idx)),   # -1 marks the pilot stream
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
        seed = rollout_seed(step, int(example_idx), chunk_idx)
        s_batch, m_batch = engine.generate_batch(
            tokens, num_samples=take, max_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k, seed=seed,
        )
        seqs.extend(s_batch); masks.extend(m_batch)
        remaining -= take; chunk_idx += 1

    rewards = [train_task.reward(conversation, tokenizer.decode(s[prefix_length:])) for s in seqs]

    # Optional: dump rollouts for RAFT post-processing. Each line is a JSON object
    # with prompt id, prompt tokens, sequence tokens, reward, seed info and step.
    if args.dump_rollouts:
        base = args.dump_rollouts_dir or os.path.join(get_base_dir(), "raft_data", args.run)
        os.makedirs(os.path.join(base, str(step)), exist_ok=True)
        fname = os.path.join(base, str(step), f"rollouts_rank{ddp_rank}.jsonl")
        try:
            with open(fname, "a") as fh:
                for si, seq in enumerate(seqs):
                    item = {
                        "prompt_id": int(example_idx),
                        "prompt_tokens": list(map(int, tokens)),
                        "sequence_tokens": list(map(int, seq)),
                        "reward": float(rewards[si]),
                        "step": int(step),
                        "sample_index": int(si),
                        "seed": int(rollout_seed(step, int(example_idx), si)),
                    }
                    fh.write(json.dumps(item) + "\n")
        except Exception:
            # Do not crash training for IO errors; just log.
            print0(f"Warning: failed to write rollouts to {fname}")

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
    top_k=50,
    seed=1234,
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
        # Engine.generate re-seeds a fresh Generator on every call, so leaving the seed
        # at its default makes every example share one random stream: row i always draws
        # the i-th variate of the same sequence. That turns pass@1 into a fixed-quantile
        # probe rather than a sample (it reliably picks tail tokens, scoring below chance),
        # and correlates the rows so pass@k plateaus. Vary the seed per example.
        generated_token_sequences, masks = engine.generate_batch(
            tokens,
            num_samples=num_samples,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=seed + idx,
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
sophia_batch_size = args.examples_per_step if args.sophia_batch_size == -1 else args.sophia_batch_size
if sophia_batch_size <= 0:
    raise ValueError("--sophia-batch-size must be positive or -1 for automatic resolution")
if args.matrix_optimizer == "sophia" or args.layer_optimizers:
    print0(f"Sophia-G curvature batch: dynamic sampled rollout tokens (initial value: {sophia_batch_size})")
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
    matrix_optimizer=args.matrix_optimizer,
    layer_optimizers=args.layer_optimizers,
    sophia_lr=args.sophia_lr,
    sophia_rho=args.sophia_rho,
    sophia_hessian_update_interval=args.sophia_hessian_update_interval,
    sophia_batch_size=sophia_batch_size,
)

# Set the initial learning rate as a fraction of the base learning rate
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]

# Learning rate scheduler: simple rampdown to zero over num_steps
def get_lr_multiplier(it):
    lrm = 1.0 - it / num_steps
    return lrm

# -----------------------------------------------------------------------------
# Capacity growth + diagnostics

initial_depth = model.config.n_layer   # pinned: model.config.n_layer moves when we grow
grow_steps = parse_schedule(args.grow_at, num_steps)
lr_bump_steps = parse_schedule(args.lr_bump_at, num_steps)
if lr_bump_steps:
    print0(f"LR bump schedule: steps {lr_bump_steps} | existing params x{args.lr_bump_mult}")
grow_events = []                       # one info dict per growth event, for logging and meta
grow_rng = torch.Generator(device="cpu").manual_seed(args.seed + 7919)
if grow_steps:
    print0(f"Growth schedule: steps {grow_steps} | +{args.grow_layers} layers at '{args.grow_position}' "
           f"| init={args.grow_init} | ve={'on' if args.grow_ve else 'off'} "
           f"| lr x{args.grow_lr_mult} (warmup {args.grow_warmup})")
# Evaluate on both sides of every growth event, not just on the --eval-every grid.
forced_eval_steps = set()
for g in grow_steps + lr_bump_steps:
    forced_eval_steps.update(x for x in range(g - 1, g + args.grow_eval_radius + 1) if 0 <= x < num_steps)

def build_probe(task, n=4, max_len=128):
    """A small fixed batch of real prompts, for growth verification and diagnostics."""
    pad = tokenizer.encode_special("<|assistant_end|>")
    seqs = [tokenizer.render_for_completion(task[i])[-max_len:] for i in range(min(n, len(task)))]
    width = max(len(x) for x in seqs)
    return torch.tensor([[pad] * (width - len(x)) + x for x in seqs], dtype=torch.long, device=device)

probe_ids = build_probe(val_task) if (args.grow_verify or args.diag_every > 0) else None

@torch.no_grad()
def measure_diagnostics(ids):
    """Plasticity instrumentation, on a fixed probe batch so it is comparable over time.

    Reward alone will not separate these arms; the mechanism the growth literature
    actually claims (Neuroplastic Expansion, ICLR 2025) is plasticity, whose signature
    in LLM RL is entropy collapse plus accumulating dormant units.
    """
    acts, resid, handles = {}, {}, []
    for i, b in enumerate(model.transformer.h):
        def hook(mod, inp, out, i=i):
            acts[i] = torch.relu(out).square().abs().mean(dim=(0, 1))  # the MLP hidden activation
        handles.append(b.mlp.c_fc.register_forward_hook(hook))
    handles.append(model.lm_head.register_forward_hook(lambda m, inp, out: resid.__setitem__("x", inp[0].detach())))
    was_training = model.training
    model.eval()
    try:
        logits = model(ids)
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()

    out = {}
    logprobs = logits.log_softmax(dim=-1)
    per_pos = -(logprobs.exp() * logprobs).sum(-1)          # (B, T)
    # The probe ends at <|assistant_start|>, so the final position is the first token
    # the policy would emit -- on ARC that IS the answer letter. This is the
    # distribution RL collapses; averaging over all positions instead buries it under
    # generic text prediction, which barely moves (measured: 1.33 vs 1.68 on the SFT
    # model, where the decision is near-uniform over 4 choices, ln 4 = 1.386).
    out["diag/entropy_decision"] = float(per_pos[:, -1].mean())
    out["diag/entropy_allpos"] = float(per_pos.mean())
    # ReDo dormancy: unit i is dormant when its mean activation is a negligible
    # share of the layer's mean. Ref: Sokar et al. 2023.
    tau = 0.025
    grown = set(new_layer_indices(grow_events, model.config.n_layer))
    fracs, new_fracs = [], []
    for i, a in acts.items():
        frac = float((a / a.mean().clamp(min=1e-12) <= tau).float().mean())
        fracs.append(frac)
        if i in grown:
            new_fracs.append(frac)
    if fracs:
        out["diag/dormant_frac"] = sum(fracs) / len(fracs)
        out["diag/dormant_frac_max"] = max(fracs)
    if new_fracs:
        out["diag/dormant_frac_new"] = sum(new_fracs) / len(new_fracs)
    # Effective rank (exp of the entropy of the normalised spectrum) of the residual
    # stream entering the lm_head. On CPU: MPS has patchy linalg coverage.
    x = resid["x"].reshape(-1, resid["x"].size(-1)).float().cpu()
    x = x - x.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(x)
    q = sv / sv.sum().clamp(min=1e-12)
    out["diag/effective_rank"] = float(torch.exp(-(q * (q + 1e-12).log()).sum()))
    # Spectral norms: the direct answer to "did the grown capacity ever engage?"
    old_s, new_s = [], []
    for i, b in enumerate(model.transformer.h):
        v = float(torch.linalg.matrix_norm(b.mlp.c_proj.weight.detach().float().cpu(), ord=2))
        (new_s if i in grown else old_s).append(v)
    if old_s:
        out["diag/c_proj_spectral_old"] = sum(old_s) / len(old_s)
    if new_s:
        out["diag/c_proj_spectral_new"] = sum(new_s) / len(new_s)
    return out

@torch.no_grad()
def weight_and_grad_stats():
    """Cheap per-step stats: Frobenius norms of the output projections, split old vs
    grown, and gradient norms per optimizer group. Accumulated on device, one sync."""
    grown = set(new_layer_indices(grow_events, model.config.n_layer))
    old_n, new_n_ = [], []
    for i, b in enumerate(model.transformer.h):
        v = b.mlp.c_proj.weight.detach().float().norm()
        (new_n_ if i in grown else old_n).append(v)
    out = {}
    if old_n:
        out["weights/c_proj_fro_old"] = float(torch.stack(old_n).mean())
    if new_n_:
        out["weights/c_proj_fro_new"] = float(torch.stack(new_n_).mean())
    return out

@torch.no_grad()
def grad_norm_stats():
    total = torch.zeros((), device=device)
    grown = torch.zeros((), device=device)
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue
            sq = p.grad.detach().float().pow(2).sum()
            total += sq
            if group.get("grown_at") is not None:
                grown += sq
    return {"grad/norm_total": float(total.sqrt()), "grad/norm_grown": float(grown.sqrt())}

# Calculate the number of examples each rank handles to achieve the desired examples_per_step
print0(f"Total sequences per step: {args.examples_per_step * args.num_samples}") # total batch size in sequences/step
assert args.examples_per_step % ddp_world_size == 0, "Desired examples per step must be divisible by the number of ranks"
examples_per_rank = args.examples_per_step // ddp_world_size # per GPU
print0(f"Calculated examples per rank: {examples_per_rank}")

# Kick off the training loop
batch_iterator = get_step_batches()
for step in range(num_steps):
    step_t0 = time.time()

    # Evaluate the model once in a while and log to wandb. Growth events are
    # additionally bracketed, so the transient is sampled densely enough to see.
    # --eval-every <= 0 means no evaluation at all; growth bracketing densifies the
    # grid, it does not re-enable a grid the user turned off.
    do_eval = args.eval_every > 0 and (step % args.eval_every == 0 or step in forced_eval_steps)
    if do_eval:
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
        # Sampling diversity. On a measured 140-step GRPO run this fell 0.71 -> 0.37
        # while pass@1 rose only 0.26 -> 0.31: the policy is mostly being sharpened,
        # not made more capable. It is the clearest plasticity signal this task offers.
        log_passk["passk/diversity_gap"] = passk[-1].item() - passk[0].item()
        print0(f"Step {step} | diversity gap (pass@{args.device_batch_size} - pass@1): "
               f"{log_passk['passk/diversity_gap']:.4f}")
        wandb_run.log({
            "step": step,
            **log_passk,
        })

    train_t0 = time.time()   # excludes the eval above: the grown arms get extra forced
                             # evals from the bracket, which would otherwise show up as
                             # the grown model being slower
    # Grow the model, before this step's rollouts are drawn so they come from the
    # grown policy. With --grow-init copy/fresh the inserted blocks are exactly the
    # identity, so nothing about the policy changes here; --grow-verify asserts that
    # rather than assuming it.
    grow_metrics = {}
    if step in lr_bump_steps:
        # Only the pre-existing groups: a grown group already carries --grow-lr-mult,
        # and bumping it too would confound the two knobs.
        bumped = 0
        for group in optimizer.param_groups:
            if group.get("grown_at") is None:
                group["initial_lr"] *= args.lr_bump_mult
                bumped += 1
        print0(f"[lr-bump] step {step}: x{args.lr_bump_mult} on {bumped} existing param groups")
        grow_metrics["lr_bump/event"] = 1.0
    if step in grow_steps:
        before_logits = capture_logits(model, probe_ids) if args.grow_verify else None
        info = grow_depth(
            model, optimizer, args.grow_layers,
            position=args.grow_position, init=args.grow_init, add_ve=bool(args.grow_ve),
            lr_mult=args.grow_lr_mult, warmup=args.grow_warmup, step=step, generator=grow_rng,
        )
        grow_events.append(info)
        grow_metrics = {
            "grow/event": 1.0,
            "grow/n_layer": float(info["n_layer_after"]),
            "grow/total_params_M": info["total_params"] / 1e6,
            "grow/new_params_M": info["new_params"] / 1e6,
        }
        if before_logits is not None:
            after_logits = capture_logits(model, probe_ids)
            delta = float((after_logits - before_logits).abs().max())
            grow_metrics["grow/logit_delta_max"] = delta
            preserving = args.grow_init in ("copy", "fresh")
            print0(f"[grow] max|dlogits| across the seam: {delta:.3e}"
                   + ("  (expected ~0)" if preserving else "  (init=random, perturbation intended)"))
            if preserving and delta > 1e-3:
                print0(f"[grow] WARNING: --grow-init {args.grow_init} is supposed to preserve the "
                       f"function but moved the logits by {delta:.3e}. Treat this run as suspect.")

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

    # Mean -log pi(sampled token) over the step: a Monte-Carlo estimate of the policy's
    # token entropy, free because logp is computed for the objective anyway. Biased by
    # top-k truncation, so it is a trend proxy; --diag-every measures true entropy.
    logp_accum = torch.zeros((), device=device)
    token_accum = torch.zeros((), device=device)

    grad_metrics = {}
    sophia_hessian_refreshed = False
    sophia_curvature_inputs = None
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
                if sophia_curvature_inputs is None:
                    sophia_curvature_inputs = inputs.detach()
                logp = -model(inputs, targets, loss_reduction='none').view_as(inputs)  # (B, T)
                if inner_epoch == 0:
                    with torch.no_grad():
                        # exactly 0 at ignore_index positions, so this sums over scored tokens only
                        logp_accum += logp.detach().sum()
                        token_accum += (targets >= 0).sum()
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
        set_learning_rates(optimizer, step, lrm)
        if inner_epoch == args.tpo_epochs - 1:
            grad_metrics = grad_norm_stats()   # measured on the gradient actually stepped on
        optimizer.step()
        model.zero_grad(set_to_none=True)
        if optimizer.should_update_sophia_hessian():
            # Every rank must participate in the optimizer's gradient all-reduce.
            # If a rank has no rollout data this step, defer the refresh instead of
            # injecting a synthetic batch into the GNB estimate.
            have_inputs = torch.tensor(int(sophia_curvature_inputs is not None), device=device)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(have_inputs, op=dist.ReduceOp.MIN)
            if have_inputs.item():
                optimizer.set_sophia_batch_size(sophia_curvature_inputs.numel() * ddp_world_size)
                curvature_logits = model(sophia_curvature_inputs)
                sampled_targets = torch.distributions.Categorical(logits=curvature_logits).sample()
                curvature_loss = F.cross_entropy(
                    curvature_logits.flatten(0, 1), sampled_targets.flatten(), reduction="mean",
                )
                curvature_loss.backward()
                optimizer.update_sophia_hessian()
                model.zero_grad(set_to_none=True)
                sophia_hessian_refreshed = True

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
    # Entropy proxy, and the capacity/cost bookkeeping that makes the arms comparable
    # on something other than steps: growth raises the per-step cost, so an equal-steps
    # comparison quietly hands the grown arms more compute.
    n_tokens_scored = float(token_accum)
    neg_logp_mean = float(logp_accum) / max(n_tokens_scored, 1.0)
    diag_metrics = {}
    if args.diag_every > 0 and (step % args.diag_every == 0 or step in forced_eval_steps):
        diag_metrics = measure_diagnostics(probe_ids)
        print0(f"Step {step} | diag | entropy {diag_metrics['diag/entropy_decision']:.3f} "
               f"| dormant {diag_metrics['diag/dormant_frac']:.3f} "
               f"| eff_rank {diag_metrics['diag/effective_rank']:.1f} "
               f"| c_proj spec old {diag_metrics.get('diag/c_proj_spectral_old', float('nan')):.3f}"
               + (f" new {diag_metrics['diag/c_proj_spectral_new']:.3f}"
                  if 'diag/c_proj_spectral_new' in diag_metrics else ""))
    now = time.time()
    step_seconds, train_seconds = now - step_t0, now - train_t0
    print0(f"Step {step}/{num_steps} | Average reward: {mean_reward} "
           f"| Average sequence length: {mean_sequence_length:.2f} | {train_seconds:.1f}s")
    wandb_run.log({
        "step": step,
        "reward": mean_reward,
        "sequence_length": mean_sequence_length,
        # -mean log pi over sampled tokens: entropy proxy, higher = less collapsed
        "policy/neg_logp_mean": neg_logp_mean,
        "policy/tokens_scored": n_tokens_scored,
        "model/n_layer": float(model.config.n_layer),
        "model/params_M": sum(p.numel() for p in model.parameters()) / 1e6,
        "model/flops_per_token": float(model.estimate_flops()),
        "cost/step_seconds": step_seconds,     # including evaluation
        "cost/train_seconds": train_seconds,   # rollouts + backward only; use this to match compute
        **grad_metrics,
        **weight_and_grad_stats(),
        **grow_metrics,
        **diag_metrics,
        "allocator": args.allocator,
        "objective": args.objective,
        "empty_rank": float(empty_rank),
        "sophia/hessian_refreshed": float(sophia_hessian_refreshed),
        # Rows pushed through the extra no-grad forward TPO needs to read p^old
        # (and p^theta again on every extra gradient epoch). Roughly one extra
        # forward per rollout per epoch, and not charged to the rollout budget --
        # same reason rollout_accounting exists for GVM's pilot pass.
        "tpo/scoring_rows": float(tpo_forward_rows - rows_before),
        **(optimizer.sophia_metrics() if optimizer.has_sophia() else {}),
        **alloc_metrics,
    })

    wandb_run.log({
        "step": step,
        "lrm": lrm,
    })

    # Master process saves the model once in a while. Skip first step. Save last step.
    if master_process and args.save_every > 0 and ((step > 0 and step % args.save_every == 0) or step == num_steps - 1):
        base_dir = get_base_dir()
        # Pinned to the depth the run STARTED at: model.config.n_layer moves when the
        # model grows, which would scatter one run's checkpoints across several dirs.
        output_dirname = args.output_tag or args.model_tag or f"d{initial_depth}"
        checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", output_dirname)
        model_config_kwargs = model.config.__dict__ # slightly naughty, abusing the simplicity of GPTConfig, TODO nicer
        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            None, # note: we don't bother to save the optimizer state
            {
                "model_config": model_config_kwargs,
                "grow_events": grow_events,
                "initial_depth": initial_depth,
                "user_config": user_config, # so a checkpoint records which arm produced it
            }
        )
        print(f"✅ Saved model checkpoint to {checkpoint_dir}")

wandb_run.finish() # wandb run finish
compute_cleanup()
