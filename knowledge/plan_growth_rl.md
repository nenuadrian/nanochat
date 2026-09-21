# Growing the model during RL — plan, hazards, and what to expect

Status: **implemented** (`nanochat/grow.py`, `tests/test_grow.py`, `runs/growth_rl.sh`,
flags in `scripts/chat_rl.py`). Base model: `d20narrow` SFT (n_layer=20, n_embd=512,
n_head=8, window_pattern="L", 264M params). Hardware: Mac Mini, MPS, single rank.

Settled: GRPO substrate, 2 epochs (280 steps), +4 blocks mid-stack at the 1/3 mark,
`--grow-lr-mult 50`, no value embeddings on grown layers.

Verified on the real checkpoint: growth is bitwise function-preserving
(`max|dlogits| = 0.000e+00`), generation still works after layer renumbering, grown
checkpoints round-trip through save/load unchanged, and grown layers reach spectral
norm 0.52 in 5 steps at `lr_mult=50` against a trained layer's 5.59 — where the base
RL rate would have reached 5e-3. Default flags reproduce the previous behaviour
exactly, including `--seed 0` reproducing the original (previously unseedable)
rollout stream.

Deviation from this doc, worth knowing: `--grow-ve` now works for **any** layer count,
odd included. Pinning `ve_layers` in the config bypasses the `has_ve` parity formula
entirely, so the "grow by even numbers only" rule in §3 is no longer a constraint —
the remapping is explicit. Two bugs not anticipated here and caught during
implementation: `attn.layer_idx` indexes the KV cache and must be renumbered after a
mid-stack insertion (otherwise inference reads the wrong cache slots), and new blocks
must be initialised on the generator's device before being moved (a CPU generator
cannot fill an MPS tensor, which crashed the `--grow-init random` arm).

---

## 0. TL;DR

Three things, in order of how much they should change your plan.

1. **This architecture gives you function-preserving depth growth for free.** A freshly
   initialised `Block` is *exactly* the identity map, because `init_weights` zeros both
   `attn.c_proj` and `mlp.c_proj` (`nanochat/gpt.py:230`) and `Block.forward` only ever
   adds their outputs to the residual stream. No Net2Net machinery needed. This is the
   same trick LLaMA Pro uses.

2. **At the default RL learning rate, a grown layer cannot leave identity.** Measured from
   your SFT checkpoint, a trained `mlp.c_proj` has spectral norm ≈ 5.4. Muon's update has
   spectral norm ≈ `lr` per step, and the effective RL LR is
   `matrix_lr(0.02) × init_lr_frac(0.05) = 1e-3`, decaying linearly to zero. Grow at step
   47 of 140 and the *best case* (every step perfectly aligned) reaches ≈ 0.03 — **0.6% of
   a trained layer**. Realistically it's a random walk, so far less. Without a fix, the
   experiment measures nothing and you get a flat null result for a boring reason.

3. **The literature's mechanism is plasticity, not capacity** — and for LLM RL
   specifically, the capacity story looks weak. Reframing the hypothesis around plasticity
   is what makes this worth running.

Verdict on the stated hypothesis ("loss spikes, then recovers, then ends better"): with
identity init the spike is *mathematically zero*. If you see a spike, it's a bug — most
likely one of the five silent breakages in §3. To get a genuine spike-and-recover you have
to deliberately use non-preserving init, which is worth including as its own arm.

---

## 1. What the research actually says

Your survey is accurate on the RL-from-scratch side. Two papers you didn't have change the
picture for *LLM* RL:

**[Is One Layer Enough?](https://arxiv.org/html/2607.01232)** (arXiv:2607.01232) — the most
directly relevant result, and it cuts against the capacity hypothesis. Across 7 models
(Qwen3 1.7/4/8B, Qwen2.5 1.5/3B, DS-Distill-7B), 3 algorithms (GRPO, Dr. GRPO, GiGPO) and
math/code/agentic tasks, training a **single well-chosen layer recovers 100–114% of
full-parameter RL gains** (1.14 on Qwen3-1.7B). Training the top-10 layers beats
full-parameter training (69.1% vs 66.4% on Qwen3-8B). Layer rankings are stable across
datasets (Spearman ρ=0.76) and are a property of the model, not the task.

Two implications:
- RL post-training is not capacity-bound. It's a low-dimensional reweighting of capability
  the model already has. Adding capacity is attacking a bottleneck that may not exist.
- **High-contribution layers concentrate in the middle of the stack.** If you grow, grow in
  the middle. This is the strongest available prior on *where*.

**[Can Scale Save Us From Plasticity Loss in LLMs?](https://arxiv.org/html/2606.24752v1)**
(arXiv:2606.24752) — 5M→314M params, 8 scales, continual multilingual. Answer: no. Onset of
plasticity loss follows a sublinear power law (`T = 1.3e-5 · P^0.8269`), so scale *delays*
proportionally but never prevents. Plasticity loss appears even in stationary training.
They fail to find a "smoking gun" diagnostic; suggested mitigations are dormant-unit
reinit (ReDo, Continual Backprop), weight decay, attention-head reinit.

That's the honest framing for your experiment: **you are not testing whether more capacity
helps. You are testing whether injecting fresh, unsaturated parameters mid-RL restores
plasticity** — the same mechanism [Neuroplastic Expansion](https://proceedings.iclr.cc/paper_files/paper/2025/file/e094d1e30e88949ed466067aef6be546-Paper-Conference.pdf)
(ICLR 2025) identifies, where growth beats Reset/ReDo/LayerNorm/Plasticity-Injection and
keeps the active-neuron ratio at 80–100% vs ~50–60%. In LLM RL the plasticity pathology has
a name and a well-known signature: **entropy collapse**. That's your primary metric, not reward.

Two corrections to your survey worth noting:
- **GrowNN** ([arXiv:2506.11706](https://ar5iv.labs.arxiv.org/html/2506.11706)): baselines are
  *step*-matched ("at least the same budget"), not FLOP-matched; growth is scheduled by BOHB
  fidelity rungs, not a clean fixed interval; and the paper contains **no analysis of
  post-growth dips and no discussion of optimizer-state handling**. Both are gaps you'd be
  filling, not replicating. Also note static depth-1 beat growth on MiniHack early on.
- The "no public work on capacity growth during RLHF" gap claim holds up. Depth up-scaling
  is well established for *pretraining/continual* learning
  ([LLaMA Pro](https://arxiv.org/html/2401.02415v1), SOLAR, MIDAS/LIDAS), never mid-RL.

---

## 2. The growth operator

### Depth (v1)

Insert `k` blocks in the middle, LLaMA-Pro style:

- **Copy** the weights of the block at the insertion point (gives the new layer sane
  q/k/v features from step one instead of random projections that RL will never train).
- **Zero** `attn.c_proj` and `mlp.c_proj` → exact identity.
- Set the new `resid_lambdas` entries to **1.0** and `x0_lambdas` to **0.0**. Required:
  the trunk applies `x = resid[i]*x + x0[i]*x0` *before* every block
  (`nanochat/gpt.py:497` region), and the init formula at `gpt.py:238` gives 1.15→1.05,
  not 1.0. Your trained values have drifted a long way (x0_lambdas range −1.24 to 3.29),
  so copying a neighbour's value here would *not* be identity.
- **Grow by even `k` only.** See §3.

Cost for `k=4` on d20narrow: +12.6M trunk params (+20% trunk FLOPs) and — if you let new
layers get value embeddings — **+33.6M more**, because each VE table is 32768×512 = 16.8M.
Value embeddings are already 167.8M of the model's 264M. Default should be
`--grow-ve=off`: sparse embedding-table gradients are the last thing RL can train.

### MLP width (v2, and cleaner than it looks)

Underrated option for this architecture: grow `c_fc` rows and `c_proj` columns of existing
blocks, zero-initialising the new `c_proj` columns. Exactly function-preserving, and it
touches **nothing** else — no residual-stream width change, so no RMSNorm shift, no rotary
recompute, no embedding/lm_head resize, no VE. `relu²` is elementwise. Cheapest honest
width experiment available here.

### What not to do in v1

Residual-stream width (`n_embd`) growth requires resizing wte, lm_head, every matrix, VE
tables and rotary `head_dim`, and interacts badly with Muon (exact channel duplication
gives rank-deficient gradients, and orthogonalisation does something arbitrary with that).
Head-count growth changes `head_dim = n_embd // n_head` and breaks rotary. Skip both.

---

## 3. Five things that will silently break (this is the dangerous section)

None of these throw an exception. All of them change the function while looking fine.

| # | Hazard | Where | Fix |
|---|---|---|---|
| 1 | **VE parity flip.** `has_ve(i, n) = i%2 == (n-1)%2`. Grow by an *odd* number and the set of layers with value embeddings inverts completely — every existing layer loses or gains a VE. | `gpt.py:53`, `gpt.py:192` | Grow by even `k` **and** remap existing `value_embeds` ModuleDict keys when inserting mid-stack (old layer 11 → key "15"). Never rebuild the dict from the formula. |
| 2 | **Backout layer shifts.** `backout_layer = n_layer // 2`, and the final hidden state is `x − backout_lambda · x_backout` with `backout_lambda` = 0.49 in your checkpoint — not a small term. Insert 4 layers mid-stack and this silently switches from old-layer-10's output to old-layer-9's. | `gpt.py:497` | Pin as an explicit `GPTConfig` field. |
| 3 | **Final-layer window override.** `window_sizes[-1] = (long, 0)` forces the last layer to full context. Append at the end and the old last layer reverts to its pattern char. | `gpt.py:313` | Moot for d20narrow (`window_pattern="L"`, all layers long) but pin per-layer windows in config anyway. |
| 4 | **Muon momentum rows misalign.** Muon group state lives on `params[0]` as a `(chunk_size, *shape)` stack, indexed positionally (`optim.py:379`, `optim.py:302`). Params are collected in module order (`gpt.py:447`), so a mid-stack insertion shifts every later param's index and momentum rows attach to the wrong matrices. | `optim.py` | See §4 — put new params in *new* groups. |
| 5 | **Checkpoints won't round-trip.** `build_model` reconstructs the model from `meta["model_config"]` and re-derives VE set, backout and windows from `n_layer`. Any pinning you do in memory is lost on reload, so a grown checkpoint silently loads as a *different function*. | `checkpoint_manager.py:75-108` | Every pinned quantity must become a `GPTConfig` field. `_patch_missing_config_keys` already handles the back-compat path. |

Also cosmetic but will wreck your run organisation: `chat_rl.py:700` derives the checkpoint
directory from `model.config.n_layer`, so a growing run scatters checkpoints across
`d20/`, `d24/`, `d28/`. Pin the output dirname.

**The single highest-value thing to build is one unit test**: run a forward pass on fixed
tokens, grow, run again, assert logits match to fp tolerance. That one assertion catches
hazards 1–3 at once, and it's how you'll know a post-growth reward dip is real.

---

## 4. Optimizer surgery

The model is not wrapped in DDP and not compiled in `chat_rl.py` — the optimizer does its
own all-reduce inside `step()`. That removes most of the usual pain.

**Do not call `setup_optimizer()` again after growth.** A fresh optimizer discards all
AdamW moments and all Muon momentum for *every* parameter. That perturbation is far larger
than the growth itself, and you'd be measuring an optimizer reset.

Recommended design — **new params go into brand-new param groups**:

- New Muon groups (same shape as existing ones, separate `dict`). State is keyed on the new
  group's `params[0]`, so existing groups are untouched: no buffer resizing, no row
  realignment, hazard #4 disappears.
- Free bonus: a separate group is exactly where you set a different LR for new params (§5).
- New `value_embeds` (if enabled) append safely to the existing AdamW group — AdamW state is
  per-parameter, so appending never disturbs anything.
- `resid_lambdas` / `x0_lambdas` are single `(n_layer,)` tensors that must be *replaced*,
  which orphans their AdamW state. They're tiny: copy `exp_avg`/`exp_avg_sq` into zero-padded
  new tensors, swap the entry in the group's `params` list, move `self.state[old] →
  self.state[new]`. ~15 lines.
- Expect one `torch.compile` recompile per new shape (`dynamic=False` on both fused kernels).
  One-time cost per growth event, not a correctness issue.

**Single-rank only in v1.** With `world_size > 1`, growing a group changes
`chunk_size = ceil(K/N)`, which shifts every rank's ownership boundary and requires
resharding the momentum stack. You're on one machine; don't pay for this yet.

---

## 5. The learning-rate problem — read this before writing any code

From your checkpoint, trained spectral norms: `attn.c_proj` ≈ 4.7–5.0, `mlp.c_proj` ≈ 5.4,
`mlp.c_fc` ≈ 10.6. A grown layer starts at 0.

Muon's step has spectral norm ≈ `group_lr × max(1, m/n)^0.5`, essentially independent of
gradient magnitude. With `matrix_lr=0.02` and `init_lr_frac=0.05`, peak effective LR is
**1e-3**, decayed linearly to zero by `1 - step/num_steps`.

Grow at step 47 of 140 → 93 steps remain at mean multiplier ≈ 0.33 → **best-case** reach
`93 × 1e-3 × 0.33 ≈ 0.031` against a target of ~5. That is **0.6%** of a trained layer, and
only if every single update points the same way.

Three ways out, and you should probably do all three:

1. `--grow-lr-mult` (default ~50) on the new param groups. This is the knob that decides
   whether the experiment measures anything, so it deserves its own ablation sweep — it is
   entirely possible that *LR*, not capacity, explains any effect you see, and you want to
   be able to say so.
2. Per-growth-event LR schedule: warmup over ~5 steps (avoids the Muon fixed-size first
   step hitting a zero matrix), then decay on the run's global schedule.
3. Longer horizon: `--num-epochs 2` or 3. 140 steps is not enough room for a growth event
   to pay off.

**Corollary for interpreting results:** if you run at default LR and see no difference, that
is *not* evidence about capacity. Log `||new_c_proj||_2` every step. If it stays near zero,
the arm is a no-op and you must report it as such.

---

## 6. Experiment design

### Arms

| Arm | Description | Question it answers |
|---|---|---|
| `static` | d20narrow, full horizon | baseline + noise floor |
| `grow@0` | grow to d24 at step 0, then train | is it the capacity, or the *timing*? |
| `grow@⅓` | grow +4 mid-stack at step ⅓ | the main treatment |
| `grow@⅓,⅔` | +2 at ⅓, +2 at ⅔ | does staged beat single-shot? |
| `grow-random` | same schedule, **non**-preserving init (no zero `c_proj`) | your literal spike-then-recover hypothesis |
| `lr-control` | `static`, but `matrix_lr × grow-lr-mult` from step ⅓ | **critical** — separates growth from "you raised the LR" |

That last arm is the one most likely to kill a positive result, which is exactly why it
should be in the first batch rather than added later.

### Compute matching

Growth raises per-step cost, so report the x-axis four ways: steps, rollouts, wall-clock,
cumulative FLOPs (`model.estimate_flops()` already exists and updates with `n_layer`).
**Equal wall-clock is the fair comparison and the one that flatters growth least.** Say so
explicitly in whatever you write up.

### Metrics — the actual deliverable

Reward alone will not resolve this at 3 seeds. Log per step:

- **Policy entropy** of sampled completions — the plasticity signal. Primary metric.
- **`||c_proj||_2` per layer, new layers broken out** — did the new capacity ever engage?
- **Dormant-unit fraction** (ReDo-style: MLP hidden units with near-zero mean activation),
  per layer. This is NE's headline diagnostic.
- **Effective rank** of the residual stream.
- **Gradient norm per param group**, new vs old.
- **KL from the pre-growth policy** — measures the real size of the perturbation.
- **Per-layer ablation at eval** (zero a layer's `c_proj`, measure reward drop) — the direct
  port of "Is One Layer Enough?"'s layer-contribution metric. Run it once at the end: it
  tells you whether grown layers ended up mattering, and whether they landed in the
  high-contribution middle band.

### Evaluation cadence

`--eval-every 60` gives ~3 points in 140 steps — useless for seeing a dip. Use
`--eval-every 10` with `--eval-examples 200`, and **force an eval at steps g−1, g, g+1**
around every growth event. You cannot characterise a transient you sample three times.

### Seeds and statistics

3 seeds minimum, 5 preferred. Report IQM with bootstrap CIs (GrowNN's convention). ARC-Easy
reward is noisy — your current runs swing between p=0.0 and p=0.94 across prompts within a
single step.

### Feasibility on your Mac Mini

Measured from the two runs going right now (competing for the same box): **GRPO ≈ 14 s/step,
TPO ≈ 22 s/step**, so ~10 s/step solo. One 140-step epoch ≈ 25–35 min; grown arms ~20% slower.

- 2 epochs (280 steps) ≈ 60–75 min per run.
- First batch: 4 arms × 3 seeds = 12 runs ≈ **12–15 h**, i.e. one overnight, run sequentially.
- Full 6 arms × 5 seeds = 30 runs ≈ 35 h. Only worth it after batch 1 shows something.

Task: stay on **ARC-Easy**. Your own note at `chat_rl.py:57` is right that GSM8K sits at ~2%
and leaves nothing to measure; current runs show ARC-Easy reward ≈ 0.29–0.36 with real spread.

---

## 7. Phased plan

**Phase 0 — instrumentation only, no growth (½ day).** Add entropy / dormancy / effective-rank
/ per-group grad-norm / weight-norm logging and dense eval to `chat_rl.py`. Run `static` × 3
seeds. *You cannot detect a post-growth dip without first knowing the noise floor.* This phase
is independently useful — it's the plasticity instrumentation your RL pipeline is currently
missing, growth or no growth.

**Phase 1 — `nanochat/grow.py` (1 day).** `grow_depth(model, optimizer, k, at, init=...)`.
New `GPTConfig` fields (`backout_layer`, `ve_layers`, `window_sizes`) + `_patch_missing_config_keys`
entries. Optimizer surgery per §4. Pin the checkpoint dirname.

**Phase 2 — correctness (½ day).** The identity test (§3). A checkpoint round-trip test:
grow → save → load → assert logits unchanged. An optimizer-state test: assert existing
params' moments are bit-identical across a growth call. A 20-step smoke run.

**Phase 3 — batch 1 (overnight).** `static`, `grow@⅓`, `grow-random`, `lr-control` × 3 seeds,
2 epochs. Decide from entropy and `||c_proj||` curves whether there's anything here at all.

**Phase 4 — only if batch 1 is interesting.** Remaining arms, `--grow-lr-mult` sweep, MLP-width
operator, adaptive triggers (grow when entropy or dormancy crosses a threshold, rather than on
a fixed schedule — the open question your survey correctly identifies as unresolved).

---

## 8. Decisions I need from you

1. **Horizon.** 2 epochs (280 steps) is my recommendation — 140 leaves no room for growth to pay
   off. Costs ~2× per run.
2. **`--grow-lr-mult` default.** I'd start at 50 based on §5. Alternative: keep default LR and
   accept a likely-null first batch.
3. **Insertion point.** Middle, per "Is One Layer Enough?". Append-at-end is simpler but the
   literature says end layers are where RL gains are *smallest*.
4. **Value embeddings on new layers.** I'd default off (+33.6M params of sparse-gradient
   embedding table for `k=4`). Worth one arm later to check.
5. **Objective.** GRPO or TPO as the substrate? GRPO is ~1.6× faster and is the cleaner
   baseline; TPO's frozen-target fixed point interacts with growth in ways worth a separate
   look (its gradient vanishes on solved groups, which changes what "plasticity" even means here).

---

## 9. Pre-flight audit (run before committing 13 GPU-hours)

Two paired 40-step runs on the real checkpoint (`grow@10` vs `static`, same seed),
plus re-analysis of the completed 140-step GRPO run. Four things changed as a result.

### Confirmed working

**Identity growth is exact end to end.** In the paired run, per-step reward is
bit-identical (`max|diff| = 0.00e+00`) through the growth step and diverges only
afterwards — which is the correct prediction, since an identity block leaves both the
forward pass and the gradients to the old parameters unchanged, so the first
divergence can only come from the *next* optimizer step.

**Grown layers engage at the chosen LR.** `c_proj` spectral norm climbs
0.014 → 0.505 over 25 post-growth steps in a 40-step run. Extrapolating the LR budget
of the real schedule (growth at step 92 of 280) puts the endpoint near 3 against a
trained layer's 5.28 — engaged, not dominant. `--grow-lr-mult 50` is the right order.

**Statistical power is adequate.** Paired difference sd 0.083/step, integrated
autocorrelation time ~1.7, 188 post-growth steps × 3 seeds → detectable effect 0.009,
about **11% of the whole RL effect** (which is +0.081). 5 seeds buys 9%: not worth
67% more wall-clock.

### Corrections to earlier claims in this document

**Pairing helps much less than §6 implied.** Decomposing the 140-step run suggested
prompt-difficulty variance accounted for ~100% of the step-to-step noise and would
cancel between arms at a shared seed. Measured directly on the paired runs, it cuts
the noise only **1.2x**. Once the policies diverge, the same prompt yields genuinely
different success rates, and that difference is itself the noise. Pair anyway — it is
free — but do not count on it.

### Two instrumentation bugs, both fixed

**`diag/entropy` was measuring the wrong positions.** It averaged over all probe
positions, which is dominated by ordinary text prediction and barely moves. On the SFT
model: 1.682 averaged over positions versus **1.326 at the decision position**, where
the model puts 100% of its mass on A–E and sits near-uniform over four choices
(ln 4 = 1.386). Now logged as `diag/entropy_decision`, with the old figure kept as
`diag/entropy_allpos`. Had this shipped, the headline plasticity metric would have
been flat by construction.

**`cost/step_seconds` included evaluation time.** The grown arms get extra forced evals
from the growth bracket, so they would have looked slower for a reason unrelated to
being bigger — corrupting exactly the equal-wall-clock comparison §6 calls the fair
one. Split into `cost/train_seconds` (rollouts + backward) and `cost/step_seconds`.

### The finding that should change the endpoint

**On this task RL mostly destroys diversity rather than adding capability.** From the
completed 140-step GRPO run:

| step | pass@1 | pass@8 | gap |
|---|---|---|---|
| 0 | 0.2625 | 0.7125 | 0.450 |
| 60 | 0.2925 | 0.3650 | 0.073 |
| 120 | 0.3075 | 0.4100 | 0.103 |

pass@1 gains **+0.045** while pass@8 loses **−0.30**. The SFT policy is near-uniform
over the four choices and RL sharpens it; most of what changes is the sharpening.

Consequence for the experiment: reward and pass@1 have a tiny dynamic range (0.081
total, of which we can resolve 11%), while the **diversity gap has a 0.35 swing of
which we can resolve 8%**. `passk/diversity_gap` is now logged explicitly. Judge these
arms primarily on whether growth arrests the collapse while holding pass@1 — that is
both the more sensitive endpoint and the one the plasticity literature actually
predicts growth should move.

Note also that the collapse is mostly complete by step 60 of 140 (43% of the run).
If the 280-step schedule collapses on the same fraction, growth at 0.33 lands
mid-collapse, which is defensible. If the first evals show it collapsing earlier,
add a `--grow-at=0.15` arm.

### Open concern

Dormant-unit fraction (~1%) and effective rank (~207/512) are flat over 40 steps, so
the classic NE-style plasticity signals show nothing at this horizon. The diversity
collapse above is the plasticity signal on this task; if the static arm's dormancy and
effective rank are still flat at 280 steps, then the *capacity* framing is all that is
left, and "Is One Layer Enough?" predicts that comes out null.
