# Target Policy Optimization (TPO) — arXiv:2604.06159, Kaddour

Code: https://github.com/JeanKaddour/tpo · Implemented here as `nanochat/tpo.py`
+ `--objective tpo` in `scripts/chat_rl.py`.

---

## 1. The idea in one paragraph

Given a prompt and `K` scored completions, policy gradient answers two questions
at once — *which* completions should gain probability mass, and *how far* the
parameters should move to make that happen. The second answer is the optimizer's
job (learning rate, clipping, ratio), so the first gets distorted by it. TPO
separates them. Write down the distribution you *want* over the group,

```
u  = zscore(rewards)                     # within-group, population std, 0 if flat
q  ∝ p_old * exp(u / η)                   # η = 1 throughout the paper
```

and fit the policy to it by cross-entropy, `L = -Σ_i q_i log p_i^θ`, where
`p^θ = softmax(ℓ^θ)` over the group's sequence log-probs. That's it. No critic,
no importance ratio, no clip, no reference model.

Two facts do all the work:

- **`∂L/∂ℓ_i = p_i^θ − q_i`.** The gradient vanishes exactly when the policy
  matches the target. A mean-subtracted policy gradient has no fixed point: it
  keeps pushing on a group it has already solved. The paper measures this
  directly (Fig. 7a) — TPO's gradient norm collapses to ~0 once error plateaus,
  GRPO's does not.
- **`q` is the closed-form argmax of `E_q[u] − η·KL(q‖p_old)`** over the simplex
  on the *sampled* candidates (Prop. 1). So TPO is MPO's E-step without MPO's
  critic — the finite candidate set is what makes the target available in closed
  form.

A zero-variance group (all-fail or all-pass) gets `σ=0 → u=0 → q=p_old →` exactly
zero gradient, for free, with no masking rule.

## 2. Why it should beat GRPO, mechanistically

The sharpest result in the paper is §3.2. In a one-hot tabular bandit every
method has the *same* within-context direction `e_y − π`; they differ only in a
scalar `β(p)` that decides how much of a normalized step each context gets:

| method | β(p) | β at p=0.1 |
|---|---|---|
| CE oracle | `1` | 1.00 |
| **TPO** | `p(λ−1)/(1−p+λp)`, λ≈28 | **0.73** |
| GRPO | `sqrt(p/(1−p))` | 0.33 |
| DG | `p/(1+p)` | 0.09 |

GRPO and DG both **vanish as p→0**: they spend the step budget on contexts that
are nearly solved already and barely touch the hard ones. TPO's coefficient stays
flat, so it tracks the equal-weight oracle. That is the same failure mode nanochat
already instruments as `realised/p_frac_degenerate` — and it is directly relevant
to the GVM/VIP work on this branch, because **allocation and weighting are two
shots at the same target**: GVM/VIP decide *how many rollouts* a hard prompt gets,
TPO decides *how much gradient* a hard prompt's rollouts produce. They compose.

## 3. Results worth knowing

- **Dense reward: a wash.** MNIST bandit, token-reversal with bag-of-tokens
  reward — TPO is faster but everything converges. GSM8K RLVR (Qwen3-1.7B, K=16):
  TPO hits 50% ~10 steps earlier, both land at 85–87%.
- **Sparse/terminal reward: not a wash.** Exact-match error, prompt-matched,
  H=10: TPO 7.4%, GRPO 50.4%, PPO/DG no learning at all. At H=7: 6.9 vs 14.5.
- **Reasoning Gym graph colouring, Qwen3-1.7B: GRPO scores ~0 for 300 steps,
  TPO reaches 0.96.** This is the result that should make us care.
- **Ablations all bite.** Removing the `p_old` anchor (`q ∝ exp(u)`) is
  consistently harmful. Keeping the candidates and the z-scores but reverting to
  scalar weighting ("Group PG") is the *worst* variant — so the gain is the
  target-matching, not the grouping. GRPO without its KL penalty collapses.
- **Robustness.** η anywhere in [0.25, 2] converges within 1.5x of the best.
  TPO stays under 2.3% error across 1/2/4/8/16 gradient epochs; GRPO swings
  4.3% → 37.6% → 6.3% → 3.3% → 1.1% over the same sweep. Multi-epoch reuse works
  without ratios or clipping because the frozen target *is* the trust region.
- **Zero-variance masking is a trap.** Explicitly masking all-same-reward groups
  makes GRPO much worse (6.3% → 29.7%): after the first epoch those groups act as
  an anchor back to the rollout snapshot. TPO gets that for free.

## 4. What is now in this repo

```
nanochat/tpo.py        standardize_scores / tpo_target / tpo_weights  (~90 lines)
tests/test_tpo.py      49 tests, no model, no GPU
scripts/chat_rl.py     --objective {grpo,tpo} + --tpo-{eta,anchor,logp,epochs}
```

Rollouts, tasks, allocators, logging and the optimizer are all shared, so
`--objective` is a clean A/B on the loss alone.

### The one implementation trick

`p^θ` couples all `K` candidates through one softmax, so a naive implementation
needs the whole group live in a single autograd graph — impossible when
`n_i=16 > device_batch_size=8`. But since `∂L/∂ℓ_i = p_i^θ − q_i` exactly,

```
L_surrogate = -Σ_i w_i · ℓ_i^θ ,   w = (q − p^θ).detach()
```

has the **same gradient** as the real cross-entropy loss. That is the identical
`coefficient × log-prob` shape the existing REINFORCE path already uses, so TPO
drops in with `w` where the advantage was, and groups can be split across as many
forward passes as memory demands. Verified against the real model forward:
max parameter-gradient difference between the microbatched surrogate and the
single-graph loss is **5.6e-12**. `w` sums to zero, just like `r − r̄`.

Cost: one extra no-grad forward per rollout to read `ℓ^old` (and again per extra
gradient epoch, to re-read `p^θ`). Logged as `tpo/scoring_rows`, in the same
spirit as `rollout_accounting` for GVM's pilot pass. We deliberately do *not*
reuse the Engine's sampling logits — it samples with a KV cache and applies
temperature/top-k, so its probabilities are not `π_θ`, and the mismatch would
surface as a silently wrong `p_old` rather than as an error.

### ⚠ The caveat to check before trusting any number

`ℓ_i` is a **sum** of token log-probs, so its spread across a group grows like
`√T`. Once that spread is large the group softmax collapses onto whichever sample
was shortest/likeliest, `q ≈ p_old` however the rewards fell, and `w → 0` —
TPO silently stops updating. Measured (K=16, 25% pass rate, Monte Carlo):

| std(ℓ) | mean p_max | mean ‖q−p‖₁ | groups inactive |
|---:|---:|---:|---:|
| 1 nat | 0.25 | 0.98 | 0% |
| 2 | 0.47 | 0.75 | 1% |
| 3 | 0.61 | 0.56 | 10% |
| 5 | 0.76 | 0.34 | 37% |
| 10 | 0.88 | 0.17 | 67% |
| 20 | 0.94 | 0.08 | 83% |

and `std(ℓ) ≈ √T × per-token log-prob std (~1.2 nats)`, so:

- **ARC** (completion is one letter, T≈3): std(ℓ)≈2 nats → ~1% inactive. **Safe,
  and it is already the recommended RL task on this branch.**
- **GSM8K** (T≈150–256): std(ℓ)≈15–19 nats → **67–83% of groups do nothing.**

The paper doesn't discuss this; their LLM runs are GSM8K-length, so either the
surviving groups carry it or their completions are tighter than this estimate.
Either way it is measurable, so `chat_rl` logs `tpo/p_max_mean`, `tpo/p_max_max`,
`tpo/p_entropy_mean`, `tpo/weight_l1_mean` and `tpo/frac_inactive` every step.
**Look at `tpo/p_max_mean` first.** If it sits near 1, `--tpo-logp mean` divides
`ℓ` by the completion length before the group softmax (GSPO's reasoning) and
restores the signal — but it is no longer the paper's objective, so it is a knob,
not the default.

## 5. Experiments to run

```bash
# A/B on the loss alone. ARC-Challenge, same budget as the allocator runs.
python -m scripts.chat_rl --task=arc-challenge --objective=grpo --run=rl-grpo \
    --examples-per-step=16 --num-samples=16

python -m scripts.chat_rl --task=arc-challenge --objective=tpo --run=rl-tpo \
    --examples-per-step=16 --num-samples=16
```

Then, roughly in order of expected value:

```bash
# Multi-epoch extraction -- the paper's ~5x early-convergence claim, and the
# thing nanochat cannot currently do at all (no ratio, no clip).
--objective=tpo --tpo-epochs=4

# Does the p_old anchor earn its place here too? (paper: yes, consistently)
--objective=tpo --tpo-anchor=0

# Only if tpo/p_max_mean came back near 1.
--objective=tpo --tpo-logp=mean

# TPO x allocation. Both attack degenerate prompts; do they stack or overlap?
--objective=tpo --allocator=gvm
--objective=tpo --allocator=vip
```

What to watch, beyond reward: `tpo/frac_inactive` should track
`realised/p_frac_degenerate` and little else — if it is much higher, the group
softmax is collapsing, not the rewards. Gradient norm should decay as reward
rises (§3.1 of the paper); if it doesn't, the fixed point isn't being reached.

### The sharpest test we could run

TPO's advantage is *specifically* sparse terminal reward, and ARC at p≈0.5 is the
easy regime where the paper says everything ties. GSM8K is genuinely sparse at
this scale (~2% → ~83% of prompts at p=0) but is exactly where the length
collapse bites. So the honest experiment is **GSM8K with `--tpo-logp mean`**, or
a short-completion sparse task. Worth noting that nanochat added ARC precisely
*because* GSM8K had no difficulty signal for the allocators — but TPO is the one
method here that claims to work when almost everything fails, so GSM8K may be the
right benchmark for it rather than the wrong one.

## 6. Deliberately not implemented

- **Token-level TPO** (`TPO_token`), which groups `K` next-token candidates at
  each prefix. It is the paper's strongest variant on dense/sequential reward,
  but it needs a per-token reward signal that neither GSM8K nor ARC provides.
- **DG** (arXiv, Osband et al.), the other baseline. Complementary to TPO —
  DG fixes misallocation *across* contexts, TPO *within* a context — so it is the
  natural next drop-in if this line proves out.
- **Multi-epoch for GRPO.** It would need the PPO ratio+clip that nanochat
  deliberately deleted; `--tpo-epochs > 1` is rejected for `--objective grpo`
  rather than silently running uncorrected off-policy.

## 7. Verification done

- 49 unit tests: gradient identity, Prop. 1 optimality against 200 random simplex
  points per seed, zero-variance neutrality, population-std z-scoring, η limits,
  anchor ablation, frozen-target reuse, K=1 safety, collapse diagnostics.
- Real-model integration: microbatched surrogate == single-graph cross-entropy
  loss to 5.6e-12 on parameter gradients, with ragged completion lengths.
- Functional: running the multi-epoch path on a tiny GPT drives `KL(q‖p^θ)` to
  1e-7 and the gradient norm from 1.6e+01 to ~1e-5 — the self-extinguishing
  property of Fig. 7a, reproduced.
- Not verified: anything requiring a trained checkpoint. No SFT model is present
  on this machine, so the training loop has not been run end to end.
