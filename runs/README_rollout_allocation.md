# Rollout budget allocation: uniform vs GVM vs VIP

Run these in order. Everything before "Experiments" is just getting a model that
can do GSM8K at all; the allocators only become interesting once it can.

All three allocators spend **the same total rollouts per step**
(`examples_per_step x num_samples`). They differ only in how that budget is split
across the prompts in a step:

| allocator | where `p_q` comes from | allocation |
|---|---|---|
| `uniform` | not used | `n_q = C/B` |
| `gvm` | **measured** by `N'` pilot rollouts per prompt | `n_q ∝ G_q / sqrt(p_q + α/p_q^(β-1))` (arXiv:2505.02391 Prop. 1) |
| `vip` | **predicted** by a GP over prompt embeddings | minimise `Σ a_q (n_q-1)/n_q²`, `a_q = 4σ²p_q(1-p_q)` (arXiv:2602.01601 Thm 5.1) |

The honest difference: GVM's pilot pass is real compute that the budget `C` does
not count. `cost/rollouts_total` and `cost/pilot_overhead` report it, so compare
on those, not just on `C`.

---

## 0. Setup (once)

```bash
cd nanochat
git checkout gvm
uv venv && uv sync --extra cpu     # --extra gpu on a CUDA box
source .venv/bin/activate
python -m pytest tests/test_allocation.py -q
```

Those tests cover the allocators and the GP with no model and no GPU. They should
all pass before you spend an hour on training.

---

## 1. Tokenizer

```bash
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
python -m nanochat.dataset -n 8
python -m scripts.tok_train --max-chars=2000000000
python -m scripts.tok_eval
```

---

## 2. Base model (pretraining)

`speedrun.sh` is written for 8xH100. On an iMac use the CPU/MPS shape below.

**On depth.** Memory is not the constraint — even d26 needs only ~17 GB. Compute
is. nanochat trains compute-optimal (`tokens = params x 10.5`), so FLOPs grow as
`63 x params^2`:

| depth | params | opt. tokens | hrs @3 TFLOPS | hrs @10 TFLOPS |
|------:|-------:|------------:|--------------:|---------------:|
| 6     | 74M    | 0.8B        | 31            | 9.5            |
| 8     | 126M   | 1.3B        | 92            | 28             |
| 12    | 286M   | 3.0B        | 478           | 143            |
| 20    | 896M   | 9.4B        | 4688          | 1407           |

Measure your own throughput before committing to a size — run 20 iterations,
read tok/s, and multiply by `6 x params`:

```bash
python -m scripts.base_train --depth=6 --head-dim=64 --window-pattern=L \
    --max-seq-len=512 --device-batch-size=32 --total-batch-size=16384 \
    --num-iterations=20 --core-metric-every=-1 --run=dummy
```

**depth 6 is the realistic target on an iMac.** The `--num-iterations=5000`
setting below is deliberately ~10x undertrained relative to compute-optimal,
which is the right trade here: see section 5, capability is not what this
experiment is measuring.

```bash
python -m scripts.base_train \
    --depth=6 --head-dim=64 --window-pattern=L \
    --max-seq-len=512 --device-batch-size=32 --total-batch-size=16384 \
    --num-iterations=5000 --eval-every=100 --core-metric-every=-1 \
    --run=base
```

### Base model metrics

```bash
python -m scripts.base_eval --device-batch-size=1 --split-tokens=16384 --max-per-task=16
```

Reports CORE, train/val BPB, and samples. This is your "before any finetuning"
reference point.

---

## 3. SFT

RL starts from the SFT checkpoint, not the base one. SFT is also what teaches the
model GSM8K's format — the default mixture is MMLU x3 + **GSM8K x4**.

```bash
python -m scripts.chat_sft --num-iterations=1500 --eval-every=200 --run=sft
python -m scripts.chat_eval -i sft
```

Write down the GSM8K number here. It decides whether section 5 is worth running.

---

## 4. Check the pipeline wires up (2 minutes)

Before any real run. Confirms the allocator plumbing works end to end.

```bash
for A in uniform gvm vip; do
  python -m scripts.chat_rl --task=arc-challenge --allocator=$A \
      --examples-per-step=4 --num-samples=4 --device-batch-size=2 \
      --gvm-pilot-samples=2 --vip-embed-prompts=64 \
      --max-new-tokens=64 --num-epochs=1 --eval-every=-1 --save-every=-1 --run=dummy
done
```

Kill each after a couple of steps. You are looking for a per-step line like
`prompt 0 (idx 17) | n=6 p=0.250` — with `n` varying across prompts for `gvm`
and `vip`, and constant for `uniform`.

---

## 5. Read this before running the experiments — pick the right task

GVM and VIP both allocate on how hard each prompt is. **If the model solves
almost nothing, every prompt looks equally hard and all three allocators collapse
to the same thing.** Under mean-subtracted advantage a prompt with `p=0` or `p=1`
contributes exactly zero gradient no matter how many rollouts it gets.

This is why `--task` exists and why **GSM8K is the wrong choice at this scale**.
Measured on a d12 (already well beyond what an iMac reaches):

| task | d12 score | usable for allocation? |
|------|----------:|------------------------|
| GSM8K         | 2.3%  | no — ~83% of prompts sit at p=0 |
| ARC-Challenge | 50.5% | **yes — p near 0.5, where VIP's objective peaks** |
| ARC-Easy      | 61.2% | yes |
| MMLU          | 35.7% | yes (no reward() yet) |

So run the experiments with `--task=arc-challenge`. GSM8K stays useful as a
held-out eval, just not as the RL task.

After the uniform run in section 6, check these two metrics:

- `realised/p_spread` — if ~0, there is no difference in difficulty to allocate on.
- `realised/p_frac_degenerate` — fraction at `p=0` or `p=1`, which contribute nothing.
- `realised/rollouts_wasted` — rollouts spent on those prompts.

Use the `realised/*` series, not `alloc/p_*`. The latter is not comparable across
allocators: it is *predicted* for VIP, *pilot-measured* for GVM, and absent for
uniform. The `realised/*` series is measured the same way in all three.

For reference: at 2.3% GSM8K accuracy with `N'=8`, 83% of prompts land at `p=0`
and 92% of the survivors sit at exactly `p=0.125`. That is not enough spread for
any allocator to beat uniform, and a null result would say nothing about the
methods. If you see that, the fix is a more capable model (deeper, more SFT),
not a different allocator setting.

---

## 6. Experiments

Same budget in all three: 16 prompts x 16 rollouts = 256 rollouts per step.

```bash
# baseline
python -m scripts.chat_rl --task=arc-challenge --allocator=uniform \
    --examples-per-step=16 --num-samples=16 --run=rl-uniform

# GVM: 4 pilot rollouts per prompt to measure p and G  (note the extra cost)
python -m scripts.chat_rl --task=arc-challenge --allocator=gvm --gvm-pilot-samples=4 \
    --gvm-alpha=0.001 --gvm-beta=2.0 \
    --examples-per-step=16 --num-samples=16 --run=rl-gvm

# VIP: no pilot pass, p predicted by the GP
python -m scripts.chat_rl --task=arc-challenge --allocator=vip --alloc-min=3 \
    --vip-embed-prompts=1024 \
    --examples-per-step=16 --num-samples=16 --run=rl-vip
```

### Ablations worth running

```bash
# Is GVM's gain from the accept rate, or from the gradient-norm term?
python -m scripts.chat_rl --task=arc-challenge --allocator=gvm --gvm-grad-norm=off --run=rl-gvm-noG

# How the per-prompt contributions are weighted. Default `per_prompt` matches
# GVM's Algorithm 1 line 8. `per_token` reproduces what verl actually does,
# where a prompt's weight grows with the rollouts it was allocated -- i.e. the
# set GVM deliberately oversamples. `inv_np` is Lemma 1's estimator explicitly.
python -m scripts.chat_rl --task=arc-challenge --allocator=gvm --estimator-weight=per_token --run=rl-gvm-pertoken
python -m scripts.chat_rl --task=arc-challenge --allocator=gvm --estimator-weight=inv_np   --run=rl-gvm-invnp
```

---

## 7. What to look at

**Did the allocator do anything**
`alloc/concentration` vs `alloc/uniform_concentration`. Equal means the
allocation was effectively uniform and any difference in outcome is noise.
Also `alloc/min`, `alloc/max`, `alloc/frac_zero`.

**Was there signal to allocate on**
`realised/p_spread`, `realised/p_frac_degenerate`, `realised/rollouts_wasted`.
See section 5, and note these are the comparable ones — `alloc/p_*` is predicted
for VIP but measured for GVM.

**What it actually cost**
`cost/rollouts_total` and `cost/pilot_overhead`. GVM at `N'=4` with 16 rollouts
per prompt is a 25% overhead; at `N'=8` it is 50%. Plot reward against
`cost/rollouts_total`, not against step, or GVM gets a free pass on the pilot.

**Is VIP's prediction usable**
`vip_gp/mae`, `vip_gp/corr`, `vip_gp/bias`. VIP's entire premise is that a
predicted `p_q` is good enough to allocate on. If `corr` is near zero, it is not,
and VIP reduces to uniform-with-extra-steps. `corr` is logged as NaN when the
predictions have no spread, which is itself the failure mode — treat NaN as bad,
not as missing.

**Did it train**
`reward`, `pass@k`, `sequence_length`.

---

## Status of this code

- `nanochat/allocation.py` and `nanochat/vip_gp.py` are unit-tested
  (`tests/test_allocation.py`, 28 tests, no GPU needed).
- `scripts/chat_rl.py` changes are **syntax-checked but never executed** — the
  machine they were written on has no nanochat environment. Section 4 exists
  precisely to shake that out; expect to fix something on the first run.
- `chat_rl.py` is not part of `speedrun.sh` and never was. It is otherwise
  healthy: its imports all resolve and it loads the `sft` checkpoint that
  `chat_sft` produces.

## Deviations from the papers

- VIP's GP is fitted over a capped pool of `--vip-embed-prompts` prompts rather
  than the full training set, to keep the embedding pass affordable. Prompt
  indices are mapped into that pool modulo its size.
- VIP's `σ_Z_q` (per-prompt gradient-norm scale) is taken as constant, leaving
  the allocation driven by `p(1-p)`. Estimating it would need GVM's pilot pass,
  which is the cost VIP exists to avoid.
- GVM's `G_q` uses the gradient wrt the token embedding, matching the reference
  implementation's `embed_tokens`, and sums token log-probs rather than
  averaging — so it scales with response length. `--gvm-grad-norm=off` isolates
  this.
