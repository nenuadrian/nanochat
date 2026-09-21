RAFT (Reward-Aware Fine-Tuning) — Design

Goal
- Provide a reproducible, configurable fine-tuning phase that turns high-reward rollouts
  collected during GRPO/TPO into supervised updates (cross-entropy) to stabilise policy.
- Support two modes: plain supervised CE on best rollouts, and reward-weighted CE (soft
  weighting by reward or advantage).

When to run
- Interleaved with RL or as a distinct phase: e.g. GRPO → GROW → RAFT → GRPO.
- RAFT can be triggered at specific steps or after the full RL pass.

High-level algorithm
1. Collect rollouts during a collection window (configurable) using the existing
   rollout/generation machinery (Engine.generate_batch). For reproducibility, write
   all rollouts to disk in JSONL per-step directories.
2. Score each rollout with the task's reward() function (already implemented in tasks/*).
3. Select training examples using one of the selection rules:
   - top-k per prompt (by reward)
   - threshold (reward >= r0)
   - weighted sampling (prob ∝ exp(beta * reward))
4. Construct supervised examples: (context, chosen_completion). Keep token ids and/or
   text; saving token ids avoids tokenizer diffs.
5. Fine-tune the model with cross-entropy on the selected examples.
   - Modes:
     * `ce` — plain CE, unweighted examples (default)
     * `rw` — reward-weighted CE: multiply each example's loss by w(reward) (e.g. reward or exp(reward/τ))
6. Optionally repeat (multiple RAFT epochs or cycles), then continue RL or stop.

Data formats and storage
- Directory layout: <base_dir>/raft_data/<run>/<step>/rollouts.jsonl
  - Each line: JSON object with fields {
      prompt_id, prompt_text, prompt_tokens, completion_tokens, completion_text,
      reward, logp, seed, step, sample_index
    }
- Selected training set: <base_dir>/raft_data/<run>/train.jsonl (same object shape)
- Checkpoints written to chatrl_checkpoints/ or a distinct `raft_checkpoints/` dir when
  --output-tag directs it.

CLI flags (proposed additions to scripts/chat_rl.py)
- --raft-enable (bool) : whether to run RAFT at configured times
- --raft-mode {ce|rw}
- --raft-topk (int) : top-k per prompt
- --raft-threshold (float) : min reward to keep
- --raft-beta (float) : for weighted sampling or reward→weight transform (rw mode)
- --raft-epochs (int) : epochs over selected examples
- --raft-lr (float)
- --raft-batch-size (int)
- --raft-accum (int) : grad-accum steps
- --raft-max-examples (int) : cap examples used (0 = all)
- --raft-eval-every (int) : evaluation frequency during raft
- --raft-data-dir (str) : override default raft data dir
- --raft-save-every (int) : how often to checkpoint during raft
- --raft-after-steps (str) : comma-separated steps/fractions when to run RAFT (e.g. "0.5,1.0")

Integration points
- Collection: instrument the existing rollout loop to optionally write each generated
  batch to the per-step JSONL (this is cheap in memory but adds IO; make it opt-in).
- Selection: new helper select_raft_examples(rollout_paths, topk, threshold, beta)
- Training: new function run_raft_finetune(model, tokenizer, examples, args) that
  mirrors the train-time optimizer setup already used (Adam/AdamW for embedding/unembedding,
  Muon for matrix if used). Keep API parallel to current training functions.
- Checkpointing: reuse `save_checkpoint` but tag `phase=raft` and include `user_config` metadata.
- Evaluation: run the same pass@k evaluation (val_task) as GRPO uses; log to wandb with a distinct
  `phase=raft` tag and metric prefix `raft/`.

Weighting and stability
- Reward-weighted CE risks collapse to short/high-logprob answers. Mitigations:
  - Normalize weights per-batch to have mean=1.
  - Clip weights to [w_min, w_max].
  - Optionally use advantage: (r − μ) where μ is mean reward over candidate set.
- Learning-rate choices: RAFT is a supervised, high-signal step — use a conservative LR
  compared to GRPO's matrix/embedding split (suggest `--raft-lr=5e-4` for embeddings, lower
  for matrices), small number of epochs (1–3), and short `--raft-warmup` (if applicable).

Safety and reproducibility
- Always save the exact examples used for RAFT (train.jsonl) and the selection criteria
  (topk/threshold/beta) in `meta_raft.json` next to checkpoints.
- Record the RNG seeds and the rollouts used; these allow exact replays.

Example workflows and commands
- Run RAFT after a GRPO experiment that wrote rollouts:
  python -m scripts.chat_rl --run=myrun --raft-enable --raft-mode=ce --raft-topk=1 \
      --raft-epochs=2 --raft-lr=5e-4 --raft-batch-size=64 --output-tag=myrun-raft

- Interleaved: GRPO for N steps, then RAFT, then resume GRPO. Orchestrator will:
  1) run chat_rl with --max-steps=N --dump-rollouts
  2) run train_raft on the dumped data
  3) resume chat_rl from the latest checkpoint

Implementation TODOs
- [ ] Add rollout dump option and JSONL format writer (opt-in via --dump-rollouts)
- [ ] Implement `select_raft_examples` with top-k/threshold/weighted sampling
- [ ] Implement `run_raft_finetune` using existing checkpoint and optimizer helpers
- [ ] Add CLI flags and wiring in `scripts/chat_rl.py`
- [ ] Add tests: small synthetic task that creates rollouts and confirms RAFT reduces loss
- [ ] Document in README and `knowledge/raft_design.md` (this file)

Defaults (recommended)
- mode=ce, topk=1, threshold=0.9 (if tasks have 0/1 rewards use threshold=1), epochs=1–2,
  raft-lr=5e-4, raft-batch-size=64, raft-max-examples=10000

Rationale: plain CE on highest-reward rollouts is the simplest stabiliser; reward-weighted
CE is an optional experiment, but must be normalised/clipped to avoid training on a few
very-high-weight tokens.

Notes on evaluation
- Use existing pass@k eval and the same val_task; present `raft/` metrics alongside `grpo/`.
- Log selection stats: number of examples selected, reward distribution, avg length.

References and links
- TPO/GRPO code paths already live in `scripts/chat_rl.py` and `nanochat/tpo.py`.
- Growth logic is in `nanochat/grow.py`; orchestrator should call grow functions between phases.

If you confirm this design I can:
- add the CLI flags and rollout-dump wiring in `scripts/chat_rl.py` (small patch), and
- implement `select_raft_examples` + `run_raft_finetune` as callable utilities.
