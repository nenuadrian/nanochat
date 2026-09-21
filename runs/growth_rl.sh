#!/bin/bash

# Capacity growth during RL: does inserting transformer blocks mid-RL help?
# Design doc: knowledge/plan_growth_rl.md   Implementation: nanochat/grow.py
#
# Usage:
#   bash runs/growth_rl.sh              # batch 1: 4 arms x 3 seeds, sequential
#   ARMS="static grow-mid" bash runs/growth_rl.sh     # subset
#   SEEDS="1" bash runs/growth_rl.sh                  # single seed, ~4-5h
#   DRY=1 bash runs/growth_rl.sh                      # print commands, run nothing
#   SAVE=1 bash runs/growth_rl.sh                     # also keep each run's final checkpoint
#
# Checkpoints are off by default: this model is 264M params (~1.1GB per save) and
# batch 1 is decided on the logged metrics, not on the weights. Set SAVE=1 if you
# want the grown models around afterwards (~13GB for 12 runs).
#
# Sequential on purpose: these runs are compared on wall-clock per step, and
# running two at once on one box makes that number meaningless (measured on this
# Mac Mini: 14 s/step with two jobs competing vs ~10 s/step solo).

set -u
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
[ -d ".venv" ] && source .venv/bin/activate

SEEDS="${SEEDS:-1 2 3}"
ARMS="${ARMS:-static grow-mid grow-random lr-control}"
DRY="${DRY:-0}"
# Any positive value still saves the final step, so this keeps exactly one checkpoint per run.
SAVE_EVERY=$([ "${SAVE:-0}" = "1" ] && echo 100000 || echo -1)

# Horizon: ARC-Easy train is 2251 examples / 16 per step = 140 steps per epoch.
# Two epochs, because a growth event at the 1/3 mark needs room left to pay off.
COMMON="--task=arc-easy --objective=grpo --num-epochs=2
        --eval-every=20 --eval-examples=200 --diag-every=10
        --grow-eval-radius=3 --save-every=$SAVE_EVERY"

# Growth: +4 blocks mid-stack at the 1/3 mark (step ~92 of 280).
# Middle because arXiv:2607.01232 finds RL's high-contribution layers sit there.
# lr-mult 50 because at the base RL rate a grown layer cannot leave identity:
# a Muon step moves a matrix ~1e-3 in spectral norm against a trained c_proj of ~5.
GROW="--grow-at=0.33 --grow-layers=4 --grow-position=middle --grow-lr-mult=50 --grow-warmup=5 --grow-ve=0"

run () {   # run <arm> <seed> <extra args...>
    local arm=$1 seed=$2; shift 2
    local name="growth-${arm}-s${seed}"
    echo "=== $name ==="
    if [ "$DRY" = "1" ]; then
        echo "python -m scripts.chat_rl $COMMON --run=$name --output-tag=$name --seed=$seed $*"
        return
    fi
    # shellcheck disable=SC2086
    python -m scripts.chat_rl $COMMON --run="$name" --output-tag="$name" --seed="$seed" "$@" \
        2>&1 | tee "logs/${name}.log"
}

mkdir -p logs
for seed in $SEEDS; do
  for arm in $ARMS; do
    case $arm in
      # Baseline. Also establishes the noise floor: without it you cannot tell a
      # post-growth dip from ordinary ARC-Easy variance.
      static)       run "$arm" "$seed" ;;
      # The treatment: identity-preserving growth (reward is continuous by construction).
      # shellcheck disable=SC2086
      grow-mid)     run "$arm" "$seed" $GROW --grow-init=copy ;;
      # Non-preserving growth. The arm that actually tests "reward dips, then recovers".
      # shellcheck disable=SC2086
      grow-random)  run "$arm" "$seed" $GROW --grow-init=random ;;
      # The control that can kill a false positive: same LR bump, no new capacity.
      lr-control)   run "$arm" "$seed" --lr-bump-at=0.33 --lr-bump-mult=50 ;;
      # --- batch 2, only worth running if batch 1 shows something -------------
      # Was it the capacity or the timing? Same final model, grown at step 0.
      # shellcheck disable=SC2086
      grow-at0)     run "$arm" "$seed" $GROW --grow-at=0 ;;
      # Does staged beat single-shot? +2 at 1/3, +2 at 2/3.
      # shellcheck disable=SC2086
      grow-staged)  run "$arm" "$seed" $GROW --grow-at=0.33,0.66 --grow-layers=2 ;;
      # Growth at the base LR. Expected to be a no-op; that IS the finding.
      # shellcheck disable=SC2086
      grow-baselr)  run "$arm" "$seed" $GROW --grow-lr-mult=1 ;;
      *) echo "unknown arm: $arm" >&2; exit 1 ;;
    esac
  done
done
echo "done. compare in wandb project nanochat-rl, or: grep -h 'diag |' logs/growth-*.log"
