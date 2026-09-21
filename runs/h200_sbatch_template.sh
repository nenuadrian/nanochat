#!/usr/bin/env bash
## SBATCH template for H200 nodes
## Usage: sbatch /path/to/h200_sbatch_template.sh <run_name> <depth> <job_tag>
## Expects these args to be provided to construct the command.

#SBATCH --job-name=%RUN_NAME%
## Output/error/job-name are passed at submission time so they reflect the
## chosen run name and depth. Keep static resource/account directives here.
# Generic GPU gres; override in the template if you need a specific GPU type.
#SBATCH --gres=gpu:1
#SBATCH --account=gpu-h200-fse-pgdr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem=64G

set -euo pipefail

RUN_NAME=${1:-run}
DEPTH=${2:-20}
JOBTAG=${3:-baseline}

echo "Starting job: ${RUN_NAME} depth=${DEPTH} tag=${JOBTAG}"

# load modules / activate environment (adjust to your cluster)
module purge || true
# Try to load conda module if available
module load anaconda/2023.11 || true
# Robust Conda/venv activation: prefer conda env 'nanochat' if present,
# otherwise fall back to a venv at $HOME/.venv/nanochat if it exists.
if [ -n "${CONDA_PREFIX:-}" ]; then
  : # already inside conda
else
  if command -v conda >/dev/null 2>&1; then
    # Source conda sh to enable `conda activate` in non-interactive shells
    CONDA_BASE=$(conda info --base 2>/dev/null || echo "")
    if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
      . "$CONDA_BASE/etc/profile.d/conda.sh"
      conda activate nanochat || true
    fi
  fi
fi
# Fallback to a virtualenv if the conda env is not present
if [ -z "${CONDA_PREFIX:-}" ] && [ -f "$HOME/.venv/nanochat/bin/activate" ]; then
  source "$HOME/.venv/nanochat/bin/activate"
fi

# Ensure wandb is installed in the runtime environment (non-fatal if install fails)
python - <<'PY'
try:
    import wandb
except Exception:
    import sys, subprocess
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', 'wandb'])
    except Exception:
        pass
PY

# Install project dependencies from requirements-csf3.txt. Choose torch wheel
# based on whether this job is running on a GPU partition. This is best-effort
# and skips failures so the job can still proceed if pip install fails.
REQ_FILE="$PWD/requirements-csf3.txt"
if [ -f "$REQ_FILE" ]; then
  python -m pip install --upgrade pip setuptools wheel || true
  # Install common deps (excludes torch-specific wheel)
  python -m pip install --no-cache-dir -r "$REQ_FILE" || true
  # Install torch variant appropriate for GPU/CPU
  PART=${SLURM_JOB_PARTITION:-}
  if echo "${PART}" | grep -qi "^gpu"; then
    # GPU partition: install CUDA 12.8 wheel
    python -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.9.1 || true
  else
    # CPU-only: install CPU wheel
    python -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.9.1 || true
  fi
fi

# model path on csf3 (adjust as needed)
MODEL_TAG=${MODEL_TAG:-d${DEPTH}narrow}
MODEL_PATH=${MODEL_PATH:-/home/${USER}/models/${MODEL_TAG}}

cd $SLURM_SUBMIT_DIR

echo "Using model path: ${MODEL_PATH}"

# run the training script; adjust arguments as desired
python -m scripts.chat_rl \
  --run ${RUN_NAME} \
  --task gsm8k \
  --model-tag ${MODEL_TAG} \
  --device-type cuda \
  --examples-per-step 128 \
  --num-samples 8 \
  --device-batch-size 16 \
  --num-epochs 3 \
  --eval-every 500 \
  --eval-examples 400 \
  --save-every 500 \
  --output-tag ${RUN_NAME}_${DEPTH}_${JOBTAG} \
  --seed ${SEED:-42}

echo "Job finished"
