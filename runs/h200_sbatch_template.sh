#!/usr/bin/env bash
## SBATCH template for H200 nodes
## Usage: sbatch /path/to/h200_sbatch_template.sh <run_name> <depth> <job_tag>
## Expects these args to be provided to construct the command.

#SBATCH --job-name=%RUN_NAME%
#SBATCH --output=logs/%RUN_NAME%_%DEPTH%_%JOBTAG%.out
#SBATCH --error=logs/%RUN_NAME%_%DEPTH%_%JOBTAG%.err
# Partition is set at submission time (use `sbatch --partition=...`).
#SBATCH --partition=
# Generic GPU gres; override in the template if you need a specific GPU type.
#SBATCH --gres=gpu:1
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
module load anaconda/2023.11 || true
source activate /home/${USER}/.conda/envs/nanochat || true

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
