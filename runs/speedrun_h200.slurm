#!/bin/bash --login
# nanochat d24 speedrun adapted for H200s on CSF3 (SLURM).
# (--login so the `module` command is available in the batch shell)
# NOTE: jobs on the gpu-h200-fse-pgdr account run under QoS "gpu-h200-fse"
# (OverPartQOS), which caps each user at 4 H200s total across running jobs —
# it overrides the partition QoS "gpuh" cap of 2. So 4 GPUs is the max here.
# Submit with: sbatch runs/speedrun_h200.slurm
# wandb run name can be overridden: sbatch --export=ALL,WANDB_RUN=myname runs/speedrun_h200.slurm

#SBATCH -p gpuH
#SBATCH -A gpu-h200-fse-pgdr
#SBATCH --gres=gpu:4
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 1-00:00:00
#SBATCH --job-name=nanochat-d24-4xh200

set -e

# Compute nodes reach the internet via the university proxy (wandb, HF fallback)
module load tools/env/proxy2

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/scratch/nanochat-h200/.cache"

cd "$HOME/scratch/nanochat-h200"
source .venv/bin/activate

WANDB_RUN=${WANDB_RUN:-d24-4xh200}
NPROC=${SLURM_GPUS_ON_NODE:-2}

nvidia-smi -L

# Dataset shards were pre-copied to $NANOCHAT_BASE_DIR/base_data_climbmix;
# this verifies them and fills any gaps (skips files that already exist).
python -m nanochat.dataset -n 170

# Tokenizer (train + eval)
python -m scripts.tok_train
python -m scripts.tok_eval

# Base model pretraining (d24, fp8), then eval
torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_eval -- --device-batch-size=16

# SFT + eval
torchrun --standalone --nproc_per_node=$NPROC -m scripts.chat_sft -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC -m scripts.chat_eval -- -i sft

echo "SPEEDRUN_H200_DONE"
