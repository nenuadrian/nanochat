#!/usr/bin/env bash
# Submit staged growth jobs to H200 Slurm cluster.
# Usage: ssh to csf3 then run this script from the repo root.

set -euo pipefail


RUN_BASE=${1:-baseline-d20-gsm8k}
START_DEPTH=${2:-20}
END_DEPTH=${3:-24}
INCR=${4:-1}    # 1 or 2 layers at a time
JOBTAG=${5:-staged}
PARTITION=${6:-h200}

TEMPLATE="runs/h200_sbatch_template.sh"

mkdir -p logs

for d in $(seq ${START_DEPTH} ${INCR} ${END_DEPTH}); do
  RUN_NAME="${RUN_BASE}_d${d}"
  echo "Submitting depth ${d} as ${RUN_NAME}"
  sbatch --partition=${PARTITION} ${TEMPLATE} ${RUN_NAME} ${d} ${JOBTAG}
  sleep 0.5
done

echo "Also submitting baseline (no-growth) job"

sbatch --partition=${PARTITION} ${TEMPLATE} ${RUN_BASE}_baseline ${START_DEPTH} baseline

echo "Submitted jobs for depths ${START_DEPTH}..${END_DEPTH} (step ${INCR})."
