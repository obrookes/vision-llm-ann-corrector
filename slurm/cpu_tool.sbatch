#!/bin/bash
#SBATCH --job-name=vlm-corrector-cpu-tool
#SBATCH --account=brics.b5bd
#SBATCH --partition=workq
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm-cpu-%j.out
#
# Run a CPU-side tool (pytest, --fake dry runs, ...) on a compute node instead of the login node.
# No GPU, no sam3/torch, no apptainer container -- correct.py --fake and tests/ never import
# sam3/torch (see segmenter.py's module docstring / tests/test_correct.py's own docstring). CMD is
# run with bash from the repo root, with $PY set to the vlm-verify env's python (same env the repo
# README already runs tests/--fake with).
#
#   CMD='$PY -m pytest -q tests/' sbatch slurm/cpu_tool.sbatch

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
set -euo pipefail
export PY="${PY:-$SCRATCH/envs/vlm-verify/bin/python}"
mkdir -p outputs
echo "== cpu_tool: $CMD =="
bash -euo pipefail -c "$CMD"
