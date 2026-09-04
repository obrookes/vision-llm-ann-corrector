#!/bin/bash
#SBATCH --job-name=sam3-correct
#SBATCH --account=brics.b5bd
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm-%A_%a.out
#SBATCH --array=0-0
#
# Usage:
#   WORKLIST=outputs/worklist_calib.jsonl sbatch slurm/submit.sbatch
#   WORKLIST=outputs/worklist.jsonl sbatch --array=0-3 slurm/submit.sbatch   # 4-way shard
# Each array task runs one SAM3 predictor on one GPU over its shard of the worklist.

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" && exec bash slurm/run.sh
