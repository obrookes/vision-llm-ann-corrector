#!/bin/bash
# Body of a SAM3 correction job: worklist sharding, container env, apptainer exec.
# Mirrors vision-llm-ann-generator/slurm/run.sh's structure/env-var handling.
#
# `exec bash slurm/run.sh` from within slurm/submit.sbatch, or source the same env vars
# (WORKLIST, CHECKPOINT, ...) and run it directly on an interactive allocation.

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

WORKLIST="${WORKLIST:-outputs/worklist_calib.jsonl}"
CHECKPOINT="${CHECKPOINT:-$SCRATCH/weights/sam3/sam3-safari-pos.pt}"
FRAMES_DIR="${FRAMES_DIR:-$SCRATCH/frames}"
MASKS_DIR="${MASKS_DIR:-$SCRATCH/masks}"
OUT_DIR="${OUT_DIR:-$SCRATCH/masks_v2}"
TEXT_PROMPT="${TEXT_PROMPT:-person}"
MIN_CONFIDENCE="${MIN_CONFIDENCE:-low}"
IOU_MATCH="${IOU_MATCH:-0.5}"
DILATE="${DILATE:-0.10}"

SIF="${SIF:-$SCRATCH/containers/sam3.sif}"
[ -e "$SIF" ] || SIF="$SCRATCH/containers/sam3-sandbox"   # login-node mksquashfs fails (pids limit); sandbox works too
if [ ! -e "$SIF" ]; then
    echo "run.sh: no container image found." >&2
    echo "  looked for: \$SCRATCH/containers/sam3.sif and \$SCRATCH/containers/sam3-sandbox" >&2
    echo "  (built by vision-llm-ann-generator/container/; see its README.md)" >&2
    exit 1
fi

N="${SLURM_ARRAY_TASK_COUNT:-1}"; I="${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p outputs/shards "$OUT_DIR" "$SCRATCH/hf-cache" "$SCRATCH/torch-cache" "$SCRATCH/triton-cache"
SHARD="outputs/shards/$(basename "$WORKLIST" .jsonl)_${I}_of_${N}.jsonl"
awk -v n="$N" -v i="$I" 'NR % n == i' "$WORKLIST" > "$SHARD"

LOG="${LOG:-outputs/corrections_${I}_of_${N}.jsonl}"

export APPTAINERENV_HF_HOME="$SCRATCH/hf-cache"
export APPTAINERENV_HF_HUB_OFFLINE=1
export APPTAINERENV_TMPDIR=/tmp
export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR="$SCRATCH/torch-cache"
export APPTAINERENV_TRITON_CACHE_DIR="$SCRATCH/triton-cache"
# Triton (used by sam3.perflib NMS kernels) finds libcuda via `ldconfig -p`, which inside the NGC
# image points at /usr/local/cuda/compat/lib (absent on the node), and it links with -lcuda so it
# needs a `libcuda.so` name, while `apptainer --nv` only provides libcuda.so.1 under
# /.singularity.d/libs. Give it a dir with both names (symlink targets resolve inside the container).
TRITON_LIBCUDA_DIR="$SCRATCH/lib/triton-libcuda"
mkdir -p "$TRITON_LIBCUDA_DIR"
ln -sfn /.singularity.d/libs/libcuda.so.1 "$TRITON_LIBCUDA_DIR/libcuda.so.1"
ln -sfn /.singularity.d/libs/libcuda.so.1 "$TRITON_LIBCUDA_DIR/libcuda.so"
export APPTAINERENV_TRITON_LIBCUDA_PATH="$TRITON_LIBCUDA_DIR"

GPUS=0
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    GPUS=$(nvidia-smi -L | wc -l)
fi

echo "== run.sh: WORKLIST=$SHARD CHECKPOINT=$CHECKPOINT OUT_DIR=$OUT_DIR TEXT_PROMPT=$TEXT_PROMPT SIF=$SIF task=${I}/${N} GPUs=$GPUS =="

START=$(date +%s)

apptainer exec --nv --bind "/lus,/scratch,$HOME" "$SIF" \
    python3 correct.py \
        --worklist "$SHARD" \
        --frames-dir "$FRAMES_DIR" \
        --masks-dir "$MASKS_DIR" \
        --out-dir "$OUT_DIR" \
        --checkpoint "$CHECKPOINT" \
        --text-prompt "$TEXT_PROMPT" \
        --min-confidence "$MIN_CONFIDENCE" \
        --iou-match "$IOU_MATCH" \
        --dilate "$DILATE" \
        --log "$LOG" \
        ${EXTRA_ARGS}

STATUS=$?
END=$(date +%s)
echo "== run.sh: done in $((END - START))s, exit=$STATUS =="
exit $STATUS
