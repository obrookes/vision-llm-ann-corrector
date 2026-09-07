# vision-llm-ann-corrector

SAM3-driven correction of person-mask annotations that a VLM verifier flagged as wrong. Third leg
of a three-repo pipeline:

    vision-llm-ann-generator  --annotate-->  masks (SAM3 text-prompted)
    vision-llm-ann-verifier   --review------>  worklist.jsonl  (VLM flags bad/missing masks)
    vision-llm-ann-corrector  --correct------>  masks_v2  (this repo: SAM3 re-segments flagged frames)
    (loop: masks_v2 fed back into the verifier for re-review)

## The hybrid rule

For each frame in a correction worklist, don't call SAM3 blind -- reuse a single text-prompted
("person") detection pass per frame and only fall back to a geometry-driven box prompt when the
text pass doesn't already contain a good match:

1. Load the frame's original masks (COCO RLE) and decode with `rle.py` (numpy-only decoder, no
   pycocotools dependency -- copied from `vision-llm-ann-verifier`'s `images.py`).
2. Open one single-frame SAM3 session and run the text prompt once to get candidate masks for the
   whole frame (`segmenter.Sam3Segmenter.text`).
3. `"add"` actions (a person the VLM says is missing, given as a normalised `[x0,y0,x1,y1]` box):
   pick the text candidate with the highest box IoU against the proposed box (>= `--iou-match`,
   and not already claimed by >0.5 IoU by a mask being kept); if none qualifies, fall back to a
   box prompt (`segmenter.Sam3Segmenter.box`). Appended as a new instance.
4. `"resegment"` actions (an existing mask the VLM flagged, with an `issue`):
   - `wrong_object` / `duplicate`: drop the instance. No re-prompt -- there's nothing there worth
     re-segmenting.
   - `loose` / `fragment` / `merged` / anything else: prefer a text candidate with IoU > 0.5
     against the old mask; otherwise prompt with the old mask's bounding box, dilated by
     `--dilate` on each side, and replace the instance.
5. Untouched instances are copied through byte-identical (same RLE). Every output instance carries
   a `"provenance"` field recording where it came from.

See `segmenter.py`'s module docstring for a documented deviation from the original box+point
combined-prompt idea: the installed SAM3 API (`sam3` 0.1.0) doesn't support combining a box and a
point in one `add_prompt` call (`points is not None` requires `text_str is None and boxes_xywh is
None`, verified against the installed package source), so `.box()`'s `point` argument is used only
as a tie-breaker between multiple returned candidates, not as an additional prompt.

## I/O contract

Input worklist: one JSON line per frame, produced by `vision-llm-ann-verifier`'s
`tools/worklist.py` from `--images --schema v2` review runs:

    {"image": "<path>.png", "masks_json": "<path>_masks.json", "n_masks": int,
     "actions": [{"type": "resegment", "idx": int, "issue": str} |
                 {"type": "add", "box": [x0,y0,x1,y1] (0-1 normalised), "note": str}],
     "votes": {...}, "confidence": "high" | "low"}

Input masks (`<masks-dir>/<stem>_masks.json`, never modified): a flat list of
`{"instance_idx", "center_xy", "area_px", "rle": {"size": [h, w], "counts": <pycocotools-
compressed string>}}`.

Output masks (`<out-dir>/<stem>_masks.json`): same shape as the input, re-indexed `0..N-1` by
default, plus a `"provenance"` field per instance:

    {"source": "original" | "auto" | "removed", "action": "resegment" | "add", "orig_idx": int,
     "issue": str, "prompt": "text" | "box", "score": float}

`center_xy` in the output is the mask's centroid (mean of its True pixel coordinates), matching
what the input files already use. Corrections are logged one JSON line per frame processed to
`--log` (default `corrections.jsonl`): stem, in/out mask counts, per-action outcomes, candidate/
fallback counts, timing.

### `--keep-ids` and `--rle-format`

`--keep-ids`: don't re-index the output 0..N-1. Kept/resegmented instances keep their original
`instance_idx` unchanged; `"add"` instances get fresh ids `max(existing ids)+1, +2, ...`. Use
this when `instance_idx` values are SAM3 track ids from `vision-llm-ann-generator` (non-contiguous,
must be preserved across the correction loop) rather than positional `0..N-1` indices.
`resegment.idx` matching is keyed by `instance_idx` (not list position) in both modes -- see
`process_frame`'s `inst_by_idx`/`kept` dicts, both keyed by `instance_idx`.

`--rle-format {compressed,intlist}` (default `compressed`): `compressed` writes the same
pycocotools-style LEB128 `counts` string the input files already use (unchanged default
behaviour). `intlist` writes uncompressed int-list `counts`
(`{"size": [h, w], "counts": [int, ...]}`, same column-major convention, `counts[0]` = background
run) instead -- required when the output will be read by `vision-llm-ann-generator/tracks.py`'s
`rle_decode`, which explicitly rejects compressed string counts.

## Running it

Login-node (no GPU, `--fake` uses `segmenter.FakeSegmenter` instead of SAM3/torch):

    $SCRATCH/envs/vlm-verify/bin/python correct.py \
        --worklist outputs/worklist_calib.jsonl \
        --out-dir /tmp/masks_v2_smoke --fake --dry-run --limit 5

On a GPU node, via the SAM3 container built by `vision-llm-ann-generator/container/`:

    WORKLIST=outputs/worklist_calib.jsonl sbatch slurm/submit.sbatch
    # or, sharded across 4 array tasks:
    WORKLIST=outputs/worklist.jsonl sbatch --array=0-3 slurm/submit.sbatch

Resumable: a frame whose `<out-dir>/<stem>_masks.json` already exists is skipped without opening
the image or the model, so a killed/re-submitted array job picks up where it left off.

## Tests

    $SCRATCH/envs/vlm-verify/bin/python -m pytest tests/ -v

Uses `segmenter.FakeSegmenter` (no torch/sam3 import) plus a real `$SCRATCH/masks`/`$SCRATCH/frames`
pair for the RLE round-trip and end-to-end CLI tests (skipped if that data isn't present on the
machine running the tests). The pycocotools-agreement RLE test skips on the login-node env (no
pycocotools there) and passes inside the `sam3.sif` container (verified manually -- see NOTES.md).
