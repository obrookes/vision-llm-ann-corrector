"""Apply a verifier correction worklist to SAM3-generated masks (SAM3 hybrid correction).

    python correct.py --worklist outputs/worklist_calib.jsonl \\
        --frames-dir $SCRATCH/frames --masks-dir $SCRATCH/masks --out-dir $SCRATCH/masks_v2 \\
        --checkpoint $SCRATCH/weights/sam3/sam3-safari-pos.pt \\
        [--min-confidence high|low] [--text-prompt person] [--iou-match 0.5] [--dilate 0.10] \\
        [--log corrections.jsonl] [--limit N] [--dry-run] [--fake] [--device cuda] \\
        [--passthrough-manifest manifest.txt]

Reads one worklist line per frame (see vision-llm-ann-verifier-correction's tools/worklist.py for
the format: `image`, `masks_json`, `actions` = [{"type": "resegment", "idx", "issue"} |
{"type": "add", "box": [x0,y0,x1,y1] normalised 0-1}], `votes`, `confidence`). For each frame:

  1. Load the PNG + original masks (rle.py decodes the input pycocotools-compressed RLE).
  2. Open one single-frame SAM3 session (segmenter.Sam3Segmenter) and run the text prompt once
     to get candidate masks for the whole frame.
  3. "add" actions: try to match a text candidate to the proposed box (box IoU >= --iou-match,
     not already claimed by a kept mask at IoU > 0.5); otherwise fall back to a box prompt.
     Appended as a new instance.
  4. "resegment" actions: {wrong_object, duplicate} -> drop the instance, no re-prompt.
     {loose, fragment, merged, ...} -> prefer a text candidate with IoU > 0.5 vs the old mask,
     else a box prompt on the old mask's bbox dilated by --dilate, replacing the instance.
  5. Untouched instances are copied unchanged (identical RLE). Every output instance carries a
     `"provenance"` field; the output is written to `<out-dir>/<stem>_masks.json`, re-indexed
     0..N-1, and one JSON line is appended to --log per frame processed.

Resumable: a frame whose `<out-dir>/<stem>_masks.json` already exists is skipped without opening
the image or the model. Never writes to --masks-dir (the input files).

`--passthrough-manifest` (optional, run after the worklist pass): given a manifest of frame image
paths (e.g. the same manifest a verifier re-verify pass will use), for every stem listed there that
has no `<out-dir>/<stem>_masks.json` yet (i.e. it wasn't in --worklist, or --worklist has no line
for it -- a frame with no flagged issues) copies its original masks through unchanged, re-indexed
with `provenance: {"source": "original"}` per instance, same as an untouched worklist instance.
Lets a re-verify pass that reads a single --masks-dir see every sampled frame, not just corrected
ones. Runs with no SAM3/segmenter needed (skips straight past `build_segmenter`), so this step is
safe to run on the login node, or piggy-backed on any array task -- but only run it once (not once
per array shard) to avoid redundant writes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from geometry import (
    dilate_xyxy,
    iou_masks,
    iou_xyxy,
    mask_bbox_xyxy,
    mask_centroid,
    mask_in_box_fraction,
    xyxy_norm_to_px,
    xyxy_to_xywh_norm,
)
from rle import rle_decode, rle_encode

RESEGMENT_DROP_ISSUES = {"wrong_object", "duplicate"}


def load_instances(masks_json_path: Path) -> list[dict]:
    with open(masks_json_path) as f:
        data = json.load(f)
    out = []
    for inst in data:
        inst = dict(inst)
        inst["mask"] = rle_decode(inst["rle"])
        out.append(inst)
    return out


def build_output_instance(mask: np.ndarray, provenance: dict) -> dict:
    bbox = mask_bbox_xyxy(mask)
    centroid = mask_centroid(mask)
    return {
        "instance_idx": None,  # reindexed by the caller
        "center_xy": centroid if centroid is not None else [0, 0],
        "area_px": int(mask.sum()),
        "rle": rle_encode(mask),
        "provenance": provenance,
    }


def resolve_frame_paths(rec: dict, frames_dir, masks_dir):
    image_path = Path(rec["image"])
    if not image_path.exists() and frames_dir is not None:
        image_path = Path(frames_dir) / image_path.name
    masks_json_path = Path(rec["masks_json"]) if rec.get("masks_json") else None
    if masks_json_path is not None and not masks_json_path.exists() and masks_dir is not None:
        masks_json_path = Path(masks_dir) / masks_json_path.name
    return image_path, masks_json_path


def process_frame(rec: dict, segmenter, args, image_path: Path, masks_json_path: Path):
    """Returns (out_instances, log_entry). Does not touch disk."""
    t0 = time.perf_counter()
    stem = image_path.stem
    n_fallback = 0
    n_text_candidates = 0
    actions_log = []

    pil = Image.open(image_path).convert("RGB")
    width, height = pil.size

    instances = load_instances(masks_json_path)
    inst_by_idx = {inst["instance_idx"]: inst for inst in instances}

    # idx -> {"mask": np.ndarray, "provenance": dict}; starts as "keep everything unchanged"
    kept: dict[int, dict] = {
        idx: {"mask": inst["mask"], "provenance": {"source": "original", "orig_idx": idx}}
        for idx, inst in inst_by_idx.items()
    }
    added: list[dict] = []  # [{"mask", "provenance"}]

    text_candidates_cache = None

    def get_text_candidates():
        nonlocal text_candidates_cache, n_text_candidates
        if text_candidates_cache is None:
            text_candidates_cache = segmenter.text(pil, args.text_prompt)
            n_text_candidates = len(text_candidates_cache)
        return text_candidates_cache

    def overlaps_kept(mask: np.ndarray) -> bool:
        return any(iou_masks(mask, k["mask"]) > 0.5 for k in kept.values())

    for action in rec.get("actions", []):
        atype = action.get("type")

        if atype == "resegment":
            idx = action.get("idx")
            issue = action.get("issue")
            if idx not in inst_by_idx:
                actions_log.append({"type": "resegment", "idx": idx, "issue": issue, "result": "idx_not_found"})
                continue

            if issue in RESEGMENT_DROP_ISSUES:
                kept.pop(idx, None)
                actions_log.append({"type": "resegment", "idx": idx, "issue": issue, "result": "removed"})
                continue

            old_mask = inst_by_idx[idx]["mask"]
            cand = None
            prompt_kind = None
            for c in get_text_candidates():
                if iou_masks(c.mask, old_mask) > 0.5:
                    cand = c
                    prompt_kind = "text"
                    break
            if cand is None:
                bbox = mask_bbox_xyxy(old_mask)
                if bbox is None:
                    actions_log.append({"type": "resegment", "idx": idx, "issue": issue, "result": "empty_old_mask"})
                    continue
                dilated = dilate_xyxy(bbox, args.dilate, width, height)
                xywh_norm = xyxy_to_xywh_norm(dilated, width, height)
                point = inst_by_idx[idx].get("center_xy")
                cand = segmenter.box(pil, xywh_norm, point=point)
                prompt_kind = "box"
                n_fallback += 1

            if cand is None:
                actions_log.append({"type": "resegment", "idx": idx, "issue": issue, "result": "no_candidate"})
                continue

            kept[idx] = {
                "mask": cand.mask,
                "provenance": {
                    "source": "auto",
                    "action": "resegment",
                    "issue": issue,
                    "orig_idx": idx,
                    "prompt": prompt_kind,
                    "score": cand.score,
                },
            }
            actions_log.append({"type": "resegment", "idx": idx, "issue": issue, "result": "replaced", "prompt": prompt_kind})

        elif atype == "add":
            box_norm = action.get("box")
            if not box_norm or len(box_norm) != 4:
                actions_log.append({"type": "add", "result": "bad_box"})
                continue
            box_px = xyxy_norm_to_px(box_norm, width, height)

            cand = None
            prompt_kind = None
            best_iou = 0.0
            for c in get_text_candidates():
                if overlaps_kept(c.mask):
                    continue
                bbox = mask_bbox_xyxy(c.mask)
                if bbox is None:
                    continue
                cand_iou = iou_xyxy(bbox, box_px)
                if cand_iou < args.iou_match:
                    continue
                if mask_in_box_fraction(c.mask, box_px) < 0.5:
                    continue
                if cand_iou > best_iou:
                    best_iou = cand_iou
                    cand = c
                    prompt_kind = "text"

            if cand is None:
                x0n, y0n, x1n, y1n = box_norm
                xywh_norm = [x0n, y0n, x1n - x0n, y1n - y0n]
                point = [(box_px[0] + box_px[2]) / 2, (box_px[1] + box_px[3]) / 2]
                cand = segmenter.box(pil, xywh_norm, point=point)
                prompt_kind = "box"
                n_fallback += 1

            if cand is None:
                actions_log.append({"type": "add", "box": box_norm, "result": "no_candidate"})
                continue

            added.append({
                "mask": cand.mask,
                "provenance": {
                    "source": "auto",
                    "action": "add",
                    "prompt": prompt_kind,
                    "score": cand.score,
                    "note": action.get("note"),
                },
            })
            actions_log.append({"type": "add", "box": box_norm, "result": "added", "prompt": prompt_kind})

        else:
            actions_log.append({"type": atype, "result": "unknown_action_type"})

    out_instances = []
    for idx in sorted(kept):
        out_instances.append(build_output_instance(kept[idx]["mask"], kept[idx]["provenance"]))
    for a in added:
        out_instances.append(build_output_instance(a["mask"], a["provenance"]))
    for i, inst in enumerate(out_instances):
        inst["instance_idx"] = i

    log_entry = {
        "stem": stem,
        "image": str(image_path),
        "masks_json": str(masks_json_path),
        "n_masks_in": len(instances),
        "n_masks_out": len(out_instances),
        "n_removed": len(instances) - len(kept),
        "n_added": len(added),
        "n_text_candidates": n_text_candidates,
        "n_fallback_box_prompts": n_fallback,
        "actions": actions_log,
        "elapsed_s": time.perf_counter() - t0,
    }
    return out_instances, log_entry


def passthrough_masks(masks_json_path: Path) -> list[dict]:
    """Copy every instance in masks_json_path through unchanged (same mask/RLE), tagged with
    provenance {"source": "original"}, re-indexed 0..N-1 -- same shape process_frame's "untouched"
    instances get, for a frame with no worklist actions at all."""
    instances = load_instances(masks_json_path)
    out = [
        build_output_instance(inst["mask"], {"source": "original", "orig_idx": inst["instance_idx"]})
        for inst in instances
    ]
    for i, inst in enumerate(out):
        inst["instance_idx"] = i
    return out


def iter_worklist(path: Path, limit=None):
    n = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            n += 1
            if limit is not None and n >= limit:
                return


def build_segmenter(args):
    if args.fake:
        from segmenter import FakeSegmenter

        return FakeSegmenter()
    from segmenter import Sam3Segmenter

    if not args.checkpoint:
        sys.exit("correct.py: --checkpoint is required unless --fake is given")
    return Sam3Segmenter(checkpoint=args.checkpoint, device=args.device, score_thresh=args.score_thresh)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worklist", required=True, help="outputs/worklist*.jsonl (tools/worklist.py format)")
    ap.add_argument("--frames-dir", help="fallback dir if a worklist `image` path doesn't exist as given")
    ap.add_argument("--masks-dir", help="fallback dir if a worklist `masks_json` path doesn't exist as given")
    ap.add_argument("--out-dir", required=True, help="write <stem>_masks.json here")
    ap.add_argument("--checkpoint", help="SAM3 .pt checkpoint (required unless --fake)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--score-thresh", type=float, default=0.5, help="SAM3 output_prob_thresh")
    ap.add_argument("--min-confidence", choices=["high", "low"], default="low",
                     help="'high': only process worklist rows with confidence=='high'. 'low' (default): process all rows.")
    ap.add_argument("--text-prompt", default="person")
    ap.add_argument("--iou-match", type=float, default=0.5, help="min box IoU to match a text candidate to a proposed box/old mask")
    ap.add_argument("--dilate", type=float, default=0.10, help="fraction of box width/height to grow a resegment box prompt by, per side")
    ap.add_argument("--log", default="corrections.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="process at most N worklist rows")
    ap.add_argument("--dry-run", action="store_true", help="run the model and compute corrections but do not write <out-dir> or --log")
    ap.add_argument("--fake", action="store_true", help="use segmenter.FakeSegmenter instead of SAM3 (no torch/sam3 import; for tests/dry runs on the login node)")
    ap.add_argument("--passthrough-manifest",
                     help="manifest of frame image paths; after the worklist pass, copy original masks "
                          "unchanged into --out-dir for any listed stem that --worklist didn't produce "
                          "an output for (no SAM3 needed -- run once, not once per array shard)")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    segmenter = None
    n_written = n_skipped_exists = n_skipped_confidence = n_error = 0
    log_fh = None if args.dry_run else open(args.log, "a")

    try:
        for rec in iter_worklist(Path(args.worklist), limit=args.limit):
            image_path, masks_json_path = resolve_frame_paths(rec, args.frames_dir, args.masks_dir)
            stem = image_path.stem
            out_path = out_dir / f"{stem}_masks.json"

            if out_path.exists() and not args.dry_run:
                n_skipped_exists += 1
                continue
            if args.min_confidence == "high" and rec.get("confidence") != "high":
                n_skipped_confidence += 1
                continue
            if masks_json_path is None or not masks_json_path.exists():
                print(f"correct.py: no masks_json for {image_path}, skipping", file=sys.stderr)
                n_error += 1
                continue
            if not image_path.exists():
                print(f"correct.py: no image at {image_path}, skipping", file=sys.stderr)
                n_error += 1
                continue

            if segmenter is None:
                segmenter = build_segmenter(args)

            try:
                out_instances, log_entry = process_frame(rec, segmenter, args, image_path, masks_json_path)
            except Exception as e:  # noqa: BLE001
                print(f"correct.py: error on {stem}: {e}", file=sys.stderr)
                n_error += 1
                continue

            if args.dry_run:
                print(json.dumps(log_entry))
            else:
                with open(out_path, "w") as f:
                    json.dump(out_instances, f)
                log_fh.write(json.dumps(log_entry) + "\n")
                log_fh.flush()
            n_written += 1
    finally:
        if segmenter is not None:
            segmenter.close()
        if log_fh is not None:
            log_fh.close()

    print(
        f"correct.py: written={n_written} skipped_exists={n_skipped_exists} "
        f"skipped_confidence={n_skipped_confidence} errors={n_error}",
        file=sys.stderr,
    )

    if args.passthrough_manifest:
        n_pass_written = n_pass_skipped = n_pass_error = 0
        with open(args.passthrough_manifest) as f:
            manifest_lines = [line.strip() for line in f if line.strip()]
        for line in manifest_lines:
            image_path = Path(line)
            stem = image_path.stem
            out_path = out_dir / f"{stem}_masks.json"
            if out_path.exists():
                n_pass_skipped += 1
                continue
            masks_json_path = Path(args.masks_dir) / f"{stem}_masks.json" if args.masks_dir else None
            if masks_json_path is None or not masks_json_path.exists():
                print(f"correct.py: passthrough: no masks_json for {stem}, skipping", file=sys.stderr)
                n_pass_error += 1
                continue
            out_instances = passthrough_masks(masks_json_path)
            if not args.dry_run:
                with open(out_path, "w") as f:
                    json.dump(out_instances, f)
            n_pass_written += 1
        print(
            f"correct.py: passthrough written={n_pass_written} skipped_exists={n_pass_skipped} "
            f"errors={n_pass_error}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
