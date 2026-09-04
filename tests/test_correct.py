"""Tests run with the login-node python ($SCRATCH/envs/vlm-verify/bin/python -- has numpy, cv2,
PIL; pycocotools is checked for at import time and its tests skip if absent). No sam3/torch import
anywhere in this file, so it runs with no GPU and no container.

    $SCRATCH/envs/vlm-verify/bin/python -m pytest tests/ -v
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from geometry import (  # noqa: E402
    dilate_xyxy,
    iou_masks,
    iou_xyxy,
    mask_bbox_xyxy,
    mask_centroid,
    mask_in_box_fraction,
    xyxy_norm_to_px,
    xyxy_to_xywh_norm,
)  # mask_centroid also used directly below
from rle import rle_decode, rle_encode  # noqa: E402
from segmenter import FakeSegmenter  # noqa: E402
from correct import build_output_instance, load_instances, process_frame  # noqa: E402

SCRATCH = os.environ.get("SCRATCH", "/scratch/b5bd/obrookes.b5bd")
REAL_MASKS_DIR = Path(SCRATCH) / "masks"
REAL_FRAMES_DIR = Path(SCRATCH) / "frames"


def _first_real_masks_file():
    if not REAL_MASKS_DIR.is_dir():
        return None
    for p in sorted(REAL_MASKS_DIR.glob("*_masks.json")):
        stem = p.name[: -len("_masks.json")]
        if (REAL_FRAMES_DIR / f"{stem}.png").exists():
            return p
    return None


REAL_MASKS_FILE = _first_real_masks_file()
requires_real_data = pytest.mark.skipif(REAL_MASKS_FILE is None, reason=f"no {SCRATCH}/masks + frames pair found")


# ── geometry ──────────────────────────────────────────────────────────────

def test_iou_xyxy_basic():
    assert iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou_xyxy([0, 0, 10, 10], [10, 10, 20, 20]) == 0.0
    assert abs(iou_xyxy([0, 0, 10, 10], [5, 0, 15, 10]) - (50 / 150)) < 1e-9


def test_dilate_xyxy_clips_to_image():
    box = dilate_xyxy([0, 0, 10, 10], 0.5, width=12, height=100)
    assert box[0] == 0.0
    assert box[2] == 12.0  # would be 15, clipped to width


def test_mask_bbox_and_centroid_roundtrip():
    m = np.zeros((20, 30), dtype=bool)
    m[5:10, 8:15] = True
    bbox = mask_bbox_xyxy(m)
    assert bbox == [8, 5, 15, 10]
    cx, cy = mask_centroid(m)
    assert 8 <= cx < 15 and 5 <= cy < 10


def test_mask_in_box_fraction():
    m = np.zeros((10, 10), dtype=bool)
    m[0:4, 0:4] = True  # 16 px
    assert mask_in_box_fraction(m, [0, 0, 4, 4]) == 1.0
    assert mask_in_box_fraction(m, [0, 0, 2, 2]) == pytest.approx(4 / 16)


def test_xywh_norm_conversions_roundtrip():
    box_px = [10, 20, 110, 220]
    norm = xyxy_to_xywh_norm(box_px, width=200, height=400)
    back = xyxy_norm_to_px([norm[0], norm[1], norm[0] + norm[2], norm[1] + norm[3]], width=200, height=400)
    assert [round(v, 6) for v in back] == [10.0, 20.0, 110.0, 220.0]


def test_iou_masks():
    a = np.zeros((10, 10), dtype=bool)
    b = np.zeros((10, 10), dtype=bool)
    a[0:5, 0:5] = True
    b[0:5, 0:5] = True
    assert iou_masks(a, b) == 1.0
    b[:] = False
    b[5:10, 5:10] = True
    assert iou_masks(a, b) == 0.0


# ── rle ───────────────────────────────────────────────────────────────────

def test_rle_roundtrip_synthetic():
    rng = np.random.default_rng(0)
    for _ in range(5):
        h, w = rng.integers(5, 40), rng.integers(5, 40)
        mask = rng.random((h, w)) > 0.6
        rle = rle_encode(mask)
        back = rle_decode(rle)
        assert back.shape == mask.shape
        assert np.array_equal(back, mask)


def test_rle_roundtrip_all_false_and_all_true():
    for val in (False, True):
        mask = np.full((7, 9), val, dtype=bool)
        assert np.array_equal(rle_decode(rle_encode(mask)), mask)


@requires_real_data
def test_rle_decode_matches_area_px_on_real_file():
    instances = json.loads(REAL_MASKS_FILE.read_text())
    assert instances, "expected at least one instance"
    for inst in instances:
        mask = rle_decode(inst["rle"])
        assert int(mask.sum()) == inst["area_px"]


@requires_real_data
def test_rle_encode_roundtrip_on_real_file():
    instances = json.loads(REAL_MASKS_FILE.read_text())
    for inst in instances:
        mask = rle_decode(inst["rle"])
        re_decoded = rle_decode(rle_encode(mask))
        assert np.array_equal(mask, re_decoded)


def test_rle_encode_agrees_with_pycocotools_when_available():
    pycocotools = pytest.importorskip("pycocotools")
    from pycocotools import mask as mask_api

    rng = np.random.default_rng(1)
    mask = rng.random((30, 25)) > 0.5
    ours = rle_encode(mask)
    fortran = np.asfortranarray(mask.astype(np.uint8))
    theirs = mask_api.encode(fortran)
    theirs_counts = theirs["counts"]
    if isinstance(theirs_counts, bytes):
        theirs_counts = theirs_counts.decode("ascii")
    assert ours["counts"] == theirs_counts
    assert ours["size"] == list(theirs["size"])


# ── correct.py logic (FakeSegmenter) ───────────────────────────────────────

def _make_masks_json(tmp_path, instances):
    p = tmp_path / "frame_masks.json"
    out = []
    for inst in instances:
        d = dict(inst)
        mask = d.pop("mask")
        d["rle"] = rle_encode(mask)
        d.setdefault("area_px", int(mask.sum()))
        d.setdefault("center_xy", list(mask_centroid(mask) or [0, 0]))
        out.append(d)
    p.write_text(json.dumps(out))
    return p


def _square_mask(h, w, box_xyxy):
    m = np.zeros((h, w), dtype=bool)
    x0, y0, x1, y1 = box_xyxy
    m[y0:y1, x0:x1] = True
    return m


@requires_real_data
def _real_frame_and_stem():
    stem = REAL_MASKS_FILE.name[: -len("_masks.json")]
    return REAL_FRAMES_DIR / f"{stem}.png", stem


class _Args:
    text_prompt = "person"
    iou_match = 0.5
    dilate = 0.10


@requires_real_data
def test_process_frame_no_actions_leaves_masks_unchanged(tmp_path):
    image_path, stem = _real_frame_and_stem()
    instances = load_instances(REAL_MASKS_FILE)
    rec = {"image": str(image_path), "masks_json": str(REAL_MASKS_FILE), "actions": [], "confidence": "low"}
    fake = FakeSegmenter()
    out_instances, log_entry = process_frame(rec, fake, _Args(), image_path, REAL_MASKS_FILE)
    assert len(out_instances) == len(instances)
    for orig, out in zip(instances, out_instances):
        assert out["provenance"]["source"] == "original"
        assert out["rle"] == orig["rle"]  # byte-identical RLE for untouched instances
    assert fake.calls == []  # no text/box calls needed when there are no actions
    assert log_entry["n_masks_out"] == len(instances)


@requires_real_data
def test_process_frame_resegment_duplicate_removes_instance(tmp_path):
    image_path, stem = _real_frame_and_stem()
    instances = load_instances(REAL_MASKS_FILE)
    idx0 = instances[0]["instance_idx"]
    rec = {
        "image": str(image_path),
        "masks_json": str(REAL_MASKS_FILE),
        "actions": [{"type": "resegment", "idx": idx0, "issue": "duplicate"}],
        "confidence": "low",
    }
    fake = FakeSegmenter()
    out_instances, log_entry = process_frame(rec, fake, _Args(), image_path, REAL_MASKS_FILE)
    assert len(out_instances) == len(instances) - 1
    assert log_entry["n_removed"] == 1
    assert fake.calls == []  # drop path never calls the segmenter


@requires_real_data
def test_process_frame_resegment_loose_falls_back_to_box(tmp_path):
    image_path, stem = _real_frame_and_stem()
    instances = load_instances(REAL_MASKS_FILE)
    idx0 = instances[0]["instance_idx"]
    rec = {
        "image": str(image_path),
        "masks_json": str(REAL_MASKS_FILE),
        "actions": [{"type": "resegment", "idx": idx0, "issue": "loose"}],
        "confidence": "low",
    }
    fake = FakeSegmenter()
    out_instances, log_entry = process_frame(rec, fake, _Args(), image_path, REAL_MASKS_FILE)
    assert len(out_instances) == len(instances)
    replaced = [o for o in out_instances if o["provenance"].get("orig_idx") == idx0]
    assert len(replaced) == 1
    assert replaced[0]["provenance"]["source"] == "auto"
    assert replaced[0]["provenance"]["prompt"] == "box"
    # FakeSegmenter.text() returns a whole-frame mask, which never has IoU > 0.5 vs a small
    # person mask, so this must go through the box fallback exactly once.
    assert sum(1 for c in fake.calls if c[0] == "box") == 1
    assert log_entry["n_fallback_box_prompts"] == 1


def test_process_frame_add_matches_text_candidate(tmp_path):
    h, w = 100, 100
    existing = _square_mask(h, w, [10, 10, 20, 20])
    masks_json = _make_masks_json(tmp_path, [{"instance_idx": 0, "mask": existing}])
    image_path = tmp_path / "frame.png"
    from PIL import Image

    Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)).save(image_path)

    rec = {
        "image": str(image_path),
        "masks_json": str(masks_json),
        "actions": [{"type": "add", "box": [0.0, 0.0, 1.0, 1.0], "note": "whole frame"}],
        "confidence": "low",
    }
    fake = FakeSegmenter()  # .text() returns a whole-frame mask -> should match the add box directly
    out_instances, log_entry = process_frame(rec, fake, _Args(), image_path, masks_json)
    assert len(out_instances) == 2
    added = [o for o in out_instances if o["provenance"].get("action") == "add"]
    assert len(added) == 1
    assert added[0]["provenance"]["prompt"] == "text"
    assert sum(1 for c in fake.calls if c[0] == "box") == 0  # text candidate matched, no fallback needed
    assert log_entry["n_added"] == 1


def test_process_frame_add_overlapping_existing_kept_mask_is_skipped(tmp_path):
    h, w = 50, 50
    existing = _square_mask(h, w, [0, 0, 50, 50])  # whole frame, same as FakeSegmenter.text()'s candidate
    masks_json = _make_masks_json(tmp_path, [{"instance_idx": 0, "mask": existing}])
    image_path = tmp_path / "frame.png"
    from PIL import Image

    Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)).save(image_path)

    rec = {
        "image": str(image_path),
        "masks_json": str(masks_json),
        "actions": [{"type": "add", "box": [0.0, 0.0, 1.0, 1.0], "note": "dup of existing"}],
        "confidence": "low",
    }
    fake = FakeSegmenter()
    out_instances, log_entry = process_frame(rec, fake, _Args(), image_path, masks_json)
    # the only text candidate overlaps the existing kept mask entirely, so "add" must fall back
    # to a box prompt (still producing a new instance) rather than silently reusing it.
    added = [o for o in out_instances if o["provenance"].get("action") == "add"]
    assert len(added) == 1
    assert added[0]["provenance"]["prompt"] == "box"


def test_process_frame_reindexes_output_0_to_n_minus_1(tmp_path):
    h, w = 40, 40
    instances = [
        {"instance_idx": 5, "mask": _square_mask(h, w, [0, 0, 5, 5])},
        {"instance_idx": 9, "mask": _square_mask(h, w, [10, 10, 15, 15])},
    ]
    masks_json = _make_masks_json(tmp_path, instances)
    image_path = tmp_path / "frame.png"
    from PIL import Image

    Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)).save(image_path)
    rec = {"image": str(image_path), "masks_json": str(masks_json), "actions": [], "confidence": "low"}
    fake = FakeSegmenter()
    out_instances, _ = process_frame(rec, fake, _Args(), image_path, masks_json)
    assert [o["instance_idx"] for o in out_instances] == list(range(len(out_instances)))
    assert [o["provenance"]["orig_idx"] for o in out_instances] == [5, 9]


# ── build_output_instance ───────────────────────────────────────────────────

def test_build_output_instance_area_and_rle_consistent():
    mask = _square_mask(20, 20, [2, 3, 10, 8])
    inst = build_output_instance(mask, {"source": "original"})
    assert inst["area_px"] == int(mask.sum())
    assert np.array_equal(rle_decode(inst["rle"]), mask)


# ── CLI end-to-end (--fake, --dry-run and a real write) ────────────────────

@requires_real_data
def test_cli_fake_writes_output_and_is_resumable(tmp_path):
    image_path, stem = _real_frame_and_stem()
    worklist = tmp_path / "worklist.jsonl"
    rec = {
        "image": str(image_path),
        "masks_json": str(REAL_MASKS_FILE),
        "actions": [],
        "confidence": "low",
    }
    worklist.write_text(json.dumps(rec) + "\n")
    out_dir = tmp_path / "out"
    log_path = tmp_path / "corrections.jsonl"

    cmd = [
        sys.executable, str(REPO_ROOT / "correct.py"),
        "--worklist", str(worklist),
        "--out-dir", str(out_dir),
        "--log", str(log_path),
        "--fake",
    ]
    r1 = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert r1.returncode == 0, r1.stderr
    out_file = out_dir / f"{stem}_masks.json"
    assert out_file.exists()
    assert "written=1" in r1.stderr
    mtime1 = out_file.stat().st_mtime

    r2 = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert r2.returncode == 0, r2.stderr
    assert "skipped_exists=1" in r2.stderr
    assert out_file.stat().st_mtime == mtime1  # untouched on the resumed run


@requires_real_data
def test_cli_dry_run_does_not_create_out_dir(tmp_path):
    image_path, stem = _real_frame_and_stem()
    worklist = tmp_path / "worklist.jsonl"
    rec = {"image": str(image_path), "masks_json": str(REAL_MASKS_FILE), "actions": [], "confidence": "low"}
    worklist.write_text(json.dumps(rec) + "\n")
    out_dir = tmp_path / "out_dry"

    cmd = [
        sys.executable, str(REPO_ROOT / "correct.py"),
        "--worklist", str(worklist),
        "--out-dir", str(out_dir),
        "--fake", "--dry-run",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert r.returncode == 0, r.stderr
    assert not out_dir.exists()
    assert json.loads(r.stdout.strip().splitlines()[0])["stem"] == stem
