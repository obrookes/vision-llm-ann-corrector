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
from rle import rle_decode, rle_encode, rle_encode_intlist  # noqa: E402
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


GENERATOR_REPO = Path("/home/b5bd/obrookes.b5bd/vision-llm-ann-generator")
requires_generator_repo = pytest.mark.skipif(
    not (GENERATOR_REPO / "tracks.py").is_file(), reason=f"{GENERATOR_REPO}/tracks.py not found"
)


def test_rle_encode_compressed_format_unchanged():
    """fmt="compressed" (the default) must still behave exactly as rle_encode did before the
    --rle-format flag was added: same compressed-string counts, same roundtrip."""
    rng = np.random.default_rng(7)
    mask = rng.random((17, 23)) > 0.5
    default = rle_encode(mask)
    explicit = rle_encode(mask, fmt="compressed")
    assert default == explicit
    assert isinstance(default["counts"], str)
    assert np.array_equal(rle_decode(default), mask)


@requires_generator_repo
def test_rle_encode_intlist_roundtrips_with_own_and_generator_decoder():
    sys.path.insert(0, str(GENERATOR_REPO))
    import tracks as generator_tracks  # noqa: E402  (generator repo, not a package of this repo)

    rng = np.random.default_rng(3)
    for _ in range(5):
        h, w = int(rng.integers(5, 40)), int(rng.integers(5, 40))
        mask = rng.random((h, w)) > 0.5

        rle = rle_encode_intlist(mask)
        assert isinstance(rle["counts"], list)
        assert all(isinstance(c, int) for c in rle["counts"])

        assert np.array_equal(rle_decode(rle), mask)  # this repo's own decoder

        theirs = generator_tracks.rle_decode(rle)
        assert theirs.dtype == np.bool_ or theirs.dtype == bool
        assert np.array_equal(theirs.astype(bool), mask)

    # rle_encode(mask, fmt="intlist") must be the same call
    mask = rng.random((11, 9)) > 0.5
    assert rle_encode(mask, fmt="intlist") == rle_encode_intlist(mask)


def test_rle_encode_unknown_fmt_raises():
    mask = np.zeros((3, 3), dtype=bool)
    with pytest.raises(ValueError):
        rle_encode(mask, fmt="bogus")


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
    keep_ids = False
    rle_format = "compressed"


class _KeepIdsArgs(_Args):
    keep_ids = True


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

# ── hard_case action ─────────────────────────────────────────────────────

class _ZonedFakeSegmenter:
    """FakeSegmenter-like stand-in whose .text() returns several fixed candidates at known
    positions instead of a single whole-frame rectangle, so box/zone/score selection logic can be
    exercised precisely. Records every call like FakeSegmenter."""

    def __init__(self, candidates_by_thresh):
        self.calls = []
        self._candidates_by_thresh = candidates_by_thresh  # {score_thresh_or_None: [Candidate,...]}

    def text(self, pil, prompt, score_thresh=None):
        self.calls.append(("text", prompt, score_thresh))
        return list(self._candidates_by_thresh.get(score_thresh, []))

    def box(self, pil, xywh_norm, point=None):
        self.calls.append(("box", tuple(xywh_norm), point))
        w, h = pil.size
        x0n, y0n, wn, hn = xywh_norm
        x0, y0 = int(round(x0n * w)), int(round(y0n * h))
        x1, y1 = int(round((x0n + wn) * w)), int(round((y0n + hn) * h))
        mask = np.zeros((h, w), dtype=bool)
        mask[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = True
        from segmenter import _candidate_from_mask
        return _candidate_from_mask(mask, 0.7)

    def close(self):
        pass


def _empty_frame(tmp_path, h=90, w=90):
    masks_json = _make_masks_json(tmp_path, [])
    image_path = tmp_path / "frame.png"
    from PIL import Image
    Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)).save(image_path)
    return image_path, masks_json


def test_hard_case_box_candidate_inside_box_accepted(tmp_path):
    h, w = 90, 90
    image_path, masks_json = _empty_frame(tmp_path, h, w)
    from segmenter import _candidate_from_mask
    inside = _square_mask(h, w, [10, 10, 20, 20])  # fully inside the box below
    outside = _square_mask(h, w, [70, 70, 89, 89])  # not overlapping the box at all
    fake = _ZonedFakeSegmenter({0.1: [_candidate_from_mask(outside, 0.5), _candidate_from_mask(inside, 0.9)]})

    rec = {
        "image": str(image_path), "masks_json": str(masks_json),
        "actions": [{"type": "hard_case", "box": [0.0, 0.0, 0.4, 0.4], "zone": None,
                     "difficulty": "hard", "score_thresh": 0.1, "source": "motion", "votes": 1}],
        "confidence": "low",
    }
    out_instances, log_entry = process_frame(rec, fake, _Args(), image_path, masks_json)
    added = [o for o in out_instances if o["provenance"].get("action") == "hard_case"]
    assert len(added) == 1
    assert added[0]["provenance"]["prompt"] == "text_lowthresh"
    assert added[0]["provenance"]["score"] == pytest.approx(0.9)
    assert added[0]["provenance"]["hard_case_source"] == "motion"
    assert added[0]["provenance"]["score_thresh"] == 0.1
    assert sum(1 for c in fake.calls if c[0] == "box") == 0
    assert log_entry["actions"][0]["result"] == "added"


def test_hard_case_box_candidate_outside_box_falls_back_to_box_prompt(tmp_path):
    h, w = 90, 90
    image_path, masks_json = _empty_frame(tmp_path, h, w)
    from segmenter import _candidate_from_mask
    outside = _square_mask(h, w, [70, 70, 89, 89])
    fake = _ZonedFakeSegmenter({0.1: [_candidate_from_mask(outside, 0.9)]})

    rec = {
        "image": str(image_path), "masks_json": str(masks_json),
        "actions": [{"type": "hard_case", "box": [0.0, 0.0, 0.2, 0.2], "zone": None,
                     "difficulty": "hard", "score_thresh": 0.1, "source": "none", "votes": 1}],
        "confidence": "low",
    }
    out_instances, log_entry = process_frame(rec, fake, _Args(), image_path, masks_json)
    added = [o for o in out_instances if o["provenance"].get("action") == "hard_case"]
    assert len(added) == 1
    assert added[0]["provenance"]["prompt"] == "box"
    assert sum(1 for c in fake.calls if c[0] == "box") == 1
    assert log_entry["n_fallback_box_prompts"] == 1


def test_hard_case_null_box_zone_filter(tmp_path):
    h, w = 90, 90
    image_path, masks_json = _empty_frame(tmp_path, h, w)
    from segmenter import _candidate_from_mask
    upper = _square_mask(h, w, [10, 0, 20, 10])     # centre y ~5, upper third (< 30)
    lower = _square_mask(h, w, [10, 70, 20, 85])    # centre y ~77, lower third (>= 60)
    fake = _ZonedFakeSegmenter({0.1: [_candidate_from_mask(upper, 0.95), _candidate_from_mask(lower, 0.5)]})

    rec = {
        "image": str(image_path), "masks_json": str(masks_json),
        "actions": [{"type": "hard_case", "box": None, "zone": "lower",
                     "difficulty": "moderate", "score_thresh": 0.1, "source": "motion", "votes": 1}],
        "confidence": "low",
    }
    out_instances, log_entry = process_frame(rec, fake, _Args(), image_path, masks_json)
    added = [o for o in out_instances if o["provenance"].get("action") == "hard_case"]
    assert len(added) == 1
    # zone=lower must pick the lower-zone candidate (score 0.5) even though the upper-zone one
    # scores higher -- proves the zone filter is applied before the max-score pick.
    assert added[0]["provenance"]["score"] == pytest.approx(0.5)


def test_hard_case_no_candidate_no_change_and_logged(tmp_path):
    h, w = 90, 90
    image_path, masks_json = _empty_frame(tmp_path, h, w)
    fake = _ZonedFakeSegmenter({0.1: []})  # no candidates at all

    rec = {
        "image": str(image_path), "masks_json": str(masks_json),
        "actions": [{"type": "hard_case", "box": None, "zone": None,
                     "difficulty": "hard", "score_thresh": 0.1, "source": "none", "votes": 1}],
        "confidence": "low",
    }
    out_instances, log_entry = process_frame(rec, fake, _Args(), image_path, masks_json)
    assert out_instances == []
    assert log_entry["n_added"] == 0
    assert log_entry["actions"][0]["result"] == "hard_case_no_candidate"


def test_hard_case_per_action_threshold_passed_to_segmenter_default_pass_unaffected(tmp_path):
    """The hard_case pass must call segmenter.text(..., score_thresh=0.1); an "add" action on the
    same frame must still use the cached default-threshold pass (score_thresh=None), proving the
    two are cached/called separately."""
    h, w = 90, 90
    image_path, masks_json = _empty_frame(tmp_path, h, w)
    from segmenter import _candidate_from_mask
    default_cand = _square_mask(h, w, [0, 0, 90, 90])  # whole frame, matches the add box below
    low_cand = _square_mask(h, w, [10, 10, 20, 20])
    fake = _ZonedFakeSegmenter({
        None: [_candidate_from_mask(default_cand, 0.9)],
        0.1: [_candidate_from_mask(low_cand, 0.9)],
    })

    rec = {
        "image": str(image_path), "masks_json": str(masks_json),
        "actions": [
            {"type": "add", "box": [0.0, 0.0, 1.0, 1.0], "note": "whole frame"},
            {"type": "hard_case", "box": [0.0, 0.0, 0.4, 0.4], "zone": None,
             "difficulty": "hard", "score_thresh": 0.1, "source": "motion", "votes": 1},
        ],
        "confidence": "low",
    }
    out_instances, log_entry = process_frame(rec, fake, _Args(), image_path, masks_json)
    text_calls = [c for c in fake.calls if c[0] == "text"]
    assert ("text", "person", None) in text_calls
    assert ("text", "person", 0.1) in text_calls
    added = {o["provenance"]["action"]: o for o in out_instances}
    assert added["add"]["provenance"]["prompt"] == "text"
    assert added["hard_case"]["provenance"]["prompt"] == "text_lowthresh"


def test_hard_case_max_new_caps_additional_instances(tmp_path):
    h, w = 90, 90
    image_path, masks_json = _empty_frame(tmp_path, h, w)
    from segmenter import _candidate_from_mask
    cand = _square_mask(h, w, [10, 10, 20, 20])
    fake = _ZonedFakeSegmenter({0.1: [_candidate_from_mask(cand, 0.9)]})

    rec = {
        "image": str(image_path), "masks_json": str(masks_json),
        "actions": [
            {"type": "hard_case", "box": [0.0, 0.0, 0.4, 0.4], "zone": None, "score_thresh": 0.1,
             "source": "motion", "votes": 1},
            {"type": "hard_case", "box": [0.0, 0.0, 0.4, 0.4], "zone": None, "score_thresh": 0.1,
             "source": "motion", "votes": 1},
        ],
        "confidence": "low",
    }
    out_instances, log_entry = process_frame(rec, fake, _Args(), image_path, masks_json)
    added = [o for o in out_instances if o["provenance"].get("action") == "hard_case"]
    assert len(added) == 1  # max_new defaults to 1, second hard_case action is a no-op
    assert log_entry["actions"][1]["result"] == "max_new_reached"


def test_unknown_action_type_does_not_break_processing(tmp_path):
    h, w = 40, 40
    image_path, masks_json = _empty_frame(tmp_path, h, w)
    fake = FakeSegmenter()
    rec = {
        "image": str(image_path), "masks_json": str(masks_json),
        "actions": [{"type": "some_future_action", "foo": "bar"}],
        "confidence": "low",
    }
    out_instances, log_entry = process_frame(rec, fake, _Args(), image_path, masks_json)
    assert out_instances == []
    assert log_entry["actions"][0]["result"] == "unknown_action_type"


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
