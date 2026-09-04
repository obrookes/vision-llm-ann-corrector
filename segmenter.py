"""The only module allowed to import sam3/torch (mirrors vision-llm-ann-generator/sam3_runner.py's
own rule and docstring convention). Both are imported lazily inside methods so correct.py, geometry.py,
rle.py and the test suite run on a CPU-only login node without sam3 or torch installed.

Exposes:
    Sam3Segmenter(checkpoint, device="cuda", score_thresh=0.5)
        .text(pil, prompt) -> list[Candidate]
        .box(pil, xywh_norm, point=None) -> Candidate | None
        .close()
    FakeSegmenter(...)          # same interface, returns synthetic rectangles; no sam3/torch import
    Candidate                    # {mask: bool (H, W) np.ndarray, box_xywh: pixel xyxy [x0,y0,x1,y1], score: float}

SAM3 API as installed (sam3 0.1.0, checked inside $SCRATCH/containers/sam3.sif on the login node --
CPU-only `import inspect; inspect.getsource(...)` of the modules below, no model build/GPU use):

    from sam3.model_builder import build_sam3_video_predictor
    predictor = build_sam3_video_predictor(checkpoint_path=<local .pt>, has_presence_token=...)
    predictor.handle_request(dict(type="start_session", resource_path=[pil_image]))  -> {"session_id"}
        NOTE: sam3/model/sam3_video_inference.py:1712-1716 `is_image_type` treats a list of length
        1 as a still image -- exactly what we want here (one frame per session, no propagation).
    predictor.handle_request(dict(type="add_prompt", session_id, frame_index=0,
                                   text=..., output_prob_thresh=0.5))
    predictor.handle_request(dict(type="add_prompt", session_id, frame_index=0,
                                   bounding_boxes=[[x, y, w, h]], bounding_box_labels=[1],
                                   output_prob_thresh=0.5))
        -> both return {"frame_index": int, "outputs": {out_obj_ids, out_probs, out_boxes_xywh
                        (normalised, unused -- we recompute box_xywh from the mask itself),
                        out_binary_masks (N, H, W) bool}}
    predictor.handle_request(dict(type="close_session", session_id))

    Coordinates: sam3/model/sam3_base_predictor.py:150-162 `add_prompt`'s `rel_coordinates=True`
    default applies to `points`/box geometric prompts, and
    sam3/model/sam3_video_inference.py:884-885 asserts `boxes_xywh` (and points, transitively) are
    normalised 0..1 -- so `.box()` takes `xywh_norm` in [0, 1], NOT pixels.

    DEVIATION FROM THE ORIGINAL DESIGN NOTE ("box prompt plus a positive point at the box centre"):
    verified against sam3/model/sam3_video_inference.py:1364-1401 (`Sam3VideoInferenceWithInstanceInteractivity.add_prompt`,
    the class build_sam3_video_predictor actually returns per model_builder.py:676-686) that when
    `points` is given, `text_str`/`boxes_xywh` MUST be None (`assert text_str is None and
    boxes_xywh is None`) -- the installed API has no call that combines a box and a point prompt in
    one shot. `.box()` therefore takes an *optional* `point` purely as a tie-breaker: if the box
    prompt returns more than one candidate object (only expected in edge cases -- a lone geometric
    box prompt does not invoke text detection), whichever candidate mask contains `point` is
    preferred; otherwise it's unused. The correction issue this was meant to help with (an
    ambiguous/loose box) is still addressed by the box prompt alone, which SAM3 turns into a
    "visual/geometric prompt" on the detector, not a point-refinement of an existing tracked object.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geometry import mask_bbox_xyxy


@dataclass
class Candidate:
    mask: np.ndarray  # bool (H, W)
    box_xywh: list  # pixel-space [x0, y0, w, h], derived from the mask's own bbox
    score: float


def _candidate_from_mask(mask: np.ndarray, score: float) -> Candidate | None:
    bbox = mask_bbox_xyxy(mask)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    return Candidate(mask=mask, box_xywh=[x0, y0, x1 - x0, y1 - y0], score=float(score))


def checkpoint_variant(checkpoint: str) -> str:
    """Copied from vision-llm-ann-generator/sam3_runner.py:checkpoint_variant (same repo family,
    same sam3 install) -- peeks at state-dict keys (mmap, no GPU) to classify the presence
    mechanism so the right builder patch/kwargs get applied for SA-FARI fine-tunes."""
    import torch

    sd = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    keys = list(sd.keys())
    if any(k.startswith("detector.segmentation_head.presence_head.") for k in keys):
        return "seg_head_presence"
    if any(k.startswith("detector.transformer.decoder.presence_token") for k in keys):
        return "decoder_presence_token"
    return "unknown"


def _patch_segmentation_head_with_presence():
    """Copied from vision-llm-ann-generator/sam3_runner.py (same function, same reasoning): makes
    the model match SA-FARI checkpoints (segmentation head with a DotProductScoring presence head,
    no decoder presence token). Idempotent."""
    from sam3 import model_builder as mb

    if getattr(mb, "_presence_head_patched", False):
        return
    orig = mb._create_segmentation_head

    def patched(*args, **kwargs):
        head = orig(*args, **kwargs)
        head.presence_head = mb._create_dot_product_scoring()
        return head

    mb._create_segmentation_head = patched

    orig_dec = mb._create_transformer_decoder

    def patched_dec(*args, **kwargs):
        dec = orig_dec(*args, **kwargs)
        dec.presence_token = None
        dec.presence_token_head = None
        dec.presence_token_out_norm = None
        return dec

    mb._create_transformer_decoder = patched_dec
    mb._presence_head_patched = True


class Sam3Segmenter:
    """Text- and box-prompted single-frame SAM3 segmentation, one session per PIL frame (reused
    across calls to .text()/.box() for the same frame -- add_prompt resets internal tracker state
    per call but not the loaded frame, see sam3_video_inference.py:864-865)."""

    def __init__(self, checkpoint: str, device: str = "cuda", score_thresh: float = 0.5):
        self.checkpoint = checkpoint
        self.device = device
        self.score_thresh = score_thresh
        self._predictor = None
        self._session_id = None
        self._session_pil_id = None

    def _load(self):
        if self._predictor is not None:
            return self._predictor
        from sam3.model_builder import build_sam3_video_predictor

        variant = checkpoint_variant(self.checkpoint)
        kwargs = dict(checkpoint_path=self.checkpoint)
        if variant == "seg_head_presence":
            kwargs["has_presence_token"] = False
            _patch_segmentation_head_with_presence()
        print(f"segmenter: checkpoint variant={variant} kwargs={kwargs}", flush=True)
        self._predictor = build_sam3_video_predictor(**kwargs)
        return self._predictor

    def _ensure_session(self, pil):
        predictor = self._load()
        if self._session_pil_id != id(pil):
            self._close_session()
            resp = predictor.handle_request(dict(type="start_session", resource_path=[pil]))
            self._session_id = resp["session_id"] if isinstance(resp, dict) else resp
            self._session_pil_id = id(pil)
        return predictor, self._session_id

    def _close_session(self):
        if self._session_id is not None and self._predictor is not None:
            try:
                self._predictor.handle_request(dict(type="close_session", session_id=self._session_id))
            except Exception:
                pass
        self._session_id = None
        self._session_pil_id = None

    def _to_candidates(self, resp) -> list[Candidate]:
        outputs = resp["outputs"]
        masks = outputs["out_binary_masks"]
        probs = outputs["out_probs"]
        if hasattr(masks, "cpu"):
            masks = masks.cpu().numpy()
        if hasattr(probs, "cpu"):
            probs = probs.cpu().numpy()
        out = []
        for i in range(masks.shape[0]):
            cand = _candidate_from_mask(np.asarray(masks[i], dtype=bool), float(probs[i]))
            if cand is not None:
                out.append(cand)
        out.sort(key=lambda c: c.score, reverse=True)
        return out

    def text(self, pil, prompt: str) -> list:
        predictor, session_id = self._ensure_session(pil)
        resp = predictor.handle_request(
            dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=prompt,
                output_prob_thresh=self.score_thresh,
            )
        )
        return self._to_candidates(resp)

    def box(self, pil, xywh_norm, point=None):
        predictor, session_id = self._ensure_session(pil)
        resp = predictor.handle_request(
            dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                bounding_boxes=[list(xywh_norm)],
                bounding_box_labels=[1],
                output_prob_thresh=self.score_thresh,
            )
        )
        candidates = self._to_candidates(resp)
        if not candidates:
            return None
        if point is not None and len(candidates) > 1:
            px, py = int(round(point[0])), int(round(point[1]))

            def contains_point(c: Candidate):
                h, w = c.mask.shape
                if 0 <= py < h and 0 <= px < w:
                    return bool(c.mask[py, px])
                return False

            candidates.sort(key=lambda c: (contains_point(c), c.score), reverse=True)
        return candidates[0]

    def close(self):
        self._close_session()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class FakeSegmenter:
    """Torch/sam3-free stand-in for tests and login-node dry runs: `.text()` returns a single
    rectangular candidate covering the whole frame (score 0.9); `.box()` returns a rectangle
    matching the requested normalised box exactly (score 0.8). Good enough to exercise correct.py's
    IoU-matching / overlap / fallback logic without a GPU."""

    def __init__(self, *args, **kwargs):
        self.calls = []

    def text(self, pil, prompt: str) -> list:
        self.calls.append(("text", prompt))
        w, h = pil.size
        mask = np.ones((h, w), dtype=bool)
        cand = _candidate_from_mask(mask, 0.9)
        return [cand] if cand is not None else []

    def box(self, pil, xywh_norm, point=None):
        self.calls.append(("box", tuple(xywh_norm), point))
        w, h = pil.size
        x0n, y0n, wn, hn = xywh_norm
        x0, y0 = int(round(x0n * w)), int(round(y0n * h))
        x1, y1 = int(round((x0n + wn) * w)), int(round((y0n + hn) * h))
        x0, x1 = max(0, min(x0, w)), max(0, min(x1, w))
        y0, y1 = max(0, min(y0, h)), max(0, min(y1, h))
        mask = np.zeros((h, w), dtype=bool)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
        return _candidate_from_mask(mask, 0.8)

    def close(self):
        pass
