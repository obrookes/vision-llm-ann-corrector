"""Small pixel-space box/mask geometry helpers shared by correct.py and segmenter.py.

All boxes here are pixel-space [x0, y0, x1, y1] (xyxy) unless a name says otherwise
(`*_xywh`, `*_norm`). No torch/sam3 imports -- safe to use from tests on the login node.
"""

from __future__ import annotations

import numpy as np


def mask_bbox_xyxy(mask: np.ndarray):
    """Bounding box of the True pixels of a bool (H, W) mask, or None if empty."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def mask_centroid(mask: np.ndarray):
    """Centre of mass of the True pixels, or None if empty. (x, y), int pixel coords."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [int(round(float(xs.mean()))), int(round(float(ys.mean())))]


def iou_xyxy(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def iou_masks(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def mask_in_box_fraction(mask: np.ndarray, box_xyxy) -> float:
    """Fraction of mask's True pixels that fall inside box_xyxy (pixel space)."""
    total = int(mask.sum())
    if total == 0:
        return 0.0
    x0, y0, x1, y1 = box_xyxy
    ys, xs = np.nonzero(mask)
    inside = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
    return float(inside.sum()) / total


def dilate_xyxy(box_xyxy, frac: float, width: int, height: int):
    """Grow box_xyxy by `frac` of its own width/height on each side, clipped to the image."""
    x0, y0, x1, y1 = box_xyxy
    bw, bh = x1 - x0, y1 - y0
    dx, dy = bw * frac, bh * frac
    return [
        max(0.0, x0 - dx),
        max(0.0, y0 - dy),
        min(float(width), x1 + dx),
        min(float(height), y1 + dy),
    ]


def xyxy_to_xywh_norm(box_xyxy, width: int, height: int):
    x0, y0, x1, y1 = box_xyxy
    return [x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height]


def xyxy_norm_to_px(box_xyxy_norm, width: int, height: int):
    x0, y0, x1, y1 = box_xyxy_norm
    return [x0 * width, y0 * height, x1 * width, y1 * height]
