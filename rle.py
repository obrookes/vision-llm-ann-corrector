"""COCO-compressed RLE decode/encode, numpy-only, no pycocotools dependency required.

Input masks (`$SCRATCH/masks/<stem>_masks.json`) store each instance's mask as a COCO RLE dict
`{"size": [h, w], "counts": <str>}` produced by pycocotools inside the SAM3 container. Compute
nodes are offline and the vLLM/login-node python does not have pycocotools installed, so
`rle_decode` is a hand-rolled decoder (copied from vision-llm-ann-verifier's images.py:rle_decode,
same repo family, same convention) that only needs numpy.

`rle_encode` writes the SAME compressed-string format back out, so `correct.py`'s output
(`$SCRATCH/masks_v2/<stem>_masks.json`) is byte-compatible with the input schema and anything
downstream (images.py, gallery.py, re-verify) that reads it doesn't need to change.

  - If pycocotools is importable (true inside the sam3.sif container the corrections actually
    run in), `rle_encode` calls it directly -- guaranteed byte-identical to what produced the
    input files.
  - Otherwise (true on the login node used for tests), a pure-numpy encoder reimplements
    pycocotools' LEB128-ish variable-length integer coding (maskApi.c's `rleToString`/
    `rleFrString`) by hand. `tests/test_correct.py` checks the two encoders agree whenever
    pycocotools is present, and always checks `rle_decode(rle_encode(mask)) == mask`.
"""

from __future__ import annotations

import numpy as np


def rle_decode(rle: dict) -> np.ndarray:
    """Decode a COCO RLE dict ({"size": [h, w], "counts": str | list}) to a bool (H, W) mask.

    Copied verbatim from vision-llm-ann-verifier/images.py:rle_decode.
    """
    h, w = rle["size"]
    s = rle["counts"]
    if isinstance(s, str):
        counts = []
        i = 0
        m = 0
        while i < len(s):
            x = 0
            k = 0
            while True:
                c = ord(s[i]) - 48
                x |= (c & 0x1F) << (5 * k)
                more = c & 0x20
                i += 1
                k += 1
                if not more:
                    if c & 0x10:
                        x |= -1 << (5 * k)
                    break
            if m > 2:
                x += counts[m - 2]
            counts.append(x)
            m += 1
    else:
        counts = list(s)
    counts = np.asarray(counts, dtype=np.int64)
    ends = np.cumsum(counts)
    starts = ends - counts
    mask = np.zeros(h * w, dtype=bool)
    for st, en in zip(starts[1::2], ends[1::2]):
        mask[st:en] = True
    return mask.reshape((w, h)).T  # COCO RLE is column-major


def _counts_from_mask(mask: np.ndarray) -> list[int]:
    """Column-major run lengths of a bool (H, W) mask, starting with the run of 0s (possibly 0
    long). Exact inverse of rle_decode's `mask.reshape((w, h)).T` -> flatten column-major."""
    flat = mask.T.reshape(-1).astype(np.uint8)  # column-major flatten of (H, W)
    if flat.size == 0:
        return [0]
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    boundaries = np.concatenate(([0], change, [flat.size]))
    run_lengths = np.diff(boundaries).tolist()
    if flat[0] == 1:
        # counts must start with a run of 0s (possibly zero-length)
        run_lengths = [0] + run_lengths
    return run_lengths


def _counts_to_string(counts: list[int]) -> str:
    """pycocotools' maskApi.c `rleToString`: delta-code every element against the one two back,
    then a signed variable-length (5 bits/byte, continuation bit 0x20) encoding of each delta."""
    out = []
    for i, c in enumerate(counts):
        x = c
        if i > 2:
            x -= counts[i - 2]
        more = True
        while more:
            byte = x & 0x1F
            x >>= 5
            if byte & 0x10:
                more = x != -1
            else:
                more = x != 0
            if more:
                byte |= 0x20
            out.append(chr(byte + 48))
    return "".join(out)


def _rle_encode_numpy(mask: np.ndarray) -> dict:
    h, w = mask.shape
    counts = _counts_from_mask(mask)
    return {"size": [h, w], "counts": _counts_to_string(counts)}


def rle_encode_intlist(mask: np.ndarray) -> dict:
    """Encode a bool (H, W) mask to a COCO RLE dict with UNCOMPRESSED int-list `counts`
    ({"size": [h, w], "counts": [int, ...]}), column-major, counts[0] = background run.

    This is the format vision-llm-ann-generator/tracks.py:rle_decode requires -- its decoder
    explicitly rejects compressed string counts (the default `rle_encode` here produces exactly
    that, which the generator's decoder can't read). No pycocotools needed either way, since
    `_counts_from_mask` already produces this convention directly (it's `rle_encode`'s compressed
    path's own intermediate representation, before `_counts_to_string` LEB128-encodes it).
    """
    mask = np.ascontiguousarray(mask.astype(bool))
    h, w = mask.shape
    counts = _counts_from_mask(mask)
    return {"size": [h, w], "counts": [int(c) for c in counts]}


def rle_encode(mask: np.ndarray, fmt: str = "compressed") -> dict:
    """Encode a bool (H, W) mask to a COCO RLE dict.

    fmt="compressed" (default): LEB128-ish compressed-string `counts`, byte-compatible with the
    input files' schema. Uses pycocotools when available (guaranteed byte-identical to the input
    files' producer); falls back to the pure-numpy re-implementation above otherwise.

    fmt="intlist": uncompressed int-list `counts` -- see `rle_encode_intlist`.
    """
    if fmt == "intlist":
        return rle_encode_intlist(mask)
    if fmt != "compressed":
        raise ValueError(f"rle_encode: unknown fmt {fmt!r}, expected 'compressed' or 'intlist'")

    mask = np.ascontiguousarray(mask.astype(bool))
    try:
        from pycocotools import mask as _mask_api
    except ImportError:
        return _rle_encode_numpy(mask)

    fortran = np.asfortranarray(mask.astype(np.uint8))
    rle = _mask_api.encode(fortran)
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(rle["size"][0]), int(rle["size"][1])], "counts": counts}
