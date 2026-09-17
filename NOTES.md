# NOTES

Working notes for this repo (vision-llm-ann-corrector). See README.md for the design/usage summary.

## Environment (2026-09-04, login node, aarch64 GH200)

- `$SCRATCH/envs/vlm-verify/bin/python`: numpy 2.5.2, opencv-python (cv2) 5.0.0, PIL -- no
  pycocotools, no torch/sam3. Used to run `tests/` and any `--fake`/`--dry-run` invocation of
  `correct.py`. `pip install pytest` was needed into this env (not present initially).
- `$SCRATCH/containers/sam3.sif`: has sam3 0.1.0 (`/usr/local/lib/python3.12/dist-packages/sam3`)
  and pycocotools. `apptainer exec` (no `--nv`, CPU-only) works fine for source inspection
  (`import inspect; inspect.getsource(...)`) and for confirming the pycocotools RLE-encoder
  agreement test -- neither builds a model or touches a GPU. Compute nodes are offline for this
  task; no `sbatch`/GPU code was run.

## SAM3 API, verified against the installed package (not just the generator repo's docstring)

`apptainer exec $SCRATCH/containers/sam3.sif python3 -c 'import inspect; import
sam3.model.sam3_base_predictor as m; print(inspect.getsource(m))'` (and the same for
`sam3.model.sam3_video_inference`, `sam3.model_builder`) confirms:

- `build_sam3_video_predictor(checkpoint_path=..., has_presence_token=...)` ->
  `Sam3VideoPredictorMultiGPU` wrapping `build_sam3_video_model`, which always returns a
  `Sam3VideoInferenceWithInstanceInteractivity` (`model_builder.py:676-686`) -- i.e. `points`-based
  prompts *are* supported by the model actually returned by the public builder, not just
  `boxes_xywh`/`text`.
- `handle_request(type="start_session", resource_path=[pil_image])`: a length-1 list hits
  `is_image_type`'s `len(resource_path) == 1` branch (`sam3_video_inference.py:1712-1716`) --
  correct.py wants exactly this (one frame, no propagation) for every session it opens.
- `handle_request(type="add_prompt", session_id, frame_index=0, text=..., output_prob_thresh=...)`
  and the `bounding_boxes=[[x,y,w,h]], bounding_box_labels=[1]` box-prompt form both go through
  `Sam3BasePredictor.add_prompt` (`sam3_base_predictor.py:150-207`), which converts lists to
  tensors, then calls `self.model.add_prompt(**filtered_kwargs)` after filtering kwargs down to
  `inspect.signature(self.model.add_prompt)`'s parameters -- confirmed the model's own `add_prompt`
  (`sam3_video_inference.py:1364-1401`) is what actually runs, and:
  - `boxes_xywh` must be normalised 0..1 (`sam3_video_inference.py:884-890` asserts
    `(boxes_xywh >= 0).all() and (boxes_xywh <= 1).all()`) -- **not pixels**. `segmenter.box()`
    therefore takes `xywh_norm`.
  - **`points` and `boxes_xywh`/`text_str` cannot be combined in one call**:
    `sam3_video_inference.py:1376-1383` asserts `text_str is None and boxes_xywh is None` whenever
    `points is not None`. This directly contradicts the original design note's "box prompt plus a
    positive point at the box centre" as a *single* call. `add_tracker_new_points` (the function
    the `points` branch delegates to, `sam3_video_inference.py:1404+`) also has no `box` parameter
    of its own, unlike the lower-level `add_new_points_or_box` in `sam3_tracking_predictor.py`
    (which *does* accept both, but isn't reachable through `add_prompt`/`handle_request`). Net
    effect: **`segmenter.Sam3Segmenter.box()` sends a box-only prompt**; its `point` argument is
    used purely as a tie-breaker if the box prompt happens to return more than one candidate mask
    (picks whichever contains the point). This is documented in `segmenter.py`'s module docstring
    as a deviation from the original design.
  - Output dict: `{"frame_index": int, "outputs": {"out_obj_ids", "out_probs",
    "out_boxes_xywh" (normalised), "out_binary_masks" (N, H, W) bool}}`
    (`sam3_video_inference.py:432-500`, `_postprocess_output`). `segmenter.py` ignores
    `out_boxes_xywh` and recomputes each candidate's pixel-space bbox from its own mask instead
    (one less unit-convention to get wrong, and it's what `rle.py`/`geometry.py` want anyway).
- `add_prompt` (the text/box path, `sam3_video_inference.py:864-865`) calls
  `self.reset_state(inference_state)` on every call ("since it's a semantic prompt, we start
  over"). This resets tracker/detection history but *not* the loaded frame/backbone features, so
  `segmenter.Sam3Segmenter` reuses one session across a frame's `.text()` call and any number of
  `.box()` calls (keyed by `id(pil)`), only opening a new `start_session` when a different PIL
  object comes in. Each `add_prompt` call is still an independent, from-scratch prompt -- no state
  leaks between a frame's `.text()` and its later `.box()` calls, or between two `.box()` calls.

## RLE encoder

`rle.py`'s pure-numpy `rle_encode` fallback re-implements pycocotools' `maskApi.c`
`rleToString`/`rleFrString` variable-length signed integer coding by hand (5 bits/byte, bit 0x20
continuation, delta-coded against the count two positions back). Verified byte-identical to
`pycocotools.mask.encode` inside the container on random test data
(`apptainer exec $SCRATCH/containers/sam3.sif python3 -c '...'`, see the repo's git history /
session transcript for the exact command) -- `tests/test_correct.py`'s
`test_rle_encode_agrees_with_pycocotools_when_available` covers this automatically wherever
pycocotools is importable, and skips otherwise (true on the login-node python used to actually run
`pytest` for this repo).

## `--keep-ids` / `--rle-format intlist` (2026-09-07)

The next worklist this repo processes (`$SCRATCH/chimp/...`) has `instance_idx` values that are
SAM3 track ids assigned by `vision-llm-ann-generator` during tracking, not positional `0..N-1`
indices -- they're sparse/non-contiguous and must survive the correction loop unchanged so
downstream consumers (re-verify, the generator's own track bookkeeping) can still key on them.
`--keep-ids` turns off `correct.py`'s default re-indexing: kept/resegmented instances keep their
`instance_idx`; `"add"` instances get ids above the highest one the frame ever used
(`max(inst_by_idx.keys())+1, +2, ...`, computed from the *original* instance set so a dropped
`wrong_object`/`duplicate` instance's id is never reused within the same frame). Both
`process_frame`'s `resegment.idx` matching and `passthrough_masks` already worked (and still work)
by `instance_idx`, not list position -- `inst_by_idx`/`kept` were already dicts keyed by
`instance_idx` before this change, so nothing there needed fixing, only the final
reindex-vs-keep step.

`--rle-format intlist` exists because `vision-llm-ann-generator/tracks.py:rle_decode`
(`/home/b5bd/obrookes.b5bd/vision-llm-ann-generator/tracks.py`) hard-rejects compressed string
`counts` (`raise ValueError(...)` if `isinstance(counts, str)`) -- it only reads the uncompressed
int-list COCO RLE form. `rle.py`'s `_counts_from_mask` already produces exactly that list (it's
the pre-LEB128 intermediate value the default `compressed` path encodes further), so
`rle_encode_intlist`/`rle_encode(mask, fmt="intlist")` just returns it directly -- no pycocotools
involved either way. Verified round-trip both ways in `tests/test_correct.py`: our own
`rle_decode` and the generator's `tracks.rle_decode` (imported via `sys.path.insert` at the
generator repo's absolute path) both reproduce the original random mask from an intlist-encoded
RLE.

## Things not done / left for whoever runs this on a GPU node

- No end-to-end run against real SAM3 has happened -- everything above is either read from the
  installed package's source or exercised through `FakeSegmenter`. First real run should be small
  (`--limit`, a single array task, `outputs/worklist_calib.jsonl`) and its `corrections.jsonl` log
  spot-checked before scaling up.
- `--score-thresh` / SAM3's `output_prob_thresh` default (0.5) is untuned.
- No text-prompt tuning beyond the generator's own default (`"person"`).
