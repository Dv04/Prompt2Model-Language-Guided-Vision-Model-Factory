#!/usr/bin/env python3
"""edge_infer.py — Autonomous edge-inference script for Prompt2Model ONNX exports.

This script is entirely SELF-CONTAINED.  It does NOT import anything from the
``prompt2model`` package.  All preprocessing parameters (input resolution,
normalisation mean/std, class labels) are read exclusively from the metadata
embedded inside the ``.onnx`` file via ``onnxruntime.InferenceSession`` and
``onnx.load``.

Usage examples
--------------
# Run with a real image (classification or detection — task auto-detected):
    python scripts/edge_infer.py --model output/smoke/classification_run/model.onnx \\
                                 --image path/to/image.png

# Run with a synthetic random image (no --image required):
    python scripts/edge_infer.py --model output/smoke/detection_run/model.onnx

# Increase detection score threshold and top-k:
    python scripts/edge_infer.py --model output/smoke/detection_run/model.onnx \\
                                 --score-threshold 0.4 --topk 5

# Suppress the metadata dump:
    python scripts/edge_infer.py --model output/smoke/classification_run/model.onnx \\
                                 --no-metadata-dump

Exit codes
----------
0   — inference completed successfully
1   — model file not found or ONNX validation failed
2   — image file not found
3   — unexpected runtime error
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Optional PIL import (only needed for real-image mode)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from PIL import Image as _PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PILImage = None  # type: ignore[assignment]
    _PIL_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# Lazy-import heavy deps so --help is instant
# ──────────────────────────────────────────────────────────────────────────────
def _import_onnx():
    try:
        import onnx
        return onnx
    except ImportError:
        _die(1, "onnx is not installed.  Run: pip install onnx")


def _import_ort():
    try:
        import onnxruntime as ort
        return ort
    except ImportError:
        _die(1, "onnxruntime is not installed.  Run: pip install onnxruntime")


# ──────────────────────────────────────────────────────────────────────────────
# Console helpers
# ──────────────────────────────────────────────────────────────────────────────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_RED    = "\033[31m"
_DIM    = "\033[2m"

def _c(text: str, *codes: str) -> str:
    """Wrap text in ANSI colour codes (no-op on non-TTY)."""
    if not sys.stdout.isatty():
        return text
    return "".join(codes) + text + _RESET


def _banner(text: str) -> None:
    width = 72
    print(_c("─" * width, _DIM))
    print(_c(f"  {text}", _BOLD, _CYAN))
    print(_c("─" * width, _DIM))


def _ok(text: str) -> None:
    print(_c(f"  ✓  {text}", _GREEN))


def _warn(text: str) -> None:
    print(_c(f"  ⚠  {text}", _YELLOW), file=sys.stderr)


def _die(code: int, text: str) -> None:
    print(_c(f"\n  ✗  ERROR: {text}", _RED, _BOLD), file=sys.stderr)
    sys.exit(code)


# ──────────────────────────────────────────────────────────────────────────────
# Metadata parsing
# ──────────────────────────────────────────────────────────────────────────────
def load_metadata(model_path: str) -> dict[str, Any]:
    """Load and parse all metadata_props embedded in the ONNX file.

    Returns a dict where list/dict values are already JSON-decoded.
    Raises SystemExit(1) if required keys are missing.
    """
    onnx = _import_onnx()
    model = onnx.load(model_path)

    raw: dict[str, str] = {p.key: p.value for p in model.metadata_props}
    if not raw:
        _die(1, "The ONNX file contains no metadata_props. "
                "Re-export using the Prompt2Model pipeline.")

    parsed: dict[str, Any] = {}
    for key, value in raw.items():
        try:
            parsed[key] = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            parsed[key] = value  # plain string (task, prompt, model_name, …)

    _validate_metadata(parsed)
    return parsed


def _validate_metadata(meta: dict[str, Any]) -> None:
    """Check all keys the edge-inference script depends on are present."""
    required = {
        "task", "input_resolution", "mean", "std", "class_dict",
    }
    missing = required - meta.keys()
    if missing:
        _die(
            1,
            f"ONNX metadata is missing required key(s): {sorted(missing)}.\n"
            "       Re-export the model using the updated Prompt2Model pipeline "
            "to embed the full metadata.",
        )

    task = meta["task"]
    if task not in ("classification", "detection"):
        _die(1, f"Unknown task in metadata: '{task}'. Expected 'classification' or 'detection'.")

    res = meta["input_resolution"]
    if not (isinstance(res, list) and len(res) == 2):
        _die(1, f"'input_resolution' must be a list of two ints, got: {res!r}")


def print_metadata(meta: dict[str, Any]) -> None:
    _banner("Embedded ONNX Metadata")
    keys_order = [
        "task", "model_name", "prompt",
        "input_resolution", "mean", "std",
        "class_dict", "labels", "label_map",
    ]
    for key in keys_order:
        if key not in meta:
            continue
        value = meta[key]
        if isinstance(value, (dict, list)):
            value_str = json.dumps(value)
        else:
            value_str = str(value)
        # Wrap long strings
        if len(value_str) > 60:
            value_str = textwrap.shorten(value_str, width=60, placeholder=" …")
        print(f"    {_c(key + ':', _BOLD):<28s} {value_str}")
    # Warn about optional keys that are absent
    optional = {"label_map", "labels", "model_name", "prompt"}
    absent = optional - meta.keys()
    if absent:
        _warn(f"Optional metadata keys absent (non-fatal): {sorted(absent)}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Image preprocessing (uses ONLY embedded metadata — no external config)
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_image(
    image_path: str,
    input_resolution: list[int],
    mean: list[float],
    std: list[float],
) -> np.ndarray:
    """Load a real image and preprocess it using only the embedded metadata.

    Returns a float32 NCHW array of shape (1, 3, H, W).
    """
    if not _PIL_AVAILABLE:
        _die(1, "Pillow is required for real-image inference.  Run: pip install Pillow")

    path = Path(image_path)
    if not path.exists():
        _die(2, f"Image file not found: {image_path}")

    h, w = input_resolution
    img = _PILImage.open(path).convert("RGB").resize((w, h))
    arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)  in [0, 1]

    mean_arr = np.array(mean, dtype=np.float32)
    std_arr  = np.array(std,  dtype=np.float32)
    arr = (arr - mean_arr) / std_arr                # normalise

    arr = arr.transpose(2, 0, 1)                    # (3, H, W)
    arr = arr[np.newaxis, ...]                       # (1, 3, H, W)
    return arr


def make_dummy_input(
    input_resolution: list[int],
    seed: int = 42,
) -> np.ndarray:
    """Generate a random float32 NCHW array as a synthetic test input.

    The values are drawn from a standard normal distribution to roughly
    mimic normalised ImageNet statistics.
    """
    rng = np.random.default_rng(seed)
    h, w = input_resolution
    return rng.standard_normal((1, 3, h, w)).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# ONNX Runtime session
# ──────────────────────────────────────────────────────────────────────────────
def create_session(model_path: str) -> Any:
    """Create an OnnxRuntime InferenceSession, preferring CPU for portability."""
    ort = _import_ort()
    providers = ["CPUExecutionProvider"]
    # Opportunistically use CUDA if available, but don't fail if absent
    available = [p.lower() for p in ort.get_available_providers()]
    if "cudaexecutionprovider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        _warn("CUDA provider detected — using GPU (set ORT_DISABLE_CUDA=1 to force CPU)")
    return ort.InferenceSession(model_path, providers=providers)


def run_inference(session: Any, input_array: np.ndarray) -> tuple[list[np.ndarray], float]:
    """Run the session and return (outputs, latency_ms)."""
    input_name = session.get_inputs()[0].name
    feed = {input_name: input_array}

    # Warm-up pass
    session.run(None, feed)

    # Timed pass
    t0 = time.perf_counter()
    outputs = session.run(None, feed)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return outputs, latency_ms


# ──────────────────────────────────────────────────────────────────────────────
# Result interpretation
# ──────────────────────────────────────────────────────────────────────────────
def interpret_classification(
    outputs: list[np.ndarray],
    class_dict: dict[str, str],
    topk: int,
) -> None:
    """Pretty-print top-k classification predictions."""
    logits = outputs[0].squeeze()           # (num_classes,)
    # Softmax
    exp   = np.exp(logits - logits.max())
    probs = exp / exp.sum()

    topk  = min(topk, len(probs))
    idxs  = np.argsort(probs)[::-1][:topk]

    _banner("Classification Results")
    print(f"    {'Rank':<6} {'Class':<28} {'Confidence':>12}")
    print(_c("    " + "─" * 50, _DIM))
    for rank, idx in enumerate(idxs, start=1):
        label   = class_dict.get(str(idx), f"class_{idx}")
        conf    = probs[idx]
        bar     = "█" * int(conf * 20)
        is_top  = rank == 1
        row = f"    {rank:<6} {label:<28} {conf:>11.2%}  {bar}"
        print(_c(row, _BOLD, _GREEN) if is_top else row)
    print()


def interpret_detection(
    outputs: list[np.ndarray],
    class_dict: dict[str, str],
    score_threshold: float,
    topk: int,
) -> None:
    """Pretty-print detection results, filtering by score threshold.

    Expected output layout from DetectionExportWrapper:
        outputs[0] → boxes  (topk, 4)   xyxy in pixel coords
        outputs[1] → scores (topk,)
        outputs[2] → labels (topk,)     integer category id
    """
    if len(outputs) < 3:
        _die(3, f"Expected 3 detection outputs (boxes, scores, labels), got {len(outputs)}.")

    boxes  = outputs[0].squeeze()   # (topk, 4)
    scores = outputs[1].squeeze()   # (topk,)
    labels = outputs[2].squeeze()   # (topk,)

    # Ensure 2-D even if topk=1
    if boxes.ndim == 1:
        boxes  = boxes[np.newaxis, :]
        scores = scores[np.newaxis]
        labels = labels[np.newaxis]

    # Filter
    mask = scores >= score_threshold
    boxes_f  = boxes[mask]
    scores_f = scores[mask]
    labels_f = labels[mask]

    # Sort by descending score and limit
    order    = np.argsort(scores_f)[::-1][:topk]
    boxes_f  = boxes_f[order]
    scores_f = scores_f[order]
    labels_f = labels_f[order]

    _banner("Detection Results")
    if len(scores_f) == 0:
        print(_c(f"    No detections above score threshold ({score_threshold:.2f}).\n", _YELLOW))
        return

    print(f"    {'#':<4} {'Class':<24} {'Score':>8}  {'Box (x1,y1,x2,y2)':>32}")
    print(_c("    " + "─" * 72, _DIM))
    for i, (box, score, lbl) in enumerate(zip(boxes_f, scores_f, labels_f), start=1):
        # Detection models shift label indices by +1 for a background class.
        # We store class_dict with 0-based dataset indices, so subtract 1
        # if the label is clearly above the class_dict range.
        label_key = str(int(lbl))
        if label_key not in class_dict:
            label_key = str(int(lbl) - 1)
        label = class_dict.get(label_key, f"class_{int(lbl)}")
        box_str = f"({box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f})"
        row = f"    {i:<4} {label:<24} {score:>8.3f}  {box_str:>32}"
        print(_c(row, _BOLD, _GREEN) if i == 1 else row)
    print()


# ──────────────────────────────────────────────────────────────────────────────
# ONNX model validation (structural check, no runtime)
# ──────────────────────────────────────────────────────────────────────────────
def validate_onnx_structure(model_path: str) -> None:
    """Run onnx.checker on the model to catch structural corruption early."""
    onnx = _import_onnx()
    try:
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        _ok("ONNX structural validation passed")
    except Exception as exc:
        _die(1, f"ONNX validation failed: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edge_infer",
        description=(
            "Autonomous edge-inference script for Prompt2Model ONNX exports.\n"
            "All configuration is read exclusively from metadata embedded in the\n"
            ".onnx file — no external config files or prompt2model package needed."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        metavar="PATH",
        help="Path to the exported .onnx model file.",
    )
    parser.add_argument(
        "--image", "-i",
        default=None,
        metavar="PATH",
        help=(
            "Path to an image file to run inference on.  "
            "If omitted, a synthetic random tensor is used (dummy mode)."
        ),
    )
    parser.add_argument(
        "--topk", "-k",
        type=int,
        default=3,
        metavar="N",
        help="Number of top predictions to show (classification) or "
             "maximum detections to display (detection).  Default: 3.",
    )
    parser.add_argument(
        "--score-threshold", "-t",
        type=float,
        default=0.3,
        dest="score_threshold",
        metavar="FLOAT",
        help="Minimum detection score to display (detection task only).  Default: 0.3.",
    )
    parser.add_argument(
        "--no-metadata-dump",
        action="store_true",
        dest="no_metadata_dump",
        help="Skip printing the full embedded metadata table.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="INT",
        help="RNG seed for synthetic dummy input generation.  Default: 42.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args   = parser.parse_args(argv)

    model_path = args.model
    if not Path(model_path).exists():
        _die(1, f"Model file not found: {model_path}")

    # ── Step 1: Load & parse embedded metadata ─────────────────────────────
    _banner("Loading ONNX Metadata")
    meta             = load_metadata(model_path)
    task             = meta["task"]
    input_resolution = meta["input_resolution"]   # [H, W]
    mean             = meta["mean"]
    std              = meta["std"]
    class_dict       = meta["class_dict"]          # {"0": "cat", …}

    _ok(f"Task              : {task}")
    _ok(f"Model backbone    : {meta.get('model_name', 'unknown')}")
    _ok(f"Input resolution  : {input_resolution[0]} × {input_resolution[1]} px")
    _ok(f"Normalisation mean: {mean}")
    _ok(f"Normalisation std : {std}")
    _ok(f"Classes ({len(class_dict)}): {list(class_dict.values())}")
    if "label_map" in meta and meta["label_map"]:
        _ok(f"Label map         : {meta['label_map']}")
    print()

    if not args.no_metadata_dump:
        print_metadata(meta)

    # ── Step 2: Validate ONNX structure ───────────────────────────────────
    _banner("ONNX Structural Validation")
    validate_onnx_structure(model_path)
    print()

    # ── Step 3: Prepare input tensor ───────────────────────────────────────
    _banner("Input Preparation")
    if args.image:
        _ok(f"Mode              : REAL IMAGE  ({args.image})")
        input_array = preprocess_image(args.image, input_resolution, mean, std)
        _ok(f"Preprocessed shape: {input_array.shape}  (using embedded mean/std)")
    else:
        _ok(f"Mode              : DUMMY / SYNTHETIC (no --image provided)")
        input_array = make_dummy_input(input_resolution, seed=args.seed)
        _ok(f"Synthetic shape   : {input_array.shape}  seed={args.seed}")
    print()

    # ── Step 4: Create ORT session ─────────────────────────────────────────
    _banner("ONNX Runtime Session")
    session = create_session(model_path)
    input_info  = session.get_inputs()[0]
    output_info = session.get_outputs()
    _ok(f"Input  : '{input_info.name}'  shape={input_info.shape}  dtype={input_info.type}")
    for out in output_info:
        _ok(f"Output : '{out.name}'  shape={out.shape}  dtype={out.type}")
    print()

    # ── Step 5: Run inference ──────────────────────────────────────────────
    _banner("Running Inference")
    try:
        outputs, latency_ms = run_inference(session, input_array)
    except Exception as exc:
        _die(3, f"OnnxRuntime inference failed: {exc}")

    _ok(f"Inference latency : {latency_ms:.2f} ms  (warm single-sample, CPU)")
    _ok(f"Output count      : {len(outputs)}")
    for i, out in enumerate(outputs):
        _ok(f"  Output[{i}] shape : {out.shape}  min={float(out.min()):.4f}  max={float(out.max()):.4f}")
    print()

    # ── Step 6: Interpret results using class_dict from metadata ───────────
    if task == "classification":
        interpret_classification(outputs, class_dict, topk=args.topk)
    else:
        interpret_detection(
            outputs,
            class_dict,
            score_threshold=args.score_threshold,
            topk=args.topk,
        )

    # ── Summary ────────────────────────────────────────────────────────────
    _banner("Inference Summary")
    mode_str = f"real image ({Path(args.image).name})" if args.image else "synthetic dummy input"
    print(f"    Model    : {Path(model_path).name}")
    print(f"    Task     : {task}")
    print(f"    Input    : {mode_str}")
    print(f"    Latency  : {latency_ms:.2f} ms")
    print(f"    Config   : 100% from embedded ONNX metadata — no external files used")
    print()
    _ok("Autonomous edge-inference completed successfully.")
    print()


if __name__ == "__main__":
    main()
