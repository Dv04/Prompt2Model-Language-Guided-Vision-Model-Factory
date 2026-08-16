#!/usr/bin/env python3
"""compress_production_models.py - apply B1's compression machinery to Dhi's
ACTUAL production edge models (aixavier/models/), not synthetic fixtures.

Why: the production Orin box is chronically overloaded and crash-looping
under compute pressure. Shrinking the models it runs is a real product win.
This script quantizes each production ONNX artifact to INT8 (dynamic,
MatMul/Gemm only - see compression.quantize_onnx docstring for why Conv
must stay FP32), measures the size reduction, evaluates the compressed
artifact against a floor, and REFUSES to ship anything that falls below it
(prompt2model.compression.decide_gate - never softened).

READ BEFORE TRUSTING A NUMBER - the eval-data honesty problem
================================================================
There is no labeled ground-truth eval set for these detector/pose/PPE/face
models on this Mac. CrowdHuman (person boxes) lives on the shared GPU box
only, and this script never touches the GPU box or the Orin (Mac-side only,
by design). aixavier ships no boxed annotations for its PPE/pose/face
models either - only unlabeled demo videos and a handful of face crops.

So instead of fabricating an "accuracy" number against data that doesn't
exist, every model here is graded on OUTPUT AGREEMENT: how closely the
INT8 ONNX graph reproduces the FP32 production graph's raw output tensors
on the SAME real (unlabeled) frames pulled from aixavier's own demo videos
/ test fixtures. The metric is mean cosine similarity between flattened
baseline and compressed outputs, averaged over sampled frames/clips. This
is a regression/fidelity check ("did compression change what the model
sees"), and is reported as exactly that - never relabeled as "accuracy".
It is fed through the same accuracy_floor_relative gate math as B1's
classification path (baseline pinned at similarity=1.0, i.e. self-fidelity)
so a compressed artifact that stops agreeing with its own FP32 parent is
refused exactly like a classifier that drops below its accuracy floor.

The one model that is genuinely an end-to-end classifier (VideoMAE
violence) could in principle use compression.evaluate_onnx_classification
for real accuracy - but that needs labeled clips, which this Mac also does
not have. It goes through the same output-agreement metric for now; flip
it to a real labeled accuracy the moment a labeled violence clip set
exists (board item: closing this gap is real future KD work, not this
script's job).

models/weights/face/model.pth is SKIPPED outright: it is an mmdetection
2.7 / mmcv 1.2.7 training checkpoint (SCRFD-34G, epoch 640) - not the
artifact the runtime loads, and that legacy training stack is not a
dependency of this repo (nor compatible with its torch 2.5). The runtime
loads models/onnx/face/scrfd_34g_gnkps.onnx directly, so THAT is what gets
compressed instead - see the face_detector_scrfd34g entry below.

ANOTHER HONEST FINDING FROM A REAL RUN - dynamic MatMul/Gemm quantization
barely touches YOLO-style detectors. yolo11n.onnx has 88 Conv nodes and
only 2 MatMul; ppe_detection.onnx has 64 Conv and ZERO MatMul/Gemm;
scrfd_34g_gnkps.onnx has 161 Conv and zero MatMul/Gemm. Since the gate's
Conv-safety restriction (compression.quantize_onnx's docstring - Conv
quantization previously collapsed a real classifier to chance accuracy)
means only MatMul/Gemm gets quantized, these Conv-heavy backbones shrink
by roughly 0% (sometimes even a few hundred bytes LARGER, from added
quantization metadata). The real payoff shows up on the two
attention/MatMul-heavy models: FaceLiVT (49 MatMul/Gemm nodes, -6.5%) and
VideoMAE (almost entirely attention, -70.5%). If the product goal is
"materially shrink the object/pose/PPE/face-detector engines", this
quantize-only path is not the lever - static (activation-calibrated)
quantization that also covers Conv, or actual pruning/distillation, would
be needed, and static Conv quantization is exactly the mode this repo
already measured collapsing a real classifier's accuracy - it would need
its own from-scratch collapse-safety validation before being trusted on
these detectors. That is future work, not something this script fakes.

Usage
-----
    .venv/bin/python scripts/compress_production_models.py \\
        --models-root /path/to/aixavier/models \\
        --output-dir output/production_compression

    # only one model family, fewer eval samples (fast smoke run):
    .venv/bin/python scripts/compress_production_models.py \\
        --models-root /path/to/aixavier/models --only object_yolo11n --samples 4

On the GPU box: copy `aixavier/models/{weights,onnx}` over (see the repo
README section this script's caller reports), point --models-root at the
copy, and run identically - no CUDA/TensorRT is required, this is a CPU
ONNX Runtime pass.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from prompt2model.compression import quantize_onnx
from prompt2model.config import CompressionConfig, ModelConstraints
from prompt2model.compression import decide_gate

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("compress_production_models")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ── model registry ───────────────────────────────────────────────────────────


@dataclass
class ModelSpec:
    name: str
    family: str  # object_detector | pose | ppe_detector | face_detector | face_embedding | violence_classifier
    onnx_rel: str | None          # path relative to models-root, or None if it must be exported
    pt_rel: str | None            # path relative to models-root (for export / provenance), or None
    input_wh: tuple[int, int]     # (width, height) for image models; ignored for clip models
    preprocess: str               # key into PREPROCESSORS
    eval_kind: str                # "video_frames" | "images" | "video_clip"
    eval_sources: tuple[str, ...] # paths relative to aixavier root (videos/, models/test_data/)
    n_samples: int = 8
    accuracy_floor_relative: float = 0.98
    notes: str = ""


MODEL_REGISTRY: list[ModelSpec] = [
    ModelSpec(
        name="object_yolo11n",
        family="object_detector",
        onnx_rel="onnx/object/yolo11n.onnx",
        pt_rel="weights/object/yolo11n.pt",
        input_wh=(640, 640),
        preprocess="yolo",
        eval_kind="video_frames",
        eval_sources=("videos/people counting.mp4",),
        notes="LIVE production object detector (configs/detectors/object.yaml).",
    ),
    ModelSpec(
        name="object_yolo26n",
        family="object_detector",
        onnx_rel="onnx/object/yolo26n_clean.onnx",
        pt_rel="weights/object/yolo26n.pt",
        input_wh=(640, 640),
        preprocess="yolo",
        eval_kind="video_frames",
        eval_sources=("videos/people counting.mp4",),
        notes="Alternate object detector, commented out in object.yaml (not live), still shipped.",
    ),
    ModelSpec(
        name="pose_yolov11n",
        family="pose",
        onnx_rel=None,  # no ONNX in-tree for this one - export from the .pt
        pt_rel="weights/pose/yolov11n-pose.pt",
        input_wh=(640, 640),
        preprocess="yolo",
        eval_kind="video_frames",
        eval_sources=("videos/people counting.mp4",),
        notes=(
            "LIVE runtime fallback (pose.py glob on models/weights/pose/*.pt) only when "
            "no RTMPose engine/ONNX is present. Production pose path is RTMPose "
            "(models/onnx/pose/rtmpose_m_end2end.onnx, 192x256 top-down crop) - a different "
            "architecture, out of scope here since it is not a plain whole-frame detector."
        ),
    ),
    ModelSpec(
        name="ppe_detection",
        family="ppe_detector",
        onnx_rel="onnx/ppe/ppe_detection.onnx",
        pt_rel="weights/ppe/ppe_detection.pt",
        input_wh=(640, 640),
        preprocess="yolo",
        eval_kind="video_frames",
        eval_sources=("videos/safetyPPE.mp4",),
        notes="LIVE production PPE detector fallback (TRT engine is the mandated primary path).",
    ),
    ModelSpec(
        name="face_detector_scrfd34g",
        family="face_detector",
        onnx_rel="onnx/face/scrfd_34g_gnkps.onnx",
        pt_rel=None,  # weights/face/model.pth is the mmdet training ckpt for this - see module docstring
        input_wh=(640, 640),
        preprocess="scrfd",
        eval_kind="images",
        eval_sources=(
            "models/test_data/face/front1.jpg",
            "models/test_data/face/front2.jpg",
            "models/test_data/face/side1.jpg",
            "models/test_data/face/person1.png",
            "models/test_data/face/person2.png",
            "models/test_data/face/crowdPerson1.png",
        ),
        notes=(
            "LIVE production face detector (configs/detectors/face.yaml). Compressed in place "
            "of weights/face/model.pth (the mmdet training checkpoint this ONNX was exported "
            "from) - see module docstring for why the .pth is skipped."
        ),
    ),
    ModelSpec(
        name="face_embedding_facelivtv2l",
        family="face_embedding",
        onnx_rel="onnx/face/facelivtv2-l.onnx",
        pt_rel="weights/face/facelivtv2-l.pt",
        input_wh=(112, 112),
        preprocess="facelivt",
        eval_kind="images",
        eval_sources=(
            "models/test_data/face/front1.jpg",
            "models/test_data/face/front2.jpg",
            "models/test_data/face/side1.jpg",
            "models/test_data/face/person1.png",
            "models/test_data/face/person2.png",
            "models/test_data/face/crowdPerson1.png",
        ),
        notes=(
            "LIVE production face embedding model (configs/detectors/face.yaml embedding block). "
            "Eval crops are direct resize of the whole test image to 112x112, NOT the real "
            "detect->align crop the runtime uses - a simplification, since this pass only needs "
            "a real image to compare baseline vs compressed embeddings on, not a correct pipeline."
        ),
    ),
    ModelSpec(
        name="violence_videomae_binary",
        family="violence_classifier",
        onnx_rel="onnx/violence/videomae_binary/model.onnx",
        pt_rel=None,
        input_wh=(224, 224),
        preprocess="videomae",
        eval_kind="video_clip",
        eval_sources=("videos/stonepelting.mp4", "videos/vandalism.mp4"),
        n_samples=16,  # frames per clip (VideoMAE clip_length), not "16 clips"
        notes=(
            "LIVE production violence classifier (configs/detectors/violence.yaml, default "
            "model). Genuinely a classifier - B1's real accuracy/KD path "
            "(train_distilled_classification + evaluate_onnx_classification) applies here "
            "once a labeled clip set exists; this pass uses the same output-agreement metric "
            "as the detectors pending that (see module docstring)."
        ),
    ),
    ModelSpec(
        name="violence_videomae_xdviolence",
        family="violence_classifier",
        onnx_rel="onnx/violence/videomae_xdviolence/model.onnx",
        pt_rel=None,
        input_wh=(224, 224),
        preprocess="videomae",
        eval_kind="video_clip",
        eval_sources=("videos/stonepelting.mp4", "videos/vandalism.mp4"),
        n_samples=16,
        notes="Fallback multi-label violence classifier (violence.yaml fallback chain).",
    ),
]

# models/weights/face/model.pth is intentionally NOT in MODEL_REGISTRY - see
# SKIPPED_MODELS below and the module docstring for why.
SKIPPED_MODELS = [
    {
        "name": "face_model_pth",
        "family": "face_detector_training_checkpoint",
        "path": "weights/face/model.pth",
        "reason": (
            "mmdetection 2.7 / mmcv 1.2.7 training checkpoint (SCRFD-34G, epoch 640, "
            "trained 2021 on 4x V100). Not the artifact the runtime loads, and mmdet/mmcv "
            "are not dependencies of this repo (nor compatible with its torch 2.5). "
            "The runtime-loaded equivalent, models/onnx/face/scrfd_34g_gnkps.onnx, is "
            "compressed directly instead (see face_detector_scrfd34g)."
        ),
    }
]


# ── preprocessing ────────────────────────────────────────────────────────────


def _yolo_norm(rgb_u8: np.ndarray) -> np.ndarray:
    """0-255 RGB HWC -> 0-1 CHW float32 (standard ultralytics preprocessing)."""
    arr = rgb_u8.astype(np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))


def _scrfd_norm(rgb_u8: np.ndarray) -> np.ndarray:
    """SCRFD preprocessing: mean=127.5, std=128 per channel (insightface convention)."""
    arr = (rgb_u8.astype(np.float32) - 127.5) / 128.0
    return np.transpose(arr, (2, 0, 1))


def _facelivt_norm(rgb_u8: np.ndarray) -> np.ndarray:
    """FaceLiVT preprocessing: mean=0.5, std=0.5 (see configs/detectors/face.yaml)."""
    arr = rgb_u8.astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return np.transpose(arr, (2, 0, 1))


def _videomae_norm(rgb_u8: np.ndarray) -> np.ndarray:
    """VideoMAE preprocessing per preprocessor_config.json: rescale 1/255, ImageNet mean/std."""
    arr = rgb_u8.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    arr = (arr - mean) / std
    return np.transpose(arr, (2, 0, 1))


PREPROCESSORS = {
    "yolo": _yolo_norm,
    "scrfd": _scrfd_norm,
    "facelivt": _facelivt_norm,
    "videomae": _videomae_norm,
}


def _resize_rgb(bgr: np.ndarray, wh: tuple[int, int]):
    import cv2

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, wh, interpolation=cv2.INTER_LINEAR)


def _sample_video_frames(path: Path, n: int, wh: tuple[int, int]) -> list[np.ndarray]:
    """Evenly-spaced frames across the whole video, resized + BGR->RGB.
    NOT a letterbox (production does letterbox+pad) - direct resize is enough
    for a same-input baseline-vs-compressed fidelity comparison."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return []
        indices = sorted(set(int(i * total / n) for i in range(min(n, total))))
        frames = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(_resize_rgb(frame, wh))
        return frames
    finally:
        cap.release()


def _sample_video_clip(path: Path, n_frames: int, wh: tuple[int, int]) -> np.ndarray | None:
    """One evenly-sampled n_frames-long clip spanning the whole video ->
    shape [n_frames, H, W, 3] uint8 RGB, or None if unreadable."""
    frames = _sample_video_frames(path, n_frames, wh)
    if len(frames) < n_frames:
        return None
    return np.stack(frames[:n_frames], axis=0)


def _load_images(paths: list[Path], wh: tuple[int, int]) -> list[np.ndarray]:
    import cv2

    out = []
    for path in paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        out.append(_resize_rgb(bgr, wh))
    return out


def build_eval_samples(spec: ModelSpec, aixavier_root: Path) -> tuple[list[np.ndarray], list[str]]:
    """Returns (list of preprocessed input tensors ready to feed the ONNX
    session, list of the sources actually used) for one model. Empty list
    means "no usable eval data" - callers must skip, never fabricate."""
    preprocess = PREPROCESSORS[spec.preprocess]
    used: list[str] = []

    if spec.eval_kind in ("video_frames", "images"):
        wh = spec.input_wh
        raw_frames: list[np.ndarray] = []
        if spec.eval_kind == "video_frames":
            for source in spec.eval_sources:
                video_path = aixavier_root / source
                if not video_path.exists():
                    continue
                frames = _sample_video_frames(video_path, spec.n_samples, wh)
                if frames:
                    raw_frames.extend(frames)
                    used.append(source)
        else:  # images
            paths = [aixavier_root / source for source in spec.eval_sources]
            existing = [p for p in paths if p.exists()]
            raw_frames = _load_images(existing, wh)
            used = [str(p.relative_to(aixavier_root)) for p in existing]
        samples = [preprocess(frame)[None, ...] for frame in raw_frames]  # add batch dim
        return samples, used

    if spec.eval_kind == "video_clip":
        clips = []
        for source in spec.eval_sources:
            video_path = aixavier_root / source
            if not video_path.exists():
                continue
            clip = _sample_video_clip(video_path, spec.n_samples, spec.input_wh)
            if clip is not None:
                clips.append(clip)
                used.append(source)
        samples = []
        for clip in clips:
            processed_frames = [preprocess(frame) for frame in clip]  # each -> (3,H,W)
            clip_tensor = np.stack(processed_frames, axis=0)[None, ...]  # (1, T, 3, H, W)
            samples.append(clip_tensor)
        return samples, used

    raise ValueError(f"unknown eval_kind: {spec.eval_kind}")


# ── ONNX export (for .pt files with no in-tree ONNX) ─────────────────────────


def quant_preprocess(onnx_path: Path, run_dir: Path) -> Path:
    """Run onnxruntime's recommended shape-inference + graph-optimization
    pass before quantization (`quant_pre_process`). Several production
    graphs here (e.g. facelivtv2-l, torch.onnx-dynamo exported with a
    literal "None"-named symbolic batch dim) fail quantize_dynamic's
    internal shape inference without this - a real, reproducible bug this
    driver works around at the input-graph level, without touching
    compression.quantize_onnx itself. This step canonicalizes/optimizes the
    graph (constant folding, redundant-node elimination); it is not expected
    to change numerical outputs, which the output-agreement metric below
    verifies for every model regardless."""
    from onnxruntime.quantization.shape_inference import quant_pre_process

    out_path = run_dir / "preprocessed" / (onnx_path.stem + "_preprocessed.onnx")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    quant_pre_process(str(onnx_path), str(out_path), skip_symbolic_shape=False)
    return out_path


def export_pt_to_onnx(pt_path: Path, output_dir: Path, imgsz: int) -> Path:
    """Export an ultralytics .pt checkpoint to ONNX into output_dir - never
    writes into the source models tree. Ultralytics always writes next to
    the source .pt, so we export into a scratch copy and move the result."""
    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_pt = output_dir / pt_path.name
    shutil.copy2(pt_path, scratch_pt)
    model = YOLO(str(scratch_pt))
    exported = model.export(format="onnx", opset=17, imgsz=imgsz, simplify=False)
    exported_path = Path(exported)
    final_path = output_dir / (pt_path.stem + ".onnx")
    if exported_path != final_path:
        shutil.move(str(exported_path), str(final_path))
    scratch_pt.unlink(missing_ok=True)
    return final_path


# ── inference + agreement metric ─────────────────────────────────────────────


def run_onnx_all_outputs(onnx_path: Path, samples: list[np.ndarray]) -> list[list[np.ndarray]]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    results = []
    for sample in samples:
        outputs = session.run(None, {input_name: sample.astype(np.float32)})
        results.append(outputs)
    return results


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.dot(a, b) / denom)


def output_agreement(baseline: list[list[np.ndarray]], compressed: list[list[np.ndarray]]) -> float:
    """Mean cosine similarity between baseline and compressed output tensors,
    across every sample and every output head. This is a FIDELITY metric
    (does the quantized graph reproduce the FP32 graph's raw outputs on real
    inputs), not accuracy against ground truth - see module docstring."""
    scores = []
    for base_outs, comp_outs in zip(baseline, compressed):
        for base_tensor, comp_tensor in zip(base_outs, comp_outs):
            scores.append(_cosine(base_tensor, comp_tensor))
    return float(np.mean(scores)) if scores else 0.0


# ── per-model report ──────────────────────────────────────────────────────────


@dataclass
class ModelCompressionReport:
    name: str
    family: str
    status: str  # "compressed" | "refused_floor" | "skipped" | "error"
    reason: str
    notes: str = ""
    original_path: str | None = None
    original_size_bytes: int | None = None
    compressed_path: str | None = None
    compressed_size_bytes: int | None = None
    size_reduction_pct: float | None = None
    metric_name: str = "output_cosine_similarity_vs_fp32_baseline"
    baseline_metric: float | None = None
    compressed_metric: float | None = None
    floor: float | None = None
    passed: bool | None = None
    chosen_path: str | None = None
    eval_sources_used: list[str] = field(default_factory=list)
    n_eval_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "status": self.status,
            "reason": self.reason,
            "notes": self.notes,
            "original_path": self.original_path,
            "original_size_bytes": self.original_size_bytes,
            "compressed_path": self.compressed_path,
            "compressed_size_bytes": self.compressed_size_bytes,
            "size_reduction_pct": self.size_reduction_pct,
            "metric_name": self.metric_name,
            "baseline_metric": self.baseline_metric,
            "compressed_metric": self.compressed_metric,
            "floor": self.floor,
            "passed": self.passed,
            "chosen_path": self.chosen_path,
            "eval_sources_used": self.eval_sources_used,
            "n_eval_samples": self.n_eval_samples,
        }


def process_model(spec: ModelSpec, models_root: Path, aixavier_root: Path, run_dir: Path) -> ModelCompressionReport:
    report = ModelCompressionReport(name=spec.name, family=spec.family, status="error", reason="", notes=spec.notes)

    # 1. resolve the ONNX artifact (export from .pt only if no ONNX in-tree).
    #    onnx_rel / pt_rel are relative to models_root; eval_sources are
    #    relative to aixavier_root (models_root's parent) - see ModelSpec docs.
    onnx_path: Path | None = None
    if spec.onnx_rel is not None:
        candidate = models_root / spec.onnx_rel
        if candidate.exists():
            onnx_path = candidate
    if onnx_path is None and spec.pt_rel is not None:
        pt_path = models_root / spec.pt_rel
        if pt_path.exists():
            try:
                onnx_path = export_pt_to_onnx(pt_path, run_dir / "exports", imgsz=spec.input_wh[0])
                logger.info("%s: exported %s -> %s", spec.name, pt_path, onnx_path)
            except Exception as exc:  # noqa: BLE001
                report.status = "skipped"
                report.reason = f"ONNX export from {pt_path} failed: {exc}"
                return report
    if onnx_path is None:
        report.status = "skipped"
        report.reason = "no ONNX artifact in-tree and no .pt to export from"
        return report

    report.original_path = str(onnx_path)
    report.original_size_bytes = onnx_path.stat().st_size

    # 2. gather real eval samples.
    samples, used_sources = build_eval_samples(spec, aixavier_root)
    report.eval_sources_used = used_sources
    report.n_eval_samples = len(samples)
    if not samples:
        report.status = "skipped"
        report.reason = (
            f"no usable eval data found (checked: {list(spec.eval_sources)}) - "
            "refusing to ship an unmeasured compressed artifact"
        )
        return report

    # 3. baseline inference (the FP32 production graph's own outputs are the
    #    fidelity reference - see module docstring on why there is no ground
    #    truth here).
    try:
        baseline_outputs = run_onnx_all_outputs(onnx_path, samples)
    except Exception as exc:  # noqa: BLE001
        report.status = "error"
        report.reason = f"baseline inference failed: {exc}"
        return report

    # 4. quantize (dynamic INT8, MatMul/Gemm only - compression.quantize_onnx),
    #    after the standard shape-inference/optimization preprocess pass.
    try:
        quantize_input = quant_preprocess(onnx_path, run_dir)
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s: quant_pre_process failed (%s); quantizing the raw ONNX graph", spec.name, exc)
        quantize_input = onnx_path

    compressed_path = run_dir / "quantized" / (onnx_path.stem + "_int8.onnx")
    compressed_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        quantize_onnx(quantize_input, compressed_path, mode="dynamic")
    except Exception as exc:  # noqa: BLE001
        report.status = "skipped"
        report.reason = f"quantization failed; shipping uncompressed artifact: {exc}"
        report.chosen_path = str(onnx_path)
        return report

    report.compressed_path = str(compressed_path)
    report.compressed_size_bytes = compressed_path.stat().st_size
    report.size_reduction_pct = (
        100.0 * (report.original_size_bytes - report.compressed_size_bytes) / report.original_size_bytes
    )

    # 5. compressed inference on the SAME samples.
    try:
        compressed_outputs = run_onnx_all_outputs(compressed_path, samples)
    except Exception as exc:  # noqa: BLE001
        report.status = "refused_floor"
        report.reason = f"compressed artifact failed to run; REFUSED, shipping uncompressed: {exc}"
        report.chosen_path = str(onnx_path)
        return report

    # 6. fidelity metric + the (never-softened) accuracy-floor gate.
    metric = output_agreement(baseline_outputs, compressed_outputs)
    report.baseline_metric = 1.0  # self-fidelity by definition
    report.compressed_metric = metric
    compression_cfg = CompressionConfig(accuracy_floor_relative=spec.accuracy_floor_relative)
    passed, floor = decide_gate(1.0, metric, compression_cfg, ModelConstraints())
    report.floor = floor
    report.passed = passed
    report.status = "compressed" if passed else "refused_floor"
    report.chosen_path = str(compressed_path) if passed else str(onnx_path)
    report.reason = (
        "compressed artifact holds the output-fidelity floor"
        if passed
        else "REFUSED: compressed artifact fell below the output-fidelity floor - shipping uncompressed FP32"
    )
    return report


# ── driver ────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-root", required=True, help="path to aixavier/models")
    parser.add_argument("--output-dir", default="output/production_compression")
    parser.add_argument("--only", action="append", default=None, help="restrict to these model name(s)")
    parser.add_argument("--samples", type=int, default=None, help="override n_samples for every model")
    args = parser.parse_args(argv)

    models_root = Path(args.models_root).resolve()
    aixavier_root = models_root.parent  # models_root is .../aixavier/models
    run_dir = Path(args.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    specs = MODEL_REGISTRY
    if args.only:
        wanted = set(args.only)
        specs = [s for s in specs if s.name in wanted]
        if not specs:
            logger.error("no registry entries match --only %s", args.only)
            return 2
    if args.samples:
        # video_clip specs use n_samples as the model's fixed clip length
        # (e.g. VideoMAE's 16-frame clip) - overriding it changes the input
        # shape the ONNX graph expects, not just "how much eval data". Only
        # video_frames/images specs treat n_samples as an eval-sample count.
        specs = [
            ModelSpec(**{**vars(spec), "n_samples": args.samples}) if spec.eval_kind != "video_clip" else spec
            for spec in specs
        ]

    reports: list[ModelCompressionReport] = []
    started = time.time()
    # onnxruntime's quant_pre_process drops a "sym_shape_infer_temp.onnx"
    # scratch file relative to the CURRENT WORKING DIRECTORY (no parameter
    # to redirect it) - chdir into run_dir so that (and anything else with
    # the same habit) lands inside our own scratch tree, never the repo.
    with contextlib.chdir(run_dir):
        for spec in specs:
            logger.info("=== %s (%s) ===", spec.name, spec.family)
            report = process_model(spec, models_root, aixavier_root, run_dir)
            reports.append(report)
            logger.info("%s -> %s: %s", spec.name, report.status, report.reason)

            out_path = run_dir / f"{spec.name}.json"
            out_path.write_text(json.dumps(report.to_dict(), indent=2))

    skipped_reports = [
        ModelCompressionReport(
            name=item["name"], family=item["family"], status="skipped", reason=item["reason"],
            original_path=item["path"],
        )
        for item in SKIPPED_MODELS
    ]
    for report in skipped_reports:
        (run_dir / f"{report.name}.json").write_text(json.dumps(report.to_dict(), indent=2))

    all_reports = reports + skipped_reports
    summary = {
        "elapsed_seconds": time.time() - started,
        "models": [r.to_dict() for r in all_reports],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n%-32s %-10s %10s %10s %8s %6s" % ("model", "status", "orig_KB", "comp_KB", "reduc%", "gate"))
    for r in all_reports:
        orig_kb = f"{r.original_size_bytes / 1024:.0f}" if r.original_size_bytes else "-"
        comp_kb = f"{r.compressed_size_bytes / 1024:.0f}" if r.compressed_size_bytes else "-"
        reduc = f"{r.size_reduction_pct:.1f}" if r.size_reduction_pct is not None else "-"
        gate = "PASS" if r.passed else ("REFUSED" if r.passed is False else "-")
        print("%-32s %-10s %10s %10s %8s %6s" % (r.name, r.status, orig_kb, comp_kb, reduc, gate))

    logger.info("wrote per-model reports + summary.json to %s", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
