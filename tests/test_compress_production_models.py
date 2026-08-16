"""Tests for scripts/compress_production_models.py - the driver that applies
B1's quantize/gate machinery to Dhi's production edge models.

These exercise the driver's own plumbing (ONNX resolution, eval-sample
gathering, quantize + quant_pre_process, the output-agreement metric, and
the never-softened accuracy-floor gate) end to end against small synthetic
models and images - never against the real production weights (those are
covered by the manual run recorded in the compression report, not CI)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
import torch.nn as nn
from PIL import Image

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import compress_production_models as cpm  # noqa: E402


# ── fixtures ──────────────────────────────────────────────────────────────────


def _export_quantizable_onnx(path: Path, degrade: bool = False) -> Path:
    """A tiny Linear-heavy net (real MatMul/Gemm to quantize, unlike the
    Conv-only production detectors this script mostly compresses - see the
    module docstring's honest finding about Conv-heavy graphs)."""
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    ).eval()
    if degrade:
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 50.0)  # blow up the weights
    torch.onnx.export(
        model, torch.rand(1, 3, 32, 32), str(path),
        input_names=["images"], output_names=["logits"], opset_version=17, dynamo=False,
    )
    return path


def _write_images(directory: Path, n: int = 4) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    paths = []
    for i in range(n):
        arr = rng.integers(0, 255, size=(48, 48, 3), dtype=np.uint8)
        path = directory / f"img{i}.png"
        Image.fromarray(arr).save(path)
        paths.append(path)
    return paths


def _make_tree(tmp_path: Path, onnx_path: Path | None, images: list[Path]) -> tuple[Path, Path, cpm.ModelSpec]:
    """A minimal aixavier-shaped tree: <root>/models/onnx/... + <root>/imgs/...
    onnx_rel/pt_rel are relative to models_root; eval_sources relative to
    aixavier_root - exactly like the real registry entries."""
    aixavier_root = tmp_path / "aixavier"
    models_root = aixavier_root / "models"
    models_root.mkdir(parents=True, exist_ok=True)

    onnx_rel = None
    if onnx_path is not None:
        dest = models_root / "onnx" / "testfam" / "tiny.onnx"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(onnx_path.read_bytes())
        onnx_rel = str(dest.relative_to(models_root))

    img_dir = aixavier_root / "imgs"
    eval_sources = tuple(str(p.relative_to(aixavier_root)) for p in images) if images else ("imgs/missing.png",)

    spec = cpm.ModelSpec(
        name="tiny_test_model",
        family="test_family",
        onnx_rel=onnx_rel,
        pt_rel=None,
        input_wh=(32, 32),
        preprocess="yolo",
        eval_kind="images",
        eval_sources=eval_sources,
        accuracy_floor_relative=0.98,
    )
    return models_root, aixavier_root, spec


# ── pure math: cosine agreement ──────────────────────────────────────────────


def test_cosine_identical_is_one():
    a = np.array([1.0, 2.0, 3.0, -4.0])
    assert cpm._cosine(a, a.copy()) == pytest.approx(1.0)


def test_cosine_orthogonal_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cpm._cosine(a, b) == pytest.approx(0.0)


def test_output_agreement_averages_across_samples_and_heads():
    baseline = [[np.array([1.0, 0.0]), np.array([0.0, 1.0])]]
    compressed = [[np.array([1.0, 0.0]), np.array([1.0, 0.0])]]  # second head now agrees on the first vec
    # head 1: cos=1.0, head 2: cos=0.0 -> mean 0.5
    assert cpm.output_agreement(baseline, compressed) == pytest.approx(0.5)


# ── end-to-end: process_model ────────────────────────────────────────────────


def test_process_model_compresses_and_passes_the_gate(tmp_path):
    onnx_path = _export_quantizable_onnx(tmp_path / "src.onnx")
    aixavier_root = tmp_path / "aixavier"
    images = _write_images(aixavier_root / "imgs")
    models_root, aixavier_root, spec = _make_tree(tmp_path, onnx_path, images)

    run_dir = tmp_path / "run"
    report = cpm.process_model(spec, models_root, aixavier_root, run_dir)

    assert report.status == "compressed"
    assert report.passed is True
    assert report.n_eval_samples == len(images)
    assert report.original_size_bytes and report.compressed_size_bytes
    assert report.chosen_path == report.compressed_path
    assert Path(report.compressed_path).exists()
    assert report.metric_name == "output_cosine_similarity_vs_fp32_baseline"
    assert report.compressed_metric is not None and report.compressed_metric > 0.98
    assert report.floor == pytest.approx(0.98)


def test_process_model_refuses_a_deliberately_degraded_artifact(tmp_path, monkeypatch):
    """Regression coverage for the hard rule: decide_gate must REFUSE and
    keep the uncompressed artifact when the "compressed" output disagrees
    with the FP32 baseline - simulated here by monkeypatching quantize_onnx
    to emit a model with wildly different weights instead of a real INT8
    quantization (deterministically forces low output agreement, unlike
    relying on a real quantization pass to happen to degrade badly)."""
    onnx_path = _export_quantizable_onnx(tmp_path / "src.onnx")
    aixavier_root = tmp_path / "aixavier"
    models_root = aixavier_root / "models"
    dest = models_root / "onnx" / "testfam" / "tiny.onnx"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(onnx_path.read_bytes())

    img_dir = aixavier_root / "imgs"
    images = _write_images(img_dir)
    spec = cpm.ModelSpec(
        name="tiny_degraded_model",
        family="test_family",
        onnx_rel=str(dest.relative_to(models_root)),
        pt_rel=None,
        input_wh=(32, 32),
        preprocess="yolo",
        eval_kind="images",
        eval_sources=tuple(str(p.relative_to(aixavier_root)) for p in images),
        accuracy_floor_relative=0.98,
    )

    def _fake_quantize_onnx(src, output_path, **_kwargs):
        # Emit a structurally-identical but numerically-wrecked "compressed"
        # artifact instead of a real INT8 quantization - deterministically
        # reproduces "compression degraded the model" without depending on
        # a specific quantization backend behaving badly.
        _export_quantizable_onnx(Path(output_path), degrade=True)
        return Path(output_path)

    monkeypatch.setattr(cpm, "quantize_onnx", _fake_quantize_onnx)

    run_dir = tmp_path / "run"
    report = cpm.process_model(spec, models_root, aixavier_root, run_dir)

    assert report.status == "refused_floor"
    assert report.passed is False
    assert "REFUSED" in report.reason
    assert report.chosen_path == report.original_path  # uncompressed artifact shipped
    assert report.compressed_metric is not None and report.compressed_metric < report.floor


def test_process_model_skips_when_no_onnx_and_no_pt(tmp_path):
    models_root, aixavier_root, spec = _make_tree(tmp_path, onnx_path=None, images=[])
    run_dir = tmp_path / "run"
    report = cpm.process_model(spec, models_root, aixavier_root, run_dir)
    assert report.status == "skipped"
    assert "no ONNX artifact" in report.reason


def test_process_model_skips_when_no_eval_data(tmp_path):
    onnx_path = _export_quantizable_onnx(tmp_path / "src.onnx")
    models_root, aixavier_root, spec = _make_tree(tmp_path, onnx_path, images=[])
    # eval_sources defaults to a nonexistent file inside _make_tree when images=[]
    run_dir = tmp_path / "run"
    report = cpm.process_model(spec, models_root, aixavier_root, run_dir)
    assert report.status == "skipped"
    assert "no usable eval data" in report.reason
    assert report.n_eval_samples == 0


def test_skipped_models_registry_documents_face_pth():
    names = {entry["name"] for entry in cpm.SKIPPED_MODELS}
    assert "face_model_pth" in names
    entry = next(e for e in cpm.SKIPPED_MODELS if e["name"] == "face_model_pth")
    assert "mmdetection" in entry["reason"]
    assert entry["path"] == "weights/face/model.pth"


def test_model_registry_names_are_unique():
    names = [spec.name for spec in cpm.MODEL_REGISTRY]
    assert len(names) == len(set(names))
