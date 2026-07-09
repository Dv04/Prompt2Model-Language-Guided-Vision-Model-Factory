"""Compression stage: quantization, distillation, and the accuracy-floor gate."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from prompt2model.compression import (
    CompressionReport,
    apply_compression,
    compute_floor,
    decide_gate,
    evaluate_onnx_classification,
    quantize_onnx,
    train_distilled_classification,
)
from prompt2model.config import CompressionConfig, ModelConstraints, TrainingConfig


def _tiny_net(num_classes: int = 2) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, num_classes),
    )


def _loader(n: int = 16, image_size: int = 32, batch: int = 8) -> DataLoader:
    generator = torch.Generator().manual_seed(0)
    images = torch.rand((n, 3, image_size, image_size), generator=generator)
    labels = torch.randint(0, 2, (n,), generator=generator)
    return DataLoader(TensorDataset(images, labels), batch_size=batch)


def _export_tiny_onnx(path: Path) -> Path:
    model = _tiny_net().eval()
    torch.onnx.export(
        model, torch.rand(1, 3, 32, 32), str(path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}}, opset_version=17, dynamo=False,
    )
    return path


# ── gate math (pure) ─────────────────────────────────────────────────────────


def test_floor_is_stricter_of_relative_and_absolute():
    compression = CompressionConfig(accuracy_floor_relative=0.9)
    # Relative floor: 0.9 * 0.8 = 0.72; absolute floor 0.75 is stricter.
    assert compute_floor(0.8, compression, ModelConstraints(accuracy_floor=0.75)) == 0.75
    # No absolute floor → relative governs.
    assert compute_floor(0.8, compression, ModelConstraints()) == pytest.approx(0.72)


def test_gate_pass_and_refuse():
    compression = CompressionConfig(accuracy_floor_relative=0.95)
    constraints = ModelConstraints()
    passed, floor = decide_gate(0.90, 0.88, compression, constraints)
    assert passed and floor == pytest.approx(0.855)
    passed, _ = decide_gate(0.90, 0.80, compression, constraints)  # below 0.855
    assert not passed


# ── quantization + same-runtime evaluation ───────────────────────────────────


def test_quantize_dynamic_and_evaluate(tmp_path):
    onnx_path = _export_tiny_onnx(tmp_path / "model.onnx")
    out = quantize_onnx(onnx_path, tmp_path / "model_int8.onnx", mode="dynamic")
    assert out.exists() and out.stat().st_size > 0
    accuracy = evaluate_onnx_classification(out, _loader())
    assert 0.0 <= accuracy <= 1.0


def test_quantize_static_requires_calibration(tmp_path):
    onnx_path = _export_tiny_onnx(tmp_path / "model.onnx")
    with pytest.raises(ValueError):
        quantize_onnx(onnx_path, tmp_path / "x.onnx", mode="static", calibration_loader=None)
    with pytest.raises(ValueError):
        quantize_onnx(onnx_path, tmp_path / "x.onnx", mode="nonsense")


def test_apply_compression_produces_gated_report(tmp_path):
    onnx_path = _export_tiny_onnx(tmp_path / "model.onnx")
    report = apply_compression(
        onnx_path,
        _loader(),
        CompressionConfig(enable_quantization=True, accuracy_floor_relative=0.0),
        ModelConstraints(),
    )
    assert isinstance(report, CompressionReport)
    assert report.attempted and report.passed is True  # floor 0 always passes
    assert report.chosen_path and report.chosen_path.endswith("_int8.onnx")
    assert Path(report.chosen_path).exists()
    assert report.baseline_size_bytes and report.compressed_size_bytes


def test_apply_compression_refusal_keeps_uncompressed(tmp_path):
    onnx_path = _export_tiny_onnx(tmp_path / "model.onnx")
    # An impossible absolute floor forces the refusal path deterministically.
    report = apply_compression(
        onnx_path,
        _loader(),
        CompressionConfig(enable_quantization=True),
        ModelConstraints(accuracy_floor=1.0),
    )
    # (Unless the tiny net is accidentally perfect on random labels - with 16
    # random labels the chance is negligible; guard anyway.)
    if report.compressed_val_accuracy is not None and report.compressed_val_accuracy < 1.0:
        assert report.passed is False
        assert report.chosen_path == str(onnx_path)
        assert "REFUSED" in report.reason


def test_apply_compression_disabled_is_noop(tmp_path):
    onnx_path = _export_tiny_onnx(tmp_path / "model.onnx")
    report = apply_compression(
        onnx_path, _loader(), CompressionConfig(), ModelConstraints()
    )
    assert not report.attempted and report.chosen_path == str(onnx_path)


def test_apply_compression_never_raises(tmp_path):
    # A bogus path must yield a refusal report, not an exception.
    report = apply_compression(
        tmp_path / "missing.onnx",
        _loader(),
        CompressionConfig(enable_quantization=True),
        ModelConstraints(),
    )
    assert report.attempted and report.passed is False
    assert "error" in report.reason


# ── distillation ─────────────────────────────────────────────────────────────


def test_distillation_smoke(tmp_path):
    teacher, student = _tiny_net(), _tiny_net()
    info = train_distilled_classification(
        teacher,
        student,
        _loader(),
        _loader(),
        TrainingConfig(epochs=1, batch_size=8, max_steps_per_epoch=2),
        CompressionConfig(enable_distillation=True),
        tmp_path,
        torch.device("cpu"),
    )
    assert Path(info["checkpoint_path"]).exists()
    assert len(info["history"]) == 1
    assert 0.0 <= info["best_val_accuracy"] <= 1.0
    assert info["temperature"] == 4.0 and info["alpha"] == 0.7


# ── prompt/config plumbing ───────────────────────────────────────────────────


def test_prompt_keywords_enable_compression():
    from prompt2model.config import DatasetConfig
    from prompt2model.parsing import parse_prompt

    dataset = DatasetConfig(root="output/toy")
    config = parse_prompt("Classify \"cat\" images, quantized int8, distilled.", dataset)
    assert config.compression.enable_quantization
    assert config.compression.enable_distillation
    plain = parse_prompt("Classify \"cat\" images.", dataset)
    assert not plain.compression.enable_quantization
    assert not plain.compression.enable_distillation


def test_planner_overlay_sets_compression_flags():
    import json

    from prompt2model.config import DatasetConfig
    from prompt2model.planner import LLMPlanner, plan_prompt

    config = plan_prompt(
        "Classify \"cat\" images.",
        DatasetConfig(root="output/toy"),
        planner=LLMPlanner(transport=lambda messages: json.dumps({"quantize": True})),
        mode="llm",
    )
    assert config.compression.enable_quantization


def test_end_to_end_quantized_run(tmp_path):
    """Full pipeline with quantization: synthetic data → train → export →
    quantize → gate. The report and chosen artifact must exist."""
    from prompt2model.config import DatasetConfig, DatasetFormat, TrainingConfig
    from prompt2model.data import create_synthetic_classification_dataset
    from prompt2model.pipeline import run_from_prompt

    data_dir = create_synthetic_classification_dataset(str(tmp_path / "data"))
    result = run_from_prompt(
        prompt='Classify "red square" and "blue circle" images, prioritize speed.',
        dataset=DatasetConfig(root=str(data_dir), format=DatasetFormat.IMAGEFOLDER, image_size=64),
        output_dir=str(tmp_path / "run"),
        training_overrides=TrainingConfig(epochs=1, batch_size=4, max_steps_per_epoch=2),
        quantize=True,
    )
    assert result.onnx_path is not None
    assert result.compression is not None and result.compression["attempted"]
    assert result.compression["chosen_path"]
    assert Path(result.compression["chosen_path"]).exists()
    assert result.metrics["compression"]["mode"] == "dynamic"
    if result.compression["passed"]:
        assert result.compressed_onnx_path
        assert result.compressed_onnx_path.endswith("_int8.onnx")
