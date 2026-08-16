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
        nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(3, num_classes),
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


def test_user_floor_below_default_is_honored():
    # Issue #14: a stated lenient floor must not be silently raised to the
    # 0.98 retention default (the old max() code returned 0.8036 here).
    compression = CompressionConfig()  # accuracy_floor_relative = 0.98
    constraints = ModelConstraints(accuracy_floor=0.60)
    assert compute_floor(0.82, compression, constraints) == 0.60
    # The real Beans int8 case: 60.2 percent accuracy vs a stated 0.60 floor
    # must pass the gate.
    passed, floor = decide_gate(0.82, 0.602, compression, constraints)
    assert passed and floor == 0.60


def test_user_floor_above_retention_default_is_honored():
    # A stricter explicit floor also binds as stated.
    compression = CompressionConfig()  # accuracy_floor_relative = 0.98
    constraints = ModelConstraints(accuracy_floor=0.99)
    assert compute_floor(0.80, compression, constraints) == 0.99
    # 0.85 clears the 0.98 * 0.80 = 0.784 retention default but not the
    # user's stated 0.99 floor, so the gate refuses.
    passed, floor = decide_gate(0.80, 0.85, compression, constraints)
    assert not passed and floor == 0.99


def test_retention_default_applies_when_no_user_floor():
    constraints = ModelConstraints()  # accuracy_floor is None
    assert compute_floor(0.8, CompressionConfig(), constraints) == pytest.approx(0.98 * 0.8)
    assert compute_floor(
        0.8, CompressionConfig(accuracy_floor_relative=0.9), constraints
    ) == pytest.approx(0.72)


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
    import onnx

    graph = onnx.load(str(out)).graph
    op_types = [node.op_type for node in graph.node]
    conv = next(node for node in graph.node if node.op_type == "Conv")
    group = next(attr.i for attr in conv.attribute if attr.name == "group")
    assert group == 3  # explicit depthwise-Conv regression coverage
    assert "Conv" in op_types
    assert "ConvInteger" not in op_types
    assert "MatMulInteger" in op_types
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


def test_distillation_recovers_from_batchnorm_collapse(tmp_path):
    """Regression test for a real B1 bug: a from-scratch (pretrained=False),
    small-batch, few-epoch classifier collapses its BatchNorm running
    statistics to an input-independent constant output - the exact failure
    mode this repo already diagnosed for the CLI smoke-test path (commit
    ba62c2f: "backbone collapses to an input-independent constant output").
    ``.train()``-mode forward passes (used for every loss computation) are
    unaffected because they use fresh per-batch statistics, so train_loss
    keeps falling normally; only ``.eval()`` (used for every accuracy check,
    including this loop's own per-epoch validation) reads the never-
    -converged running stats and collapses to chance. A real distillation run
    hit exactly this: student stuck at 33.6% (chance for 3 classes) for all
    15 epochs while train_loss fell 6.73 -> 0.60.

    ``test_distillation_smoke`` above could never have caught this: its
    fixture is random images with random labels, so it has no learnable
    signal to lose in the first place and its only assertion is a trivial
    ``0 <= acc <= 1`` bound. This test uses a real, learnable dataset and a
    genuinely collapsed pretraining precondition, and asserts the KD run
    must recover well above chance.
    """
    from prompt2model.config import DatasetConfig, DatasetFormat
    from prompt2model.data import build_classification_bundle, create_synthetic_classification_dataset
    from prompt2model.models import build_classification_model
    from prompt2model.training import train_classification_model

    torch.manual_seed(0)

    data_dir = create_synthetic_classification_dataset(
        str(tmp_path / "data"), samples_per_class=12, image_size=96
    )
    dataset_cfg = DatasetConfig(
        root=str(data_dir), format=DatasetFormat.IMAGEFOLDER, image_size=96,
        val_split=0.2, test_split=0.2, seed=0,
    )
    bundle = build_classification_bundle(dataset_cfg, batch_size=8, augmentations=None)
    device = torch.device("cpu")

    # The exact precedented collapse recipe: pretrained=False, batch_size=8,
    # a couple of epochs, a small dataset.
    train_cfg = TrainingConfig(
        epochs=2, batch_size=8, learning_rate=1e-3, weight_decay=1e-4,
        max_steps_per_epoch=None, pretrained=False,
    )

    teacher = build_classification_model("efficientnet_b0", num_classes=len(bundle.class_names), pretrained=False)
    train_classification_model(teacher, bundle.train_loader, bundle.val_loader, train_cfg, tmp_path / "teacher", device)

    student_plain = build_classification_model("mobilenet_v3_small", num_classes=len(bundle.class_names), pretrained=False)
    train_classification_model(student_plain, bundle.train_loader, bundle.val_loader, train_cfg, tmp_path / "student_plain", device)

    # Confirm the fixture actually reproduces the collapse precondition
    # (near-zero logit variance across genuinely different inputs) before
    # trusting the KD recovery assertion below.
    student_plain.eval()
    with torch.no_grad():
        logits = torch.cat([student_plain(images) for images, _ in bundle.val_loader], dim=0)
    assert logits.std(dim=0).max().item() < 1e-3, (
        "fixture did not reproduce the BatchNorm-collapse precondition this test targets"
    )

    student = build_classification_model("mobilenet_v3_small", num_classes=len(bundle.class_names), pretrained=False)
    student.load_state_dict(student_plain.state_dict())  # KD fine-tune, not from scratch - matches pipeline.py

    distill_cfg = TrainingConfig(
        epochs=15, batch_size=8, learning_rate=1e-3, weight_decay=1e-4,
        max_steps_per_epoch=None, pretrained=False,
    )
    info = train_distilled_classification(
        teacher, student, bundle.train_loader, bundle.val_loader,
        distill_cfg, CompressionConfig(enable_distillation=True), tmp_path / "distilled", device,
    )

    # Chance is 1/3 for this 3-class dataset. A correctly-recalibrating KD
    # loop recovers to near-perfect accuracy on this trivially-separable
    # data well within 15 epochs; before the fix this stayed at chance for
    # every single epoch.
    assert info["best_val_accuracy"] > 0.8


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
