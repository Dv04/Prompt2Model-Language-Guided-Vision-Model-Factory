"""Calibration + conformal abstain: the math, the pipeline fit, the edge handshake."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from prompt2model.calibration import (
    calibrate_classification,
    conformal_threshold,
    ece,
    fit_temperature,
)


def _synthetic_logits(n: int = 400, scale: float = 3.0, seed: int = 0):
    """Well-separated 2-class logits, then overconfidence-scaled by `scale`
    — the classic miscalibration a fitted temperature should undo."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n)
    base = rng.normal(0.0, 1.0, (n, 2))
    base[np.arange(n), labels] += 1.5  # signal
    return base * scale, labels


def test_fit_temperature_detects_overconfidence():
    """The NLL-optimal T is not exactly the injected scale (the pre-scale
    logits aren't perfectly calibrated themselves) — the guarantees are:
    it detects overconfidence (T substantially > 1) and lowers NLL."""
    from prompt2model.calibration import _nll

    logits, labels = _synthetic_logits(scale=3.0)
    t = fit_temperature(logits, labels)
    assert 1.3 < t < 6.0
    assert _nll(logits, labels, t) < _nll(logits, labels, 1.0)


def test_temperature_improves_ece():
    logits, labels = _synthetic_logits(scale=4.0)
    t = fit_temperature(logits, labels)

    def _conf_correct(temp):
        scaled = logits / temp
        exp = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        return probs.max(axis=1).tolist(), (probs.argmax(axis=1) == labels).tolist()

    ece_before = ece(*_conf_correct(1.0))
    ece_after = ece(*_conf_correct(t))
    assert ece_after < ece_before


def test_ece_edges():
    assert ece([], []) == 0.0
    assert ece([0.8] * 10, [True] * 8 + [False] * 2) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        ece([0.5], [])


def test_conformal_threshold_quantile():
    scores = [i / 10 for i in range(10)]  # 0.0 .. 0.9
    # n=10, alpha=0.1 → rank ceil(11*0.9)=10 → the 10th smallest = 0.9
    assert conformal_threshold(scores, alpha=0.1) == pytest.approx(0.9)
    assert conformal_threshold([], alpha=0.1) == float("inf")
    with pytest.raises(ValueError):
        conformal_threshold(scores, alpha=1.5)


def _loader(n: int = 24):
    generator = torch.Generator().manual_seed(1)
    images = torch.rand((n, 3, 16, 16), generator=generator)
    labels = torch.randint(0, 2, (n,), generator=generator)
    return DataLoader(TensorDataset(images, labels), batch_size=8)


def test_calibrate_classification_returns_contract():
    model = nn.Sequential(nn.Conv2d(3, 2, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten())
    info = calibrate_classification(model, _loader(), torch.device("cpu"))
    assert info["calibrated"] is True
    assert info["temperature"] > 0
    assert 0 < info["alpha"] < 1
    assert info["val_samples"] == 24
    assert "ece_before" in info and "ece_after" in info
    assert info["conformal_threshold"] >= 0


def test_calibrate_never_raises():
    info = calibrate_classification(object(), None, torch.device("cpu"))
    assert info["calibrated"] is False and "error" in info


# ── the edge handshake ───────────────────────────────────────────────────────


def _export_with_calibration(tmp_path, threshold: float):
    from prompt2model.edge_inference import EdgeModel
    from prompt2model.exporting import inject_metadata

    model = nn.Sequential(nn.Conv2d(3, 2, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten()).eval()
    onnx_path = tmp_path / "m.onnx"
    torch.onnx.export(model, torch.rand(1, 3, 16, 16), str(onnx_path), opset_version=17, dynamo=False)
    inject_metadata(
        str(onnx_path),
        {
            "task": "classification",
            "image_size": 16,
            "labels": ["a", "b"],
            "calibration": {
                "calibrated": True,
                "temperature": 1.0,
                "alpha": 0.1,
                "conformal_threshold": threshold,
            },
        },
    )
    return EdgeModel(onnx_path)


def _image(tmp_path):
    from PIL import Image

    path = tmp_path / "img.png"
    Image.new("RGB", (16, 16), (128, 40, 200)).save(path)
    return path


def test_edge_model_abstains_below_threshold(tmp_path):
    edge = _export_with_calibration(tmp_path, threshold=0.0)  # any doubt ⇒ abstain
    result = edge.run_inference(_image(tmp_path))
    assert result["calibrated"] is True
    # A 2-class random head can't be perfectly certain → nonconformity > 0.
    assert result["score"] < 1.0
    assert result["abstained"] is True


def test_edge_model_accepts_with_loose_threshold(tmp_path):
    edge = _export_with_calibration(tmp_path, threshold=1.0)  # never abstain
    result = edge.run_inference(_image(tmp_path))
    assert result["abstained"] is False
    assert result["label"] in ("a", "b")


def test_edge_model_uncalibrated_artifact_backward_compatible(tmp_path):
    from prompt2model.edge_inference import EdgeModel
    from prompt2model.exporting import inject_metadata

    model = nn.Sequential(nn.Conv2d(3, 2, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten()).eval()
    onnx_path = tmp_path / "m.onnx"
    torch.onnx.export(model, torch.rand(1, 3, 16, 16), str(onnx_path), opset_version=17, dynamo=False)
    inject_metadata(str(onnx_path), {"task": "classification", "image_size": 16, "labels": ["a", "b"]})
    result = EdgeModel(onnx_path).run_inference(_image(tmp_path))
    assert result["calibrated"] is False and result["abstained"] is False
    assert "label" in result and "score" in result  # legacy keys intact


def test_end_to_end_run_embeds_calibration(tmp_path):
    from prompt2model.config import DatasetConfig, DatasetFormat, TrainingConfig
    from prompt2model.data import create_synthetic_classification_dataset
    from prompt2model.edge_inference import EdgeModel
    from prompt2model.pipeline import run_from_prompt

    data_dir = create_synthetic_classification_dataset(str(tmp_path / "data"))
    result = run_from_prompt(
        prompt='Classify "red square" and "blue circle" images, prioritize speed.',
        dataset=DatasetConfig(root=str(data_dir), format=DatasetFormat.IMAGEFOLDER, image_size=64),
        output_dir=str(tmp_path / "run"),
        training_overrides=TrainingConfig(epochs=1, batch_size=4, max_steps_per_epoch=2),
    )
    assert result.metrics["calibration"]["calibrated"] is True
    edge = EdgeModel(result.onnx_path)
    assert edge.calibration.get("calibrated") is True
    assert "conformal_threshold" in edge.calibration
