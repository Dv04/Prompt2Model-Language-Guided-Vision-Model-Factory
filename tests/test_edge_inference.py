"""Week 6 - Madhuvani: Tests for zero-configuration edge inference."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from prompt2model.config import (
    DatasetConfig,
    DatasetFormat,
    ExportConfig,
    PipelineConfig,
    RequestedLabel,
    ResolvedLabel,
    TaskType,
    TrainingConfig,
)
from prompt2model.edge_inference import EdgeModel, load_model_from_onnx
from prompt2model.exporting import build_metadata_props, export_model_to_onnx
from prompt2model.models import build_classification_model


def _make_test_config() -> PipelineConfig:
    return PipelineConfig(
        prompt="Classify dogs and cats",
        task=TaskType.CLASSIFICATION,
        labels=[RequestedLabel(name="dog"), RequestedLabel(name="cat")],
        dataset=DatasetConfig(root="/tmp/test", format=DatasetFormat.IMAGEFOLDER, image_size=64),
        training=TrainingConfig(),
        export=ExportConfig(),
        resolved_labels=[
            ResolvedLabel(requested_label="dog", dataset_label="dog", score=1.0, method="identity"),
            ResolvedLabel(requested_label="cat", dataset_label="cat", score=1.0, method="identity"),
        ],
        model_name="mobilenet_v3_small",
    )


def test_edge_model_loads_metadata(tmp_path: str) -> None:
    from pathlib import Path
    tmp = Path(tmp_path)
    config = _make_test_config()
    meta = build_metadata_props(config, class_names=["dog", "cat"])

    model = build_classification_model("mobilenet_v3_small", num_classes=2, pretrained=False)
    example = torch.randn(1, 3, 64, 64)
    onnx_path = tmp / "model.onnx"

    export_model_to_onnx(model, TaskType.CLASSIFICATION, example, onnx_path, meta)

    edge_model = load_model_from_onnx(onnx_path)
    assert edge_model.task == "classification"
    assert edge_model.image_size == 64
    assert len(edge_model.labels) == 2
    assert "dog" in edge_model.labels


def test_edge_model_inference(tmp_path: str) -> None:
    from pathlib import Path
    tmp = Path(tmp_path)
    config = _make_test_config()
    meta = build_metadata_props(config, class_names=["dog", "cat"])

    model = build_classification_model("mobilenet_v3_small", num_classes=2, pretrained=False)
    example = torch.randn(1, 3, 64, 64)
    onnx_path = tmp / "model_inf.onnx"

    export_model_to_onnx(model, TaskType.CLASSIFICATION, example, onnx_path, meta)

    edge_model = EdgeModel(onnx_path)
    
    # Create a dummy image
    image_path = tmp / "test_img.png"
    Image.fromarray(np.uint8(np.random.rand(100, 100, 3) * 255)).save(image_path)
    
    results = edge_model.run_inference(image_path)
    assert "label" in results
    assert "score" in results
    assert "all_scores" in results
    assert len(results["all_scores"]) == 2


def test_edge_model_reads_real_exporter_contract(tmp_path: str) -> None:
    """Regression test for the metadata key mismatch.

    ``EdgeModel`` used to read "image_size"/"normalization_mean"/
    "normalization_std", but the real export path
    (``exporting.build_metadata_props``, called from ``pipeline.py`` and
    ``tuning.py``) writes "input_resolution"/"mean"/"std" instead. Because
    the mismatched keys silently fell back to defaults instead of raising,
    an EdgeModel built from a real artifact with a non-default resolution or
    custom normalisation would resize images to the wrong size and hit an
    ONNX Runtime shape-mismatch (the export graph is not dynamic on H/W), or
    silently normalise with the wrong statistics.

    This exports through the actual production functions
    (``build_metadata_props`` + ``export_model_to_onnx``) with a non-default
    image size and non-default mean/std, then loads with ``EdgeModel`` and
    runs inference end to end - no network, a tiny untrained backbone.
    """
    tmp = Path(tmp_path)
    config = _make_test_config()  # image_size=64, i.e. not the "128" fallback
    custom_mean = (0.1, 0.2, 0.3)
    custom_std = (0.5, 0.6, 0.7)
    meta = build_metadata_props(config, class_names=["dog", "cat"], mean=custom_mean, std=custom_std)

    model = build_classification_model("mobilenet_v3_small", num_classes=2, pretrained=False)
    example = torch.randn(1, 3, 64, 64)
    onnx_path = tmp / "model_contract.onnx"
    export_model_to_onnx(model, TaskType.CLASSIFICATION, example, onnx_path, meta)

    edge_model = EdgeModel(onnx_path)
    assert edge_model.image_size == 64
    np.testing.assert_allclose(edge_model.mean, np.array(custom_mean, dtype=np.float32))
    np.testing.assert_allclose(edge_model.std, np.array(custom_std, dtype=np.float32))

    image_path = tmp / "contract_img.png"
    Image.fromarray(np.uint8(np.random.rand(100, 100, 3) * 255)).save(image_path)

    # Would previously raise from onnxruntime: the preprocessed tensor was
    # resized to the wrong fallback size (128) against a graph fixed at 64.
    results = edge_model.run_inference(image_path)
    assert results["label"] in ("dog", "cat")
