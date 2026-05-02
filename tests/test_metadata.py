"""Unit tests for the build_metadata_props injection tool.

These tests exercise the metadata builder in isolation (no GPU, no ONNX runtime
download required) and also verify the inject → read round-trip against a minimal
synthetic ONNX model.
"""

from __future__ import annotations

import json
from pathlib import Path

import onnx
import torch
from onnx import TensorProto, helper

from prompt2model.config import (
    DatasetConfig,
    DatasetFormat,
    ExportConfig,
    ModelConstraints,
    PipelineConfig,
    RequestedLabel,
    ResolvedLabel,
    TaskType,
    TrainingConfig,
)
from prompt2model.data import IMAGENET_MEAN, IMAGENET_STD
from prompt2model.exporting import build_metadata_props, inject_metadata, read_metadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    task: TaskType = TaskType.CLASSIFICATION,
    image_size: int = 128,
    resolved_labels: list[ResolvedLabel] | None = None,
    model_name: str = "mobilenet_v3_small",
) -> PipelineConfig:
    """Return a minimal PipelineConfig with the given parameters."""
    if resolved_labels is None:
        resolved_labels = [
            ResolvedLabel(requested_label="cat", dataset_label="cat", score=1.0, method="identity"),
            ResolvedLabel(requested_label="dog", dataset_label="dog", score=1.0, method="identity"),
        ]
    return PipelineConfig(
        project_name="test",
        prompt="Classify cat and dog images.",
        task=task,
        labels=[RequestedLabel(name="cat"), RequestedLabel(name="dog")],
        constraints=ModelConstraints(),
        dataset=DatasetConfig(root="/tmp/fake", format=DatasetFormat.IMAGEFOLDER, image_size=image_size),
        training=TrainingConfig(),
        export=ExportConfig(),
        model_name=model_name,
        resolved_labels=resolved_labels,
    )


def _make_minimal_onnx(tmp_path: Path) -> Path:
    """Create a tiny valid ONNX file (single Identity node) for round-trip tests."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 128, 128])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 128, 128])
    node = helper.make_node("Identity", inputs=["X"], outputs=["Y"])
    graph = helper.make_graph([node], "test_graph", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    out = tmp_path / "minimal.onnx"
    onnx.save(model, str(out))
    return out


# ---------------------------------------------------------------------------
# Test 1 — all required keys are present for classification
# ---------------------------------------------------------------------------

def test_build_metadata_props_classification_has_all_keys() -> None:
    config = _make_config(task=TaskType.CLASSIFICATION, image_size=128)
    class_names = ["cat", "dog"]
    meta = build_metadata_props(config, class_names)

    required_keys = {
        "task", "prompt", "model_name",
        "input_resolution", "mean", "std",
        "class_dict", "labels", "label_map",
    }
    assert required_keys.issubset(meta.keys()), (
        f"Missing keys: {required_keys - meta.keys()}"
    )


# ---------------------------------------------------------------------------
# Test 2 — values are correct for classification
# ---------------------------------------------------------------------------

def test_build_metadata_props_classification_values() -> None:
    config = _make_config(task=TaskType.CLASSIFICATION, image_size=224)
    class_names = ["cat", "dog"]
    meta = build_metadata_props(config, class_names)

    assert meta["task"] == "classification"
    assert meta["input_resolution"] == [224, 224]
    assert meta["mean"] == list(IMAGENET_MEAN)
    assert meta["std"] == list(IMAGENET_STD)
    assert meta["class_dict"] == {"0": "cat", "1": "dog"}
    assert meta["labels"] == ["cat", "dog"]
    assert meta["label_map"] == {"cat": "cat", "dog": "dog"}
    assert meta["model_name"] == "mobilenet_v3_small"


# ---------------------------------------------------------------------------
# Test 3 — custom normalisation overrides propagate correctly
# ---------------------------------------------------------------------------

def test_build_metadata_props_custom_norm() -> None:
    config = _make_config()
    custom_mean = (0.5, 0.5, 0.5)
    custom_std = (0.25, 0.25, 0.25)
    meta = build_metadata_props(config, ["cat", "dog"], mean=custom_mean, std=custom_std)

    assert meta["mean"] == [0.5, 0.5, 0.5]
    assert meta["std"] == [0.25, 0.25, 0.25]


# ---------------------------------------------------------------------------
# Test 4 — label_map reflects the resolver's resolved labels correctly
# ---------------------------------------------------------------------------

def test_build_metadata_props_label_map_reflects_resolver() -> None:
    resolved = [
        ResolvedLabel(requested_label="motorcycle", dataset_label="motorbike", score=0.92, method="lexical"),
        ResolvedLabel(requested_label="car", dataset_label="automobile", score=0.88, method="clip"),
    ]
    config = _make_config(resolved_labels=resolved)
    meta = build_metadata_props(config, ["automobile", "motorbike"])

    assert meta["label_map"]["motorcycle"] == "motorbike"
    assert meta["label_map"]["car"] == "automobile"
    # class_dict should index the class_names list, not the resolved labels
    assert meta["class_dict"] == {"0": "automobile", "1": "motorbike"}


# ---------------------------------------------------------------------------
# Test 5 — inject → read round-trip preserves all metadata faithfully
# ---------------------------------------------------------------------------

def test_inject_and_read_roundtrip(tmp_path: Path) -> None:
    onnx_path = _make_minimal_onnx(tmp_path)
    config = _make_config(image_size=128)
    class_names = ["cat", "dog"]
    meta_in = build_metadata_props(config, class_names)

    inject_metadata(onnx_path, meta_in)
    meta_out = read_metadata(onnx_path)

    # Every key survives the round-trip
    for key in meta_in:
        assert key in meta_out, f"Key '{key}' lost after inject/read round-trip"

    # Structured values survive JSON deserialisation
    assert json.loads(meta_out["class_dict"]) == {"0": "cat", "1": "dog"}
    assert json.loads(meta_out["label_map"]) == {"cat": "cat", "dog": "dog"}
    assert json.loads(meta_out["input_resolution"]) == [128, 128]
    assert json.loads(meta_out["mean"]) == list(IMAGENET_MEAN)
    assert json.loads(meta_out["std"]) == list(IMAGENET_STD)

    # Scalar string values
    assert meta_out["task"] == "classification"
    assert meta_out["model_name"] == "mobilenet_v3_small"
