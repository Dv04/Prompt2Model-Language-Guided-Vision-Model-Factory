"""Week 5 - Madhuvani: Tests for enhanced ONNX metadata injection."""

import json

import torch

from prompt2model.config import (
    DatasetConfig,
    DatasetFormat,
    ExportConfig,
    ModelConstraints,
    PipelineConfig,
    ResolvedLabel,
    TaskType,
    TrainingConfig,
    RequestedLabel,
)
from prompt2model.exporting import build_full_metadata, export_model_to_onnx, read_metadata
from prompt2model.models import build_classification_model


def _make_test_config() -> PipelineConfig:
    return PipelineConfig(
        prompt="Classify cats and dogs in low light",
        task=TaskType.CLASSIFICATION,
        labels=[RequestedLabel(name="cat"), RequestedLabel(name="dog")],
        dataset=DatasetConfig(root="/tmp/test", format=DatasetFormat.IMAGEFOLDER, image_size=96),
        training=TrainingConfig(),
        export=ExportConfig(),
        resolved_labels=[
            ResolvedLabel(requested_label="cat", dataset_label="cat", score=1.0, method="identity"),
            ResolvedLabel(requested_label="dog", dataset_label="dog", score=1.0, method="identity"),
        ],
        model_name="mobilenet_v3_small",
        augmentation_tags=["low_light"],
    )


def test_build_full_metadata_keys() -> None:
    config = _make_test_config()
    meta = build_full_metadata(config)

    required_keys = {
        "task", "model_name", "prompt", "input_resolution",
        "normalization_mean", "normalization_std", "class_dictionary",
        "labels", "image_size", "preprocessing_steps",
        "augmentation_tags", "priority", "speed_accuracy_tradeoff",
    }
    assert required_keys.issubset(set(meta.keys()))


def test_build_full_metadata_values_parseable() -> None:
    config = _make_test_config()
    meta = build_full_metadata(config)

    resolution = json.loads(meta["input_resolution"])
    assert resolution["height"] == 96
    assert resolution["width"] == 96

    mean = json.loads(meta["normalization_mean"])
    assert len(mean) == 3
    assert abs(mean[0] - 0.485) < 1e-6

    std = json.loads(meta["normalization_std"])
    assert len(std) == 3

    class_dict = json.loads(meta["class_dictionary"])
    assert class_dict["cat"] == 0
    assert class_dict["dog"] == 1

    steps = json.loads(meta["preprocessing_steps"])
    assert len(steps) == 3
    assert steps[0]["op"] == "resize"
    assert steps[1]["op"] == "to_tensor"
    assert steps[2]["op"] == "normalize"


def test_metadata_injected_into_onnx(tmp_path) -> None:
    config = _make_test_config()
    meta = build_full_metadata(config)

    model = build_classification_model("mobilenet_v3_small", num_classes=2, pretrained=False)
    example = torch.randn(1, 3, 96, 96)
    onnx_path = str(tmp_path / "test_model.onnx")

    export_model_to_onnx(
        model=model,
        task=TaskType.CLASSIFICATION,
        example_input=example,
        output_path=onnx_path,
        metadata=meta,
    )

    read_meta = read_metadata(onnx_path)
    assert "input_resolution" in read_meta
    assert "normalization_mean" in read_meta
    assert "class_dictionary" in read_meta
    assert "preprocessing_steps" in read_meta

    class_dict = json.loads(read_meta["class_dictionary"])
    assert class_dict["cat"] == 0
