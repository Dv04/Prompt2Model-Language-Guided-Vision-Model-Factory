from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from prompt2model.config import PipelineConfig, TaskType
from prompt2model.data import IMAGENET_MEAN, IMAGENET_STD


def build_metadata_props(
    config: PipelineConfig,
    class_names: list[str],
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> dict[str, Any]:
    """Build the complete metadata dictionary to embed in the ONNX file.

    Args:
        config: The fully resolved pipeline configuration (including resolved_labels).
        class_names: Ordered list of class names as seen by the model
            (index 0 → class_names[0], etc.).  For detection models trained with a
            background class at index 0, pass the dataset category names only; the
            background entry is handled transparently by the model and is NOT included
            here so that consumers index directly with the predicted label integer.
        mean: Per-channel mean used during training normalisation.
            Defaults to ImageNet mean (0.485, 0.456, 0.406).
        std: Per-channel std used during training normalisation.
            Defaults to ImageNet std (0.229, 0.224, 0.225).

    Returns:
        A flat dict whose values are JSON-serialisable primitives or
        JSON-serialisable containers.  ``inject_metadata`` will serialise
        ``dict``/``list`` values automatically.

    Keys injected
    -------------
    task              : str - "classification" or "detection"
    prompt            : str - the original natural-language prompt
    model_name        : str - backbone identifier chosen by the pipeline
    input_resolution  : list - [height, width] in pixels (square images)
    mean              : list - [R, G, B] per-channel normalisation mean
    std               : list - [R, G, B] per-channel normalisation std
    class_dict        : dict - {"0": "cat", "1": "dog", …}  (str keys for JSON)
    labels            : list - flat list of dataset class names (legacy / convenience)
    label_map         : dict - {"requested_label": "dataset_label", …} from resolver
    """
    class_dict: dict[str, str] = {str(i): name for i, name in enumerate(class_names)}
    label_map: dict[str, str] = {
        resolved.requested_label: resolved.dataset_label
        for resolved in config.resolved_labels
    }
    return {
        "task": config.task.value,
        "prompt": config.prompt,
        "model_name": config.model_name or "",
        "input_resolution": [config.dataset.image_size, config.dataset.image_size],
        "mean": list(mean),
        "std": list(std),
        "class_dict": class_dict,
        "labels": class_names,
        "label_map": label_map,
    }


class DetectionExportWrapper(nn.Module):
    def __init__(self, model: nn.Module, topk: int) -> None:
        super().__init__()
        self.model = model
        self.topk = topk

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        predictions = self.model([images[0]])[0]
        boxes = predictions["boxes"][: self.topk]
        scores = predictions["scores"][: self.topk]
        labels = predictions["labels"][: self.topk]
        pad = self.topk - boxes.shape[0]
        if pad > 0:
            boxes = torch.cat([boxes, torch.zeros((pad, 4), device=boxes.device, dtype=boxes.dtype)], dim=0)
            scores = torch.cat([scores, torch.zeros((pad,), device=scores.device, dtype=scores.dtype)], dim=0)
            labels = torch.cat([labels, torch.zeros((pad,), device=labels.device, dtype=labels.dtype)], dim=0)
        return boxes, scores, labels


def build_full_metadata(config: PipelineConfig) -> dict[str, Any]:
    """Construct the complete ONNX metadata dictionary from a pipeline config.

    Embeds all information required for zero-configuration edge deployment:
    - task type and model name
    - input resolution
    - normalization statistics
    - class dictionary (label → index)
    - preprocessing steps
    - augmentation tags
    - original prompt
    """
    label_names = [r.dataset_label for r in config.resolved_labels]
    class_dictionary = {label: idx for idx, label in enumerate(label_names)}

    return {
        "task": config.task.value,
        "model_name": config.model_name or "unknown",
        "prompt": config.prompt,
        "input_resolution": json.dumps({
            "height": config.dataset.image_size,
            "width": config.dataset.image_size,
        }),
        "normalization_mean": json.dumps(list(IMAGENET_MEAN)),
        "normalization_std": json.dumps(list(IMAGENET_STD)),
        "class_dictionary": json.dumps(class_dictionary),
        "labels": json.dumps(label_names),
        "image_size": str(config.dataset.image_size),
        "preprocessing_steps": json.dumps([
            {"op": "resize", "height": config.dataset.image_size, "width": config.dataset.image_size},
            {"op": "to_tensor"},
            {"op": "normalize", "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        ]),
        "augmentation_tags": json.dumps(config.augmentation_tags),
        "priority": config.constraints.priority.value,
        "speed_accuracy_tradeoff": str(config.constraints.speed_accuracy_tradeoff),
    }


def export_model_to_onnx(
    model: nn.Module,
    task: TaskType,
    example_input: torch.Tensor,
    output_path: str | Path,
    metadata: dict[str, Any],
    topk_detections: int = 20,
    opset: int = 17,
) -> str:
    output_path = str(output_path)
    model.eval()
    wrapper: nn.Module
    output_names: list[str]
    if task == TaskType.CLASSIFICATION:
        wrapper = model
        output_names = ["logits"]
    else:
        wrapper = DetectionExportWrapper(model, topk=topk_detections)
        output_names = ["boxes", "scores", "labels"]

    torch.onnx.export(
        wrapper.cpu(),
        example_input.cpu(),
        output_path,
        export_params=True,
        input_names=["images"],
        output_names=output_names,
        dynamic_axes={"images": {0: "batch"}},
        opset_version=opset,
        dynamo=False,
    )
    inject_metadata(output_path, metadata)
    return output_path


def inject_metadata(output_path: str | Path, metadata: dict[str, Any]) -> None:
    """Inject metadata into ONNX model's metadata_props.

    Handles nested dicts/lists by JSON-serializing them automatically.
    """
    model = onnx.load(str(output_path))
    del model.metadata_props[:]
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
    onnx.save(model, str(output_path))


def read_metadata(output_path: str | Path) -> dict[str, str]:
    model = onnx.load(str(output_path))
    return {item.key: item.value for item in model.metadata_props}


def verify_onnx(output_path: str | Path, example_input: torch.Tensor) -> dict[str, Any]:
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    inputs = {session.get_inputs()[0].name: example_input.detach().cpu().numpy().astype(np.float32)}
    outputs = session.run(None, inputs)
    return {
        "metadata": read_metadata(output_path),
        "output_shapes": [list(np.array(output).shape) for output in outputs],
    }

