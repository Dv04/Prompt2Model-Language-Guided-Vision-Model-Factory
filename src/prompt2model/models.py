from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    efficientnet_b0,
    mobilenet_v3_large,
    mobilenet_v3_small,
)
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    ssdlite320_mobilenet_v3_large,
)

from prompt2model.config import PriorityPreset, TaskType

# YOLO and RT-DETR model names routed through ultralytics
YOLO_MODELS: set[str] = {"yolov11n", "yolov11s", "yolov11m", "rtdetr-l", "rtdetr-x"}

# Mapping from our model names to ultralytics weight file names
_ULTRALYTICS_WEIGHTS: dict[str, str] = {
    "yolov11n": "yolo11n.pt",
    "yolov11s": "yolo11s.pt",
    "yolov11m": "yolo11m.pt",
    "rtdetr-l": "rtdetr-l.pt",
    "rtdetr-x": "rtdetr-x.pt",
}


def is_yolo_model(name: str) -> bool:
    """Return True if the model name is handled by the ultralytics backend."""
    return name in YOLO_MODELS


def recommend_model_name(task: TaskType, priority: PriorityPreset) -> str:
    if task == TaskType.CLASSIFICATION:
        if priority == PriorityPreset.SPEED:
            return "mobilenet_v3_small"
        if priority == PriorityPreset.ACCURACY:
            return "efficientnet_b0"
        return "mobilenet_v3_large"
    # Detection: ssdlite for speed (lightweight, torchvision-native),
    # yolov11n for balanced (ultralytics NMS-free), rtdetr-l for accuracy
    if priority == PriorityPreset.SPEED:
        return "ssdlite320_mobilenet_v3_large"
    if priority == PriorityPreset.BALANCED:
        return "yolov11n"
    return "rtdetr-l"  # ACCURACY


def build_classification_model(name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    if name == "mobilenet_v3_small":
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if name == "mobilenet_v3_large":
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if name == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    raise ValueError(f"unsupported classification model: {name}")


def build_detection_model(name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """Build a torchvision-style detection model (non-YOLO). For YOLO use build_yolo_model()."""
    weights_backbone = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    if name == "ssdlite320_mobilenet_v3_large":
        return ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=weights_backbone, num_classes=num_classes)
    if name == "fasterrcnn_mobilenet_v3_large_320_fpn":
        return fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=None,
            weights_backbone=weights_backbone,
            num_classes=num_classes,
        )
    raise ValueError(f"unsupported torchvision detection model: {name}. For YOLO/RT-DETR use build_yolo_model().")


def build_yolo_model(name: str) -> "YOLO":  # type: ignore[name-defined]
    """Build a YOLO or RT-DETR model via the ultralytics backend.

    Returns an ultralytics YOLO object. Training is done through
    ``train_yolo_model()`` in training.py which uses ultralytics' native API.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required for YOLO/RT-DETR models. "
            "Install with: pip install ultralytics"
        ) from exc

    weights = _ULTRALYTICS_WEIGHTS.get(name)
    if weights is None:
        raise ValueError(f"unsupported YOLO model: {name}. Supported: {sorted(YOLO_MODELS)}")
    return YOLO(weights)
