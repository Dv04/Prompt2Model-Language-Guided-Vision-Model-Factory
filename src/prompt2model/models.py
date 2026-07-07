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

from prompt2model.config import ModelConstraints, PriorityPreset, TaskType

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


# Approximate parameter counts (millions) used for the max_parameters cap.
_PARAMS_MILLIONS: dict[str, float] = {
    "mobilenet_v3_small": 2.5,
    "mobilenet_v3_large": 5.5,
    "efficientnet_b0": 5.3,
    "ssdlite320_mobilenet_v3_large": 3.4,
    "yolov11n": 2.6,
    "rtdetr-l": 32.0,
}

# Latency envelopes (ms) that hard-cap the tier regardless of the stated
# priority: a "prioritize accuracy, under 25 ms" prompt gets the speed tier —
# the stated constraint wins over the stated preference.
_LATENCY_SPEED_MS = 30
_LATENCY_BALANCED_MS = 80

# Power budgets at or below this force the speed tier (camera-SoC class).
_POWER_SPEED_W = 5.0

_TIER_ORDER = (PriorityPreset.ACCURACY, PriorityPreset.BALANCED, PriorityPreset.SPEED)


def recommend_model(task: TaskType, constraints: ModelConstraints) -> str:
    """Constraint-driven model selection: priority sets the starting tier,
    hard constraints (latency / power / parameter cap) can only push it DOWN
    toward smaller models — never up."""
    priority = constraints.priority

    def _tier_index(p: PriorityPreset) -> int:
        return _TIER_ORDER.index(p)

    tier = _tier_index(priority)
    if constraints.power_budget_w is not None and constraints.power_budget_w <= _POWER_SPEED_W:
        tier = max(tier, _tier_index(PriorityPreset.SPEED))
    if constraints.target_latency_ms is not None:
        if constraints.target_latency_ms <= _LATENCY_SPEED_MS:
            tier = max(tier, _tier_index(PriorityPreset.SPEED))
        elif constraints.target_latency_ms <= _LATENCY_BALANCED_MS:
            tier = max(tier, _tier_index(PriorityPreset.BALANCED))

    cap = constraints.max_parameters_millions
    while True:
        name = recommend_model_name(task, _TIER_ORDER[tier])
        if cap is None or _PARAMS_MILLIONS.get(name, 0.0) <= cap or tier == len(_TIER_ORDER) - 1:
            return name
        tier += 1


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
