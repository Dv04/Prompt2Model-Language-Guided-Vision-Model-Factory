from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF

from prompt2model.config import TaskType


@dataclass(frozen=True)
class AugmentationOp:
    name: str
    probability: float = 0.5


@dataclass(frozen=True)
class AugmentationPlan:
    task: TaskType
    operations: list[AugmentationOp]

    def summary(self) -> list[str]:
        return [op.name for op in self.operations]


def build_augmentation_plan(tags: list[str], task: TaskType, default_probability: float = 0.5) -> AugmentationPlan:
    ops: list[AugmentationOp] = []
    tags = list(dict.fromkeys(tags))

    if "low_light" in tags:
        ops.extend(
            [
                AugmentationOp("brightness_contrast", default_probability),
                AugmentationOp("gaussian_blur", 0.3),
            ]
        )
    if "rain" in tags or "fog" in tags:
        ops.extend(
            [
                AugmentationOp("gaussian_blur", 0.4),
                AugmentationOp("color_jitter", default_probability),
            ]
        )
    if "motion_blur" in tags:
        ops.append(AugmentationOp("gaussian_blur", 0.6))
    if "glare" in tags:
        ops.append(AugmentationOp("color_jitter", 0.4))
    if "occlusion" in tags and task == TaskType.DETECTION:
        ops.append(AugmentationOp("horizontal_flip", 0.5))

    if not ops:
        ops.append(AugmentationOp("horizontal_flip", 0.2 if task == TaskType.DETECTION else 0.1))

    return AugmentationPlan(task=task, operations=ops)


class TorchVisionAugmentationBackend:
    """A lightweight backend that keeps detection boxes consistent."""

    def __init__(self, plan: AugmentationPlan, seed: int = 42) -> None:
        self.plan = plan
        self.rng = random.Random(seed)
        self.color_jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.02)

    def __call__(self, image: Image.Image, target: dict[str, Any] | None = None) -> tuple[Image.Image, dict[str, Any] | None]:
        for operation in self.plan.operations:
            if self.rng.random() > operation.probability:
                continue
            try:
                image, target = self._apply_operation(image, target, operation.name)
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning("Augmentation '%s' failed: %s", operation.name, exc)
        return image, target

    def _apply_operation(
        self,
        image: Image.Image,
        target: dict[str, Any] | None,
        name: str,
    ) -> tuple[Image.Image, dict[str, Any] | None]:
        if name == "brightness_contrast":
            brightness = self.rng.uniform(0.7, 1.3)
            contrast = self.rng.uniform(0.7, 1.3)
            return TF.adjust_brightness(image, brightness), TF.adjust_contrast(image, contrast), target
        if name == "gaussian_blur":
            kernel_size = 3 if self.rng.random() < 0.5 else 5
            return TF.gaussian_blur(image, kernel_size=[kernel_size, kernel_size]), target
        if name == "color_jitter":
            return self.color_jitter(image), target
        if name == "horizontal_flip":
            flipped = TF.hflip(image)
            if target is None or "boxes" not in target:
                return flipped, target
            width = image.width
            boxes = target["boxes"].clone()
            x_min = boxes[:, 0].clone()
            x_max = boxes[:, 2].clone()
            boxes[:, 0] = width - x_max
            boxes[:, 2] = width - x_min
            updated = dict(target)
            updated["boxes"] = boxes
            return flipped, updated
        return image, target


def target_to_cpu(target: dict[str, Any]) -> dict[str, Any]:
    converted = {}
    for key, value in target.items():
        converted[key] = value.cpu() if isinstance(value, torch.Tensor) else value
    return converted

