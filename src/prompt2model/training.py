from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from prompt2model.config import TaskType, TrainingConfig


@dataclass
class TrainingArtifacts:
    checkpoint_path: str
    history: list[dict[str, float]]
    best_metric: float
    device: str
    total_time_seconds: float = 0.0


def select_device(task: TaskType, requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if task == TaskType.CLASSIFICATION and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


def train_classification_model(
    model: nn.Module,
    train_loader: Any,
    val_loader: Any,
    config: TrainingConfig,
    output_dir: str | Path,
    device: torch.device,
) -> TrainingArtifacts:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history: list[dict[str, float]] = []
    best_metric = float("-inf")
    best_state = copy.deepcopy(model.state_dict())

    start_time = time.perf_counter()
    for epoch in range(config.epochs):
        # ... training loop logic stays the same ...
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        steps = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())
            predictions = logits.argmax(dim=1)
            train_correct += int((predictions == labels).sum().item())
            train_total += int(labels.numel())
            steps += 1
            if config.max_steps_per_epoch and steps >= config.max_steps_per_epoch:
                break

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_steps = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += float(loss.item())
                val_correct += int((logits.argmax(dim=1) == labels).sum().item())
                val_total += int(labels.numel())
                val_steps += 1
                if config.max_steps_per_epoch and val_steps >= config.max_steps_per_epoch:
                    break

        train_accuracy = train_correct / max(train_total, 1)
        val_accuracy = val_correct / max(val_total, 1)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": train_loss / max(steps, 1),
                "train_accuracy": train_accuracy,
                "val_loss": val_loss / max(val_steps, 1),
                "val_accuracy": val_accuracy,
            }
        )
        if val_accuracy >= best_metric:
            best_metric = val_accuracy
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    checkpoint_path = Path(output_dir) / "best_model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    elapsed = time.perf_counter() - start_time
    return TrainingArtifacts(
        checkpoint_path=str(checkpoint_path),
        history=history,
        best_metric=best_metric,
        device=str(device),
        total_time_seconds=elapsed,
    )


def train_detection_model(
    model: nn.Module,
    train_loader: Any,
    val_loader: Any,
    config: TrainingConfig,
    output_dir: str | Path,
    device: torch.device,
) -> TrainingArtifacts:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history: list[dict[str, float]] = []
    best_metric = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    start_time = time.perf_counter()
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        steps = 0
        for images, targets in train_loader:
            images = [image.to(device) for image in images]
            targets = [_to_device(target, device) for target in targets]
            optimizer.zero_grad(set_to_none=True)
            losses = model(images, targets)
            loss = sum(losses.values())
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            steps += 1
            if config.max_steps_per_epoch and steps >= config.max_steps_per_epoch:
                break

        model.train()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [image.to(device) for image in images]
                targets = [_to_device(target, device) for target in targets]
                losses = model(images, targets)
                val_loss += float(sum(losses.values()).item())
                val_steps += 1
                if config.max_steps_per_epoch and val_steps >= config.max_steps_per_epoch:
                    break

        mean_val_loss = val_loss / max(val_steps, 1)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": train_loss / max(steps, 1),
                "val_loss": mean_val_loss,
            }
        )
        if mean_val_loss <= best_metric:
            best_metric = mean_val_loss
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    checkpoint_path = Path(output_dir) / "best_detection_model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    elapsed = time.perf_counter() - start_time
    return TrainingArtifacts(
        checkpoint_path=str(checkpoint_path),
        history=history,
        best_metric=best_metric,
        device=str(device),
        total_time_seconds=elapsed,
    )


def benchmark_model(model: nn.Module, sample_input: Any, device: torch.device, repeats: int = 5) -> dict[str, float]:
    model = model.to(device)
    model.eval()
    elapsed = []
    with torch.no_grad():
        for _ in range(repeats):
            start = time.perf_counter()
            if isinstance(sample_input, list):
                _ = model([item.to(device) for item in sample_input])
            else:
                _ = model(sample_input.to(device))
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - start)
    avg_seconds = sum(elapsed) / max(len(elapsed), 1)
    parameter_count = sum(param.numel() for param in model.parameters())
    return {
        "latency_ms": avg_seconds * 1000.0,
        "fps": 1.0 / avg_seconds if avg_seconds > 0 else 0.0,
        "parameter_count": float(parameter_count),
        "parameter_count_millions": float(parameter_count) / 1_000_000.0,
    }

