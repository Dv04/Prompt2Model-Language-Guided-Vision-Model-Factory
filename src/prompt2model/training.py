from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from prompt2model.config import DatasetConfig, TaskType, TrainingConfig


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


def learning_rate_at_step(
    config: TrainingConfig,
    *,
    step: int,
    total_steps: int,
    steps_per_epoch: int,
) -> float:
    """Warmup plus cosine learning rate used by native PyTorch trainers.

    ``step`` is zero-based and the final scheduled step reaches the configured
    minimum ratio. Constant mode remains an explicit reproducibility option.
    """
    if total_steps < 1 or steps_per_epoch < 1:
        raise ValueError("total_steps and steps_per_epoch must be positive")
    if step < 0:
        raise ValueError("step must be non-negative")
    base = float(config.learning_rate)
    if config.lr_schedule == "constant":
        return base
    bounded_step = min(step, total_steps - 1)
    warmup_steps = min(config.warmup_epochs * steps_per_epoch, total_steps)
    if warmup_steps and bounded_step < warmup_steps:
        return base * ((bounded_step + 1) / warmup_steps)
    decay_steps = total_steps - warmup_steps
    if decay_steps <= 1:
        progress = 1.0
    else:
        progress = (bounded_step - warmup_steps) / (decay_steps - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    ratio = config.min_learning_rate_ratio + (
        1.0 - config.min_learning_rate_ratio
    ) * cosine
    return base * ratio


def _training_steps(loader: Any, config: TrainingConfig) -> int:
    try:
        available = len(loader)
    except (TypeError, AttributeError):
        if config.max_steps_per_epoch is None:
            raise ValueError(
                "max_steps_per_epoch is required when the training loader has no length"
            )
        available = config.max_steps_per_epoch
    if config.max_steps_per_epoch is not None:
        available = min(available, config.max_steps_per_epoch)
    if available < 1:
        raise ValueError("training loader must contain at least one batch")
    return int(available)


def _set_learning_rate(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value


def train_classification_model(
    model: nn.Module,
    train_loader: Any,
    val_loader: Any,
    config: TrainingConfig,
    output_dir: str | Path,
    device: torch.device,
    trial: Any = None,
) -> TrainingArtifacts:
    """Train a classification model.

    Args:
        trial: Optional ``optuna.Trial`` for pruning during HPO. When provided,
            validation accuracy is reported each epoch and unpromising trials
            are pruned early.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history: list[dict[str, float]] = []
    best_metric = float("-inf")
    best_state = copy.deepcopy(model.state_dict())
    steps_per_epoch = _training_steps(train_loader, config)
    total_steps = max(1, config.epochs * steps_per_epoch)
    global_step = 0

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
            current_lr = learning_rate_at_step(
                config,
                step=global_step,
                total_steps=total_steps,
                steps_per_epoch=steps_per_epoch,
            )
            _set_learning_rate(optimizer, current_lr)
            optimizer.step()
            global_step += 1

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
                "learning_rate": current_lr,
            }
        )
        if val_accuracy >= best_metric:
            best_metric = val_accuracy
            best_state = copy.deepcopy(model.state_dict())

        # Optuna integration: report intermediate value and check for pruning
        if trial is not None:
            try:
                import optuna
                trial.report(val_accuracy, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            except ImportError:
                pass

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
    trial: Any = None,
) -> TrainingArtifacts:
    """Train a torchvision-style detection model.

    Args:
        trial: Optional ``optuna.Trial`` for pruning during HPO.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history: list[dict[str, float]] = []
    best_metric = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    steps_per_epoch = _training_steps(train_loader, config)
    total_steps = max(1, config.epochs * steps_per_epoch)
    global_step = 0

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
            current_lr = learning_rate_at_step(
                config,
                step=global_step,
                total_steps=total_steps,
                steps_per_epoch=steps_per_epoch,
            )
            _set_learning_rate(optimizer, current_lr)
            optimizer.step()
            global_step += 1
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
                "learning_rate": current_lr,
            }
        )
        if mean_val_loss <= best_metric:
            best_metric = mean_val_loss
            best_state = copy.deepcopy(model.state_dict())

        # Optuna integration: report intermediate value and check for pruning
        if trial is not None:
            try:
                import optuna
                trial.report(-mean_val_loss, epoch)  # negate so higher = better
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            except ImportError:
                pass

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


def train_yolo_model(
    model_name: str,
    dataset_config: DatasetConfig,
    training_config: TrainingConfig,
    output_dir: str | Path,
    device: torch.device,
) -> TrainingArtifacts:
    """Train a YOLO or RT-DETR model using the ultralytics native training API.

    Converts our COCO-format DatasetConfig into the YAML format ultralytics
    expects, runs training, and returns TrainingArtifacts compatible with the
    rest of the pipeline.
    """
    import shutil
    import tempfile

    from prompt2model.models import _ULTRALYTICS_WEIGHTS, build_yolo_model

    try:
        import yaml
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "pyyaml"], check=True)
        import yaml

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load COCO annotations to extract class info
    annotation_path = Path(dataset_config.annotation_path or "")
    coco_data = json.loads(annotation_path.read_text())
    categories = sorted(coco_data["categories"], key=lambda c: c["id"])
    class_names = [c["name"] for c in categories]
    num_classes = len(class_names)

    # Split image IDs for train / val
    import math
    import random as _random
    image_ids = [item["id"] for item in coco_data["images"]]
    rng = _random.Random(dataset_config.seed)
    rng.shuffle(image_ids)
    total = len(image_ids)
    val_count = max(1, math.floor(total * dataset_config.val_split))
    val_ids = set(image_ids[:val_count])
    train_ids = set(image_ids[val_count:])

    # Build split-specific annotation dicts
    train_anns = [a for a in coco_data["annotations"] if a["image_id"] in train_ids]
    val_anns = [a for a in coco_data["annotations"] if a["image_id"] in val_ids]
    train_imgs = [i for i in coco_data["images"] if i["id"] in train_ids]
    val_imgs = [i for i in coco_data["images"] if i["id"] in val_ids]

    # Write temp COCO JSON files
    tmp_dir = output_dir / "_yolo_data_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    train_json = tmp_dir / "train.json"
    val_json = tmp_dir / "val.json"
    train_json.write_text(json.dumps({
        "images": train_imgs, "annotations": train_anns, "categories": coco_data["categories"]
    }))
    val_json.write_text(json.dumps({
        "images": val_imgs, "annotations": val_anns, "categories": coco_data["categories"]
    }))

    # Write YOLO data YAML (COCO JSON format)
    data_yaml_path = tmp_dir / "data.yaml"
    data_yaml = {
        "path": str(Path(dataset_config.root).resolve()),
        "train": str(train_json.resolve()),
        "val": str(val_json.resolve()),
        "nc": num_classes,
        "names": class_names,
    }
    data_yaml_path.write_text(yaml.dump(data_yaml))

    # Build and train the YOLO model
    yolo = build_yolo_model(model_name)
    device_str = "mps" if device.type == "mps" else str(device)
    yolo.train(
        data=str(data_yaml_path),
        epochs=training_config.epochs,
        batch=training_config.batch_size,
        lr0=training_config.learning_rate,
        cos_lr=training_config.lr_schedule == "cosine",
        warmup_epochs=training_config.warmup_epochs,
        weight_decay=training_config.weight_decay,
        imgsz=dataset_config.image_size,
        project=str(output_dir),
        name="yolo_run",
        device=device_str,
        verbose=False,
        exist_ok=True,
    )

    # Locate best checkpoint written by ultralytics
    best_pt = output_dir / "yolo_run" / "weights" / "best.pt"
    checkpoint_path = output_dir / "best_detection_model.pt"
    if best_pt.exists():
        shutil.copy(best_pt, checkpoint_path)
    else:
        # Fallback to last.pt
        last_pt = output_dir / "yolo_run" / "weights" / "last.pt"
        if last_pt.exists():
            shutil.copy(last_pt, checkpoint_path)

    # Load training results CSV if available
    results_csv = output_dir / "yolo_run" / "results.csv"
    history: list[dict[str, float]] = []
    best_metric = 0.0
    if results_csv.exists():
        import csv
        with results_csv.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    epoch_val = float(row.get("epoch", 0))
                    map50 = float(row.get("metrics/mAP50(B)", 0) or 0)
                    history.append({"epoch": epoch_val + 1, "val_map50": map50})
                    if map50 > best_metric:
                        best_metric = map50
                except (ValueError, TypeError):
                    continue

    # Clean up temp files
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return TrainingArtifacts(
        checkpoint_path=str(checkpoint_path),
        history=history,
        best_metric=best_metric,
        device=str(device),
    )


def benchmark_model(model: nn.Module, sample_input: Any, device: torch.device, repeats: int = 5) -> dict[str, float]:
    model = model.to(device)
    model.eval()
    
    # 1. Theoretical FLOPs calculation (using fvcore)
    flops = 0.0
    try:
        from fvcore.nn import FlopCountAnalysis
        # FlopCountAnalysis expects a specific input format
        # If sample_input is a list (detection), we pass it as is
        # If it's a tensor (classification), we wrap it if needed but fvcore usually handles it
        analysis = FlopCountAnalysis(model, sample_input)
        analysis.unsupported_ops_warnings(False)
        flops = float(analysis.total())
    except Exception:
        # Fallback if fvcore is missing or fails
        flops = 0.0

    # 2. Latency and FPS
    elapsed = []
    with torch.no_grad():
        for _ in range(repeats):
            start = time.perf_counter()
            if isinstance(sample_input, list):
                # For detection, sample_input is a list of images
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
        "flops": flops,
        "gflops": flops / 1_000_000_000.0,
        "parameter_count": float(parameter_count),
        "parameter_count_millions": float(parameter_count) / 1_000_000.0,
    }
