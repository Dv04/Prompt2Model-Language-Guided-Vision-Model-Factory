"""Optuna-based hyperparameter optimization for Prompt2Model.

This module implements budgeted hyperparameter search over learning rate,
weight decay, and number of epochs using Bayesian optimization (TPE sampler)
with median pruning. Training is bounded by a wall-clock time budget derived
from the ``budget_minutes`` field in :class:`~prompt2model.config.ModelConstraints`.

Usage example::

    from prompt2model.hpo import run_hpo

    best_config, best_artifacts = run_hpo(
        model_name="mobilenet_v3_small",
        num_classes=3,
        task=TaskType.CLASSIFICATION,
        bundle=bundle,
        budget_minutes=15,
        output_dir=Path("output/hpo_run"),
        device=device,
    )
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from prompt2model.config import TaskType, TrainingConfig
from prompt2model.models import build_classification_model, build_detection_model
from prompt2model.training import (
    TrainingArtifacts,
    train_classification_model,
    train_detection_model,
)

if TYPE_CHECKING:
    from prompt2model.data import ClassificationBundle, DetectionBundle

log = logging.getLogger(__name__)


@dataclass
class HPOResult:
    best_config: TrainingConfig
    best_artifacts: TrainingArtifacts
    n_trials_completed: int
    best_trial_number: int
    study_direction: str
    elapsed_seconds: float


def _optuna_available() -> bool:
    try:
        import optuna  # noqa: F401
        return True
    except ImportError:
        return False


def run_hpo(
    model_name: str,
    num_classes: int,
    task: TaskType,
    bundle: ClassificationBundle | DetectionBundle,
    budget_minutes: int,
    output_dir: Path,
    device: torch.device,
    n_trials: int = 15,
    pretrained: bool = False,
) -> HPOResult:
    """Run budgeted hyperparameter optimization using Optuna.

    Searches over ``learning_rate``, ``weight_decay``, and ``epochs`` using
    the TPE sampler with median pruning. The search terminates when either
    ``n_trials`` have completed or ``budget_minutes`` wall-clock time has
    elapsed, whichever comes first.

    Args:
        model_name: Name of the backbone to optimize (must be a classification
            or torchvision detection model, not a YOLO model).
        num_classes: Number of output classes.
        task: ``TaskType.CLASSIFICATION`` or ``TaskType.DETECTION``.
        bundle: Pre-built data bundle from ``build_classification_bundle()``
            or ``build_detection_bundle()``.
        budget_minutes: Maximum wall-clock time budget. Passed from
            ``ModelConstraints.budget_minutes``.
        output_dir: Directory where trial checkpoints are saved.
        device: Torch device to use for training.
        n_trials: Maximum number of Optuna trials (may terminate earlier due
            to time budget).
        pretrained: Whether to start from pretrained ImageNet weights.

    Returns:
        :class:`HPOResult` with the best config, best training artifacts, and
        study statistics.
    """
    if not _optuna_available():
        log.warning(
            "optuna is not installed. Falling back to default TrainingConfig. "
            "Install with: pip install optuna>=3.6"
        )
        return _fallback_hpo(model_name, num_classes, task, bundle, output_dir, device, pretrained)

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    budget_seconds = budget_minutes * 60
    direction = "maximize" if task == TaskType.CLASSIFICATION else "minimize"

    best_config: TrainingConfig | None = None
    best_artifacts: TrainingArtifacts | None = None
    best_trial_number: int = 0

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_config, best_artifacts, best_trial_number

        # Enforce wall-clock time budget
        if time.time() - start_time >= budget_seconds:
            raise optuna.exceptions.OptunaError("time budget exhausted")

        # Define search space
        lr = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        epochs = trial.suggest_int("epochs", 2, 8)

        config = TrainingConfig(
            learning_rate=lr,
            weight_decay=wd,
            epochs=epochs,
            batch_size=16,
            pretrained=pretrained,
            max_steps_per_epoch=None,
        )

        trial_dir = output_dir / f"trial_{trial.number}"

        try:
            if task == TaskType.CLASSIFICATION:
                model = build_classification_model(model_name, num_classes, pretrained)
                artifacts = train_classification_model(
                    model, bundle.train_loader, bundle.val_loader,
                    config, trial_dir, device, trial=trial,
                )
                metric = artifacts.best_metric  # val accuracy (maximize)
            else:
                model = build_detection_model(model_name, num_classes, pretrained)
                artifacts = train_detection_model(
                    model, bundle.train_loader, bundle.val_loader,
                    config, trial_dir, device, trial=trial,
                )
                metric = artifacts.best_metric  # val loss (minimize)

        except optuna.exceptions.TrialPruned:
            raise
        except Exception as exc:
            log.warning("Trial %d failed: %s", trial.number, exc)
            raise optuna.exceptions.TrialPruned() from exc

        # Track best result
        is_better = (
            (task == TaskType.CLASSIFICATION and (best_artifacts is None or metric > best_artifacts.best_metric))
            or (task == TaskType.DETECTION and (best_artifacts is None or metric < best_artifacts.best_metric))
        )
        if is_better:
            best_config = config
            best_artifacts = artifacts
            best_trial_number = trial.number

        return metric

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1)
    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)

    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=budget_seconds,
            catch=(Exception,),
            show_progress_bar=False,
        )
    except Exception as exc:
        log.warning("HPO study ended early: %s", exc)

    elapsed = time.time() - start_time
    n_completed = len([t for t in study.trials if t.state.name == "COMPLETE"])
    log.info(
        "HPO finished: %d trials completed in %.1fs (budget=%dm)",
        n_completed, elapsed, budget_minutes,
    )

    # If no trial succeeded, fall back to defaults
    if best_config is None or best_artifacts is None:
        log.warning("No HPO trial succeeded. Using default TrainingConfig.")
        return _fallback_hpo(model_name, num_classes, task, bundle, output_dir, device, pretrained)

    return HPOResult(
        best_config=best_config,
        best_artifacts=best_artifacts,
        n_trials_completed=n_completed,
        best_trial_number=best_trial_number,
        study_direction=direction,
        elapsed_seconds=elapsed,
    )


def _fallback_hpo(
    model_name: str,
    num_classes: int,
    task: TaskType,
    bundle: Any,
    output_dir: Path,
    device: torch.device,
    pretrained: bool,
) -> HPOResult:
    """Run a single training pass with default config when Optuna is unavailable."""
    start = time.time()
    config = TrainingConfig(pretrained=pretrained)
    fallback_dir = output_dir / "fallback"

    if task == TaskType.CLASSIFICATION:
        model = build_classification_model(model_name, num_classes, pretrained)
        artifacts = train_classification_model(
            model, bundle.train_loader, bundle.val_loader, config, fallback_dir, device
        )
    else:
        model = build_detection_model(model_name, num_classes, pretrained)
        artifacts = train_detection_model(
            model, bundle.train_loader, bundle.val_loader, config, fallback_dir, device
        )

    return HPOResult(
        best_config=config,
        best_artifacts=artifacts,
        n_trials_completed=1,
        best_trial_number=0,
        study_direction="maximize" if task == TaskType.CLASSIFICATION else "minimize",
        elapsed_seconds=time.time() - start,
    )
