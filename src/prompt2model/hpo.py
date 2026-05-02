"""Week 5 — Venkata: Optuna hyperparameter optimization integration.

Week 6 will add optional Ray Tune with ASHA on top of this module.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from prompt2model.config import PriorityPreset, TaskType, TrainingConfig
from prompt2model.training import (
    TrainingArtifacts,
    select_device,
    train_classification_model,
    train_detection_model,
)

logger = logging.getLogger(__name__)

_OPTUNA_AVAILABLE = False
try:
    import optuna

    _OPTUNA_AVAILABLE = True
except ImportError:
    pass

_RAY_TUNE_AVAILABLE = False
try:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler

    _RAY_TUNE_AVAILABLE = True
except ImportError:
    pass


@dataclass
class HPOResult:
    """Outcome of a hyperparameter search."""

    best_params: dict[str, Any]
    best_metric: float
    n_trials_completed: int
    total_time_seconds: float
    trial_history: list[dict[str, Any]]


@dataclass
class SearchSpace:
    """Defines the Optuna search space for a given backbone."""

    lr_low: float = 1e-5
    lr_high: float = 1e-2
    weight_decay_low: float = 1e-6
    weight_decay_high: float = 1e-2
    batch_sizes: list[int] | None = None

    @classmethod
    def for_backbone(cls, model_name: str, priority: PriorityPreset) -> SearchSpace:
        if priority == PriorityPreset.SPEED:
            return cls(lr_low=5e-4, lr_high=5e-2, batch_sizes=[8, 16, 32])
        if priority == PriorityPreset.ACCURACY:
            return cls(lr_low=1e-5, lr_high=5e-3, batch_sizes=[4, 8, 16])
        return cls(lr_low=1e-4, lr_high=1e-2, batch_sizes=[8, 16])


def _build_objective(
    model_factory: Any,
    train_loader: Any,
    val_loader: Any,
    base_config: TrainingConfig,
    task: TaskType,
    device: torch.device,
    search_space: SearchSpace,
    output_dir: Path,
) -> Any:
    """Build an Optuna objective function."""

    def objective(trial: Any) -> float:
        lr = trial.suggest_float("learning_rate", search_space.lr_low, search_space.lr_high, log=True)
        wd = trial.suggest_float("weight_decay", search_space.weight_decay_low, search_space.weight_decay_high, log=True)
        if search_space.batch_sizes:
            bs = trial.suggest_categorical("batch_size", search_space.batch_sizes)
        else:
            bs = base_config.batch_size

        trial_config = TrainingConfig(
            batch_size=bs,
            epochs=base_config.epochs,
            learning_rate=lr,
            weight_decay=wd,
            num_workers=base_config.num_workers,
            pretrained=base_config.pretrained,
            device=base_config.device,
            max_steps_per_epoch=base_config.max_steps_per_epoch,
        )

        model = copy.deepcopy(model_factory())
        trial_dir = output_dir / f"trial_{trial.number}"

        if task == TaskType.CLASSIFICATION:
            result = train_classification_model(model, train_loader, val_loader, trial_config, trial_dir, device)
            return result.best_metric  # higher is better (accuracy)
        else:
            result = train_detection_model(model, train_loader, val_loader, trial_config, trial_dir, device)
            return -result.best_metric  # lower val_loss is better, negate for maximize

    return objective


def run_optuna_study(
    model_factory: Any,
    train_loader: Any,
    val_loader: Any,
    base_config: TrainingConfig,
    task: TaskType,
    device: torch.device,
    search_space: SearchSpace | None = None,
    n_trials: int = 5,
    timeout_seconds: int | None = None,
    output_dir: str | Path = "output/hpo",
) -> HPOResult:
    """Run Optuna hyperparameter search.

    Args:
        model_factory: Callable that returns a fresh model instance.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        base_config: Base training configuration (epochs, workers, etc.).
        task: Classification or detection.
        device: Torch device.
        search_space: Search space definition (auto-generated if None).
        n_trials: Maximum number of trials.
        timeout_seconds: Hard time budget in seconds.
        output_dir: Directory for trial artifacts.

    Returns:
        HPOResult with the best hyperparameters and trial history.
    """
    if not _OPTUNA_AVAILABLE:
        raise ImportError(
            "Optuna is required for HPO. Install with: pip install optuna"
        )

    if search_space is None:
        search_space = SearchSpace()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    direction = "maximize" if task == TaskType.CLASSIFICATION else "maximize"
    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))

    objective = _build_objective(
        model_factory=model_factory,
        train_loader=train_loader,
        val_loader=val_loader,
        base_config=base_config,
        task=task,
        device=device,
        search_space=search_space,
        output_dir=out,
    )

    start = time.perf_counter()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds)
    elapsed = time.perf_counter() - start

    trial_history = []
    for trial in study.trials:
        trial_history.append({
            "number": trial.number,
            "params": trial.params,
            "value": trial.value,
            "state": str(trial.state),
        })

    best = study.best_trial
    logger.info("HPO complete: %d trials in %.1fs, best value=%.4f", len(study.trials), elapsed, best.value)

    return HPOResult(
        best_params=best.params,
        best_metric=best.value,
        n_trials_completed=len(study.trials),
        total_time_seconds=elapsed,
        trial_history=trial_history,
    )


def best_params_to_config(base: TrainingConfig, best_params: dict[str, Any]) -> TrainingConfig:
    """Merge HPO best parameters back into a TrainingConfig."""
    return TrainingConfig(
        batch_size=best_params.get("batch_size", base.batch_size),
        epochs=base.epochs,
        learning_rate=best_params.get("learning_rate", base.learning_rate),
        weight_decay=best_params.get("weight_decay", base.weight_decay),
        num_workers=base.num_workers,
        pretrained=base.pretrained,
        device=base.device,
        max_steps_per_epoch=base.max_steps_per_epoch,
    )


def run_ray_tune_study(
    model_factory: Any,
    train_loader: Any,
    val_loader: Any,
    base_config: TrainingConfig,
    task: TaskType,
    device: torch.device,
    search_space: SearchSpace | None = None,
    n_trials: int = 5,
    timeout_seconds: int | None = None,
    output_dir: str | Path = "output/ray_hpo",
) -> HPOResult:
    """Run Ray Tune with ASHA scheduler and Optuna search backend.

    This is an OPTIONAL integration. Falls back to pure Optuna if Ray is unavailable.

    Returns:
        HPOResult with the best hyperparameters and trial history.
    """
    if not _RAY_TUNE_AVAILABLE:
        logger.info("Ray Tune not available, falling back to pure Optuna")
        return run_optuna_study(
            model_factory=model_factory,
            train_loader=train_loader,
            val_loader=val_loader,
            base_config=base_config,
            task=task,
            device=device,
            search_space=search_space,
            n_trials=n_trials,
            timeout_seconds=timeout_seconds,
            output_dir=output_dir,
        )

    if search_space is None:
        search_space = SearchSpace()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from ray.tune.search.optuna import OptunaSearch

    optuna_search = OptunaSearch(
        metric="metric",
        mode="max",
    )

    scheduler = ASHAScheduler(
        max_t=base_config.epochs,
        grace_period=1,
        reduction_factor=2,
    )

    def trainable(config: dict[str, Any]) -> None:
        trial_config = TrainingConfig(
            batch_size=config.get("batch_size", base_config.batch_size),
            epochs=base_config.epochs,
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            num_workers=base_config.num_workers,
            pretrained=base_config.pretrained,
            device=base_config.device,
            max_steps_per_epoch=base_config.max_steps_per_epoch,
        )
        model = copy.deepcopy(model_factory())
        if task == TaskType.CLASSIFICATION:
            result = train_classification_model(
                model, train_loader, val_loader, trial_config, out / "ray_trial", device
            )
            tune.report(metric=result.best_metric)
        else:
            result = train_detection_model(
                model, train_loader, val_loader, trial_config, out / "ray_trial", device
            )
            tune.report(metric=-result.best_metric)

    param_space = {
        "learning_rate": tune.loguniform(search_space.lr_low, search_space.lr_high),
        "weight_decay": tune.loguniform(search_space.weight_decay_low, search_space.weight_decay_high),
    }
    if search_space.batch_sizes:
        param_space["batch_size"] = tune.choice(search_space.batch_sizes)

    start = time.perf_counter()

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    analysis = tune.run(
        trainable,
        config=param_space,
        num_samples=n_trials,
        search_alg=optuna_search,
        scheduler=scheduler,
        time_budget_s=timeout_seconds,
        verbose=0,
        local_dir=str(out / "ray_results"),
    )
    elapsed = time.perf_counter() - start

    best_config = analysis.best_config
    best_result = analysis.best_result

    trial_history = []
    for trial in analysis.trials:
        trial_history.append({
            "number": trial.trial_id,
            "params": trial.config,
            "value": trial.last_result.get("metric"),
            "state": str(trial.status),
        })

    return HPOResult(
        best_params=best_config,
        best_metric=best_result.get("metric", 0.0),
        n_trials_completed=len(analysis.trials),
        total_time_seconds=elapsed,
        trial_history=trial_history,
    )
