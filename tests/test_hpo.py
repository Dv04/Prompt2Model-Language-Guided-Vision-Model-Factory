"""Week 5 — Venkata: Smoke tests for Optuna HPO integration."""

import pytest

from prompt2model.config import DatasetConfig, DatasetFormat, TaskType, TrainingConfig
from prompt2model.data import build_classification_bundle, create_synthetic_classification_dataset
from prompt2model.models import build_classification_model
from prompt2model.training import select_device

optuna = pytest.importorskip("optuna")
from prompt2model.hpo import HPOResult, SearchSpace, run_optuna_study, best_params_to_config


def test_optuna_smoke_classification(tmp_path: str) -> None:
    from pathlib import Path
    tmp = Path(tmp_path)
    data_dir = create_synthetic_classification_dataset(str(tmp / "cls_data"), samples_per_class=6, image_size=64)
    dataset_cfg = DatasetConfig(root=str(data_dir), format=DatasetFormat.IMAGEFOLDER, image_size=64)
    bundle = build_classification_bundle(dataset_cfg, batch_size=4, augmentations=None)

    def model_factory():
        return build_classification_model("mobilenet_v3_small", num_classes=len(bundle.class_names), pretrained=False)

    device = select_device(TaskType.CLASSIFICATION, requested="cpu")
    base_config = TrainingConfig(epochs=1, batch_size=4, max_steps_per_epoch=2)
    space = SearchSpace(lr_low=1e-3, lr_high=1e-2, batch_sizes=None)

    result = run_optuna_study(
        model_factory=model_factory,
        train_loader=bundle.train_loader,
        val_loader=bundle.val_loader,
        base_config=base_config,
        task=TaskType.CLASSIFICATION,
        device=device,
        search_space=space,
        n_trials=2,
        timeout_seconds=120,
        output_dir=str(tmp / "hpo_output"),
    )

    assert isinstance(result, HPOResult)
    assert result.n_trials_completed == 2
    assert "learning_rate" in result.best_params
    assert result.total_time_seconds > 0
    assert len(result.trial_history) == 2


def test_best_params_to_config() -> None:
    base = TrainingConfig(epochs=5, batch_size=8)
    best = {"learning_rate": 0.005, "weight_decay": 0.001, "batch_size": 16}
    merged = best_params_to_config(base, best)
    assert merged.learning_rate == 0.005
    assert merged.weight_decay == 0.001
    assert merged.batch_size == 16
    assert merged.epochs == 5  # preserved from base
