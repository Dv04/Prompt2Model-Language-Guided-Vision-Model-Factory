"""Smoke test for the Optuna HPO module."""
from __future__ import annotations

from pathlib import Path

import pytest

from prompt2model.config import DatasetConfig, DatasetFormat, TrainingConfig, TaskType
from prompt2model.data import create_synthetic_classification_dataset, build_classification_bundle
from prompt2model.training import select_device
from prompt2model.hpo import run_hpo, _optuna_available


def test_hpo_runs_and_returns_valid_config(tmp_path: Path) -> None:
    """HPO should complete at least one trial and return a valid config + artifacts."""
    if not _optuna_available():
        pytest.skip("optuna not installed")

    dataset_root = create_synthetic_classification_dataset(
        tmp_path / "data", samples_per_class=6, image_size=64
    )
    config = DatasetConfig(
        root=str(dataset_root),
        format=DatasetFormat.IMAGEFOLDER,
        image_size=64,
        val_split=0.2,
        test_split=0.1,
    )
    bundle = build_classification_bundle(config, batch_size=4, augmentations=None)
    device = select_device(TaskType.CLASSIFICATION, requested="cpu")

    result = run_hpo(
        model_name="mobilenet_v3_small",
        num_classes=len(bundle.class_names),
        task=TaskType.CLASSIFICATION,
        bundle=bundle,
        budget_minutes=1,  # very short budget for smoke test
        output_dir=tmp_path / "hpo",
        device=device,
        n_trials=3,
        pretrained=False,
    )

    assert result.best_config is not None
    assert result.best_artifacts is not None
    assert result.best_artifacts.best_metric >= 0.0
    assert Path(result.best_artifacts.checkpoint_path).exists()
    assert result.n_trials_completed >= 1
    assert result.elapsed_seconds > 0


def test_hpo_pipeline_integration(tmp_path: Path) -> None:
    """run_from_prompt with enable_hpo=True should complete end-to-end."""
    if not _optuna_available():
        pytest.skip("optuna not installed")

    from prompt2model.data import create_synthetic_classification_dataset
    from prompt2model.pipeline import run_from_prompt

    dataset_root = create_synthetic_classification_dataset(
        tmp_path / "cls_data", samples_per_class=8, image_size=64
    )
    result = run_from_prompt(
        prompt="Classify red square, blue circle, green triangle images under low light and prioritize speed.",
        dataset=DatasetConfig(
            root=str(dataset_root),
            format=DatasetFormat.IMAGEFOLDER,
            image_size=64,
        ),
        output_dir=str(tmp_path / "hpo_run"),
        training_overrides=TrainingConfig(epochs=2, batch_size=4, max_steps_per_epoch=2, device="cpu"),
        enable_clip=False,
        enable_hpo=True,
    )
    assert Path(result.report_path).exists()
    assert result.hpo_info is not None
    assert result.hpo_info["n_trials_completed"] >= 1
