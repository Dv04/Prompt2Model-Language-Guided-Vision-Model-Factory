"""Week 6 — Venkata: Tests for Ray Tune ASHA scheduler integration.

This test should gracefully pass if Ray or Tune isn't installed by falling back to Optuna.
"""

import pytest
from prompt2model.config import DatasetConfig, DatasetFormat, TaskType, TrainingConfig
from prompt2model.data import build_classification_bundle, create_synthetic_classification_dataset
from prompt2model.models import build_classification_model
from prompt2model.training import select_device
from prompt2model.hpo import run_ray_tune_study, HPOResult


def test_ray_tune_fallback_or_run(tmp_path: str) -> None:
    from pathlib import Path
    tmp = Path(tmp_path)
    data_dir = create_synthetic_classification_dataset(str(tmp / "cls_data"), samples_per_class=4, image_size=64)
    dataset_cfg = DatasetConfig(root=str(data_dir), format=DatasetFormat.IMAGEFOLDER, image_size=64)
    bundle = build_classification_bundle(dataset_cfg, batch_size=4, augmentations=None)

    def model_factory():
        return build_classification_model("mobilenet_v3_small", num_classes=len(bundle.class_names), pretrained=False)

    device = select_device(TaskType.CLASSIFICATION, requested="cpu")
    base_config = TrainingConfig(epochs=1, batch_size=4, max_steps_per_epoch=2)

    # Use a small number of trials and a timeout
    result = run_ray_tune_study(
        model_factory=model_factory,
        train_loader=bundle.train_loader,
        val_loader=bundle.val_loader,
        base_config=base_config,
        task=TaskType.CLASSIFICATION,
        device=device,
        n_trials=2,
        timeout_seconds=30,
        output_dir=str(tmp / "ray_output"),
    )

    assert isinstance(result, HPOResult)
    assert result.n_trials_completed > 0
    assert "learning_rate" in result.best_params
