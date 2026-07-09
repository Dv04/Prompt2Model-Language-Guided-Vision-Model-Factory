"""Week 8 - Venkata: Stress tests for budget enforcement and reliability."""

import pytest
import time
from pathlib import Path
from prompt2model.config import DatasetConfig, DatasetFormat, TaskType, TrainingConfig
from prompt2model.data import build_classification_bundle, create_synthetic_classification_dataset
from prompt2model.models import build_classification_model
from prompt2model.training import select_device, train_classification_model

def test_budget_enforcement_smoke(tmp_path: str) -> None:
    from pathlib import Path
    tmp = Path(tmp_path)
    data_dir = create_synthetic_classification_dataset(str(tmp / "data"), samples_per_class=4, image_size=64)
    dataset_cfg = DatasetConfig(root=str(data_dir), format=DatasetFormat.IMAGEFOLDER, image_size=64)
    bundle = build_classification_bundle(dataset_cfg, batch_size=4, augmentations=None)
    
    model = build_classification_model("mobilenet_v3_small", num_classes=len(bundle.class_names), pretrained=False)
    device = select_device(TaskType.CLASSIFICATION, requested="cpu")
    
    # Very small budget (not really enforced in the training loop yet, but we check reliability)
    config = TrainingConfig(epochs=1, batch_size=4, max_steps_per_epoch=2)
    
    start_time = time.time()
    result = train_classification_model(model, bundle.train_loader, bundle.val_loader, config, tmp, device)
    end_time = time.time()
    
    # Ensure it completed without crashing under minimal data
    assert result.checkpoint_path is not None
    assert len(result.history) > 0
