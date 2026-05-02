"""Unit tests for the Ray Tune integration and terminal-step export.

These tests verify the terminal export logic and result promotion without
requiring a full Ray installation.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from prompt2model.config import (
    DatasetConfig,
    DatasetFormat,
    ExportConfig,
    ModelConstraints,
    PipelineConfig,
    RequestedLabel,
    ResolvedLabel,
    TaskType,
    TrainingConfig,
)
from prompt2model.tuning import _terminal_export, export_best_trial


@pytest.fixture
def dummy_model() -> nn.Module:
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 3)
        def forward(self, x): return x
    return Model()


@pytest.fixture
def sample_config() -> PipelineConfig:
    return PipelineConfig(
        project_name="test",
        prompt="Classify cats.",
        task=TaskType.CLASSIFICATION,
        labels=[RequestedLabel(name="cat")],
        constraints=ModelConstraints(),
        dataset=DatasetConfig(root="/tmp", format=DatasetFormat.IMAGEFOLDER, image_size=96),
        training=TrainingConfig(),
        export=ExportConfig(),
        model_name="mobilenet_v3_small",
        resolved_labels=[ResolvedLabel(requested_label="cat", dataset_label="cat", score=1.0, method="identity")],
    )


def test_terminal_export_writes_onnx(dummy_model, sample_config, tmp_path):
    class_names = ["cat"]
    example_input = torch.randn(1, 3, 96, 96)
    
    onnx_path = _terminal_export(
        dummy_model, sample_config, class_names, example_input, tmp_path
    )
    
    assert Path(onnx_path).exists()
    assert Path(onnx_path).name == "model.onnx"
    
    # Verify it's a valid ONNX
    import onnx
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    
    # Check metadata keys
    meta = {p.key: p.value for p in model.metadata_props}
    assert "class_dict" in meta
    assert "label_map" in meta
    assert "input_resolution" in meta


def test_export_best_trial_copies_file(tmp_path, monkeypatch):
    # Setup mock Ray result
    mock_result = MagicMock()
    trial_dir = tmp_path / "trial_dir"
    trial_dir.mkdir()
    onnx_file = trial_dir / "model.onnx"
    onnx_file.write_text("fake onnx")
    
    mock_result.metrics = {"onnx_path": str(onnx_file)}
    
    mock_grid = MagicMock()
    mock_grid.get_best_result.return_value = mock_result
    
    # Mock HAS_RAY to True for this test
    monkeypatch.setattr("prompt2model.tuning.HAS_RAY", True)
    
    dest_path = tmp_path / "promoted.onnx"
    promoted = export_best_trial(mock_grid, dest_path)
    
    assert Path(promoted).exists()
    assert Path(promoted).read_text() == "fake onnx"


def test_terminal_export_idempotent(dummy_model, sample_config, tmp_path):
    class_names = ["cat"]
    example_input = torch.randn(1, 3, 96, 96)
    
    path1 = _terminal_export(dummy_model, sample_config, class_names, example_input, tmp_path)
    path2 = _terminal_export(dummy_model, sample_config, class_names, example_input, tmp_path)
    
    assert path1 == path2
    assert Path(path1).exists()
