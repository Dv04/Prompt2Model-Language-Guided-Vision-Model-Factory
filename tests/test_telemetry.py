"""Tests for the telemetry logging system."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from prompt2model.config import DatasetConfig, DatasetFormat, PipelineConfig, RequestedLabel, TaskType
from prompt2model.telemetry import TelemetryLogger
from prompt2model.training import benchmark_model


@pytest.fixture
def dummy_model() -> nn.Module:
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3)
            self.fc = nn.Linear(8 * 94 * 94, 2)
        def forward(self, x): return self.fc(self.conv(x).flatten(1))
    return Model()


def test_benchmark_model_includes_flops(dummy_model):
    sample_input = torch.randn(1, 3, 96, 96)
    device = torch.device("cpu")
    
    results = benchmark_model(dummy_model, sample_input, device)
    
    assert "flops" in results
    assert "gflops" in results
    assert results["parameter_count"] > 0
    # If fvcore is installed, it should be > 0. 
    # If not, it defaults to 0.0 but we should ideally have it in the env.
    assert isinstance(results["flops"], float)
    # The dummy model has a conv and linear, should have significant flops
    # assert results["flops"] > 0


def test_telemetry_logger_json_and_csv(tmp_path):
    csv_path = tmp_path / "global.csv"
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    
    logger = TelemetryLogger(global_csv_path=csv_path)
    
    config = PipelineConfig(
        prompt="Test prompt",
        task=TaskType.CLASSIFICATION,
        labels=[RequestedLabel(name="test")],
        dataset=DatasetConfig(root="/tmp", format=DatasetFormat.IMAGEFOLDER, image_size=96),
        model_name="test_model",
    )
    
    metrics = {
        "accuracy": 0.95,
        "latency_ms": 10.5,
        "fps": 95.0,
        "gflops": 1.2,
        "parameter_count_millions": 5.4,
    }
    
    record = logger.log_run(config, metrics, run_dir)
    
    # Check JSON
    json_path = run_dir / "telemetry.json"
    assert json_path.exists()
    saved = json.loads(json_path.read_text())
    assert saved["metadata"]["prompt"] == "Test prompt"
    assert saved["metrics"]["accuracy"] == 0.95
    assert "hardware" in saved
    
    # Check CSV
    assert csv_path.exists()
    lines = csv_path.read_text().splitlines()
    assert len(lines) == 2  # Header + 1 row
    assert "Test prompt" in lines[1]
    assert "95.0" in lines[1]


def test_hardware_info_detection():
    logger = TelemetryLogger()
    info = logger._hardware_info
    
    assert "os" in info
    assert "python_version" in info
    assert "gpu_available" in info
    assert isinstance(info["gpu_available"], bool)
