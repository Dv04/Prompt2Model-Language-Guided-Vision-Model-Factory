"""Week 7 — Integration: End-to-end pipeline test.

Validates the complete handshake between parsing, HPO, training,
enhanced metadata export, and edge verification.
"""

import pytest
from prompt2model.config import DatasetConfig, DatasetFormat, TrainingConfig
from prompt2model.data import create_synthetic_classification_dataset
from prompt2model.pipeline import Prompt2ModelFactory, PipelineResult


def test_full_pipeline_integration(tmp_path: str) -> None:
    from pathlib import Path
    tmp = Path(tmp_path)
    
    # 1. Setup synthetic data
    data_dir = create_synthetic_classification_dataset(str(tmp / "data"), samples_per_class=8, image_size=64)
    dataset_cfg = DatasetConfig(root=str(data_dir), format=DatasetFormat.IMAGEFOLDER, image_size=64)
    
    # 2. Setup Factory
    factory = Prompt2ModelFactory()
    
    # 3. Build Config from Prompt (triggering HPO with high budget)
    prompt = "Classify red squares and blue circles. Prioritize accuracy and use a 20 minute budget."
    config = factory.build_config(prompt, dataset=dataset_cfg)
    
    # Manually ensure HPO conditions are met for the test
    config.constraints.budget_minutes = 20
    config.training.epochs = 1
    config.training.max_steps_per_epoch = 2
    config.export.output_dir = str(tmp / "output")
    
    # 4. Run Pipeline
    result = factory.run(config)
    
    # 5. Verify Handshake
    assert isinstance(result, PipelineResult)
    assert result.hpo_results is not None
    assert result.hpo_results.n_trials_completed >= 1
    
    assert result.onnx_path is not None
    assert Path(result.onnx_path).exists()
    
    # Check that edge verification was recorded in metrics
    assert result.metrics.get("edge_inference_verified") is True
    
    # Telemetry verification
    assert result.telemetry is not None
    assert "fps" in result.telemetry
    assert "parameters_total" in result.telemetry
    assert result.telemetry["fps"] >= 0
    
    # Verify report existence
    assert Path(result.report_path).exists()
    report_content = Path(result.report_path).read_text()
    assert "Prompt2Model Evaluation Report" in report_content
    assert "Resolved Labels" in report_content
