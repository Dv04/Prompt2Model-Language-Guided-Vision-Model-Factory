from pathlib import Path

from prompt2model.config import DatasetConfig, DatasetFormat, TrainingConfig
from prompt2model.data import create_synthetic_classification_dataset
from prompt2model.pipeline import run_from_prompt


def test_classification_pipeline_runs_end_to_end(tmp_path: Path) -> None:
    dataset_root = create_synthetic_classification_dataset(tmp_path / "classification_data", samples_per_class=8, image_size=80)
    result = run_from_prompt(
        prompt="Classify red square, blue circle, and green triangle images under low light and prioritize speed.",
        dataset=DatasetConfig(root=str(dataset_root), format=DatasetFormat.IMAGEFOLDER, image_size=80),
        output_dir=str(tmp_path / "run"),
        training_overrides=TrainingConfig(epochs=2, batch_size=4, max_steps_per_epoch=2, device="cpu"),
        enable_clip=False,
    )
    assert Path(result.report_path).exists()
    assert Path(result.checkpoint_path).exists()
    assert result.metrics["accuracy"] >= 0.0
    assert result.onnx_path is not None
    assert result.onnx_verification is not None

    # Legacy key still present
    assert "labels" in result.onnx_verification["metadata"]

    # New rich metadata keys introduced by build_metadata_props
    new_keys = {"class_dict", "mean", "std", "input_resolution", "label_map"}
    missing = new_keys - result.onnx_verification["metadata"].keys()
    assert not missing, f"ONNX metadata missing keys: {missing}"

