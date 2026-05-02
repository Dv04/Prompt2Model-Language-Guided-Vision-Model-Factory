from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from prompt2model.config import PipelineConfig, TaskType
from prompt2model.exporting import build_metadata_props, export_model_to_onnx
from prompt2model.training import train_classification_model, train_detection_model

# Lazy import for ray
try:
    import ray
    from ray import tune
    from ray.tune import Trainable
    HAS_RAY = True
except ImportError:
    HAS_RAY = False
    # Define dummy classes for type hinting if ray is missing
    class Trainable: pass


def _terminal_export(
    model: nn.Module,
    config: PipelineConfig,
    class_names: list[str],
    example_input: torch.Tensor,
    trial_dir: Path,
) -> str:
    """Internal helper to export the model to ONNX at the end of a trial.

    Args:
        model: Trained model (best checkpoint loaded).
        config: Pipeline configuration.
        class_names: List of class names.
        example_input: Sample input for ONNX trace.
        trial_dir: Directory where the trial is running.

    Returns:
        Path to the exported ONNX file.
    """
    onnx_path = trial_dir / "model.onnx"
    metadata = build_metadata_props(config, class_names)
    
    export_model_to_onnx(
        model=model,
        task=config.task,
        example_input=example_input,
        output_path=onnx_path,
        metadata=metadata,
        topk_detections=config.export.topk_detections,
        opset=config.export.onnx_opset,
    )
    return str(onnx_path)


def _trainable_function(
    tune_config: dict[str, Any],
    pipeline_config: PipelineConfig,
    class_names: list[str],
    train_loader: Any,
    val_loader: Any,
    example_input: torch.Tensor,
):
    """Function API for Ray Tune training."""
    # Update pipeline config with search space parameters
    # This assumes search space keys match PipelineConfig fields (lr, weight_decay, etc)
    # or specific overrides in training config
    for key, value in tune_config.items():
        if hasattr(pipeline_config.training, key):
            setattr(pipeline_config.training, key, value)
        elif hasattr(pipeline_config.dataset, key):
            setattr(pipeline_config.dataset, key, value)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Import model builders here to avoid circular imports if any
    from prompt2model.models import build_classification_model, build_detection_model
    
    if pipeline_config.task == TaskType.CLASSIFICATION:
        model = build_classification_model(
            pipeline_config.model_name or "mobilenet_v3_small",
            num_classes=len(class_names),
            pretrained=pipeline_config.training.pretrained,
        )
        artifacts = train_classification_model(
            model, train_loader, val_loader, pipeline_config.training, Path.cwd(), device
        )
        best_metric_name = "val_accuracy"
    else:
        model = build_detection_model(
            pipeline_config.model_name or "ssdlite320_mobilenet_v3_large",
            num_classes=len(class_names) + 1,
            pretrained=pipeline_config.training.pretrained,
        )
        artifacts = train_detection_model(
            model, train_loader, val_loader, pipeline_config.training, Path.cwd(), device
        )
        best_metric_name = "val_loss"

    # Load best checkpoint before export
    model.load_state_dict(torch.load(artifacts.checkpoint_path, map_location=device))
    
    # Terminal Step: ONNX Metadata Export
    onnx_path = _terminal_export(
        model, pipeline_config, class_names, example_input, Path.cwd()
    )
    
    # Report final metrics
    final_metrics = artifacts.history[-1]
    final_metrics.update({
        best_metric_name: artifacts.best_metric,
        "onnx_path": onnx_path,
        "done": True
    })
    
    if HAS_RAY:
        tune.report(final_metrics)


def run_hpo(
    config: PipelineConfig,
    class_names: list[str],
    train_loader: Any,
    val_loader: Any,
    example_input: torch.Tensor,
    search_space: dict[str, Any],
    num_samples: int = 8,
    metric: str | None = None,
    mode: str | None = None,
    resources_per_trial: dict[str, Any] | None = None,
    storage_path: str | Path | None = None,
) -> Any:
    """Launch a Ray Tune HPO run.
    
    Returns:
        ray.tune.ResultGrid (if Ray is available)
    """
    if not HAS_RAY:
        raise ImportError("Ray Tune is not installed. Install with: pip install 'ray[tune]'")

    if metric is None:
        metric = "val_accuracy" if config.task == TaskType.CLASSIFICATION else "val_loss"
    if mode is None:
        mode = "max" if config.task == TaskType.CLASSIFICATION else "min"

    trainable_with_params = tune.with_parameters(
        _trainable_function,
        pipeline_config=config,
        class_names=class_names,
        train_loader=train_loader,
        val_loader=val_loader,
        example_input=example_input,
    )

    tuner = tune.Tuner(
        trainable_with_params,
        tune_config=tune.TuneConfig(
            metric=metric,
            mode=mode,
            num_samples=num_samples,
        ),
        param_space=search_space,
        run_config=tune.RunConfig(
            storage_path=str(storage_path) if storage_path else None,
        ),
    )
    
    return tuner.fit()


def export_best_trial(
    result_grid: Any,
    output_path: str | Path,
) -> str:
    """Find the best trial and copy its ONNX artifact to a canonical location.

    Args:
        result_grid: The result of tuner.fit().
        output_path: Destination path for the best model.onnx.

    Returns:
        Path to the promoted ONNX file.
    """
    if not HAS_RAY:
        raise ImportError("Ray Tune is not installed.")

    best_result = result_grid.get_best_result()
    best_onnx_path = Path(best_result.metrics["onnx_path"])
    
    if not best_onnx_path.exists():
        raise FileNotFoundError(f"Best trial ONNX not found at {best_onnx_path}")

    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_onnx_path, dest)
    
    return str(dest)
