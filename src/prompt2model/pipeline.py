from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prompt2model.augmentations import TorchVisionAugmentationBackend, build_augmentation_plan
from prompt2model.config import DatasetConfig, PipelineConfig, TaskType, TrainingConfig
from prompt2model.data import (
    build_classification_bundle,
    build_detection_bundle,
    load_dataset_labels,
)
from prompt2model.edge_inference import EdgeModel, load_model_from_onnx
from prompt2model.evaluation import run_classification_inference, run_detection_inference
from prompt2model.exporting import build_metadata_props, export_model_to_onnx, verify_onnx
from prompt2model.label_resolution import LabelResolver
from prompt2model.models import (
    build_classification_model,
    build_detection_model,
    build_yolo_model,
    is_yolo_model,
    recommend_model,
)
from prompt2model.planner import LLMPlanner, PlannerMode, plan_prompt
from prompt2model.reporting import write_markdown_report
from prompt2model.training import (
    TrainingArtifacts,
    benchmark_model,
    select_device,
    train_classification_model,
    train_detection_model,
    train_yolo_model,
)
from prompt2model.telemetry import TelemetryLogger


@dataclass
class PipelineResult:
    run_dir: str
    config_path: str
    checkpoint_path: str
    report_path: str
    metrics: dict[str, Any]
    onnx_path: str | None
    onnx_verification: dict[str, Any] | None
    hpo_info: dict[str, Any] | None = None
    telemetry: dict[str, Any] | None = None
    # B1 compression stage: the gate report + the INT8 artifact when it
    # passed the accuracy floor (None = not attempted or refused).
    compression: dict[str, Any] | None = None
    compressed_onnx_path: str | None = None
    # B1 deployment stage: which runtime the final artifact targets, whether
    # it was built locally, and the build recipe when it wasn't.
    deployment: dict[str, Any] | None = None

    @property
    def hpo_results(self) -> Any:
        return self.hpo_info


class Prompt2ModelFactory:
    def __init__(self, label_resolver: LabelResolver | None = None) -> None:
        self.label_resolver = label_resolver or LabelResolver()
        self.telemetry = TelemetryLogger(
            global_csv_path=Path(__file__).resolve().parents[2] / "data" / "telemetry_history.csv"
        )

    def build_config(
        self,
        prompt: str,
        dataset: DatasetConfig,
        task_hint: TaskType | None = None,
        training_overrides: TrainingConfig | None = None,
        planner: LLMPlanner | None = None,
        planner_mode: PlannerMode = "auto",
    ) -> PipelineConfig:
        config = plan_prompt(
            prompt,
            dataset=dataset,
            task_hint=task_hint,
            training_overrides=training_overrides,
            planner=planner,
            mode=planner_mode,
        )
        dataset_labels = load_dataset_labels(dataset)
        config.resolved_labels = self.label_resolver.resolve(config.labels, dataset_labels)
        if not config.model_name:
            config.model_name = recommend_model(config.task, config.constraints)
        return config

    def run(self, config: PipelineConfig, enable_hpo: bool = False) -> PipelineResult:
        run_dir = Path(config.export.output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "pipeline_config.json"

        if config.task == TaskType.DETECTION and config.dataset.image_size < 320:
            config.dataset.image_size = 320
        if config.task == TaskType.DETECTION and config.training.batch_size < 2:
            config.training.batch_size = 2

        config_path.write_text(config.model_dump_json(indent=2))

        plan = build_augmentation_plan(config.augmentation_tags, config.task)
        augmentation_backend = TorchVisionAugmentationBackend(plan, seed=config.dataset.seed)
        device = select_device(config.task, requested=config.training.device)

        hpo_info: dict[str, Any] | None = None
        calibration_info: dict[str, Any] | None = None  # classification-only for now

        if config.task == TaskType.CLASSIFICATION:
            bundle = build_classification_bundle(
                config.dataset,
                batch_size=config.training.batch_size,
                augmentations=augmentation_backend,
                num_workers=config.training.num_workers,
            )

            if enable_hpo:
                # Run Optuna HPO - respects budget_minutes from the parsed prompt
                from prompt2model.hpo import run_hpo
                hpo_result = run_hpo(
                    model_name=config.model_name or "mobilenet_v3_small",
                    num_classes=len(bundle.class_names),
                    task=config.task,
                    bundle=bundle,
                    budget_minutes=config.constraints.budget_minutes,
                    output_dir=run_dir / "hpo",
                    device=device,
                    pretrained=config.training.pretrained,
                )
                training = hpo_result.best_artifacts
                model = build_classification_model(
                    config.model_name or "mobilenet_v3_small",
                    num_classes=len(bundle.class_names),
                    pretrained=hpo_result.best_config.pretrained,
                )
                import torch
                model.load_state_dict(torch.load(training.checkpoint_path, map_location="cpu", weights_only=True))
                hpo_info = {
                    "n_trials_completed": hpo_result.n_trials_completed,
                    "best_trial_number": hpo_result.best_trial_number,
                    "best_learning_rate": hpo_result.best_config.learning_rate,
                    "best_weight_decay": hpo_result.best_config.weight_decay,
                    "best_epochs": hpo_result.best_config.epochs,
                    "elapsed_seconds": hpo_result.elapsed_seconds,
                    "budget_minutes": config.constraints.budget_minutes,
                }
            else:
                model = build_classification_model(
                    config.model_name or "mobilenet_v3_small",
                    num_classes=len(bundle.class_names),
                    pretrained=config.training.pretrained,
                )
                training = train_classification_model(
                    model, bundle.train_loader, bundle.val_loader, config.training, run_dir, device
                )

            # Compression stage 1 - knowledge distillation (opt-in): a
            # bigger teacher from the registry's accuracy tier fine-tunes the
            # already-trained student via temperature-scaled soft targets.
            if config.compression.enable_distillation:
                from prompt2model.compression import train_distilled_classification
                from prompt2model.config import PriorityPreset
                from prompt2model.models import recommend_model_name

                teacher_name = config.compression.distillation_teacher or recommend_model_name(
                    TaskType.CLASSIFICATION, PriorityPreset.ACCURACY
                )
                teacher = build_classification_model(
                    teacher_name,
                    num_classes=len(bundle.class_names),
                    pretrained=config.training.pretrained,
                )
                train_classification_model(
                    teacher, bundle.train_loader, bundle.val_loader,
                    config.training, run_dir / "teacher", device,
                )
                student = build_classification_model(
                    config.model_name or "mobilenet_v3_small",
                    num_classes=len(bundle.class_names),
                    pretrained=config.training.pretrained,
                )
                student.load_state_dict(model.state_dict())  # KD fine-tune, not from scratch
                distillation_info = train_distilled_classification(
                    teacher, student, bundle.train_loader, bundle.val_loader,
                    config.training, config.compression, run_dir / "distilled", device,
                )
                import torch
                student.load_state_dict(
                    torch.load(distillation_info["checkpoint_path"], map_location="cpu", weights_only=True)
                )
                model = student.to(device)
                distillation_info["teacher"] = teacher_name
                metrics_distillation = distillation_info
            else:
                metrics_distillation = None

            metrics = run_classification_inference(model.to(device), bundle.test_loader, device)
            if metrics_distillation is not None:
                metrics["distillation"] = {
                    "teacher": metrics_distillation["teacher"],
                    "best_val_accuracy": metrics_distillation["best_val_accuracy"],
                    "temperature": metrics_distillation["temperature"],
                    "alpha": metrics_distillation["alpha"],
                }
            sample_batch, _ = next(iter(bundle.test_loader))
            benchmark = benchmark_model(model, sample_batch[:1], device=device)

            # Phase 4 - calibration + conformal abstain threshold, fitted on
            # the validation split. Ships in the artifact metadata; EdgeModel
            # applies it at inference (the guarantee handshake).
            from prompt2model.calibration import calibrate_classification
            calibration_info = calibrate_classification(model, bundle.val_loader, device)
            metrics["calibration"] = calibration_info

        else:
            # Detection path
            bundle = build_detection_bundle(
                config.dataset,
                batch_size=config.training.batch_size,
                augmentations=augmentation_backend,
                num_workers=config.training.num_workers,
            )
            model_name = config.model_name or "fasterrcnn_mobilenet_v3_large_320_fpn"

            if is_yolo_model(model_name):
                # YOLO / RT-DETR path - uses ultralytics native training
                training = train_yolo_model(
                    model_name=model_name,
                    dataset_config=config.dataset,
                    training_config=config.training,
                    output_dir=run_dir,
                    device=device,
                )
                # Load underlying PyTorch model for eval / benchmark
                yolo = build_yolo_model(model_name)
                import torch
                if Path(training.checkpoint_path).exists():
                    yolo_loaded = type(yolo)(training.checkpoint_path)
                    model = yolo_loaded.model
                else:
                    model = yolo.model
                metrics = run_detection_inference(model.to(device), bundle.test_loader, device)
                sample_images, _ = next(iter(bundle.test_loader))
                benchmark = benchmark_model(model, sample_images[:1], device=device)
            else:
                if enable_hpo:
                    from prompt2model.hpo import run_hpo
                    hpo_result = run_hpo(
                        model_name=model_name,
                        num_classes=len(bundle.class_names) + 1,
                        task=config.task,
                        bundle=bundle,
                        budget_minutes=config.constraints.budget_minutes,
                        output_dir=run_dir / "hpo",
                        device=device,
                        pretrained=config.training.pretrained,
                    )
                    training = hpo_result.best_artifacts
                    model = build_detection_model(
                        model_name, num_classes=len(bundle.class_names) + 1,
                        pretrained=hpo_result.best_config.pretrained,
                    )
                    import torch
                    model.load_state_dict(torch.load(training.checkpoint_path, map_location="cpu", weights_only=True))
                    hpo_info = {
                        "n_trials_completed": hpo_result.n_trials_completed,
                        "budget_minutes": config.constraints.budget_minutes,
                        "elapsed_seconds": hpo_result.elapsed_seconds,
                    }
                else:
                    model = build_detection_model(
                        model_name,
                        num_classes=len(bundle.class_names) + 1,
                        pretrained=config.training.pretrained,
                    )
                    training = train_detection_model(
                        model, bundle.train_loader, bundle.val_loader, config.training, run_dir, device
                    )

                metrics = run_detection_inference(model.to(device), bundle.test_loader, device)
                sample_images, _ = next(iter(bundle.test_loader))
                benchmark = benchmark_model(model, sample_images[:1], device=device)

        metrics = {**metrics, **benchmark}
        if hpo_info:
            metrics["hpo"] = hpo_info

        telemetry = {
            "total_time_seconds": training.total_time_seconds,
            "device": training.device,
            "latency_ms": metrics.get("latency_ms"),
            "fps": metrics.get("fps"),
            "gflops": metrics.get("gflops"),
            "parameter_count": metrics.get("parameter_count"),
            "parameter_count_millions": metrics.get("parameter_count_millions"),
        }

        onnx_path: str | None = None
        verification: dict[str, Any] | None = None
        if config.export.export_onnx:
            metadata = {
                "task": config.task.value,
                "prompt": config.prompt,
                "model_name": config.model_name,
                "labels": [resolved.dataset_label for resolved in config.resolved_labels],
                "image_size": config.dataset.image_size,
            }
            if hpo_info:
                metadata["hpo_trials"] = hpo_info.get("n_trials_completed", 0)
                metadata["hpo_budget_minutes"] = hpo_info.get("budget_minutes", 0)
            metadata = build_metadata_props(config, bundle.class_names)
            if calibration_info and calibration_info.get("calibrated"):
                metadata["calibration"] = calibration_info
            try:
                if config.task == TaskType.CLASSIFICATION:
                    sample_batch, _ = next(iter(bundle.test_loader))
                    example_input = sample_batch[:1]
                else:
                    sample_images, _ = next(iter(bundle.test_loader))
                    example_input = sample_images[0].unsqueeze(0)
                onnx_path = str(run_dir / "model.onnx")
                export_model_to_onnx(
                    model=model,
                    task=config.task,
                    example_input=example_input,
                    output_path=onnx_path,
                    metadata=metadata,
                    topk_detections=config.export.topk_detections,
                    opset=config.export.onnx_opset,
                )
                # Raw ONNX verification
                verification = verify_onnx(onnx_path, example_input)

                # Handshake: Edge Inference Validation
                edge_model = EdgeModel(onnx_path)
                metrics["edge_inference_verified"] = True
            except Exception as exc:
                metrics["onnx_export_error"] = str(exc)
                onnx_path = None
                verification = None

        # Compression stage 2 - quantize + accuracy-floor gate, evaluated in
        # ONNX Runtime (the runtime the artifact ships in). The gate REFUSES
        # to ship a compressed model below the floor; the report says why.
        compression_info: dict[str, Any] | None = None
        compressed_onnx_path: str | None = None
        if onnx_path and config.compression.enable_quantization:
            if config.task == TaskType.CLASSIFICATION:
                from prompt2model.compression import apply_compression
                from prompt2model.exporting import inject_metadata

                compression_report = apply_compression(
                    onnx_path, bundle.val_loader, config.compression, config.constraints
                )
                compression_info = compression_report.to_dict()
                metrics["compression"] = compression_info
                if compression_report.passed and compression_report.compressed_path:
                    compressed_onnx_path = compression_report.compressed_path
                    try:
                        inject_metadata(
                            compressed_onnx_path,
                            {**metadata, "compression": compression_info},
                        )
                    except Exception:  # metadata is best-effort on the copy
                        pass
            else:
                compression_info = {
                    "attempted": False,
                    "reason": "detection quantization is not accuracy-gated yet - skipped",
                }
                metrics["compression"] = compression_info

        # Phase 3 - deployment target: compile the final artifact for the
        # requested runtime. ONNX Runtime is universal; accelerator targets
        # (TensorRT, ...) build locally when the toolchain exists, else emit
        # a reproducible build recipe to run on the device.
        deployment_info: dict[str, Any] | None = None
        if onnx_path:
            from prompt2model.targets import resolve_target

            final_artifact = compressed_onnx_path or onnx_path
            target = resolve_target(config.data_context.deployment_target)
            deployment_info = target.prepare(final_artifact, run_dir, metadata).to_dict()
            metrics["deployment"] = deployment_info

        report_path = write_markdown_report(
            run_dir / "evaluation_report.md",
            config=config,
            metrics=metrics,
            training_history=training.history,
            export_path=onnx_path,
            hpo_results=hpo_info,
            telemetry=telemetry,
        )
        
        # Log telemetry
        self.telemetry.log_run(config, metrics, run_dir)
        
        return PipelineResult(
            run_dir=str(run_dir),
            config_path=str(config_path),
            checkpoint_path=training.checkpoint_path,
            report_path=report_path,
            metrics=metrics,
            onnx_path=onnx_path,
            onnx_verification=verification,
            hpo_info=hpo_info,
            telemetry=telemetry,
            compression=compression_info,
            compressed_onnx_path=compressed_onnx_path,
            deployment=deployment_info,
        )


def run_from_prompt(
    prompt: str,
    dataset: DatasetConfig,
    output_dir: str,
    task_hint: TaskType | None = None,
    training_overrides: TrainingConfig | None = None,
    enable_clip: bool = False,
    enable_hpo: bool = False,
    planner: LLMPlanner | None = None,
    planner_mode: PlannerMode = "auto",
    quantize: bool = False,
    distill: bool = False,
    target: str | None = None,
) -> PipelineResult:
    """Run the full pipeline from a natural language prompt.

    Args:
        enable_hpo: When True, runs Optuna HPO instead of fixed-config training.
            Budget is taken from ``budget_minutes`` in the parsed prompt (default 15 min).
        planner: Optional LLM planner; defaults to env config (P2M_LLM_ENDPOINT /
            P2M_LLM_MODEL). See ``planner.plan_prompt`` for mode semantics.
    """
    resolver = LabelResolver(enable_clip=enable_clip)
    factory = Prompt2ModelFactory(label_resolver=resolver)
    config = factory.build_config(
        prompt,
        dataset,
        task_hint=task_hint,
        training_overrides=training_overrides,
        planner=planner,
        planner_mode=planner_mode,
    )
    config.export.output_dir = output_dir
    if quantize:
        config.compression.enable_quantization = True
    if distill:
        config.compression.enable_distillation = True
    if target:
        config.data_context.deployment_target = target
    return factory.run(config, enable_hpo=enable_hpo)
