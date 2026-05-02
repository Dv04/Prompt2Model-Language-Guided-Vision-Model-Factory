from __future__ import annotations

import argparse
import json
from pathlib import Path

from prompt2model.config import DatasetConfig, DatasetFormat, TaskType, TrainingConfig
from prompt2model.data import create_synthetic_classification_dataset, create_synthetic_detection_dataset
from prompt2model.pipeline import run_from_prompt


def _add_shared_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--task", choices=[item.value for item in TaskType], default=None)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dataset-format", choices=[item.value for item in DatasetFormat], required=True)
    parser.add_argument("--annotation-path")
    parser.add_argument("--output-dir", default="output/manual_run")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps-per-epoch", type=int, default=10)
    parser.add_argument("--device")
    parser.add_argument("--enable-hpo", action="store_true", help="Enable Optuna hyperparameter optimization")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prompt2model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-toy-data")
    generate.add_argument("--task", choices=["classification", "detection", "all"], default="all")
    generate.add_argument("--output-dir", default="output/toy_data")

    run = subparsers.add_parser("run")
    _add_shared_run_args(run)

    smoke = subparsers.add_parser("smoke-test")
    smoke.add_argument("--output-dir", default="output/smoke")

    return parser


def _run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    dataset = DatasetConfig(
        root=args.dataset_root,
        format=DatasetFormat(args.dataset_format),
        annotation_path=args.annotation_path,
        image_size=args.image_size,
    )
    training = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_steps_per_epoch=args.max_steps_per_epoch,
        device=args.device,
    )
    task_hint = TaskType(args.task) if args.task else None
    
    enable_hpo = getattr(args, "enable_hpo", False)
    
    result = run_from_prompt(
        prompt=args.prompt,
        dataset=dataset,
        output_dir=args.output_dir,
        task_hint=task_hint,
        training_overrides=training,
        enable_hpo=enable_hpo,
    )
    return {
        "run_dir": result.run_dir,
        "report_path": result.report_path,
        "onnx_path": result.onnx_path,
        "metrics": result.metrics,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "generate-toy-data":
        output_dir = Path(args.output_dir)
        payload = {}
        if args.task in {"classification", "all"}:
            payload["classification_root"] = str(create_synthetic_classification_dataset(output_dir / "classification"))
        if args.task in {"detection", "all"}:
            dataset_root, annotation_path = create_synthetic_detection_dataset(output_dir / "detection")
            payload["detection_root"] = str(dataset_root)
            payload["detection_annotations"] = str(annotation_path)
        print(json.dumps(payload, indent=2))
        return

    if args.command == "run":
        print(json.dumps(_run_pipeline(args), indent=2))
        return

    if args.command == "smoke-test":
        base = Path(args.output_dir)
        classification_root = create_synthetic_classification_dataset(base / "classification_data")
        detection_root, annotation_path = create_synthetic_detection_dataset(base / "detection_data")

        classification_args = argparse.Namespace(
            prompt="Classify red square, blue circle, and green triangle images under low light and prioritize speed.",
            task="classification",
            dataset_root=str(classification_root),
            dataset_format="imagefolder",
            annotation_path=None,
            output_dir=str(base / "classification_run"),
            image_size=96,
            epochs=2,
            batch_size=8,
            max_steps_per_epoch=4,
            device=None,
        )
        detection_args = argparse.Namespace(
            prompt="Detect squares and circles in low light images and prioritize speed.",
            task="detection",
            dataset_root=str(detection_root),
            dataset_format="coco",
            annotation_path=str(annotation_path),
            output_dir=str(base / "detection_run"),
            image_size=128,
            epochs=1,
            batch_size=2,
            max_steps_per_epoch=2,
            device="cpu",
        )
        print(
            json.dumps(
                {
                    "classification": _run_pipeline(classification_args),
                    "detection": _run_pipeline(detection_args),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
