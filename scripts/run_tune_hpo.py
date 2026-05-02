#!/usr/bin/env python3
"""run_tune_hpo.py: Ray Tune HPO driven by a parsed prompt.

This used to be hard-wired to the Beans dataset, which meant that
every prompt produced a bean-disease classifier regardless of what
the user asked for. The script now parses the prompt, picks the best
matching cached dataset via ``dataset_registry.select_dataset``, and
routes the HPO study through that dataset. When the requested labels
are not in any cached dataset the substitution is announced explicitly
so that downstream demos do not silently mismatch the prompt.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
os.environ["PYTHONPATH"] = (
    str(SRC) + (f":{os.environ['PYTHONPATH']}" if "PYTHONPATH" in os.environ else "")
)

from prompt2model.config import (
    DatasetConfig, DatasetFormat, PipelineConfig, RequestedLabel,
    ResolvedLabel, TaskType,
)
from prompt2model.parsing import parse_prompt
from prompt2model.tuning import HAS_RAY, export_best_trial, run_hpo
from dataset_registry import select_dataset

if not HAS_RAY:
    print("Ray Tune is not installed. pip install 'ray[tune]'.")
    sys.exit(1)

from ray import tune


def _build_loaders(record_root: Path, image_size: int):
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    full = datasets.ImageFolder(str(record_root), transform=transform)
    n = len(full)
    val_count = max(1, n // 5)
    train_count = n - val_count
    train_set, val_set = torch.utils.data.random_split(
        full, [train_count, val_count],
        generator=torch.Generator().manual_seed(0),
    )
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)
    return train_loader, val_loader, full.classes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        default=(
            "Classify cats, dogs, and ships in low light scenes "
            "and prioritize speed for an edge device."
        ),
    )
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--output-dir", default="output/tune_hpo")
    args = parser.parse_args()

    placeholder_root = REPO_ROOT / "data/dummy"
    placeholder_root.mkdir(parents=True, exist_ok=True)
    placeholder_dataset = DatasetConfig(
        root=str(placeholder_root),
        format=DatasetFormat.IMAGEFOLDER,
        image_size=args.image_size,
    )
    parsed = parse_prompt(args.prompt, placeholder_dataset)
    parsed_labels = [label.name for label in parsed.labels]
    print(f"[parse] task={parsed.task.value} labels={parsed_labels}")

    workdir = REPO_ROOT / "output" / "demo_datasets"
    record = select_dataset(parsed_labels, REPO_ROOT, workdir)
    if record.substituted:
        print(f"[registry] WARNING: {record.notes}")
    else:
        print(f"[registry] {record.notes}")

    train_loader, val_loader, class_names = _build_loaders(
        record.root, args.image_size,
    )
    num_classes = len(class_names)
    example_input = torch.randn(1, 3, args.image_size, args.image_size)

    config = PipelineConfig(
        prompt=args.prompt,
        task=TaskType.CLASSIFICATION,
        labels=[RequestedLabel(name=name) for name in class_names],
        dataset=DatasetConfig(
            root=str(record.root),
            format=DatasetFormat.IMAGEFOLDER,
            image_size=args.image_size,
        ),
        model_name="mobilenet_v3_small",
    )
    config.resolved_labels = [
        ResolvedLabel(
            requested_label=name, dataset_label=name,
            score=1.0, method="identity",
        )
        for name in class_names
    ]

    import ray
    ray.init(
        runtime_env={"env_vars": {"PYTHONPATH": str(SRC)}},
        ignore_reinit_error=True,
    )

    search_space = {
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "weight_decay": tune.uniform(1e-5, 1e-3),
    }

    print(f"[hpo] starting Ray Tune with {args.num_samples} samples")
    results = run_hpo(
        config=config,
        class_names=class_names,
        train_loader=train_loader,
        val_loader=val_loader,
        example_input=example_input,
        search_space=search_space,
        num_samples=args.num_samples,
        storage_path=Path.cwd() / args.output_dir,
    )

    promoted = Path(args.output_dir) / "promoted_model.onnx"
    promoted.parent.mkdir(parents=True, exist_ok=True)
    print("[hpo] promoting best trial")
    onnx_path = export_best_trial(results, output_path=promoted)
    print(f"[hpo] best ONNX written to {onnx_path}")

    import subprocess
    subprocess.run(
        [sys.executable, "scripts/edge_infer.py",
         "--model", str(onnx_path), "--no-metadata-dump"],
        check=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
