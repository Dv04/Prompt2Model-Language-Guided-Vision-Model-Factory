"""Label-aware dataset registry for the Prompt2Model demo scripts.

Given a list of requested labels parsed from a natural-language prompt,
this module picks the best-matching cached real dataset and lays it out
in the ImageFolder format that the pipeline expects.

The registry is intentionally explicit: when a user asks for classes
that are not in any cached dataset (e.g. "helmets"), we do not pretend
to have a helmet model. We pick the closest available dataset, log
that substitution clearly, and expose it via the ``substituted`` flag
on the returned record so demo scripts can surface it to the user.
"""
from __future__ import annotations

import logging
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DatasetRecord:
    name: str
    classes: list[str]
    root: Path
    substituted: bool
    notes: str


_BEANS_CLASSES = ["angular_leaf_spot", "bean_rust", "healthy"]
_CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

_KEYWORDS = {
    "beans":  ["bean", "leaf", "leaves", "plant", "crop", "disease", "rust", "angular", "agric"],
    "cifar10": [
        "cat", "dog", "ship", "plane", "airplane", "bird", "horse", "deer",
        "frog", "truck", "automobile", "car", "vehicle", "animal",
    ],
}


def _score(labels: list[str], dataset_key: str) -> int:
    blob = " ".join(label.lower() for label in labels)
    return sum(1 for keyword in _KEYWORDS[dataset_key] if keyword in blob)


def _materialize_cifar10(repo_root: Path, output_dir: Path,
                         per_class: int = 60) -> Path:
    """Lay a small CIFAR-10 ImageFolder split out from the cached batches."""
    if output_dir.exists() and any(output_dir.iterdir()):
        return output_dir

    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "numpy and Pillow are required to materialize CIFAR-10"
        ) from exc

    cifar_root = repo_root / "data/real_datasets/cifar10/cifar-10-batches-py"
    if not cifar_root.exists():
        raise FileNotFoundError(f"CIFAR-10 cache not found at {cifar_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {i: 0 for i in range(10)}
    for batch_name in [f"data_batch_{i}" for i in range(1, 6)]:
        with open(cifar_root / batch_name, "rb") as handle:
            payload = pickle.load(handle, encoding="bytes")
        data = payload[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = payload[b"labels"]
        for arr, label in zip(data, labels):
            if counts[label] >= per_class:
                continue
            class_dir = output_dir / _CIFAR10_CLASSES[label]
            class_dir.mkdir(parents=True, exist_ok=True)
            target = class_dir / f"img_{counts[label]:04d}.png"
            Image.fromarray(arr).save(target)
            counts[label] += 1
        if all(value >= per_class for value in counts.values()):
            break
    return output_dir


def _materialize_beans(repo_root: Path, output_dir: Path,
                       per_class: int = 30) -> Path:
    """Lay a small Beans ImageFolder split out from the HF cache."""
    if output_dir.exists() and any(output_dir.iterdir()):
        return output_dir

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("'datasets' package is required for Beans") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    split = load_dataset("beans", split="train")
    counts = {i: 0 for i in range(len(_BEANS_CLASSES))}
    for item in split:
        label = int(item["labels"])
        if counts[label] >= per_class:
            continue
        class_dir = output_dir / _BEANS_CLASSES[label]
        class_dir.mkdir(parents=True, exist_ok=True)
        target = class_dir / f"img_{counts[label]:04d}.png"
        item["image"].convert("RGB").save(target)
        counts[label] += 1
        if all(value >= per_class for value in counts.values()):
            break
    return output_dir


def select_dataset(labels: list[str], repo_root: Path,
                   workdir: Path) -> DatasetRecord:
    """Pick the best matching cached dataset for the requested labels."""
    scores = {key: _score(labels, key) for key in _KEYWORDS}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_key, top_score = ranked[0]
    substituted = top_score == 0

    if substituted:
        top_key = "cifar10"
        notes = (
            f"No cached dataset matches the requested labels {labels}. "
            f"Falling back to CIFAR-10 to demonstrate the trained pipeline; "
            f"the parsed labels are recorded in pipeline_config.json so the "
            f"substitution is auditable."
        )
    else:
        notes = f"Best matching cached dataset for {labels}: {top_key}"

    if top_key == "beans":
        root = _materialize_beans(repo_root, workdir / "beans_imagefolder")
        return DatasetRecord(
            name="beans", classes=_BEANS_CLASSES, root=root,
            substituted=substituted, notes=notes,
        )

    root = _materialize_cifar10(repo_root, workdir / "cifar10_imagefolder")
    return DatasetRecord(
        name="cifar10", classes=_CIFAR10_CLASSES, root=root,
        substituted=substituted, notes=notes,
    )


def reset_workdir(workdir: Path) -> None:
    if workdir.exists():
        shutil.rmtree(workdir)
