#!/usr/bin/env python3
"""run_final_demo.py: end-to-end Prompt2Model demo with REAL outputs.

Earlier versions of this script printed pre-baked numbers and bounding
boxes regardless of the prompt. This rewrite drives the actual
pipeline: parse the prompt, select a cached real dataset that matches
the parsed labels, train a small classifier or detector, export ONNX,
and report the metrics that the harness actually produced.

If the requested labels are not present in any cached dataset, the
script clearly says so and substitutes the closest available dataset
rather than fabricating a result. Every printed number is read from
the pipeline result object, not hard-coded.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from prompt2model.config import DatasetConfig, DatasetFormat, TaskType, TrainingConfig
from prompt2model.parsing import parse_prompt
from prompt2model.pipeline import run_from_prompt
from dataset_registry import select_dataset

BLUE, GREEN, YELLOW, RED, BOLD, RESET = (
    "\033[94m", "\033[92m", "\033[93m", "\033[91m", "\033[1m", "\033[0m"
)


def header(text: str) -> None:
    print("\n" + "=" * 64)
    print(f"  {BOLD}{BLUE}{text}{RESET}")
    print("=" * 64)


def step(text: str) -> None:
    print(f"\n{BOLD}{YELLOW}> {text}{RESET}")


def ok(text: str) -> None:
    print(f"{BOLD}{GREEN}  [OK]{RESET} {text}")


def warn(text: str) -> None:
    print(f"{BOLD}{RED}  [warn]{RESET} {text}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        default=(
            "Classify cats, dogs, and ships in low light scenes "
            "and prioritize speed for an edge device."
        ),
    )
    parser.add_argument("--output-dir", default="output/final_demo")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps-per-epoch", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=96)
    args = parser.parse_args()

    header("PROMPT2MODEL: LANGUAGE-GUIDED VISION FACTORY (real run)")
    print(f"{BOLD}Prompt:{RESET} {args.prompt}")

    step("Parsing prompt into the typed intermediate representation")
    placeholder_root = REPO_ROOT / "data/dummy"
    placeholder_root.mkdir(parents=True, exist_ok=True)
    placeholder_dataset = DatasetConfig(
        root=str(placeholder_root),
        format=DatasetFormat.IMAGEFOLDER,
        image_size=args.image_size,
    )
    parsed = parse_prompt(args.prompt, placeholder_dataset)
    label_names = [label.name for label in parsed.labels]
    ok(f"task = {parsed.task.value}")
    ok(f"requested labels = {label_names}")
    ok(f"priority = {parsed.constraints.priority.value}")
    if parsed.data_context.environment_tags:
        tags = sorted(parsed.data_context.environment_tags)
        ok(f"environment tags = {tags}")
    else:
        ok("environment tags = (none)")

    step("Selecting a cached real dataset that matches the parsed labels")
    workdir = REPO_ROOT / "output" / "demo_datasets"
    record = select_dataset(label_names, REPO_ROOT, workdir)
    if record.substituted:
        warn(record.notes)
    else:
        ok(record.notes)
    ok(f"dataset root = {record.root}")
    ok(f"dataset classes = {record.classes}")

    step("Running the real Prompt2Model pipeline (parse > train > eval > ONNX)")
    if parsed.task == TaskType.DETECTION:
        warn(
            "The parsed task is detection but the cached datasets are "
            "classification only. Routing the demo through the "
            "classification path so the report numbers are real; the "
            "detection path is exercised by the smoke pipeline."
        )
    dataset = DatasetConfig(
        root=str(record.root),
        format=DatasetFormat.IMAGEFOLDER,
        image_size=args.image_size,
    )
    training = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_steps_per_epoch=args.max_steps_per_epoch,
        device=None,
    )
    started = time.perf_counter()
    result = run_from_prompt(
        prompt=args.prompt,
        dataset=dataset,
        output_dir=args.output_dir,
        task_hint=TaskType.CLASSIFICATION,
        training_overrides=training,
    )
    elapsed = time.perf_counter() - started

    step("Real pipeline outputs")
    ok(f"run_dir = {result.run_dir}")
    ok(f"report  = {result.report_path}")
    ok(f"onnx    = {result.onnx_path}")
    ok(f"wall-clock = {elapsed:.1f} s on this host")
    metrics = result.metrics or {}
    interesting = {
        key: metrics[key]
        for key in (
            "accuracy", "macro_f1", "latency_ms", "fps",
            "parameter_count", "flops",
        )
        if key in metrics
    }
    print(f"\n{BOLD}Pipeline metrics (verbatim from the harness):{RESET}")
    print(json.dumps(interesting, indent=2, default=str))

    onnx_path = Path(result.onnx_path) if result.onnx_path else None
    if onnx_path and onnx_path.exists():
        ok(f"ONNX file size = {onnx_path.stat().st_size / (1024 * 1024):.2f} MB")
    else:
        warn("ONNX export missing; check the run directory for failures.")

    header("EDGE INFERENCE WALK-THROUGH")
    print(
        f"Run the exported model on a sample image with no extra config:\n"
        f"  {BOLD}python scripts/edge_infer.py --model {onnx_path}{RESET}"
    )
    if onnx_path is not None:
        print(
            f"\nRun on a video with the same metadata-self-contained file:\n"
            f"  {BOLD}python scripts/video_infer.py --model {onnx_path} "
            f"--video <input.mp4>{RESET}"
        )

    header("PROJECT FINALIZED")
    print(
        "Every number above came from the actual run; the script does "
        "not print pre-baked metrics. Re-run with --prompt to compile "
        "a different model end-to-end."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
