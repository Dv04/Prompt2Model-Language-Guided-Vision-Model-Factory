#!/usr/bin/env python3
"""run_tune_hpo.py — Demo script for Ray Tune HPO with terminal-step ONNX export."""

import sys
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

import os

# Add src to path for local development and for Ray workers
src_path = str(Path(__file__).resolve().parents[1] / "src")
sys.path.insert(0, src_path)
os.environ["PYTHONPATH"] = src_path + (f":{os.environ['PYTHONPATH']}" if "PYTHONPATH" in os.environ else "")

from prompt2model.config import DatasetConfig, DatasetFormat, PipelineConfig, RequestedLabel, TaskType
from prompt2model.tuning import HAS_RAY, export_best_trial, run_hpo

if not HAS_RAY:
    print("✗ Error: Ray Tune is not installed. Install with 'pip install ray[tune]'")
    sys.exit(1)

from ray import tune


def main():
    # Provide the src path to Ray workers explicitly
    src_path = str(Path(__file__).resolve().parents[1] / "src")
    import ray
    ray.init(runtime_env={"env_vars": {"PYTHONPATH": src_path}}, ignore_reinit_error=True)

    # 1. Setup Beans Dataset
    dataset = load_dataset("beans")
    class_names = list(dataset["train"].features["labels"].names)
    num_classes = len(class_names)

    # Simplified loader construction for demo
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_beans_real_benchmark import BeansDataset
    # For the sake of this demo script being standalone, we'll use a generic approach if possible
    # but since BeansDataset is specifically defined in Venkata's scripts, we'll mock a simple one
    
    from torch.utils.data import Dataset as TorchDataset
    from torchvision import transforms
    
    class SimpleDataset(TorchDataset):
        def __init__(self, hf_split):
            self.hf_split = hf_split
            self.transform = transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        def __len__(self): return len(self.hf_split)
        def __getitem__(self, idx):
            item = self.hf_split[idx]
            image = item["image"].convert("RGB")
            return self.transform(image), torch.tensor(int(item["labels"]))

    train_set = SimpleDataset(dataset["train"].select(range(100)))  # Small subset for demo
    val_set = SimpleDataset(dataset["validation"].select(range(20)))
    
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)
    
    example_input = torch.randn(1, 3, 160, 160)

    # 2. Define Pipeline Config
    config = PipelineConfig(
        prompt="Classify beans diseases.",
        task=TaskType.CLASSIFICATION,
        labels=[RequestedLabel(name=name) for name in class_names],
        dataset=DatasetConfig(root="data/beans", format=DatasetFormat.IMAGEFOLDER, image_size=160),
        model_name="mobilenet_v3_small",
    )
    # Ensure resolved_labels is populated for metadata injection
    from prompt2model.config import ResolvedLabel
    config.resolved_labels = [
        ResolvedLabel(requested_label=n, dataset_label=n, score=1.0, method="identity")
        for n in class_names
    ]

    # 3. Define HPO Search Space
    search_space = {
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "weight_decay": tune.uniform(1e-5, 1e-3),
    }

    # 4. Run HPO
    print("🚀 Starting Ray Tune HPO...")
    results = run_hpo(
        config=config,
        class_names=class_names,
        train_loader=train_loader,
        val_loader=val_loader,
        example_input=example_input,
        search_space=search_space,
        num_samples=2,  # Small number for demo
        storage_path=Path.cwd() / "output" / "tune_hpo"
    )

    # 5. Export Best Trial
    print("🏆 Finding best trial and promoting ONNX artifact...")
    best_onnx = export_best_trial(
        results, 
        output_path=Path.cwd() / "output" / "tune_hpo" / "promoted_model.onnx"
    )
    print(f"✓ Best model exported to: {best_onnx}")

    # 6. Verify with edge_infer.py
    print("🔬 Verifying with edge_infer.py...")
    import subprocess
    cmd = [sys.executable, "scripts/edge_infer.py", "--model", best_onnx, "--no-metadata-dump"]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
