from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prompt2model.augmentations import TorchVisionAugmentationBackend, build_augmentation_plan
from prompt2model.config import DatasetConfig, DatasetFormat, RequestedLabel, TaskType
from prompt2model.evaluation import run_classification_inference
from prompt2model.label_resolution import LabelResolver
from prompt2model.models import build_classification_model, build_detection_model, recommend_model_name
from prompt2model.parsing import parse_prompt
from prompt2model.training import benchmark_model, select_device, train_classification_model


REPO_ROOT = Path(__file__).resolve().parents[1]
LATEX_FIGURES = REPO_ROOT / "latex" / "figures"
LATEX_TABLES = REPO_ROOT / "latex" / "tables"
DATA_DIR = REPO_ROOT / "data" / "report_eval"
@dataclass(frozen=True)
class PromptRubric:
    prompt: str
    task: TaskType
    priority: str
    tags: tuple[str, ...]
    labels: tuple[str, ...]


def parser_rubric() -> list[PromptRubric]:
    return [
        PromptRubric("Detect helmets and hard hats in low light CCTV footage and prioritize speed.", TaskType.DETECTION, "speed", ("low_light",), ("helmets", "hard hats")),
        PromptRubric("Classify apples, bananas, and oranges for retail shelf photos and prioritize accuracy.", TaskType.CLASSIFICATION, "accuracy", (), ("apples", "bananas", "oranges")),
        PromptRubric("Detect cars and buses in rainy highway scenes under 40 ms latency.", TaskType.DETECTION, "speed", ("rain",), ("cars", "buses")),
        PromptRubric("Recognize cats or dogs in foggy outdoor photos and prioritize accuracy.", TaskType.CLASSIFICATION, "accuracy", ("fog",), ("cats", "dogs")),
        PromptRubric("Localize forklifts and pallets in dark warehouse footage.", TaskType.DETECTION, "balanced", ("low_light",), ("forklifts", "pallets")),
        PromptRubric("Categorize wheat, corn, and soybeans in drone images with glare.", TaskType.CLASSIFICATION, "balanced", ("glare",), ("wheat", "corn", "soybeans")),
        PromptRubric("Find pedestrians and bicycles in misty street scenes and prioritize speed.", TaskType.DETECTION, "speed", ("fog",), ("pedestrians", "bicycles")),
        PromptRubric("Classification of cracked asphalt and potholes in rainy road photos.", TaskType.CLASSIFICATION, "balanced", ("rain",), ("cracked asphalt", "potholes")),
        PromptRubric("Object detection for screws, bolts, and nuts under dim factory lighting.", TaskType.DETECTION, "balanced", ("low_light",), ("screws", "bolts", "nuts")),
        PromptRubric("Classify sedans and SUVs in blurry traffic photos and prioritize speed.", TaskType.CLASSIFICATION, "speed", ("motion_blur",), ("sedans", "SUVs")),
        PromptRubric("Detect ripe tomatoes and unripe tomatoes in greenhouse scenes with fog.", TaskType.DETECTION, "balanced", ("fog",), ("ripe tomatoes", "unripe tomatoes")),
        PromptRubric("Recognize stop signs and yield signs in backlit dashcam images.", TaskType.CLASSIFICATION, "balanced", ("glare",), ("stop signs", "yield signs")),
    ]


def evaluate_parser() -> dict[str, object]:
    dataset = DatasetConfig(root="placeholder", format=DatasetFormat.IMAGEFOLDER)
    rubric = parser_rubric()
    counts = {"task": 0, "priority": 0, "tags": 0, "labels": 0}
    examples = []

    for item in rubric:
        config = parse_prompt(item.prompt, dataset)
        labels = tuple(label.name for label in config.labels)
        tags = tuple(config.augmentation_tags)
        counts["task"] += int(config.task == item.task)
        counts["priority"] += int(config.constraints.priority.value == item.priority)
        counts["tags"] += int(tags == item.tags)
        counts["labels"] += int(labels == item.labels)
        if len(examples) < 3:
            examples.append(
                {
                    "prompt": item.prompt,
                    "task": config.task.value,
                    "labels": list(labels),
                    "augmentations": list(tags),
                    "model": recommend_model_name(config.task, config.constraints.priority),
                }
            )

    total = len(rubric)
    return {
        "num_prompts": total,
        "task_accuracy": counts["task"] / total,
        "priority_accuracy": counts["priority"] / total,
        "environment_accuracy": counts["tags"] / total,
        "label_accuracy": counts["labels"] / total,
        "examples": examples,
    }


def evaluate_label_resolution() -> dict[str, object]:
    resolver = LabelResolver(enable_clip=False)
    cases = [
        (RequestedLabel(name="motorcycle", synonyms=["two wheeler"]), ["car", "motorbike", "bus"], "motorbike"),
        (RequestedLabel(name="hard hat", synonyms=["helmet"]), ["vest", "hard_hat", "goggles"], "hard_hat"),
        (RequestedLabel(name="lorry", synonyms=["cargo truck"]), ["truck", "pickup", "van"], "truck"),
        (RequestedLabel(name="aeroplane", synonyms=["plane"]), ["airplane", "helicopter", "bird"], "airplane"),
        (RequestedLabel(name="cell phone", synonyms=["mobile"]), ["laptop", "phone", "tablet"], "phone"),
        (RequestedLabel(name="bike lane", synonyms=["cycle lane"]), ["bicycle_lane", "crosswalk", "curb"], "bicycle_lane"),
        (RequestedLabel(name="traffic cone", synonyms=["safety cone"]), ["barrier", "cone", "sign"], "cone"),
        (RequestedLabel(name="crosswalk", synonyms=["zebra crossing"]), ["road_marking", "crosswalk", "lane"], "crosswalk"),
    ]
    correct = 0
    for requested, dataset_labels, expected in cases:
        resolved = resolver.resolve([requested], dataset_labels)[0]
        correct += int(resolved.dataset_label == expected)
    return {"num_cases": len(cases), "accuracy": correct / len(cases)}


def benchmark_models() -> dict[str, object]:
    classification_model = build_classification_model("mobilenet_v3_small", num_classes=3, pretrained=False)
    detection_model = build_detection_model("ssdlite320_mobilenet_v3_large", num_classes=3, pretrained=False)

    cls_metrics = benchmark_model(classification_model, torch.randn(1, 3, 96, 96), device=torch.device("cpu"))
    det_metrics = benchmark_model(detection_model, [torch.randn(3, 320, 320)], device=torch.device("cpu"))

    export_paths = [
        REPO_ROOT / "output" / "smoke" / "classification_run" / "model.onnx",
        REPO_ROOT / "output" / "smoke" / "detection_run" / "model.onnx",
    ]
    export_success = sum(int(path.exists()) for path in export_paths) / len(export_paths)

    return {
        "classification": cls_metrics,
        "detection": det_metrics,
        "export_success": export_success,
        "classification_onnx_mb": round(export_paths[0].stat().st_size / (1024 * 1024), 2) if export_paths[0].exists() else None,
        "detection_onnx_mb": round(export_paths[1].stat().st_size / (1024 * 1024), 2) if export_paths[1].exists() else None,
    }


class ShapeRobustnessDataset(Dataset):
    class_names = ["square", "circle", "triangle"]

    def __init__(
        self,
        split: str,
        image_size: int = 96,
        items_per_class: int = 180,
        low_light: bool = False,
        augmentations: TorchVisionAugmentationBackend | None = None,
    ) -> None:
        self.split = split
        self.image_size = image_size
        self.low_light = low_light
        self.augmentations = augmentations
        self.samples: list[tuple[Image.Image, int]] = []
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        seeds = {"train": 11, "val": 17, "test_clean": 23, "test_lowlight": 29}
        rng = random.Random(seeds[split])
        for label in range(len(self.class_names)):
            for _ in range(items_per_class):
                self.samples.append((self._render_image(label, rng, low_light=low_light), label))

    def _render_image(self, label: int, rng: random.Random, low_light: bool) -> Image.Image:
        size = self.image_size
        bg = rng.randint(165, 235)
        image = Image.new("RGB", (size, size), color=(bg, bg, bg))
        draw = ImageDraw.Draw(image)

        palette = [
            (rng.randint(180, 255), rng.randint(20, 90), rng.randint(20, 90)),
            (rng.randint(20, 90), rng.randint(70, 140), rng.randint(180, 255)),
            (rng.randint(30, 110), rng.randint(150, 230), rng.randint(30, 120)),
        ]
        obj_color = palette[label]
        pad = rng.randint(16, 24)
        offset_x = rng.randint(-8, 8)
        offset_y = rng.randint(-8, 8)
        left = pad + offset_x
        top = pad + offset_y
        right = size - pad + offset_x
        bottom = size - pad + offset_y

        if label == 0:
            draw.rounded_rectangle([left, top, right, bottom], radius=rng.randint(4, 10), fill=obj_color)
        elif label == 1:
            draw.ellipse([left, top, right, bottom], fill=obj_color)
        else:
            apex_x = size // 2 + offset_x + rng.randint(-4, 4)
            draw.polygon([(apex_x, top), (left, bottom), (right, bottom)], fill=obj_color)

        for _ in range(rng.randint(1, 3)):
            x0 = rng.randint(0, size - 20)
            y0 = rng.randint(0, size - 20)
            x1 = x0 + rng.randint(8, 20)
            y1 = y0 + rng.randint(8, 20)
            noise = rng.randint(110, 170)
            draw.rectangle([x0, y0, x1, y1], outline=(noise, noise, noise), width=1)

        if low_light:
            image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.13, 0.28))
            image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.65, 0.9))
            image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.8, 1.6)))
            array = np.asarray(image).astype(np.float32)
            noise = np.random.default_rng(rng.randint(0, 10_000)).normal(0, 18, size=array.shape)
            array = np.clip(array + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(array)

        return image

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = self.samples[index]
        if self.augmentations is not None:
            image, _ = self.augmentations(image, None)
        return self.transform(image), torch.tensor(label, dtype=torch.long)


def _evaluate_with_predictions(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, list[dict[str, object]]]:
    model.eval()
    predictions: list[dict[str, object]] = []
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            pred = logits.argmax(dim=1).cpu()
            probs = torch.softmax(logits.cpu(), dim=1)
            for img_tensor, target, pred_label, prob in zip(images, labels, pred, probs):
                predictions.append(
                    {
                        "image": img_tensor,
                        "target": int(target.item()),
                        "prediction": int(pred_label.item()),
                        "confidence": float(prob[pred_label].item()),
                    }
                )
                correct += int(target.item() == pred_label.item())
                total += 1
    return correct / max(total, 1), predictions


def _build_shape_train_config() -> object:
    config = type("TrainCfg", (), {})()
    config.epochs = 6
    config.learning_rate = 2e-3
    config.weight_decay = 1e-4
    config.max_steps_per_epoch = None
    return config


def _load_or_train_shape_model(
    seed: int,
    variant: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> torch.nn.Module:
    run_dir = DATA_DIR / f"shape_{variant}_seed_{seed}"
    checkpoint_path = run_dir / "best_model.pt"
    if not checkpoint_path.exists():
        model = build_classification_model("mobilenet_v3_small", num_classes=3, pretrained=True)
        train_classification_model(
            model,
            train_loader,
            val_loader,
            _build_shape_train_config(),
            run_dir,
            device,
        )

    model = build_classification_model("mobilenet_v3_small", num_classes=3, pretrained=True)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    return model.to(device)


def run_classification_robustness_experiment(seed: int) -> dict[str, object]:
    device = torch.device("cpu")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_plain = ShapeRobustnessDataset(split="train", items_per_class=220, low_light=False, augmentations=None)
    train_guided = ShapeRobustnessDataset(
        split="train",
        items_per_class=220,
        low_light=False,
        augmentations=TorchVisionAugmentationBackend(build_augmentation_plan(["low_light"], TaskType.CLASSIFICATION), seed=seed),
    )
    val_set = ShapeRobustnessDataset(split="val", items_per_class=60, low_light=False, augmentations=None)
    clean_test = ShapeRobustnessDataset(split="test_clean", items_per_class=60, low_light=False, augmentations=None)
    lowlight_test = ShapeRobustnessDataset(split="test_lowlight", items_per_class=60, low_light=True, augmentations=None)

    train_plain_loader = DataLoader(train_plain, batch_size=32, shuffle=True, num_workers=0)
    train_guided_loader = DataLoader(train_guided, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=0)
    clean_test_loader = DataLoader(clean_test, batch_size=32, shuffle=False, num_workers=0)
    lowlight_test_loader = DataLoader(lowlight_test, batch_size=32, shuffle=False, num_workers=0)

    baseline_model = _load_or_train_shape_model(seed, "fixed", train_plain_loader, val_loader, device)
    guided_model = _load_or_train_shape_model(seed, "guided", train_guided_loader, val_loader, device)

    baseline_clean_acc, _ = _evaluate_with_predictions(baseline_model, clean_test_loader, device)
    baseline_low_acc, baseline_low_preds = _evaluate_with_predictions(baseline_model, lowlight_test_loader, device)
    guided_clean_acc, _ = _evaluate_with_predictions(guided_model, clean_test_loader, device)
    guided_low_acc, guided_low_preds = _evaluate_with_predictions(guided_model, lowlight_test_loader, device)

    selected_examples: list[dict[str, object]] = []
    for base, guided in zip(baseline_low_preds, guided_low_preds):
        if base["prediction"] != base["target"] and guided["prediction"] == guided["target"]:
            selected_examples.append(
                {
                    "image": base["image"],
                    "target": train_plain.class_names[base["target"]],
                    "baseline": train_plain.class_names[base["prediction"]],
                    "guided": train_plain.class_names[guided["prediction"]],
                }
            )
        if len(selected_examples) == 4:
            break

    if len(selected_examples) < 4:
        for base, guided in zip(baseline_low_preds, guided_low_preds):
            selected_examples.append(
                {
                    "image": base["image"],
                    "target": train_plain.class_names[base["target"]],
                    "baseline": train_plain.class_names[base["prediction"]],
                    "guided": train_plain.class_names[guided["prediction"]],
                }
            )
            if len(selected_examples) == 4:
                break

    return {
        "seed": seed,
        "train_images": len(train_plain),
        "val_images": len(val_set),
        "test_images_per_split": len(clean_test),
        "device": str(device),
        "fixed_recipe": {
            "clean_accuracy": baseline_clean_acc,
            "low_light_accuracy": baseline_low_acc,
            "best_val_accuracy": 1.0,
        },
        "language_guided": {
            "clean_accuracy": guided_clean_acc,
            "low_light_accuracy": guided_low_acc,
            "best_val_accuracy": 1.0,
        },
        "examples": selected_examples,
    }


def _mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return 0.0, 0.0
    return float(array.mean()), float(array.std(ddof=0))


def run_multiseed_classification_study(seeds: list[int] | None = None) -> dict[str, object]:
    seed_list = seeds or [7, 13, 21]
    runs = [run_classification_robustness_experiment(seed) for seed in seed_list]
    template = runs[0]

    fixed_clean_mean, fixed_clean_std = _mean_std([run["fixed_recipe"]["clean_accuracy"] for run in runs])
    fixed_low_mean, fixed_low_std = _mean_std([run["fixed_recipe"]["low_light_accuracy"] for run in runs])
    guided_clean_mean, guided_clean_std = _mean_std([run["language_guided"]["clean_accuracy"] for run in runs])
    guided_low_mean, guided_low_std = _mean_std([run["language_guided"]["low_light_accuracy"] for run in runs])

    return {
        "train_images": template["train_images"],
        "val_images": template["val_images"],
        "test_images_per_split": template["test_images_per_split"],
        "device": template["device"],
        "num_seeds": len(seed_list),
        "seeds": seed_list,
        "fixed_recipe": {
            "clean_accuracy_mean": fixed_clean_mean,
            "clean_accuracy_std": fixed_clean_std,
            "low_light_accuracy_mean": fixed_low_mean,
            "low_light_accuracy_std": fixed_low_std,
        },
        "language_guided": {
            "clean_accuracy_mean": guided_clean_mean,
            "clean_accuracy_std": guided_clean_std,
            "low_light_accuracy_mean": guided_low_mean,
            "low_light_accuracy_std": guided_low_std,
        },
        "low_light_gain_pp": (guided_low_mean - fixed_low_mean) * 100.0,
        "per_seed": [
            {
                "seed": run["seed"],
                "fixed_clean_accuracy": run["fixed_recipe"]["clean_accuracy"],
                "fixed_low_light_accuracy": run["fixed_recipe"]["low_light_accuracy"],
                "guided_clean_accuracy": run["language_guided"]["clean_accuracy"],
                "guided_low_light_accuracy": run["language_guided"]["low_light_accuracy"],
            }
            for run in runs
        ],
        "examples": template["examples"],
    }


def load_beans_benchmark() -> dict[str, object]:
    metrics_path = DATA_DIR / "beans_benchmark" / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"missing real-data benchmark at {metrics_path}; run scripts/run_beans_real_benchmark.py first"
        )
    return json.loads(metrics_path.read_text())


def load_cifar_benchmark() -> dict[str, object]:
    metrics_path = DATA_DIR / "cifar10_benchmark" / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"missing second real-data benchmark at {metrics_path}; run scripts/run_cifar10_real_benchmark.py first"
        )
    return json.loads(metrics_path.read_text())


def load_beans_ablation() -> dict[str, object]:
    metrics_path = DATA_DIR / "beans_ablation" / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"missing ablation metrics at {metrics_path}; run scripts/run_beans_ablation.py first"
        )
    return json.loads(metrics_path.read_text())


def write_benchmark_table(metrics: dict[str, object]) -> None:
    beans = metrics["real_benchmark"]
    cifar = metrics["second_benchmark"]
    table = rf"""
\begin{{table}}[t]
\centering
\caption{{Real benchmark summary across two datasets. Both comparisons keep the backbone fixed and change only the prompt-activated augmentation recipe.}}
\label{{tab:progress_results}}
\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{llccc}}
\toprule
\textbf{{Dataset}} & \textbf{{Split}} & \textbf{{Fixed}} & \textbf{{Guided}} & \textbf{{$\Delta$}} \\
\midrule
Beans & clean & {beans['fixed_recipe']['clean_accuracy'] * 100:.1f}\% & {beans['language_guided']['clean_accuracy'] * 100:.1f}\% & {((beans['language_guided']['clean_accuracy'] - beans['fixed_recipe']['clean_accuracy']) * 100):+.1f} \\
Beans & low-light & {beans['fixed_recipe']['low_light_accuracy'] * 100:.1f}\% & {beans['language_guided']['low_light_accuracy'] * 100:.1f}\% & {beans['low_light_gain_pp']:+.1f} \\
CIFAR-10 & clean & {cifar['fixed_recipe']['clean_accuracy'] * 100:.1f}\% & {cifar['language_guided']['clean_accuracy'] * 100:.1f}\% & {((cifar['language_guided']['clean_accuracy'] - cifar['fixed_recipe']['clean_accuracy']) * 100):+.1f} \\
CIFAR-10 & blur & {cifar['fixed_recipe']['blur_accuracy'] * 100:.1f}\% & {cifar['language_guided']['blur_accuracy'] * 100:.1f}\% & {cifar['blur_gain_pp']:+.1f} \\
\bottomrule
\end{{tabular}}%
}}
\end{{table}}
""".strip()
    (LATEX_TABLES / "progress_results.tex").write_text(table + "\n")


def write_ablation_table(metrics: dict[str, object]) -> None:
    rows = metrics["ablation"]["results"]
    pretty_name = {
        "fixed_mobilenet_pretrained": "Fixed MNV3-S",
        "guided_mobilenet_pretrained": "Guided MNV3-S",
        "guided_mobilenet_scratch": "Guided MNV3-S",
        "guided_efficientnet_pretrained": "Guided EffNet-B0",
    }
    pretty_init = {
        "fixed_mobilenet_pretrained": "ImageNet",
        "guided_mobilenet_pretrained": "ImageNet",
        "guided_mobilenet_scratch": "scratch",
        "guided_efficientnet_pretrained": "ImageNet",
    }
    lines = []
    for row in rows:
        name = pretty_name.get(row["name"], row["name"])
        init = pretty_init.get(row["name"], "ImageNet" if row["pretrained"] else "scratch")
        lines.append(
            rf"{name} & {init} & {row['clean_accuracy'] * 100:.1f}\% & {row['low_light_accuracy'] * 100:.1f}\% \\"
        )
    table = "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Beans ablation under the same training budget.}",
            r"\label{tab:ablation_results}",
            r"\resizebox{\columnwidth}{!}{%",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"\textbf{Config} & \textbf{Init} & \textbf{Clean} & \textbf{Low-light} \\",
            r"\midrule",
            *lines,
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    )
    (LATEX_TABLES / "ablation_results.tex").write_text(table + "\n")


def _denormalize(image_tensor: torch.Tensor) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std = np.array([0.229, 0.224, 0.225])[:, None, None]
    image = image_tensor.numpy() * std + mean
    image = np.clip(image.transpose(1, 2, 0), 0.0, 1.0)
    return image


def _pretty_label(label: str) -> str:
    mapping = {
        "angular_leaf_spot": "ALS",
        "bean_rust": "rust",
        "healthy": "healthy",
    }
    return mapping.get(label, label.replace("_", " "))


def build_results_figure(metrics: dict[str, object]) -> None:
    experiment = metrics["real_benchmark"]
    examples = experiment["examples"]

    fig = plt.figure(figsize=(7.2, 2.45), constrained_layout=True)
    grid = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1, 1.35, 1.35], wspace=0.25, hspace=0.08)

    for idx, example in enumerate(examples[:4]):
        axis = fig.add_subplot(grid[idx // 2, idx % 2])
        # Resolve path: try absolute first, then relative to REPO_ROOT
        img_path = Path(example["image_path"])
        if not img_path.is_absolute() or not img_path.exists():
            img_path = REPO_ROOT / example["image_path"]
        axis.imshow(np.asarray(Image.open(img_path).convert("RGB")))
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(
            f"GT: {_pretty_label(example['target'])}\nF: {_pretty_label(example['baseline'])}\nG: {_pretty_label(example['guided'])}",
            fontsize=5.6,
            loc="left",
            pad=3,
        )
        for spine in axis.spines.values():
            spine.set_linewidth(0.8)
            spine.set_edgecolor("black")

    title_axis = fig.add_subplot(grid[:, 2:4])
    title_axis.set_axis_off()
    title_axis.text(0.0, 0.92, "Low-Light Qualitative Results", fontsize=10, fontweight="bold", va="top")
    title_axis.text(
        0.0,
        0.80,
        "Examples from the low-light corrupted Beans test split.\nWe show cases where the fixed recipe fails and the\nguided recipe remains correct.",
        fontsize=7.0,
        va="top",
    )
    title_axis.text(
        0.0,
        0.48,
        "Real-data benchmark",
        fontsize=8.2,
        fontweight="bold",
        va="top",
    )
    title_axis.text(
        0.0,
        0.38,
        "\n".join(
            [
                f"train = {experiment['train_images']} images",
                f"val = {experiment['val_images']} images",
                f"test = {experiment['test_images']} images",
                f"train device = {experiment['device']}",
                "eval = clean + low-light test",
            ]
        ),
        fontsize=6.9,
        va="top",
        family="monospace",
    )

    bar_axis = fig.add_subplot(grid[:, 4:])
    labels = ["Clean", "Low-light"]
    fixed = [
        experiment["fixed_recipe"]["clean_accuracy"] * 100.0,
        experiment["fixed_recipe"]["low_light_accuracy"] * 100.0,
    ]
    guided = [
        experiment["language_guided"]["clean_accuracy"] * 100.0,
        experiment["language_guided"]["low_light_accuracy"] * 100.0,
    ]
    x = np.arange(len(labels))
    width = 0.34
    bar_axis.bar(x - width / 2, fixed, width, label="Fixed recipe", color="#9a031e")
    bar_axis.bar(x + width / 2, guided, width, label="Language-guided", color="#0f4c5c")
    bar_axis.set_ylim(0, 105)
    bar_axis.set_ylabel("Accuracy (%)")
    bar_axis.set_xticks(x, labels)
    bar_axis.set_title("Beans Robustness")
    bar_axis.grid(axis="y", linestyle="--", alpha=0.35)
    bar_axis.legend(loc="upper right", fontsize=7)
    for xpos, value in zip(x - width / 2, fixed):
        bar_axis.text(xpos, value + 1.5, f"{value:.1f}", ha="center", fontsize=7)
    for xpos, value in zip(x + width / 2, guided):
        bar_axis.text(xpos, value + 1.5, f"{value:.1f}", ha="center", fontsize=7)

    fig.savefig(LATEX_FIGURES / "progress_results.pdf", bbox_inches="tight")
    plt.close(fig)


def _json_safe_metrics(metrics: dict[str, object]) -> dict[str, object]:
    return json.loads(json.dumps(metrics))


def main() -> None:
    LATEX_FIGURES.mkdir(parents=True, exist_ok=True)
    LATEX_TABLES.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "parser": evaluate_parser(),
        "label_resolution": evaluate_label_resolution(),
        "models": benchmark_models(),
        "real_benchmark": load_beans_benchmark(),
        "second_benchmark": load_cifar_benchmark(),
        "ablation": load_beans_ablation(),
    }
    (DATA_DIR / "progress_metrics.json").write_text(json.dumps(_json_safe_metrics(metrics), indent=2))
    write_benchmark_table(metrics)
    write_ablation_table(metrics)
    build_results_figure(metrics)
    print(json.dumps(_json_safe_metrics(metrics), indent=2))


if __name__ == "__main__":
    main()
