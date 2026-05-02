from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from prompt2model.augmentations import TorchVisionAugmentationBackend, build_augmentation_plan
from prompt2model.config import TaskType
from prompt2model.models import build_classification_model
from prompt2model.training import select_device, train_classification_model


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "report_eval" / "beans_benchmark"
EXAMPLES_DIR = OUTPUT_DIR / "examples"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _apply_low_light(image: Image.Image, seed: int) -> Image.Image:
    rng = random.Random(seed)
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.22, 0.38))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.65, 0.9))
    image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.4, 1.0)))
    array = np.asarray(image).astype(np.float32)
    noise = np.random.default_rng(seed).normal(0, 10, size=array.shape)
    array = np.clip(array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


class BeansDataset(Dataset):
    def __init__(
        self,
        split,
        image_size: int = 160,
        augmentations: TorchVisionAugmentationBackend | None = None,
        low_light_eval: bool = False,
        seed: int = 42,
    ) -> None:
        self.split = split
        self.augmentations = augmentations
        self.low_light_eval = low_light_eval
        self.seed = seed
        self.class_names = list(split.features["labels"].names)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.split)

    def _load_image(self, index: int) -> Image.Image:
        image = self.split[index]["image"].convert("RGB")
        if self.low_light_eval:
            image = _apply_low_light(image, self.seed + index)
        return image

    def display_image(self, index: int) -> Image.Image:
        return self._load_image(index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self._load_image(index)
        if self.augmentations is not None:
            image, _ = self.augmentations(image, None)
        label = int(self.split[index]["labels"])
        return self.transform(image), torch.tensor(label, dtype=torch.long)


def _evaluate_with_predictions(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, list[dict[str, int]]]:
    model.eval()
    total = 0
    correct = 0
    predictions: list[dict[str, int]] = []
    sample_index = 0
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            preds = logits.argmax(dim=1).cpu()
            for label, pred in zip(labels, preds):
                label_int = int(label.item())
                pred_int = int(pred.item())
                predictions.append({"index": sample_index, "target": label_int, "prediction": pred_int})
                correct += int(label_int == pred_int)
                total += 1
                sample_index += 1
    return correct / max(total, 1), predictions


def _build_train_config() -> object:
    config = type("TrainCfg", (), {})()
    config.epochs = 4
    config.learning_rate = 1e-3
    config.weight_decay = 1e-4
    config.max_steps_per_epoch = None
    return config


def _load_or_train_model(
    variant: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> torch.nn.Module:
    run_dir = OUTPUT_DIR / variant
    checkpoint_path = run_dir / "best_model.pt"
    if not checkpoint_path.exists():
        model = build_classification_model("mobilenet_v3_small", num_classes=num_classes, pretrained=True)
        train_classification_model(
            model,
            train_loader,
            val_loader,
            _build_train_config(),
            run_dir,
            device,
        )

    model = build_classification_model("mobilenet_v3_small", num_classes=num_classes, pretrained=True)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    return model.to(device)


def _save_examples(dataset: BeansDataset, baseline_preds: list[dict[str, int]], guided_preds: list[dict[str, int]]) -> list[dict[str, str]]:
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    examples: list[dict[str, str]] = []
    for baseline, guided in zip(baseline_preds, guided_preds):
        if baseline["prediction"] == baseline["target"]:
            continue
        if guided["prediction"] != guided["target"]:
            continue
        index = baseline["index"]
        image = dataset.display_image(index)
        filename = f"beans_{index:03d}.png"
        image.save(EXAMPLES_DIR / filename)
        # Store path relative to REPO_ROOT so metrics.json is portable across machines
        rel_path = (EXAMPLES_DIR / filename).relative_to(REPO_ROOT)
        examples.append(
            {
                "image_path": str(rel_path),
                "target": dataset.class_names[baseline["target"]],
                "baseline": dataset.class_names[baseline["prediction"]],
                "guided": dataset.class_names[guided["prediction"]],
            }
        )
        if len(examples) == 4:
            break
    return examples


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    seed = 17
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = load_dataset("beans")
    class_names = list(dataset["train"].features["labels"].names)
    device = select_device(TaskType.CLASSIFICATION)

    train_plain = BeansDataset(dataset["train"], augmentations=None, seed=seed)
    train_guided = BeansDataset(
        dataset["train"],
        augmentations=TorchVisionAugmentationBackend(
            build_augmentation_plan(["low_light"], TaskType.CLASSIFICATION),
            seed=seed,
        ),
        seed=seed,
    )
    val_set = BeansDataset(dataset["validation"], augmentations=None, seed=seed)
    test_clean = BeansDataset(dataset["test"], augmentations=None, seed=seed)
    test_low_light = BeansDataset(dataset["test"], augmentations=None, low_light_eval=True, seed=seed)

    train_plain_loader = DataLoader(train_plain, batch_size=16, shuffle=True, num_workers=0)
    train_guided_loader = DataLoader(train_guided, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=0)
    test_clean_loader = DataLoader(test_clean, batch_size=16, shuffle=False, num_workers=0)
    test_low_light_loader = DataLoader(test_low_light, batch_size=16, shuffle=False, num_workers=0)

    baseline_model = _load_or_train_model("fixed", train_plain_loader, val_loader, device, len(class_names))
    guided_model = _load_or_train_model("guided", train_guided_loader, val_loader, device, len(class_names))

    baseline_clean_acc, _ = _evaluate_with_predictions(baseline_model, test_clean_loader, device)
    baseline_low_acc, baseline_low_preds = _evaluate_with_predictions(baseline_model, test_low_light_loader, device)
    guided_clean_acc, _ = _evaluate_with_predictions(guided_model, test_clean_loader, device)
    guided_low_acc, guided_low_preds = _evaluate_with_predictions(guided_model, test_low_light_loader, device)

    examples = _save_examples(test_low_light, baseline_low_preds, guided_low_preds)
    metrics = {
        "dataset": "beans",
        "class_names": class_names,
        "device": str(device),
        "train_images": len(train_plain),
        "val_images": len(val_set),
        "test_images": len(test_clean),
        "fixed_recipe": {
            "clean_accuracy": baseline_clean_acc,
            "low_light_accuracy": baseline_low_acc,
        },
        "language_guided": {
            "clean_accuracy": guided_clean_acc,
            "low_light_accuracy": guided_low_acc,
        },
        "low_light_gain_pp": (guided_low_acc - baseline_low_acc) * 100.0,
        "examples": examples,
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
