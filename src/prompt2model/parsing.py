from __future__ import annotations

import re
from collections.abc import Sequence

from prompt2model.config import (
    CompressionConfig,
    DataContext,
    DatasetConfig,
    ModelConstraints,
    PipelineConfig,
    PriorityPreset,
    RequestedLabel,
    TaskType,
    TrainingConfig,
)


ENVIRONMENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "low_light": ("low light", "dark", "night", "nighttime", "dim"),
    "rain": ("rain", "rainy", "wet"),
    "fog": ("fog", "foggy", "mist", "misty"),
    "motion_blur": ("motion blur", "blurry", "blur", "fast-moving"),
    "glare": ("glare", "sun glare", "backlit"),
    "occlusion": ("occlusion", "occluded", "crowd"),
}


def infer_task_from_prompt(prompt: str, task_hint: TaskType | None = None) -> TaskType:
    if task_hint is not None:
        return task_hint
    normalized = prompt.lower()
    if any(keyword in normalized for keyword in ("detect", "detection", "localize", "bounding box", "bbox", "find", "segment", "segmentation")):
        return TaskType.DETECTION
    return TaskType.CLASSIFICATION


def _extract_priority(prompt: str) -> PriorityPreset:
    normalized = prompt.lower()
    speed_terms = ("prioritize speed", "real-time", "lightweight", "edge", "maximize speed")
    accuracy_terms = ("prioritize accuracy", "high accuracy", "best accuracy", "maximize accuracy")
    if any(term in normalized for term in accuracy_terms):
        return PriorityPreset.ACCURACY
    if any(term in normalized for term in speed_terms):
        return PriorityPreset.SPEED
    return PriorityPreset.BALANCED


def _extract_speed_accuracy_tradeoff(priority: PriorityPreset) -> float:
    if priority == PriorityPreset.SPEED:
        return 0.2
    if priority == PriorityPreset.ACCURACY:
        return 0.8
    return 0.5


def _extract_latency_ms(prompt: str) -> int | None:
    match = re.search(r"(\d+)\s*ms", prompt.lower())
    return int(match.group(1)) if match else None


def _extract_budget_minutes(prompt: str) -> int:
    match = re.search(r"(\d+)\s*(minute|min|hour|hr)", prompt.lower())
    if not match:
        return 15
    value = int(match.group(1))
    unit = match.group(2).lower()
    if unit.startswith("h"):
        return value * 60
    return value


def _wants_quantization(prompt: str) -> bool:
    normalized = prompt.lower()
    return any(term in normalized for term in ("quantize", "quantized", "int8", "8-bit", "compress the model"))


def _wants_distillation(prompt: str) -> bool:
    normalized = prompt.lower()
    return any(term in normalized for term in ("distill", "distilled", "distillation"))


def _extract_environment_tags(prompt: str) -> list[str]:
    normalized = prompt.lower()
    tags = [tag for tag, keywords in ENVIRONMENT_KEYWORDS.items() if any(keyword in normalized for keyword in keywords)]
    return sorted(set(tags))


def _split_label_phrase(raw: str) -> list[str]:
    # Remove common filler phrases that are not prepositions
    cleaned = re.sub(r"\b(different types of|various types of|various|kinds of|including)\b", "", raw, flags=re.IGNORECASE)
    # Remove 'medical' only if followed by 'images'
    cleaned = re.sub(r"\bmedical\s+(?=images?)", "", cleaned, flags=re.IGNORECASE)
    # Remove common prepositions and cutoff trailing context
    cleaned = re.sub(r"\b(in|under|with|while|for|on|during|prioritize|and ensure|within a .* budget)\b.*$", "", cleaned, flags=re.IGNORECASE)
    # Remove standalone filler words
    cleaned = re.sub(r"\b(fast-moving|condition|conditions|scenes?|footage|photos?)\b", "", cleaned, flags=re.IGNORECASE)
    # Only remove 'images' or 'objects' if they are followed by other words (likely labels)
    cleaned = re.sub(r"\b(images?|objects?)\b(?=\s+\w)", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("/", ",")
    parts = re.split(r",| and |\bor\b", cleaned)
    labels = []
    for part in parts:
        label = part.strip(" .:-\"'")
        if label:
            labels.append(label)
    return labels


def _extract_labels_from_quotes(prompt: str) -> list[str]:
    return [item.strip() for item in re.findall(r"\"([^\"]+)\"", prompt) if item.strip()]


def extract_requested_labels(prompt: str, task: TaskType) -> list[RequestedLabel]:
    normalized = prompt.strip()
    
    # Priority 1: Quotes
    quoted = _extract_labels_from_quotes(normalized)
    if quoted:
        return [RequestedLabel(name=q) for q in quoted]

    # Priority 2: Keyword-based extraction
    patterns: Sequence[str]
    if task == TaskType.DETECTION:
        patterns = (
            r"(?:detect|find|localize)\s+(.+?)(?:\.|;|$)",
            r"(?:\w+\s+)?(?:object )?detection\s+(?:of|for|on)?\s?(.+?)(?:\.|;|$)",
        )
    else:
        patterns = (
            r"(?:classify|recognize|categorize|find)\s+(.+?)(?:\.|;|$)",
            r"(?:\w+\s+)?(?:image )?classification\s+(?:of|for|on)?\s?(.+?)(?:\.|;|$)",
        )

    extracted: list[str] = []
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            extracted.extend(_split_label_phrase(match.group(1)))
            break

    if not extracted:
        extracted = ["target"]

    unique = []
    seen = set()
    for label in extracted:
        normalized_label = label.lower()
        if normalized_label not in seen:
            seen.add(normalized_label)
            unique.append(RequestedLabel(name=label))
    return unique


def parse_prompt(
    prompt: str,
    dataset: DatasetConfig,
    task_hint: TaskType | None = None,
    training_overrides: TrainingConfig | None = None,
) -> PipelineConfig:
    task = infer_task_from_prompt(prompt, task_hint)
    priority = _extract_priority(prompt)
    constraints = ModelConstraints(
        priority=priority,
        speed_accuracy_tradeoff=_extract_speed_accuracy_tradeoff(priority),
        target_latency_ms=_extract_latency_ms(prompt),
        budget_minutes=_extract_budget_minutes(prompt),
    )
    data_context = DataContext(environment_tags=_extract_environment_tags(prompt))
    labels = extract_requested_labels(prompt, task)
    training = training_overrides or TrainingConfig()
    return PipelineConfig(
        prompt=prompt,
        task=task,
        labels=labels,
        constraints=constraints,
        data_context=data_context,
        dataset=dataset,
        training=training,
        compression=CompressionConfig(
            enable_quantization=_wants_quantization(prompt),
            enable_distillation=_wants_distillation(prompt),
        ),
        augmentation_tags=data_context.environment_tags,
    )
