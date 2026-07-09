import pytest

from prompt2model.config import DatasetConfig, DatasetFormat, TaskType
from prompt2model.parsing import _extract_accuracy_floor, parse_prompt


def test_parse_prompt_extracts_task_labels_and_constraints() -> None:
    config = parse_prompt(
        prompt="Detect helmets and hard hats in low light CCTV footage and prioritize speed under 50 ms latency.",
        dataset=DatasetConfig(root="data/demo", format=DatasetFormat.COCO, annotation_path="data/demo/annotations.json"),
    )
    assert config.task == TaskType.DETECTION
    assert [label.name.lower() for label in config.labels] == ["helmets", "hard hats"]
    assert config.constraints.priority.value == "speed"
    assert config.constraints.target_latency_ms == 50
    assert "low_light" in config.augmentation_tags


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("Classify defects and keep at least 70% accuracy.", 0.70),
        ("Quantize the model but maintain a minimum of 60% accuracy.", 0.60),
        ("Detect forklifts, 70% accuracy floor, run under 30ms.", 0.70),
        ("Classify red square and blue circle images.", None),
    ],
)
def test_extract_accuracy_floor_reads_absolute_contract_term(prompt: str, expected: float | None) -> None:
    assert _extract_accuracy_floor(prompt) == expected


def test_parse_prompt_sets_accuracy_floor_without_llm_planner() -> None:
    config = parse_prompt(
        prompt="Classify bean leaf disease and keep at least 70% accuracy.",
        dataset=DatasetConfig(root="data/demo", format=DatasetFormat.IMAGEFOLDER),
    )
    assert config.constraints.accuracy_floor == pytest.approx(0.70)

