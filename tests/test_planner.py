"""LLM planner front end: schema validation, repair loop, fallback, overlay,
and constraint-driven model selection. No network — transports are fakes."""
from __future__ import annotations

import json

import pytest

from prompt2model.config import (
    DatasetConfig,
    ModelConstraints,
    PriorityPreset,
    TaskType,
)
from prompt2model.models import recommend_model
from prompt2model.planner import (
    ENV_ENDPOINT,
    ENV_MODEL,
    LLMPlanner,
    PlannerError,
    PlannerOutput,
    plan_prompt,
)

DATASET = DatasetConfig(root="output/toy", image_size=96)

GOOD_PLAN = {
    "task": "detection",
    "labels": [{"name": "person without helmet", "synonyms": ["bare head"]}],
    "priority": "accuracy",
    "target_latency_ms": 100,
    "recall_bias": 0.9,
    "accuracy_floor": 0.95,
    "deployment_target": "jetson",
    "environment_tags": ["low_light"],
}

PROMPT = (
    "Watch the dock for any person without a helmet after 10 pm, under 100 ms, "
    "and never miss a single one."
)


@pytest.fixture(autouse=True)
def _no_env_planner(monkeypatch):
    monkeypatch.delenv(ENV_ENDPOINT, raising=False)
    monkeypatch.delenv(ENV_MODEL, raising=False)


class _FakeTransport:
    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.calls: list[list[dict[str, str]]] = []

    def __call__(self, messages):
        self.calls.append(messages)
        return self.replies.pop(0)


# ── LLMPlanner.plan ──────────────────────────────────────────────────────────


def test_plan_happy_path():
    transport = _FakeTransport(json.dumps(GOOD_PLAN))
    plan = LLMPlanner(transport=transport).plan(PROMPT)
    assert plan.task == TaskType.DETECTION
    assert plan.labels[0].name == "person without helmet"
    assert plan.recall_bias == 0.9
    assert len(transport.calls) == 1


def test_plan_strips_markdown_fences():
    transport = _FakeTransport(f"```json\n{json.dumps(GOOD_PLAN)}\n```")
    plan = LLMPlanner(transport=transport).plan(PROMPT)
    assert plan.accuracy_floor == 0.95


def test_plan_repair_round_trip():
    transport = _FakeTransport("not json at all", json.dumps(GOOD_PLAN))
    plan = LLMPlanner(transport=transport).plan(PROMPT)
    assert plan.task == TaskType.DETECTION
    assert len(transport.calls) == 2
    # The repair request carries the failure back to the model.
    assert "not valid" in transport.calls[1][-1]["content"]


def test_plan_fails_after_two_bad_replies():
    transport = _FakeTransport("nope", json.dumps({"recall_bias": 7}))  # out of range
    with pytest.raises(PlannerError):
        LLMPlanner(transport=transport).plan(PROMPT)


def test_unconfigured_planner_raises():
    with pytest.raises(PlannerError):
        LLMPlanner().plan(PROMPT)


# ── plan_prompt modes ────────────────────────────────────────────────────────


def test_mode_regex_never_calls_planner():
    transport = _FakeTransport(json.dumps(GOOD_PLAN))
    config = plan_prompt(
        PROMPT, DATASET, planner=LLMPlanner(transport=transport), mode="regex"
    )
    assert transport.calls == []
    assert config.prompt == PROMPT  # complete regex-parsed config


def test_mode_auto_without_planner_is_pure_regex():
    from prompt2model.parsing import parse_prompt

    auto = plan_prompt(PROMPT, DATASET, mode="auto")
    regex = parse_prompt(PROMPT, DATASET)
    assert auto.model_dump() == regex.model_dump()


def test_mode_auto_falls_back_on_planner_failure():
    def _boom(messages):
        raise PlannerError("endpoint down")

    config = plan_prompt(PROMPT, DATASET, planner=LLMPlanner(transport=_boom), mode="auto")
    assert config.labels  # regex result stands


def test_mode_llm_propagates_failure():
    def _boom(messages):
        raise PlannerError("endpoint down")

    with pytest.raises(PlannerError):
        plan_prompt(PROMPT, DATASET, planner=LLMPlanner(transport=_boom), mode="llm")


def test_mode_llm_requires_configuration():
    with pytest.raises(PlannerError):
        plan_prompt(PROMPT, DATASET, mode="llm")


# ── overlay semantics ────────────────────────────────────────────────────────


def test_overlay_applies_plan_fields():
    transport = _FakeTransport(json.dumps(GOOD_PLAN))
    config = plan_prompt(PROMPT, DATASET, planner=LLMPlanner(transport=transport), mode="llm")
    assert config.task == TaskType.DETECTION
    assert [l.name for l in config.labels] == ["person without helmet"]
    assert config.constraints.priority == PriorityPreset.ACCURACY
    assert config.constraints.speed_accuracy_tradeoff == 0.8
    assert config.constraints.target_latency_ms == 100
    assert config.constraints.recall_bias == 0.9
    assert config.constraints.accuracy_floor == 0.95
    assert config.data_context.deployment_target == "jetson"
    assert "low_light" in config.data_context.environment_tags
    assert "low_light" in config.augmentation_tags


def test_overlay_nulls_keep_regex_values():
    # A plan that says nothing keeps the regex parse intact.
    transport = _FakeTransport(json.dumps({}))
    config = plan_prompt(
        "Classify \"cat\" and \"dog\" photos, prioritize speed, within 5 minutes.",
        DATASET,
        planner=LLMPlanner(transport=transport),
        mode="llm",
    )
    assert [l.name for l in config.labels] == ["cat", "dog"]
    assert config.constraints.priority == PriorityPreset.SPEED
    assert config.constraints.budget_minutes == 5


def test_overlay_task_hint_outranks_llm():
    transport = _FakeTransport(json.dumps({"task": "detection"}))
    config = plan_prompt(
        PROMPT,
        DATASET,
        task_hint=TaskType.CLASSIFICATION,
        planner=LLMPlanner(transport=transport),
        mode="llm",
    )
    assert config.task == TaskType.CLASSIFICATION


# ── constraint-driven model selection ────────────────────────────────────────


def test_recommend_priority_only_matches_legacy():
    c = ModelConstraints(priority=PriorityPreset.ACCURACY)
    assert recommend_model(TaskType.CLASSIFICATION, c) == "efficientnet_b0"
    assert recommend_model(TaskType.DETECTION, c) == "rtdetr-l"


def test_recommend_tight_latency_forces_speed_tier():
    c = ModelConstraints(priority=PriorityPreset.ACCURACY, target_latency_ms=25)
    assert recommend_model(TaskType.CLASSIFICATION, c) == "mobilenet_v3_small"
    assert recommend_model(TaskType.DETECTION, c) == "ssdlite320_mobilenet_v3_large"


def test_recommend_moderate_latency_caps_at_balanced():
    c = ModelConstraints(priority=PriorityPreset.ACCURACY, target_latency_ms=60)
    assert recommend_model(TaskType.DETECTION, c) == "yolov11n"


def test_recommend_power_budget_forces_speed_tier():
    c = ModelConstraints(priority=PriorityPreset.BALANCED, power_budget_w=2.0)
    assert recommend_model(TaskType.CLASSIFICATION, c) == "mobilenet_v3_small"


def test_recommend_parameter_cap_walks_down_tiers():
    c = ModelConstraints(priority=PriorityPreset.ACCURACY, max_parameters_millions=5.0)
    # rtdetr-l (32M) > 5M → yolov11n (2.6M) fits.
    assert recommend_model(TaskType.DETECTION, c) == "yolov11n"
    # Cap tighter than every option → smallest tier wins anyway.
    c2 = ModelConstraints(priority=PriorityPreset.ACCURACY, max_parameters_millions=1.0)
    assert recommend_model(TaskType.CLASSIFICATION, c2) == "mobilenet_v3_small"


def test_constraint_fields_validate():
    with pytest.raises(ValueError):
        ModelConstraints(recall_bias=1.5)
    with pytest.raises(ValueError):
        ModelConstraints(power_budget_w=0)
    with pytest.raises(ValueError):
        ModelConstraints(accuracy_floor=-0.1)
