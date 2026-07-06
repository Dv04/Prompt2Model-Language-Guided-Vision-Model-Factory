"""LLM planner front end — natural-language intent → validated typed spec.

The regex parser (``parsing.parse_prompt``) is fast and dependency-free but
brittle: it misses labels phrased indirectly, can't read "never miss a single
one" as a recall preference, and knows nothing it wasn't pattern-matched for.
This module puts a language model in front of it:

    prompt ──► LLM ──► strict JSON ──► PlannerOutput (validated) ──► overlay
                                                                     onto the
                            regex-parsed PipelineConfig ──► PipelineConfig

Design rules:

* **The LLM is optional.** No endpoint configured → the regex parser runs
  alone, byte-for-byte the previous behaviour. The factory must work fully
  offline.
* **The LLM proposes, the schema disposes.** Whatever comes back is validated
  against :class:`PlannerOutput`; one repair round-trip is attempted on
  invalid output, then the planner fails loudly (``mode="llm"``) or falls
  back silently to regex (``mode="auto"``).
* **Overlay, never replace.** The regex parse always produces a complete
  config; the planner only overrides fields it actually extracted (non-null).
  A half-answer from a small model can't hole the config.
* **Any OpenAI-compatible endpoint.** ``/v1/chat/completions`` — works with
  Ollama, vLLM, OpenAI, llama.cpp server, etc. Configured via
  ``P2M_LLM_ENDPOINT`` / ``P2M_LLM_MODEL`` (or constructor args). Transport
  is injectable for tests: no network, no new dependencies (urllib only).
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Callable, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from prompt2model.config import (
    DatasetConfig,
    PipelineConfig,
    PriorityPreset,
    RequestedLabel,
    TaskType,
    TrainingConfig,
)
from prompt2model.parsing import parse_prompt

logger = logging.getLogger("prompt2model.planner")

# messages -> assistant text. Injectable for tests.
Transport = Callable[[list[dict[str, str]]], str]

ENV_ENDPOINT = "P2M_LLM_ENDPOINT"
ENV_MODEL = "P2M_LLM_MODEL"
ENV_TIMEOUT = "P2M_LLM_TIMEOUT"

PlannerMode = Literal["auto", "llm", "regex"]


class PlannerError(RuntimeError):
    """The LLM planner could not produce a valid plan."""


class PlannedLabel(BaseModel):
    name: str
    synonyms: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("label name cannot be empty")
        return value


class PlannerOutput(BaseModel):
    """The typed spec the LLM must produce. Every field optional: ``null``
    means "the prompt doesn't say", and the regex-parsed value stands."""

    task: TaskType | None = None
    labels: list[PlannedLabel] = Field(default_factory=list)
    priority: PriorityPreset | None = None
    target_latency_ms: int | None = Field(default=None, ge=1)
    budget_minutes: int | None = Field(default=None, ge=1)
    recall_bias: float | None = Field(default=None, ge=0.0, le=1.0)
    power_budget_w: float | None = Field(default=None, gt=0.0)
    accuracy_floor: float | None = Field(default=None, ge=0.0, le=1.0)
    max_parameters_millions: float | None = Field(default=None, gt=0.0)
    deployment_target: str | None = None
    environment_tags: list[str] = Field(default_factory=list)
    quantize: bool | None = None
    distill: bool | None = None
    notes: str | None = None


_SYSTEM_PROMPT = """\
You turn a user's plain-language request for a vision model into ONE JSON \
object and nothing else — no prose, no markdown fences.

Schema (every field may be null when the request does not state it):
{
  "task": "classification" | "detection" | null,
  "labels": [{"name": "<concrete class name>", "synonyms": ["..."]}],
  "priority": "speed" | "balanced" | "accuracy" | null,
  "target_latency_ms": int | null,          // "under 50 ms" -> 50
  "budget_minutes": int | null,             // training budget; "2 hours" -> 120
  "recall_bias": float 0..1 | null,         // "never miss one" -> 0.9+; "no false alarms" -> 0.1
  "power_budget_w": float | null,           // "runs at 2 watts" -> 2
  "accuracy_floor": float 0..1 | null,      // "keep at least 95% accuracy" -> 0.95
  "max_parameters_millions": float | null,  // "under 5M parameters" -> 5
  "deployment_target": string | null,       // e.g. "cpu", "jetson", "tensorrt", "hexagon"
  "environment_tags": ["low_light"|"rain"|"fog"|"motion_blur"|"glare"|"occlusion"],
  "quantize": bool | null,                  // asks for int8/compressed artifact
  "distill": bool | null,                   // asks for a distilled model
  "notes": string | null
}

Rules:
- labels are the CONCRETE things to recognize ("person without helmet",
  "forklift"), never filler words like "images" or "objects".
- Distinguish counting/locating requests (detection) from whole-image
  labelling (classification). If genuinely ambiguous use null.
- Output the JSON object only.
"""


def _strip_fences(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


class LLMPlanner:
    """Plans an intent spec through any OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        endpoint: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        transport: Transport | None = None,
    ) -> None:
        self.endpoint = (endpoint or "").rstrip("/")
        self.model = model or ""
        self.timeout = timeout
        self._transport = transport

    @classmethod
    def from_env(cls) -> "LLMPlanner":
        timeout_raw = os.getenv(ENV_TIMEOUT, "")
        try:
            timeout = float(timeout_raw) if timeout_raw else 60.0
        except ValueError:
            timeout = 60.0
        return cls(
            endpoint=os.getenv(ENV_ENDPOINT, ""),
            model=os.getenv(ENV_MODEL, ""),
            timeout=timeout,
        )

    @property
    def is_configured(self) -> bool:
        return bool(self._transport) or bool(self.endpoint and self.model)

    def _chat(self, messages: list[dict[str, str]]) -> str:
        if self._transport is not None:
            return self._transport(messages)
        if not self.endpoint or not self.model:
            raise PlannerError(
                f"LLM planner not configured — set {ENV_ENDPOINT} and {ENV_MODEL} "
                "(any OpenAI-compatible endpoint, e.g. an Ollama host)."
            )
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.endpoint}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            raise PlannerError(f"LLM endpoint error: {exc}") from exc
        try:
            return str(payload["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise PlannerError(f"unexpected chat-completions response shape: {exc}") from exc

    def plan(self, prompt: str) -> PlannerOutput:
        """Prompt → validated PlannerOutput, with one repair round-trip."""
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        raw = self._chat(messages)
        try:
            return PlannerOutput.model_validate(json.loads(_strip_fences(raw)))
        except (json.JSONDecodeError, ValidationError) as first_error:
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous output was not valid against the schema: "
                        f"{first_error}. Return ONLY the corrected JSON object."
                    ),
                }
            )
            repaired = self._chat(messages)
            try:
                return PlannerOutput.model_validate(json.loads(_strip_fences(repaired)))
            except (json.JSONDecodeError, ValidationError) as second_error:
                raise PlannerError(
                    f"LLM produced invalid plan twice: {second_error}"
                ) from second_error


def _overlay(config: PipelineConfig, plan: PlannerOutput, task_hint: TaskType | None) -> PipelineConfig:
    """Apply the planner's non-null fields on top of the regex-parsed config.
    An explicit ``task_hint`` from the caller always outranks the LLM."""
    if plan.task is not None and task_hint is None:
        config.task = plan.task
    if plan.labels:
        config.labels = [
            RequestedLabel(name=label.name, synonyms=list(label.synonyms))
            for label in plan.labels
        ]
    constraints = config.constraints
    if plan.priority is not None:
        constraints.priority = plan.priority
        constraints.speed_accuracy_tradeoff = {
            PriorityPreset.SPEED: 0.2,
            PriorityPreset.BALANCED: 0.5,
            PriorityPreset.ACCURACY: 0.8,
        }[plan.priority]
    for field in (
        "target_latency_ms",
        "budget_minutes",
        "recall_bias",
        "power_budget_w",
        "accuracy_floor",
        "max_parameters_millions",
    ):
        value = getattr(plan, field)
        if value is not None:
            setattr(constraints, field, value)
    if plan.deployment_target:
        config.data_context.deployment_target = plan.deployment_target.strip().lower()
    if plan.quantize is not None:
        config.compression.enable_quantization = plan.quantize
    if plan.distill is not None:
        config.compression.enable_distillation = plan.distill
    if plan.environment_tags:
        merged = set(config.data_context.environment_tags) | {
            t.strip().lower() for t in plan.environment_tags if t.strip()
        }
        config.data_context.environment_tags = sorted(merged)
        config.augmentation_tags = sorted(set(config.augmentation_tags) | merged)
    if plan.notes:
        config.data_context.notes = [*config.data_context.notes, plan.notes]
    return config


def plan_prompt(
    prompt: str,
    dataset: DatasetConfig,
    task_hint: TaskType | None = None,
    training_overrides: TrainingConfig | None = None,
    planner: LLMPlanner | None = None,
    mode: PlannerMode = "auto",
) -> PipelineConfig:
    """The planner-aware replacement for ``parse_prompt``.

    - ``regex``: deterministic parser only (previous behaviour).
    - ``llm``: the LLM plan is required; PlannerError propagates.
    - ``auto``: use the LLM when one is configured, fall back to regex on any
      planner failure — a config is always produced.
    """
    config = parse_prompt(
        prompt, dataset=dataset, task_hint=task_hint, training_overrides=training_overrides
    )
    if mode == "regex":
        return config

    planner = planner or LLMPlanner.from_env()
    if not planner.is_configured:
        if mode == "llm":
            raise PlannerError(
                f"mode='llm' but no planner configured — set {ENV_ENDPOINT} and {ENV_MODEL}."
            )
        return config

    try:
        plan = planner.plan(prompt)
    except PlannerError:
        if mode == "llm":
            raise
        logger.warning("LLM planner failed; falling back to regex parse", exc_info=True)
        return config
    return _overlay(config, plan, task_hint)
