from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    DETECTION = "detection"


class DatasetFormat(str, Enum):
    IMAGEFOLDER = "imagefolder"
    CSV = "csv"
    COCO = "coco"


class PriorityPreset(str, Enum):
    SPEED = "speed"
    BALANCED = "balanced"
    ACCURACY = "accuracy"


class RequestedLabel(BaseModel):
    name: str
    synonyms: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("requested label name cannot be empty")
        return value


class ResolvedLabel(BaseModel):
    requested_label: str
    dataset_label: str
    score: float
    method: str


class ModelConstraints(BaseModel):
    priority: PriorityPreset = PriorityPreset.BALANCED
    speed_accuracy_tradeoff: float = 0.5
    target_latency_ms: int | None = None
    max_parameters_millions: float | None = None
    budget_minutes: int = 15
    llm_temperature: float = 0.1
    # Recall-vs-precision preference: 1.0 = "never miss one" (recall),
    # 0.0 = "never cry wolf" (precision), None = no stated preference.
    recall_bias: float | None = None
    # Device power envelope in watts (e.g. 2 for a camera SoC, 15 for a
    # Jetson). Tight budgets force the speed tier during model selection.
    power_budget_w: float | None = None
    # Minimum acceptable accuracy (absolute, 0-1). Downstream stages
    # (compression) must refuse to ship an artifact below this floor.
    accuracy_floor: float | None = None

    @field_validator("speed_accuracy_tradeoff")
    @classmethod
    def _tradeoff_bounds(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("speed_accuracy_tradeoff must be between 0 and 1")
        return value

    @field_validator("budget_minutes")
    @classmethod
    def _budget_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("budget_minutes must be positive")
        return value

    @field_validator("recall_bias", "accuracy_floor")
    @classmethod
    def _unit_interval(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("value must be between 0 and 1")
        return value

    @field_validator("power_budget_w")
    @classmethod
    def _power_positive(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError("power_budget_w must be positive")
        return value


class DataContext(BaseModel):
    environment_tags: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    deployment_target: str = "host"


class DatasetConfig(BaseModel):
    root: str
    format: DatasetFormat = DatasetFormat.IMAGEFOLDER
    annotation_path: str | None = None
    image_size: int = 128
    val_split: float = 0.2
    test_split: float = 0.1
    seed: int = 42

    @field_validator("root")
    @classmethod
    def _normalize_root(cls, value: str) -> str:
        return str(Path(value))

    @model_validator(mode="after")
    def _validate_splits(self) -> "DatasetConfig":
        if self.val_split < 0 or self.test_split < 0:
            raise ValueError("dataset splits must be non-negative")
        if self.val_split + self.test_split >= 1:
            raise ValueError("val_split + test_split must be less than 1")
        if self.format == DatasetFormat.COCO and not self.annotation_path:
            raise ValueError("annotation_path is required for COCO datasets")
        return self


class TrainingConfig(BaseModel):
    batch_size: int = 8
    epochs: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    pretrained: bool = False
    device: str | None = None
    max_steps_per_epoch: int | None = 10


class ExportConfig(BaseModel):
    output_dir: str = "output/runs"
    export_onnx: bool = True
    onnx_opset: int = 17
    topk_detections: int = 20


class CompressionConfig(BaseModel):
    """The distill→quantize→gate stage. Off by default; the planner (or CLI)
    turns it on when the prompt asks for an edge-grade artifact."""

    enable_quantization: bool = False
    quantization_mode: str = "dynamic"  # "dynamic" | "static"
    enable_distillation: bool = False
    # Teacher for distillation: a registry model name (trained on the same
    # data as part of the run) — defaults to the accuracy tier for the task.
    distillation_teacher: str | None = None
    distillation_temperature: float = 4.0
    distillation_alpha: float = 0.7  # weight of the soft (teacher) loss
    # The compressed artifact must retain at least this fraction of the
    # uncompressed model's validation accuracy — else the factory REFUSES to
    # ship it and keeps the uncompressed artifact. An absolute floor can
    # additionally come from ModelConstraints.accuracy_floor.
    accuracy_floor_relative: float = 0.98

    @field_validator("quantization_mode")
    @classmethod
    def _mode_known(cls, value: str) -> str:
        if value not in ("dynamic", "static"):
            raise ValueError("quantization_mode must be 'dynamic' or 'static'")
        return value

    @field_validator("accuracy_floor_relative", "distillation_alpha")
    @classmethod
    def _unit(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("value must be between 0 and 1")
        return value

    @field_validator("distillation_temperature")
    @classmethod
    def _temp_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("distillation_temperature must be positive")
        return value


class PipelineConfig(BaseModel):
    project_name: str = "Prompt2Model"
    prompt: str
    task: TaskType
    labels: list[RequestedLabel]
    constraints: ModelConstraints = Field(default_factory=ModelConstraints)
    data_context: DataContext = Field(default_factory=DataContext)
    dataset: DatasetConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    model_name: str | None = None
    augmentation_tags: list[str] = Field(default_factory=list)
    resolved_labels: list[ResolvedLabel] = Field(default_factory=list)

