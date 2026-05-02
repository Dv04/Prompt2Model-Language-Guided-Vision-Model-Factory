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
    model_name: str | None = None
    augmentation_tags: list[str] = Field(default_factory=list)
    resolved_labels: list[ResolvedLabel] = Field(default_factory=list)

