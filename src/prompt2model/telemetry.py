"""Week 8 — Madhuvani: Model profiling and telemetry logging.

Benchmarks inference latency (FPS), parameter counts, and model size.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class TelemetryReport:
    """Consolidated model performance and complexity report."""

    parameters_total: int
    parameters_trainable: int
    model_size_mb: float
    inference_latency_ms: float
    fps: float
    device: str


class ModelProfiler:
    """Profiles a PyTorch model for hardware performance metrics."""

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def profile(self, sample_input: torch.Tensor, iterations: int = 20) -> TelemetryReport:
        """Measure inference speed and model size."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Estimate model size (assuming float32)
        size_mb = (total_params * 4) / (1024 * 1024)

        sample_input = sample_input.to(self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = self.model(sample_input)

        # Benchmark
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(iterations):
                _ = self.model(sample_input)
        elapsed = time.perf_counter() - start_time
        
        avg_latency = (elapsed / iterations) * 1000  # ms
        fps = 1000 / avg_latency if avg_latency > 0 else 0.0

        return TelemetryReport(
            parameters_total=total_params,
            parameters_trainable=trainable_params,
            model_size_mb=float(size_mb),
            inference_latency_ms=float(avg_latency),
            fps=float(fps),
            device=str(self.device),
        )

    def to_dict(self, report: TelemetryReport) -> dict[str, Any]:
        return {
            "parameters_total": report.parameters_total,
            "parameters_trainable": report.parameters_trainable,
            "model_size_mb": round(report.model_size_mb, 2),
            "inference_latency_ms": round(report.inference_latency_ms, 2),
            "fps": round(report.fps, 2),
            "device": report.device,
        }
