from __future__ import annotations

import csv
import datetime
import json
import platform
from pathlib import Path
from typing import Any

import torch

from prompt2model.config import PipelineConfig


class TelemetryLogger:
    """Structured telemetry logger for capturing performance across runs."""

    def __init__(self, global_csv_path: str | Path | None = None) -> None:
        self.global_csv_path = Path(global_csv_path) if global_csv_path else None
        self._hardware_info = self._get_hardware_info()

    def _get_hardware_info(self) -> dict[str, Any]:
        """Detect system hardware specifications."""
        info = {
            "os": platform.system(),
            "os_release": platform.release(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "cpu_count": platform.os.cpu_count(),
        }

        # RAM info via psutil
        try:
            import psutil
            mem = psutil.virtual_memory()
            info["ram_total_gb"] = round(mem.total / (1024**3), 2)
        except ImportError:
            info["ram_total_gb"] = None

        # GPU info via torch
        info["gpu_available"] = torch.cuda.is_available()
        if info["gpu_available"]:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        else:
            info["gpu_name"] = None
            info["gpu_memory_gb"] = None

        return info

    def log_run(
        self,
        config: PipelineConfig,
        metrics: dict[str, Any],
        run_dir: str | Path,
    ) -> dict[str, Any]:
        """Capture telemetry for a single run and save to disk.

        Args:
            config: The pipeline configuration used.
            metrics: All performance and benchmarking metrics collected.
            run_dir: The directory where the run artifacts are stored.

        Returns:
            The combined telemetry record.
        """
        run_dir = Path(run_dir)
        timestamp = datetime.datetime.now().isoformat()

        record = {
            "timestamp": timestamp,
            "metadata": {
                "prompt": config.prompt,
                "task": config.task.value,
                "model_name": config.model_name,
                "priority": config.constraints.priority.value,
            },
            "hardware": self._hardware_info,
            "metrics": metrics,
        }

        # Save individual JSON
        json_path = run_dir / "telemetry.json"
        json_path.write_text(json.dumps(record, indent=2))

        # Append to global CSV if path is set
        if self.global_csv_path:
            self._append_to_csv(record)

        return record

    def _append_to_csv(self, record: dict[str, Any]) -> None:
        """Append a flattened run record to the global telemetry CSV."""
        self.global_csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Flatten the record for CSV
        metrics = record.get("metrics", {})
        flat = {
            "timestamp": record["timestamp"],
            "prompt": record["metadata"]["prompt"],
            "task": record["metadata"]["task"],
            "model_name": record["metadata"]["model_name"],
            "os": record["hardware"]["os"],
            "gpu": record["hardware"]["gpu_name"] or "CPU",
            "accuracy": metrics.get("accuracy") or metrics.get("map@0.5"),
            "macro_f1": metrics.get("macro_f1"),
            "latency_ms": metrics.get("latency_ms"),
            "fps": metrics.get("fps"),
            "gflops": metrics.get("gflops"),
            "params_m": metrics.get("parameter_count_millions"),
        }

        file_exists = self.global_csv_path.exists()
        with open(self.global_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=flat.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(flat)
