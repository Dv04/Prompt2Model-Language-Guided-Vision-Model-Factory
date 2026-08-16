from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def isolate_pipeline_telemetry(monkeypatch, tmp_path):
    """Keep pipeline tests from appending to the tracked telemetry history."""
    import prompt2model.pipeline as pipeline
    from prompt2model.telemetry import TelemetryLogger

    def temporary_telemetry_logger(*_args, **_kwargs):
        return TelemetryLogger(global_csv_path=tmp_path / "telemetry_history.csv")

    monkeypatch.setattr(pipeline, "TelemetryLogger", temporary_telemetry_logger)
