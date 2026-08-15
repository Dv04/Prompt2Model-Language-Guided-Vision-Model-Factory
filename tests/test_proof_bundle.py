from __future__ import annotations

import json
from pathlib import Path

from prompt2model.proof_bundle import build_proof_bundle, verify_proof_bundle


def _file(root: Path, name: str, content: str) -> Path:
    path = root / name
    path.write_text(content, encoding="utf-8")
    return path


def test_complete_bundle_verifies_and_accepts(tmp_path):
    config = _file(tmp_path, "config.json", "{}")
    checkpoint = _file(tmp_path, "model.pt", "weights")
    report = _file(tmp_path, "report.md", "measured")
    engine = _file(tmp_path, "model.engine", "engine")
    bundle = build_proof_bundle(
        run_dir=tmp_path,
        artifacts={
            "config": config,
            "checkpoint": checkpoint,
            "evaluation_report": report,
        },
        target={"target": "tensorrt", "built": True, "artifact_path": str(engine)},
        evaluator={"independent": True, "id": "eval-v1"},
        rights={"complete": True, "id": "rights-v1"},
        hardware={"complete": True, "id": "orin-v1"},
        selective_risk={"passed": True, "id": "risk-v1"},
    )
    result = verify_proof_bundle(bundle)
    assert result.valid and result.accepted and not result.errors


def test_incomplete_bundle_is_valid_but_refuses_release(tmp_path):
    config = _file(tmp_path, "config.json", "{}")
    checkpoint = _file(tmp_path, "model.pt", "weights")
    report = _file(tmp_path, "report.md", "measured")
    bundle = build_proof_bundle(
        run_dir=tmp_path,
        artifacts={
            "config": config,
            "checkpoint": checkpoint,
            "evaluation_report": report,
        },
        target={"target": "tensorrt", "built": False, "artifact_path": None},
    )
    result = verify_proof_bundle(bundle)
    payload = json.loads(bundle.read_text(encoding="utf-8"))
    assert result.valid and not result.accepted
    assert "target_artifact_built" in payload["release"]["refused_reasons"]
    assert "independent_evaluation" in payload["release"]["refused_reasons"]


def test_tampered_artifact_is_rejected(tmp_path):
    config = _file(tmp_path, "config.json", "{}")
    checkpoint = _file(tmp_path, "model.pt", "weights")
    report = _file(tmp_path, "report.md", "measured")
    engine = _file(tmp_path, "model.engine", "engine")
    bundle = build_proof_bundle(
        run_dir=tmp_path,
        artifacts={
            "config": config,
            "checkpoint": checkpoint,
            "evaluation_report": report,
        },
        target={"target": "tensorrt", "built": True, "artifact_path": str(engine)},
        evaluator={"independent": True},
        rights={"complete": True},
        hardware={"complete": True},
        selective_risk={"passed": True},
    )
    checkpoint.write_text("poisoned", encoding="utf-8")
    result = verify_proof_bundle(bundle)
    assert not result.valid and not result.accepted
    assert "artifact digest mismatch: checkpoint" in result.errors
