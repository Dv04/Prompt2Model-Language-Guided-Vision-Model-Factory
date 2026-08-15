"""Machine-verifiable evidence bundle for a Prompt2Model compilation run.

The bundle is deliberately fail-closed. A model file and a successful export
are evidence artifacts, not an accepted release. Promotion requires every
declared gate, and verification recomputes both file digests and that decision.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "dhi.prompt2model.proof-bundle.v1"


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(path: Path, *, run_dir: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    root = run_dir.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"proof artifact escapes run directory: {path}") from exc
    if not resolved.is_file():
        raise ValueError(f"proof artifact is not a file: {path}")
    return {
        "path": relative.as_posix(),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def build_proof_bundle(
    *,
    run_dir: str | Path,
    artifacts: Mapping[str, str | Path | None],
    target: Mapping[str, Any] | None,
    evaluator: Mapping[str, Any] | None = None,
    rights: Mapping[str, Any] | None = None,
    hardware: Mapping[str, Any] | None = None,
    selective_risk: Mapping[str, Any] | None = None,
) -> Path:
    """Write a canonical proof bundle and a recomputable promotion decision."""
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    artifact_records = {
        role: _artifact_record(Path(path), run_dir=root)
        for role, path in sorted(artifacts.items())
        if path is not None
    }
    target = dict(target or {})
    evaluator = dict(evaluator or {})
    rights = dict(rights or {})
    hardware = dict(hardware or {})
    selective_risk = dict(selective_risk or {})

    target_path = target.get("artifact_path")
    target_artifact_hashed = False
    if target_path:
        candidate = Path(target_path)
        if candidate.exists():
            record = _artifact_record(candidate, run_dir=root)
            artifact_records.setdefault("target_artifact", record)
            target_artifact_hashed = True

    gates = {
        "config_hashed": "config" in artifact_records,
        "checkpoint_hashed": "checkpoint" in artifact_records,
        "evaluation_report_hashed": "evaluation_report" in artifact_records,
        "target_resolved": bool(target.get("target")),
        "target_artifact_built": bool(target.get("built")),
        "target_artifact_hashed": target_artifact_hashed,
        "independent_evaluation": bool(evaluator.get("independent")),
        "rights_complete": bool(rights.get("complete")),
        "hardware_evidence_complete": bool(hardware.get("complete")),
        "selective_risk_gate_passed": bool(selective_risk.get("passed")),
    }
    refused_reasons = [name for name, passed in gates.items() if not passed]
    body: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "artifacts": artifact_records,
        "target": target,
        "evaluator": evaluator,
        "rights": rights,
        "hardware": hardware,
        "selective_risk": selective_risk,
        "gates": gates,
        "release": {
            "accepted": not refused_reasons,
            "refused_reasons": refused_reasons,
        },
    }
    body["bundle_sha256"] = hashlib.sha256(_canonical_bytes(body)).hexdigest()
    output = root / "proof_bundle.json"
    output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


@dataclass(frozen=True)
class ProofVerification:
    valid: bool
    accepted: bool
    errors: tuple[str, ...]


def verify_proof_bundle(path: str | Path) -> ProofVerification:
    """Recompute the canonical digest, artifact hashes, and release decision."""
    bundle_path = Path(path).resolve(strict=True)
    root = bundle_path.parent
    try:
        payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return ProofVerification(False, False, (f"invalid bundle JSON: {exc}",))
    errors: list[str] = []
    if payload.get("schema") != SCHEMA_VERSION:
        errors.append("unsupported schema")
    claimed_digest = payload.pop("bundle_sha256", None)
    actual_digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    if claimed_digest != actual_digest:
        errors.append("bundle digest mismatch")

    for role, record in payload.get("artifacts", {}).items():
        relative = Path(str(record.get("path", "")))
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            errors.append(f"artifact path escapes bundle: {role}")
            continue
        if not candidate.is_file():
            errors.append(f"artifact missing: {role}")
            continue
        if sha256_file(candidate) != record.get("sha256"):
            errors.append(f"artifact digest mismatch: {role}")
        if candidate.stat().st_size != record.get("bytes"):
            errors.append(f"artifact size mismatch: {role}")

    gates = payload.get("gates", {})
    expected_accepted = bool(gates) and all(value is True for value in gates.values())
    release = payload.get("release", {})
    if release.get("accepted") is not expected_accepted:
        errors.append("release decision does not match gates")
    expected_reasons = sorted(name for name, passed in gates.items() if not passed)
    if sorted(release.get("refused_reasons", [])) != expected_reasons:
        errors.append("release refusal reasons do not match gates")
    return ProofVerification(not errors, expected_accepted and not errors, tuple(errors))
