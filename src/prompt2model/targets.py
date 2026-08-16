"""Pluggable deployment targets - the factory compiles to a TARGET, not a box.

ONNX Runtime is the universal backend: every artifact the factory emits runs
there unmodified. Accelerator-specific backends (TensorRT first) are one
registry entry each: they take the final ONNX artifact and produce whatever
that runtime needs - building it on the spot when the toolchain is present,
otherwise emitting a reproducible build recipe to run on the target device.
A Jetson is ONE choice here, not an assumption.

Adding a target = subclass :class:`ExportTarget`, register an alias.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("prompt2model.targets")


class UnknownTargetError(ValueError):
    """Raised before a run mutates state for an unsupported deployment target."""


@dataclass
class TargetArtifact:
    target: str            # registry name, e.g. "onnxruntime" | "tensorrt"
    runtime: str           # runtime the artifact runs in
    artifact_path: str | None  # deployable target artifact; None until built
    built: bool            # True when the deployable artifact exists locally
    recipe_path: str | None = None  # build script for on-device compilation
    notes: str = ""
    source_artifact_path: str | None = None  # compiler input, never implied deployable

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "runtime": self.runtime,
            "artifact_path": self.artifact_path,
            "built": self.built,
            "recipe_path": self.recipe_path,
            "notes": self.notes,
            "source_artifact_path": self.source_artifact_path,
        }


class ExportTarget:
    """One deployment backend. ``prepare`` must never raise into the
    pipeline - a failed build returns a not-built artifact with notes."""

    name = "abstract"

    def prepare(self, onnx_path: str | Path, run_dir: str | Path, metadata: dict[str, Any]) -> TargetArtifact:
        raise NotImplementedError


class OnnxRuntimeTarget(ExportTarget):
    """The universal target: the ONNX file itself, verified to load."""

    name = "onnxruntime"

    def prepare(self, onnx_path: str | Path, run_dir: str | Path, metadata: dict[str, Any]) -> TargetArtifact:
        onnx_path = Path(onnx_path)
        try:
            import onnxruntime as ort

            ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            return TargetArtifact(
                target=self.name,
                runtime="onnxruntime",
                artifact_path=str(onnx_path),
                built=True,
                notes="runs anywhere ONNX Runtime runs (CPU/GPU/NPU EPs)",
                source_artifact_path=str(onnx_path),
            )
        except Exception as exc:  # noqa: BLE001 - never raise into the pipeline
            return TargetArtifact(
                target=self.name,
                runtime="onnxruntime",
                artifact_path=None,
                built=False,
                notes=f"ONNX Runtime failed to load the artifact: {exc}",
                source_artifact_path=str(onnx_path),
            )


# Well-known trtexec locations checked after PATH (JetPack installs it
# outside PATH by default).
_TRTEXEC_FALLBACKS = ("/usr/src/tensorrt/bin/trtexec",)


def _find_trtexec() -> str | None:
    found = shutil.which("trtexec")
    if found:
        return found
    for candidate in _TRTEXEC_FALLBACKS:
        if Path(candidate).exists():
            return candidate
    return None


class TensorRTTarget(ExportTarget):
    """NVIDIA TensorRT: builds the engine when ``trtexec`` is available,
    otherwise emits ``build_tensorrt.sh`` to run on the target device.

    TensorRT engines are device-specific (compute capability + TRT version),
    so building on the deployment device via the recipe is the NORMAL path;
    a local build only happens when the factory itself runs on such a device.
    """

    name = "tensorrt"

    def __init__(self, fp16: bool = True, workspace_mb: int = 2048, build_timeout_s: int = 1800) -> None:
        self.fp16 = fp16
        self.workspace_mb = workspace_mb
        self.build_timeout_s = build_timeout_s

    def _command(self, onnx_path: Path, engine_path: Path) -> list[str]:
        command = [
            "trtexec",
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            f"--memPoolSize=workspace:{self.workspace_mb}",
        ]
        if self.fp16:
            command.append("--fp16")
        return command

    def prepare(self, onnx_path: str | Path, run_dir: str | Path, metadata: dict[str, Any]) -> TargetArtifact:
        onnx_path, run_dir = Path(onnx_path), Path(run_dir)
        engine_path = run_dir / (onnx_path.stem + ".engine")
        recipe_path = run_dir / "build_tensorrt.sh"

        command = self._command(onnx_path, engine_path)
        recipe_path.write_text(
            "#!/bin/sh\n"
            "# Build the TensorRT engine ON THE TARGET DEVICE (engines are\n"
            "# specific to the GPU + TensorRT version they are built on).\n"
            "# trtexec ships with TensorRT (JetPack: /usr/src/tensorrt/bin).\n"
            f"{' '.join(command)}\n"
        )
        recipe_path.chmod(0o755)

        trtexec = _find_trtexec()
        if trtexec is None:
            return TargetArtifact(
                target=self.name,
                runtime="tensorrt",
                artifact_path=None,
                built=False,
                recipe_path=str(recipe_path),
                notes="trtexec not found on this host - run build_tensorrt.sh on the target device",
                source_artifact_path=str(onnx_path),
            )

        try:
            completed = subprocess.run(
                [trtexec, *command[1:]],
                capture_output=True,
                timeout=self.build_timeout_s,
                check=False,
            )
            if completed.returncode == 0 and engine_path.exists():
                return TargetArtifact(
                    target=self.name,
                    runtime="tensorrt",
                    artifact_path=str(engine_path),
                    built=True,
                    recipe_path=str(recipe_path),
                    notes="engine built locally; valid only for this GPU + TensorRT version",
                    source_artifact_path=str(onnx_path),
                )
            tail = completed.stderr.decode("utf-8", errors="replace")[-400:]
            return TargetArtifact(
                target=self.name,
                runtime="tensorrt",
                artifact_path=None,
                built=False,
                recipe_path=str(recipe_path),
                notes=f"trtexec failed (rc={completed.returncode}): {tail}",
                source_artifact_path=str(onnx_path),
            )
        except Exception as exc:  # noqa: BLE001
            return TargetArtifact(
                target=self.name,
                runtime="tensorrt",
                artifact_path=None,
                built=False,
                recipe_path=str(recipe_path),
                notes=f"trtexec invocation error: {exc}",
                source_artifact_path=str(onnx_path),
            )


_TARGETS: dict[str, ExportTarget] = {}
_ALIASES: dict[str, str] = {
    "onnxruntime": "onnxruntime",
    "onnx": "onnxruntime",
    "ort": "onnxruntime",
    "cpu": "onnxruntime",
    "host": "onnxruntime",
    "tensorrt": "tensorrt",
    "trt": "tensorrt",
    "jetson": "tensorrt",
    "orin": "tensorrt",
}


def register_target(name: str, target: ExportTarget, aliases: tuple[str, ...] = ()) -> None:
    _TARGETS[name] = target
    _ALIASES[name] = name
    for alias in aliases:
        _ALIASES[alias.lower()] = name


register_target("onnxruntime", OnnxRuntimeTarget())
register_target("tensorrt", TensorRTTarget())


def resolve_target(name: str | None) -> ExportTarget:
    """Resolve an explicitly supported target or refuse before compilation.

    Falling back from an unknown accelerator to ONNX Runtime would falsely
    imply target compatibility, so unsupported and blank names are errors.
    ``None`` remains the deliberate default ONNX Runtime target.
    """
    key = "onnxruntime" if name is None else name.strip().lower()
    canonical = _ALIASES.get(key)
    if canonical is None:
        supported = ", ".join(sorted(_ALIASES))
        raise UnknownTargetError(
            f"unsupported deployment target {name!r}; supported targets: {supported}"
        )
    return _TARGETS[canonical]
