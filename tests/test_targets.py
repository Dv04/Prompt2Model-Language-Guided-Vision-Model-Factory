"""Deployment targets: registry resolution, ORT universal path, TRT recipe/build."""
from __future__ import annotations

import os
import stat
from pathlib import Path

import torch
import torch.nn as nn
import pytest

from prompt2model.targets import (
    OnnxRuntimeTarget,
    TensorRTTarget,
    UnknownTargetError,
    resolve_target,
)


def _export_tiny_onnx(path: Path) -> Path:
    model = nn.Sequential(nn.Conv2d(3, 2, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten()).eval()
    torch.onnx.export(model, torch.rand(1, 3, 16, 16), str(path), opset_version=17, dynamo=False)
    return path


def test_resolve_aliases():
    assert resolve_target("cpu").name == "onnxruntime"
    assert resolve_target("host").name == "onnxruntime"
    assert resolve_target(None).name == "onnxruntime"
    assert resolve_target("jetson").name == "tensorrt"
    assert resolve_target("TRT").name == "tensorrt"
    assert resolve_target("orin").name == "tensorrt"
    with pytest.raises(UnknownTargetError, match="quantum-npu-9000"):
        resolve_target("quantum-npu-9000")
    with pytest.raises(UnknownTargetError, match="unsupported deployment target"):
        resolve_target(" ")


def test_unknown_target_refuses_before_pipeline_side_effect(tmp_path):
    from prompt2model.config import PipelineConfig
    from prompt2model.pipeline import Prompt2ModelFactory

    output = tmp_path / "must_not_exist"
    config = PipelineConfig.model_validate(
        {
            "prompt": "classify widgets",
            "task": "classification",
            "labels": [{"name": "widget"}],
            "dataset": {"root": str(tmp_path)},
            "data_context": {"deployment_target": "quantum-npu-9000"},
            "export": {"output_dir": str(output)},
        }
    )
    with pytest.raises(UnknownTargetError):
        Prompt2ModelFactory().run(config)
    assert not output.exists()


def test_onnxruntime_target_builds(tmp_path):
    onnx_path = _export_tiny_onnx(tmp_path / "m.onnx")
    artifact = OnnxRuntimeTarget().prepare(onnx_path, tmp_path, {})
    assert artifact.built is True
    assert artifact.runtime == "onnxruntime"
    assert artifact.artifact_path == str(onnx_path)


def test_onnxruntime_target_bad_file_reports_not_built(tmp_path):
    bad = tmp_path / "not_a_model.onnx"
    bad.write_text("garbage")
    artifact = OnnxRuntimeTarget().prepare(bad, tmp_path, {})
    assert artifact.built is False and artifact.notes


def test_tensorrt_target_emits_recipe_without_trtexec(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "emptybin"))  # no trtexec anywhere
    monkeypatch.setattr("prompt2model.targets._TRTEXEC_FALLBACKS", ())
    onnx_path = _export_tiny_onnx(tmp_path / "m.onnx")
    artifact = TensorRTTarget().prepare(onnx_path, tmp_path, {})
    assert artifact.built is False
    assert artifact.runtime == "tensorrt"
    assert artifact.artifact_path is None
    assert artifact.source_artifact_path == str(onnx_path)
    recipe = Path(artifact.recipe_path)
    assert recipe.exists()
    content = recipe.read_text()
    assert f"--onnx={onnx_path}" in content
    assert "--saveEngine=" in content and "--fp16" in content
    assert os.access(recipe, os.X_OK)


def test_tensorrt_target_builds_with_fake_trtexec(tmp_path, monkeypatch):
    """A fake trtexec on PATH that writes the engine file exercises the
    local-build path without needing TensorRT."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "trtexec"
    fake.write_text(
        "#!/bin/sh\n"
        'for arg in "$@"; do case "$arg" in --saveEngine=*) '
        'touch "${arg#--saveEngine=}";; esac; done\n'
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    # Fake bin FIRST, but keep the system dirs - the fake script needs `touch`.
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")

    onnx_path = _export_tiny_onnx(tmp_path / "m.onnx")
    artifact = TensorRTTarget().prepare(onnx_path, tmp_path, {})
    assert artifact.built is True
    assert artifact.artifact_path.endswith(".engine")
    assert Path(artifact.artifact_path).exists()


def test_end_to_end_run_with_tensorrt_target(tmp_path, monkeypatch):
    """Full pipeline with target=jetson on a host without trtexec: the run
    must still succeed, with the recipe emitted and built=False recorded."""
    monkeypatch.setattr("prompt2model.targets._TRTEXEC_FALLBACKS", ())
    monkeypatch.setenv("PATH", "/usr/bin:/bin")  # no trtexec on CI/mac hosts

    from prompt2model.config import DatasetConfig, DatasetFormat, TrainingConfig
    from prompt2model.data import create_synthetic_classification_dataset
    from prompt2model.pipeline import run_from_prompt

    data_dir = create_synthetic_classification_dataset(str(tmp_path / "data"))
    result = run_from_prompt(
        prompt='Classify "red square" and "blue circle" images, prioritize speed.',
        dataset=DatasetConfig(root=str(data_dir), format=DatasetFormat.IMAGEFOLDER, image_size=64),
        output_dir=str(tmp_path / "run"),
        training_overrides=TrainingConfig(epochs=1, batch_size=4, max_steps_per_epoch=2),
        target="jetson",
    )
    assert result.deployment is not None
    assert result.deployment["runtime"] == "tensorrt"
    assert result.deployment["built"] is False
    assert result.deployment["artifact_path"] is None
    assert result.deployment["source_artifact_path"].endswith("model.onnx")
    assert Path(result.deployment["recipe_path"]).exists()
