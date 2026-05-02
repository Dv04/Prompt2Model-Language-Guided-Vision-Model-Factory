"""Tests for the edge_infer.py autonomous edge-inference script.

These tests verify:
1. Metadata loading and parsing from a real ONNX file
2. Input tensor shape/dtype from both real-image and dummy-input paths
3. Classification result interpretation against known logits
4. Detection result interpretation with score filtering
5. Full end-to-end invocation against the smoke ONNX artifacts
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

# Make the scripts/ directory importable without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import edge_infer  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

SMOKE_CLS_ONNX = Path("output/smoke/classification_run/model.onnx")
SMOKE_DET_ONNX = Path("output/smoke/detection_run/model.onnx")


def _make_minimal_onnx(tmp_path: Path, metadata: dict) -> Path:
    """Create a tiny valid ONNX (Identity) with injected metadata."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 96, 96])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 96, 96])
    node = helper.make_node("Identity", inputs=["X"], outputs=["Y"])
    graph = helper.make_graph([node], "g", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    # Inject metadata
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
    out = tmp_path / "test_model.onnx"
    onnx.save(model, str(out))
    return out


FULL_META = {
    "task": "classification",
    "model_name": "mobilenet_v3_small",
    "prompt": "Classify cats and dogs.",
    "input_resolution": [96, 96],
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "class_dict": {"0": "cat", "1": "dog"},
    "labels": ["cat", "dog"],
    "label_map": {"cat": "cat", "dog": "dog"},
}


# ──────────────────────────────────────────────────────────────────────────────
# 1 — Metadata loading
# ──────────────────────────────────────────────────────────────────────────────

def test_load_metadata_parses_all_keys(tmp_path: Path) -> None:
    onnx_path = _make_minimal_onnx(tmp_path, FULL_META)
    meta = edge_infer.load_metadata(str(onnx_path))

    assert meta["task"] == "classification"
    assert meta["input_resolution"] == [96, 96]
    assert meta["mean"] == [0.485, 0.456, 0.406]
    assert meta["std"] == [0.229, 0.224, 0.225]
    assert meta["class_dict"] == {"0": "cat", "1": "dog"}
    assert meta["label_map"] == {"cat": "cat", "dog": "dog"}


def test_load_metadata_dies_on_empty_props(tmp_path: Path) -> None:
    """A model with zero metadata_props should cause SystemExit(1)."""
    onnx_path = _make_minimal_onnx(tmp_path, {})
    with pytest.raises(SystemExit) as exc_info:
        edge_infer.load_metadata(str(onnx_path))
    assert exc_info.value.code == 1


def test_load_metadata_dies_on_missing_required_key(tmp_path: Path) -> None:
    bad_meta = {k: v for k, v in FULL_META.items() if k != "input_resolution"}
    onnx_path = _make_minimal_onnx(tmp_path, bad_meta)
    with pytest.raises(SystemExit) as exc_info:
        edge_infer.load_metadata(str(onnx_path))
    assert exc_info.value.code == 1


# ──────────────────────────────────────────────────────────────────────────────
# 2 — Dummy input generation
# ──────────────────────────────────────────────────────────────────────────────

def test_make_dummy_input_shape_and_dtype() -> None:
    arr = edge_infer.make_dummy_input([128, 128], seed=0)
    assert arr.shape == (1, 3, 128, 128), f"Unexpected shape: {arr.shape}"
    assert arr.dtype == np.float32


def test_make_dummy_input_is_deterministic() -> None:
    a = edge_infer.make_dummy_input([64, 64], seed=7)
    b = edge_infer.make_dummy_input([64, 64], seed=7)
    np.testing.assert_array_equal(a, b)


def test_make_dummy_input_different_seeds_differ() -> None:
    a = edge_infer.make_dummy_input([64, 64], seed=1)
    b = edge_infer.make_dummy_input([64, 64], seed=2)
    assert not np.array_equal(a, b)


# ──────────────────────────────────────────────────────────────────────────────
# 3 — Real-image preprocessing (uses embedded resolution & norm stats)
# ──────────────────────────────────────────────────────────────────────────────

def test_preprocess_image_shape_and_dtype(tmp_path: Path) -> None:
    from PIL import Image

    # Create a tiny synthetic PNG
    img = Image.new("RGB", (200, 150), color=(128, 64, 32))
    img_path = tmp_path / "test.png"
    img.save(img_path)

    arr = edge_infer.preprocess_image(
        str(img_path),
        input_resolution=[96, 96],
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    assert arr.shape == (1, 3, 96, 96), f"Unexpected shape: {arr.shape}"
    assert arr.dtype == np.float32


def test_preprocess_image_applies_normalisation(tmp_path: Path) -> None:
    """A pure-white image should produce a specific normalised value."""
    from PIL import Image

    img = Image.new("RGB", (10, 10), color=(255, 255, 255))
    img_path = tmp_path / "white.png"
    img.save(img_path)

    arr = edge_infer.preprocess_image(
        str(img_path),
        input_resolution=[10, 10],
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )
    # (1.0 - 0.5) / 0.5 = 1.0
    np.testing.assert_allclose(arr, 1.0, atol=1e-5)


def test_preprocess_image_missing_file_exits_2() -> None:
    with pytest.raises(SystemExit) as exc_info:
        edge_infer.preprocess_image(
            "/nonexistent/path/image.png",
            [96, 96],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        )
    assert exc_info.value.code == 2


# ──────────────────────────────────────────────────────────────────────────────
# 4 — Classification result interpretation
# ──────────────────────────────────────────────────────────────────────────────

def test_interpret_classification_top1_is_highest_logit(capsys: pytest.CaptureFixture) -> None:
    # logits: class "dog" (index 1) has highest value
    logits = np.array([[0.1, 2.5, -0.3]], dtype=np.float32)
    class_dict = {"0": "cat", "1": "dog", "2": "bird"}
    edge_infer.interpret_classification([logits], class_dict, topk=3)
    captured = capsys.readouterr()
    # Data rows contain a "%" sign; filter to those only
    data_lines = [l for l in captured.out.splitlines() if "%" in l]
    # First data row should be the highest-scoring class: dog
    assert "dog" in data_lines[0]


def test_interpret_classification_topk_limits_output(capsys: pytest.CaptureFixture) -> None:
    logits = np.array([[1.0, 0.5, 0.2, -0.1]], dtype=np.float32)
    class_dict = {"0": "a", "1": "b", "2": "c", "3": "d"}
    edge_infer.interpret_classification([logits], class_dict, topk=2)
    captured = capsys.readouterr()
    # Data rows contain a "%" sign; filter to those only
    data_lines = [l for l in captured.out.splitlines() if "%" in l]
    assert len(data_lines) == 2


# ──────────────────────────────────────────────────────────────────────────────
# 5 — Detection result interpretation
# ──────────────────────────────────────────────────────────────────────────────

def test_interpret_detection_filters_by_threshold(capsys: pytest.CaptureFixture) -> None:
    # Two boxes: one above, one below threshold
    boxes  = np.array([[10, 10, 50, 50], [5, 5, 20, 20]], dtype=np.float32)
    scores = np.array([0.85, 0.15], dtype=np.float32)
    labels = np.array([1, 2], dtype=np.int64)
    class_dict = {"0": "square", "1": "circle"}

    edge_infer.interpret_detection(
        [boxes, scores, labels],
        class_dict,
        score_threshold=0.3,
        topk=5,
    )
    captured = capsys.readouterr()
    # Only the 0.85-score box should appear
    assert "circle" in captured.out or "0.850" in captured.out
    # The 0.15-score box should be filtered out
    assert "0.150" not in captured.out


def test_interpret_detection_no_detections_above_threshold(capsys: pytest.CaptureFixture) -> None:
    boxes  = np.zeros((3, 4), dtype=np.float32)
    scores = np.array([0.05, 0.10, 0.08], dtype=np.float32)
    labels = np.array([1, 1, 2], dtype=np.int64)
    class_dict = {"0": "square", "1": "circle"}

    edge_infer.interpret_detection(
        [boxes, scores, labels],
        class_dict,
        score_threshold=0.5,
        topk=5,
    )
    captured = capsys.readouterr()
    assert "No detections" in captured.out


# ──────────────────────────────────────────────────────────────────────────────
# 6 — Full end-to-end CLI invocation against smoke ONNX artifacts
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not SMOKE_CLS_ONNX.exists(),
    reason="Smoke classification ONNX not found — run the pipeline first.",
)
def test_end_to_end_classification_dummy(capsys: pytest.CaptureFixture) -> None:
    """Run the full main() against the classification smoke ONNX with a dummy input."""
    edge_infer.main([
        "--model", str(SMOKE_CLS_ONNX),
        "--no-metadata-dump",
        "--topk", "3",
    ])
    captured = capsys.readouterr()
    assert "Inference latency" in captured.out
    assert "Classification Results" in captured.out
    assert "completed successfully" in captured.out


@pytest.mark.skipif(
    not SMOKE_DET_ONNX.exists(),
    reason="Smoke detection ONNX not found — run the pipeline first.",
)
def test_end_to_end_detection_dummy(capsys: pytest.CaptureFixture) -> None:
    """Run the full main() against the detection smoke ONNX with a dummy input."""
    edge_infer.main([
        "--model", str(SMOKE_DET_ONNX),
        "--no-metadata-dump",
        "--score-threshold", "0.05",
        "--topk", "5",
    ])
    captured = capsys.readouterr()
    assert "Inference latency" in captured.out
    assert "Detection Results" in captured.out
    assert "completed successfully" in captured.out


@pytest.mark.skipif(
    not SMOKE_CLS_ONNX.exists(),
    reason="Smoke classification ONNX not found.",
)
def test_end_to_end_uses_only_embedded_metadata(capsys: pytest.CaptureFixture) -> None:
    """Confirm the script reads input_resolution/mean/std from ONNX, not hard-coded values."""
    edge_infer.main([
        "--model", str(SMOKE_CLS_ONNX),
        "--no-metadata-dump",
    ])
    captured = capsys.readouterr()
    # The script must print the embedded resolution and norm stats
    assert "96" in captured.out          # input_resolution from metadata
    assert "0.485" in captured.out       # mean from metadata
    assert "0.229" in captured.out       # std from metadata
