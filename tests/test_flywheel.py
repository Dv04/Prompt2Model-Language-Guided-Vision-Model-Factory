"""Flywheel hook: hard-case capture, manifest, bounds, export, edge wiring."""
from __future__ import annotations

import json

from PIL import Image

from prompt2model.flywheel import HardCaseStore


def _img(color=(200, 30, 30)):
    return Image.new("RGB", (16, 16), color)


def test_save_and_manifest(tmp_path):
    store = HardCaseStore(tmp_path / "store")
    path = store.save(_img(), prediction="cat", confidence=0.3, abstained=True)
    assert path is not None and path.exists()
    records = store.records()
    assert len(records) == 1
    assert records[0]["prediction"] == "cat"
    assert records[0]["abstained"] is True
    assert store.count() == 1


def test_capture_policy():
    store = HardCaseStore("unused", capture_below=0.5)
    assert store.should_capture(abstained=True, confidence=0.99)
    assert store.should_capture(abstained=False, confidence=0.4)
    assert not store.should_capture(abstained=False, confidence=0.9)
    assert not store.should_capture(abstained=False, confidence=None)


def test_store_is_bounded(tmp_path):
    store = HardCaseStore(tmp_path / "store", max_items=2)
    assert store.save(_img(), prediction="a", confidence=0.1, abstained=False) is not None
    assert store.save(_img(), prediction="b", confidence=0.1, abstained=False) is not None
    assert store.save(_img(), prediction="c", confidence=0.1, abstained=False) is None
    assert store.count() == 2


def test_retrain_ready_and_summary(tmp_path):
    store = HardCaseStore(tmp_path / "store")
    for i in range(3):
        store.save(_img(), prediction="cat" if i < 2 else "dog", confidence=0.2, abstained=i == 0)
    assert not store.retrain_ready(threshold=5)
    assert store.retrain_ready(threshold=3)
    summary = store.summary()
    assert summary["count"] == 3 and summary["abstained"] == 1
    assert summary["by_prediction"] == {"cat": 2, "dog": 1}


def test_export_imagefolder(tmp_path):
    store = HardCaseStore(tmp_path / "store")
    store.save(_img(), prediction="cat", confidence=0.2, abstained=True)
    store.save(_img((30, 200, 30)), prediction="dog", confidence=0.3, abstained=False)

    unlabeled = store.export_imagefolder(tmp_path / "export_unlabeled")
    assert len(list((unlabeled / "unlabeled").glob("*.jpg"))) == 2

    pseudo = store.export_imagefolder(tmp_path / "export_pseudo", pseudo_label=True)
    assert len(list((pseudo / "cat").glob("*.jpg"))) == 1
    assert len(list((pseudo / "dog").glob("*.jpg"))) == 1


def test_corrupt_manifest_lines_are_skipped(tmp_path):
    store = HardCaseStore(tmp_path / "store")
    store.save(_img(), prediction="cat", confidence=0.2, abstained=False)
    with (tmp_path / "store" / "manifest.jsonl").open("a") as handle:
        handle.write("NOT JSON\n")
    assert store.count() == 1  # corrupt line ignored, store still usable


def test_edge_model_captures_hard_cases(tmp_path):
    """EdgeModel + threshold-0 calibration ⇒ every frame abstains ⇒ captured."""
    import torch
    import torch.nn as nn

    from prompt2model.edge_inference import EdgeModel
    from prompt2model.exporting import inject_metadata

    model = nn.Sequential(nn.Conv2d(3, 2, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten()).eval()
    onnx_path = tmp_path / "m.onnx"
    torch.onnx.export(model, torch.rand(1, 3, 16, 16), str(onnx_path), opset_version=17, dynamo=False)
    inject_metadata(
        str(onnx_path),
        {
            "task": "classification",
            "image_size": 16,
            "labels": ["a", "b"],
            "calibration": {"calibrated": True, "temperature": 1.0, "conformal_threshold": 0.0},
        },
    )
    image_path = tmp_path / "frame.png"
    _img().save(image_path)

    store = HardCaseStore(tmp_path / "store")
    result = EdgeModel(onnx_path).run_inference(image_path, hard_case_store=store)
    assert result["abstained"] is True
    assert store.count() == 1
    record = store.records()[0]
    assert record["abstained"] is True and record["source"].endswith("frame.png")


def test_flywheel_cli(tmp_path, capsys):
    from prompt2model.cli import main
    import sys

    store = HardCaseStore(tmp_path / "store")
    store.save(_img(), prediction="cat", confidence=0.2, abstained=True)

    argv = sys.argv
    try:
        sys.argv = ["prompt2model", "flywheel", "--store", str(tmp_path / "store")]
        main()
        status = json.loads(capsys.readouterr().out)
        assert status["count"] == 1

        sys.argv = [
            "prompt2model", "flywheel", "--store", str(tmp_path / "store"),
            "--action", "export", "--output-dir", str(tmp_path / "exported"),
        ]
        main()
        exported = json.loads(capsys.readouterr().out)
        assert exported["count"] == 1
        assert (tmp_path / "exported" / "unlabeled").exists()
    finally:
        sys.argv = argv
