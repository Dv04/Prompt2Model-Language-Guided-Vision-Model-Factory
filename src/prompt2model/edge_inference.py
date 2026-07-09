"""Week 6 - Madhuvani: Zero-configuration edge inference script.

Uses only ONNX Runtime and embedded metadata to run inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image


class EdgeModel:
    """Wrapper that loads and runs an ONNX model using embedded metadata."""

    def __init__(self, model_path: str | Path) -> None:
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.metadata = self._read_metadata()
        self.task = self.metadata.get("task", "classification")

        # Canonical metadata keys are the ones the real export path writes
        # (``exporting.build_metadata_props``, used by ``pipeline.py`` and
        # ``tuning.py``): "input_resolution" (a [height, width] list),
        # "mean", "std". The legacy names below ("image_size",
        # "normalization_mean", "normalization_std") were only ever produced
        # by the now-unused ``exporting.build_full_metadata`` helper, but are
        # accepted here as a tolerant fallback in case an artifact was built
        # with it. Every real pipeline artifact carries the canonical keys.
        resolution = self.metadata.get("input_resolution")
        if resolution is not None:
            self.image_size = int(json.loads(resolution)[0])
        else:
            self.image_size = int(self.metadata.get("image_size", "128"))

        mean = self.metadata.get("mean")
        if mean is None:
            mean = self.metadata.get("normalization_mean", "[0.485, 0.456, 0.406]")
        self.mean = np.array(json.loads(mean), dtype=np.float32)

        std = self.metadata.get("std")
        if std is None:
            std = self.metadata.get("normalization_std", "[0.229, 0.224, 0.225]")
        self.std = np.array(json.loads(std), dtype=np.float32)

        self.labels = json.loads(self.metadata.get("labels", "[]"))
        # B1 phase 4 - the calibration block the factory embedded (may be
        # absent on artifacts from older runs → uncalibrated behaviour).
        try:
            self.calibration = json.loads(self.metadata.get("calibration", "{}") or "{}")
        except json.JSONDecodeError:
            self.calibration = {}

    def _read_metadata(self) -> dict[str, str]:
        model = self.session.get_modelmeta()
        return model.custom_metadata_map

    def preprocess(self, image: Image.Image) -> np.ndarray:
        # 1. Resize
        img = image.convert("RGB").resize((self.image_size, self.image_size))
        # 2. To Tensor / Float
        img_np = np.array(img).astype(np.float32) / 255.0
        # 3. Normalize
        img_np = (img_np - self.mean) / self.std
        # 4. Transpose (H, W, C) -> (C, H, W)
        img_np = img_np.transpose(2, 0, 1)
        # 5. Batch dimension
        return np.expand_dims(img_np, axis=0).astype(np.float32)

    def run_inference(self, image_path: str | Path, hard_case_store: Any | None = None) -> dict[str, Any]:
        """Load image, preprocess using metadata, and run inference.

        When the artifact carries a calibration block, logits are
        temperature-scaled and the conformal threshold yields ``abstained``
 - the model says "I don't know" instead of guessing. Passing a
        ``flywheel.HardCaseStore`` captures abstained/low-confidence frames
        for the retrain loop.
        """
        image = Image.open(image_path)
        input_tensor = self.preprocess(image)
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_tensor})

        if self.task == "classification":
            logits = outputs[0][0]
            calibrated = bool(self.calibration.get("calibrated"))
            temperature = float(self.calibration.get("temperature", 1.0) or 1.0) if calibrated else 1.0
            scaled = logits / max(temperature, 1e-6)
            exp_logits = np.exp(scaled - np.max(scaled))
            probs = exp_logits / exp_logits.sum()
            best_idx = int(np.argmax(probs))
            score = float(probs[best_idx])
            abstained = False
            if calibrated:
                threshold = float(self.calibration.get("conformal_threshold", float("inf")))
                abstained = (1.0 - score) > threshold
            result = {
                "label": self.labels[best_idx] if self.labels else str(best_idx),
                "score": score,
                "all_scores": {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}
                if self.labels
                else {},
                "calibrated": calibrated,
                "abstained": abstained,
            }
            if hard_case_store is not None and hard_case_store.should_capture(
                abstained=abstained, confidence=score
            ):
                hard_case_store.save(
                    image,
                    prediction=result["label"],
                    confidence=score,
                    abstained=abstained,
                    source=str(image_path),
                )
            return result
        else:
            # Detection: [boxes, scores, labels]
            boxes, scores, labels = outputs
            results = []
            for i in range(len(scores[0])):
                if scores[0][i] > 0.3:  # Threshold
                    results.append({
                        "box": boxes[0][i].tolist(),
                        "score": float(scores[0][i]),
                        "label": self.labels[int(labels[0][i])]
                        if len(self.labels) > int(labels[0][i])
                        else str(int(labels[0][i])),
                    })
            return {"detections": results}


def load_model_from_onnx(path: str | Path) -> EdgeModel:
    return EdgeModel(path)
