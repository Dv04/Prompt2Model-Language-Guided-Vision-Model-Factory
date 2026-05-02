"""Week 6 — Madhuvani: Zero-configuration edge inference script.

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
        self.image_size = int(self.metadata.get("image_size", "128"))
        self.mean = np.array(json.loads(self.metadata.get("normalization_mean", "[0.485, 0.456, 0.406]")), dtype=np.float32)
        self.std = np.array(json.loads(self.metadata.get("normalization_std", "[0.229, 0.224, 0.225]")), dtype=np.float32)
        self.labels = json.loads(self.metadata.get("labels", "[]"))

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

    def run_inference(self, image_path: str | Path) -> dict[str, Any]:
        """Load image, preprocess using metadata, and run inference."""
        image = Image.open(image_path)
        input_tensor = self.preprocess(image)
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_tensor})

        if self.task == "classification":
            logits = outputs[0][0]
            # Softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()
            best_idx = int(np.argmax(probs))
            return {
                "label": self.labels[best_idx] if self.labels else str(best_idx),
                "score": float(probs[best_idx]),
                "all_scores": {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}
                if self.labels
                else {},
            }
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
