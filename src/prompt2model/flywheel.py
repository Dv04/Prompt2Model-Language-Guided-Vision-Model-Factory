"""Flywheel hook — the factory's models get better because they run.

When a deployed model abstains (conformal gate) or squeaks under a
confidence floor, that frame is exactly the training data the NEXT version
needs. :class:`HardCaseStore` captures those frames + a JSONL manifest;
``export_imagefolder`` turns them into retraining input; ``retrain_ready``
is the trigger. Labeling/curation of the captured pool is deliberately out
of scope here (that is the active-learning product, C1+C3) — this module is
the capture half B1 owns: nothing captured, nothing to learn from.

Bounded by construction: ``max_items`` caps the store (drop-newest, like
the factory's other queues) so a confused model can't fill a disk.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("prompt2model.flywheel")

MANIFEST_NAME = "manifest.jsonl"


class HardCaseStore:
    """Dataset-shaped store of hard (abstained / low-confidence) samples."""

    def __init__(
        self,
        root: str | Path,
        max_items: int = 1000,
        capture_below: float = 0.5,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_items = max_items
        # Confidence floor for capture when the model did NOT abstain (or is
        # uncalibrated): score under this ⇒ still worth keeping.
        self.capture_below = capture_below
        self._manifest_path = self.root / MANIFEST_NAME

    # -- write ---------------------------------------------------------------

    def should_capture(self, *, abstained: bool, confidence: float | None) -> bool:
        if abstained:
            return True
        return confidence is not None and confidence < self.capture_below

    def save(
        self,
        image: Any,
        *,
        prediction: str | None,
        confidence: float | None,
        abstained: bool,
        source: str = "edge",
        extra: dict[str, Any] | None = None,
    ) -> Path | None:
        """Persist one hard case (JPEG + manifest row). Returns the image
        path, or None when the store is full / the image can't be saved.
        Never raises — capture is best-effort by design."""
        try:
            if self.count() >= self.max_items:
                logger.debug("hard-case store full (%d) — dropping capture", self.max_items)
                return None
            stamp = f"{time.time():.6f}".replace(".", "_")
            image_path = self.root / f"hard_{stamp}.jpg"
            self._write_image(image, image_path)
            record = {
                "image": image_path.name,
                "prediction": prediction,
                "confidence": confidence,
                "abstained": abstained,
                "source": source,
                "captured_at": time.time(),
                **({"extra": extra} if extra else {}),
            }
            with self._manifest_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            return image_path
        except Exception as exc:  # noqa: BLE001
            logger.warning("hard-case capture failed: %s", exc)
            return None

    @staticmethod
    def _write_image(image: Any, path: Path) -> None:
        from PIL import Image

        if isinstance(image, (str, Path)):
            Image.open(image).convert("RGB").save(path, "JPEG")
            return
        if isinstance(image, Image.Image):
            image.convert("RGB").save(path, "JPEG")
            return
        import numpy as np

        array = np.asarray(image)
        if array.dtype != np.uint8:
            array = (np.clip(array, 0.0, 1.0) * 255).astype(np.uint8)
        Image.fromarray(array).convert("RGB").save(path, "JPEG")

    # -- read ----------------------------------------------------------------

    def records(self) -> list[dict[str, Any]]:
        if not self._manifest_path.exists():
            return []
        out = []
        for line in self._manifest_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def count(self) -> int:
        return len(self.records())

    def retrain_ready(self, threshold: int = 50) -> bool:
        return self.count() >= threshold

    def summary(self) -> dict[str, Any]:
        records = self.records()
        abstained = sum(1 for r in records if r.get("abstained"))
        by_prediction: dict[str, int] = {}
        for r in records:
            key = str(r.get("prediction") or "unknown")
            by_prediction[key] = by_prediction.get(key, 0) + 1
        return {
            "root": str(self.root),
            "count": len(records),
            "abstained": abstained,
            "by_prediction": by_prediction,
            "max_items": self.max_items,
        }

    # -- export --------------------------------------------------------------

    def export_imagefolder(self, output_dir: str | Path, pseudo_label: bool = False) -> Path:
        """Materialize the pool as an imagefolder for the retrain step.

        Default: everything under ``unlabeled/`` (hard cases need human or
        active-learning labels — that's the C-track's job). ``pseudo_label``
        buckets by the model's own prediction instead: cheap, biased, useful
        only for semi-supervised recipes — the manifest keeps the provenance
        either way.
        """
        output_dir = Path(output_dir)
        copied = 0
        for record in self.records():
            source = self.root / str(record.get("image", ""))
            if not source.exists():
                continue
            bucket = (
                str(record.get("prediction") or "unknown") if pseudo_label else "unlabeled"
            )
            destination = output_dir / bucket
            destination.mkdir(parents=True, exist_ok=True)
            (destination / source.name).write_bytes(source.read_bytes())
            copied += 1
        logger.info("flywheel export: %d hard cases → %s", copied, output_dir)
        return output_dir
