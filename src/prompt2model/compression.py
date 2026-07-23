"""Compression stage - distill → quantize → accuracy-floor gate.

The factory's promise is not "a smaller model", it is "a smaller model OR a
refusal": every compressed artifact is evaluated on the validation split in
the SAME runtime it ships in (ONNX Runtime), and if it can't hold the stated
accuracy floor the factory keeps the uncompressed artifact and says so. The
gate result travels with the artifact (ONNX metadata + run report).

Pieces:

* :func:`train_distilled_classification` - knowledge distillation for the
  classification path: soft-target KL (temperature-scaled) + hard-label CE.
  Teacher is trained (or loaded) from the registry's accuracy tier.
* :func:`quantize_onnx` - INT8 post-training quantization of an exported
  ONNX file via onnxruntime.quantization (dynamic = weights-only, works
  anywhere; static = activation calibration from the validation loader).
* :func:`evaluate_onnx_classification` - accuracy of an ONNX file over a
  torch DataLoader, run through ONNX Runtime (the honest, same-runtime gate).
* :func:`decide_gate` / :func:`apply_compression` - the floor decision and
  the orchestration.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prompt2model.config import CompressionConfig, ModelConstraints

logger = logging.getLogger("prompt2model.compression")


# ── knowledge distillation (classification) ────────────────────────────────


def train_distilled_classification(
    teacher: Any,
    student: Any,
    train_loader: Any,
    val_loader: Any,
    training_config: Any,
    compression_config: CompressionConfig,
    output_dir: Path,
    device: Any,
) -> dict[str, Any]:
    """KD fine-tuning: student mimics the teacher's temperature-softened
    distribution while still learning the hard labels.

    loss = alpha * T^2 * KL(student_T || teacher_T) + (1 - alpha) * CE

    Returns a dict with checkpoint_path, history and best_val_accuracy - 
    intentionally the same shape the reporting layer already understands.
    """
    import torch
    import torch.nn.functional as F
    from torch.optim.swa_utils import update_bn

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_distilled_model.pt"

    temperature = compression_config.distillation_temperature
    alpha = compression_config.distillation_alpha

    teacher = teacher.to(device).eval()
    student = student.to(device)
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    history: list[dict[str, float]] = []
    best_val_accuracy = 0.0
    started = time.time()

    for epoch in range(training_config.epochs):
        student.train()
        epoch_loss, steps = 0.0, 0
        for step, (images, targets) in enumerate(train_loader):
            if (
                training_config.max_steps_per_epoch is not None
                and step >= training_config.max_steps_per_epoch
            ):
                break
            images, targets = images.to(device), targets.to(device)
            with torch.no_grad():
                teacher_logits = teacher(images)
            student_logits = student(images)
            soft = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1),
                reduction="batchmean",
            ) * (temperature * temperature)
            hard = F.cross_entropy(student_logits, targets)
            loss = alpha * soft + (1.0 - alpha) * hard
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            steps += 1

        # BatchNorm running_mean/running_var are updated only as a
        # gradient-free exponential moving average during .train()-mode
        # forward passes; they are a SEPARATE mechanism from the
        # backprop/optimizer step above. With small batches, few epochs, or
        # a from-scratch (non-pretrained) backbone, that moving average
        # never converges to the real activation statistics, so .eval()
        # (below - and every downstream evaluator: the ONNX export, the
        # accuracy-floor gate, the deployed EdgeModel) can silently collapse
        # to an input-independent constant output EVEN THOUGH the
        # .train()-mode loss keeps falling normally each step, since the
        # loss computation above never touches the stale running stats.
        # This is the exact "chance-level despite falling train_loss" bug:
        # this repo already diagnosed and patched one instance of it for the
        # CLI smoke-test path (commit ba62c2f - "backbone collapses to an
        # input-independent constant output"), but train_distilled_classification
        # inherited the same vulnerability since it was never defended here.
        # update_bn() recomputes fresh running statistics from a genuine
        # data pass (temporarily forcing cumulative-average momentum) using
        # the CURRENT weights, independent of how many training epochs ran
        # or whether the backbone started pretrained.
        update_bn(val_loader, student, device=device)

        student.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, targets in val_loader:
                logits = student(images.to(device))
                correct += int((logits.argmax(dim=1).cpu() == targets).sum())
                total += int(targets.numel())
        val_accuracy = correct / total if total else 0.0
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / max(steps, 1),
                "val_accuracy": val_accuracy,
            }
        )
        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            import torch as _torch

            _torch.save(student.state_dict(), checkpoint_path)

    return {
        "checkpoint_path": str(checkpoint_path),
        "history": history,
        "best_val_accuracy": best_val_accuracy,
        "total_time_seconds": time.time() - started,
        "temperature": temperature,
        "alpha": alpha,
    }


# ── ONNX quantization ───────────────────────────────────────────────────────


def quantize_onnx(
    onnx_path: str | Path,
    output_path: str | Path,
    *,
    mode: str = "dynamic",
    calibration_loader: Any | None = None,
    max_calibration_batches: int = 8,
) -> Path:
    """INT8-quantize an exported ONNX file.

    ``dynamic`` quantizes weights only (no calibration data needed, runs on
    any host). ``static`` additionally calibrates activations from
    ``calibration_loader`` (a torch DataLoader yielding (images, targets)).
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    onnx_path, output_path = Path(onnx_path), Path(output_path)
    if mode == "dynamic":
        quantize_dynamic(str(onnx_path), str(output_path), weight_type=QuantType.QInt8)
        return output_path
    if mode != "static":
        raise ValueError(f"unknown quantization mode: {mode}")
    if calibration_loader is None:
        raise ValueError("static quantization requires a calibration_loader")

    import numpy as np
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationDataReader, quantize_static

    input_name = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    ).get_inputs()[0].name

    class _LoaderReader(CalibrationDataReader):
        def __init__(self) -> None:
            self._iterator = None

        def get_next(self):
            if self._iterator is None:
                def _gen():
                    for index, (images, _targets) in enumerate(calibration_loader):
                        if index >= max_calibration_batches:
                            break
                        yield {input_name: np.asarray(images.numpy(), dtype=np.float32)}
                self._iterator = _gen()
            return next(self._iterator, None)

    quantize_static(str(onnx_path), str(output_path), _LoaderReader())
    return output_path


def evaluate_onnx_classification(onnx_path: str | Path, loader: Any) -> float:
    """Accuracy of an ONNX classifier over a torch DataLoader, computed in
    ONNX Runtime - the same runtime the artifact ships in."""
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    correct, total = 0, 0
    for images, targets in loader:
        outputs = session.run(None, {input_name: np.asarray(images.numpy(), dtype=np.float32)})
        predictions = np.argmax(outputs[0], axis=1)
        correct += int((predictions == targets.numpy()).sum())
        total += int(targets.numpy().size)
    return correct / total if total else 0.0


# ── the gate ────────────────────────────────────────────────────────────────


@dataclass
class CompressionReport:
    attempted: bool
    mode: str | None = None
    baseline_val_accuracy: float | None = None
    compressed_val_accuracy: float | None = None
    floor: float | None = None
    passed: bool | None = None
    chosen_path: str | None = None
    compressed_path: str | None = None
    baseline_size_bytes: int | None = None
    compressed_size_bytes: int | None = None
    reason: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "mode": self.mode,
            "baseline_val_accuracy": self.baseline_val_accuracy,
            "compressed_val_accuracy": self.compressed_val_accuracy,
            "floor": self.floor,
            "passed": self.passed,
            "chosen_path": self.chosen_path,
            "compressed_path": self.compressed_path,
            "baseline_size_bytes": self.baseline_size_bytes,
            "compressed_size_bytes": self.compressed_size_bytes,
            "reason": self.reason,
            **({"extra": self.extra} if self.extra else {}),
        }


def compute_floor(
    baseline_accuracy: float,
    compression: CompressionConfig,
    constraints: ModelConstraints,
) -> float:
    """The accuracy the compressed artifact must reach: the stricter of the
    relative retention floor and any absolute floor stated in the prompt."""
    relative = compression.accuracy_floor_relative * baseline_accuracy
    absolute = constraints.accuracy_floor or 0.0
    return max(relative, absolute)


def decide_gate(
    baseline_accuracy: float,
    compressed_accuracy: float,
    compression: CompressionConfig,
    constraints: ModelConstraints,
) -> tuple[bool, float]:
    floor = compute_floor(baseline_accuracy, compression, constraints)
    return compressed_accuracy >= floor, floor


def apply_compression(
    onnx_path: str | Path,
    val_loader: Any,
    compression: CompressionConfig,
    constraints: ModelConstraints,
) -> CompressionReport:
    """Quantize an exported classifier and gate it against the floor.

    Never raises into the pipeline: any failure returns a report with
    ``attempted=True, passed=False`` and the uncompressed artifact chosen.
    """
    onnx_path = Path(onnx_path)
    if not compression.enable_quantization:
        return CompressionReport(attempted=False, chosen_path=str(onnx_path), reason="quantization disabled")

    compressed_path = onnx_path.with_name(onnx_path.stem + "_int8.onnx")
    try:
        baseline_accuracy = evaluate_onnx_classification(onnx_path, val_loader)
        quantize_onnx(
            onnx_path,
            compressed_path,
            mode=compression.quantization_mode,
            calibration_loader=val_loader if compression.quantization_mode == "static" else None,
        )
        compressed_accuracy = evaluate_onnx_classification(compressed_path, val_loader)
    except Exception as exc:  # gate failure must never kill the run
        logger.warning("compression failed; shipping uncompressed artifact: %s", exc)
        return CompressionReport(
            attempted=True,
            mode=compression.quantization_mode,
            passed=False,
            chosen_path=str(onnx_path),
            reason=f"compression error: {exc}",
        )

    passed, floor = decide_gate(baseline_accuracy, compressed_accuracy, compression, constraints)
    chosen = compressed_path if passed else onnx_path
    reason = (
        "compressed artifact holds the accuracy floor"
        if passed
        else "REFUSED: compressed artifact fell below the accuracy floor - shipping uncompressed"
    )
    return CompressionReport(
        attempted=True,
        mode=compression.quantization_mode,
        baseline_val_accuracy=baseline_accuracy,
        compressed_val_accuracy=compressed_accuracy,
        floor=floor,
        passed=passed,
        chosen_path=str(chosen),
        compressed_path=str(compressed_path),
        baseline_size_bytes=onnx_path.stat().st_size,
        compressed_size_bytes=compressed_path.stat().st_size if compressed_path.exists() else None,
        reason=reason,
    )
