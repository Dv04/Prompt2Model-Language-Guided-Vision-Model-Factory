"""Calibration + uncertainty head - every factory model ships knowing when
it doesn't know.

A trained classifier's raw confidences lie (typically overconfident). This
module fits two things on the validation split and embeds them in the ONNX
metadata so they travel WITH the artifact:

* **temperature** - a single scalar T dividing the logits before softmax,
  fitted to minimize validation NLL (grid search: dependency-free, exact
  enough for one parameter). ECE before/after is recorded so the effect is
  auditable.
* **conformal abstain threshold** - the split-conformal (1 − alpha)
  quantile of validation nonconformity (1 − calibrated max-probability).
  At inference, nonconformity above the threshold ⇒ ABSTAIN; under
  exchangeability, accepted predictions err at most ~alpha of the time.

``EdgeModel`` reads the same metadata and exposes the abstain decision - 
the "guarantee handshake": the model + its honesty contract are one file.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger("prompt2model.calibration")


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / max(temperature, 1e-6)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def _nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    probs = _softmax(logits, temperature)
    picked = probs[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(picked, 1e-12, 1.0)).mean())


def ece(confidences: Sequence[float], correct: Sequence[bool], bins: int = 15) -> float:
    """Expected Calibration Error: occupancy-weighted mean |accuracy − mean
    confidence| over equal-width bins (Guo et al. 2017). 0 = calibrated."""
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must be the same length")
    n = len(confidences)
    if n == 0 or bins <= 0:
        return 0.0
    conf_sum = [0.0] * bins
    hits = [0] * bins
    counts = [0] * bins
    for c, ok in zip(confidences, correct):
        c = min(max(float(c), 0.0), 1.0)
        b = min(int(c * bins), bins - 1)
        counts[b] += 1
        conf_sum[b] += c
        hits[b] += 1 if ok else 0
    total = 0.0
    for b in range(bins):
        if counts[b]:
            total += (counts[b] / n) * abs(hits[b] / counts[b] - conf_sum[b] / counts[b])
    return total


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    grid: Sequence[float] | None = None,
) -> float:
    """Single-parameter grid search minimizing validation NLL. The grid is
    log-spaced over [0.05, 10] - wide enough for badly miscalibrated heads
    in either direction."""
    if grid is None:
        grid = np.logspace(math.log10(0.05), math.log10(10.0), 120)
    best_t, best_nll = 1.0, _nll(logits, labels, 1.0)
    for t in grid:
        candidate = _nll(logits, labels, float(t))
        if candidate < best_nll:
            best_t, best_nll = float(t), candidate
    return best_t


def conformal_threshold(nonconformity: Sequence[float], alpha: float = 0.1) -> float:
    """Split-conformal (1 − alpha) quantile with the (n+1) finite-sample
    correction. Empty input → +inf (never abstain until calibrated)."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    scores = sorted(float(s) for s in nonconformity)
    n = len(scores)
    if n == 0:
        return float("inf")
    rank = max(1, min(math.ceil((n + 1) * (1.0 - alpha)), n))
    return scores[rank - 1]


def calibrate_classification(
    model: Any,
    val_loader: Any,
    device: Any,
    alpha: float = 0.1,
) -> dict[str, Any]:
    """Fit temperature + conformal threshold on the validation split.

    Returns the metadata block the artifact ships with. Never raises - 
    on any failure returns ``{"calibrated": False, "error": ...}`` so the
    pipeline keeps going and the artifact honestly says it's uncalibrated.
    """
    try:
        import torch

        model = model.to(device).eval()
        logits_list, labels_list = [], []
        with torch.no_grad():
            for images, targets in val_loader:
                logits_list.append(model(images.to(device)).cpu().numpy())
                labels_list.append(targets.numpy())
        logits = np.concatenate(logits_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        if len(labels) < 2:
            return {"calibrated": False, "error": "validation split too small"}

        raw_probs = _softmax(logits, 1.0)
        raw_conf = raw_probs.max(axis=1)
        raw_correct = raw_probs.argmax(axis=1) == labels

        temperature = fit_temperature(logits, labels)
        cal_probs = _softmax(logits, temperature)
        cal_conf = cal_probs.max(axis=1)
        cal_correct = cal_probs.argmax(axis=1) == labels

        threshold = conformal_threshold(1.0 - cal_conf, alpha=alpha)
        return {
            "calibrated": True,
            "temperature": round(temperature, 4),
            "alpha": alpha,
            "conformal_threshold": round(float(threshold), 6),
            "ece_before": round(ece(raw_conf.tolist(), raw_correct.tolist()), 4),
            "ece_after": round(ece(cal_conf.tolist(), cal_correct.tolist()), 4),
            "val_samples": int(len(labels)),
        }
    except Exception as exc:  # noqa: BLE001 - calibration must not kill the run
        logger.warning("calibration failed: %s", exc)
        return {"calibrated": False, "error": str(exc)}
