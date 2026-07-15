"""Metrics for clean performance, targeted trigger success, and visibility."""

from __future__ import annotations

import math

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.mean(y_true == y_pred)) if y_true.size else math.nan


def attack_success_rate(
    y_true: np.ndarray, y_pred_triggered: np.ndarray, *, target_class: int
) -> float:
    """Targeted ASR over all examples whose true class is not the target."""
    y_true = np.asarray(y_true)
    y_pred_triggered = np.asarray(y_pred_triggered)
    if y_true.shape != y_pred_triggered.shape:
        raise ValueError("y_true and y_pred_triggered must have the same shape")
    eligible = y_true != target_class
    if not np.any(eligible):
        return math.nan
    return float(np.mean(y_pred_triggered[eligible] == target_class))


def conditional_attack_success_rate(
    y_true: np.ndarray,
    y_pred_clean: np.ndarray,
    y_pred_triggered: np.ndarray,
    *,
    target_class: int,
) -> float:
    """ASR only where the model was correct before the trigger was added."""
    y_true = np.asarray(y_true)
    y_pred_clean = np.asarray(y_pred_clean)
    y_pred_triggered = np.asarray(y_pred_triggered)
    if not (y_true.shape == y_pred_clean.shape == y_pred_triggered.shape):
        raise ValueError("All label arrays must have the same shape")
    eligible = (y_true != target_class) & (y_pred_clean == y_true)
    if not np.any(eligible):
        return math.nan
    return float(np.mean(y_pred_triggered[eligible] == target_class))


def per_source_class_asr(
    y_true: np.ndarray, y_pred_triggered: np.ndarray, *, target_class: int
) -> dict[int, float]:
    """Return targeted ASR for every non-target source class."""
    y_true = np.asarray(y_true)
    y_pred_triggered = np.asarray(y_pred_triggered)
    result: dict[int, float] = {}
    for source_class in np.unique(y_true):
        source_class = int(source_class)
        if source_class == target_class:
            continue
        mask = y_true == source_class
        result[source_class] = float(np.mean(y_pred_triggered[mask] == target_class))
    return result


def target_logit_margin(logits: np.ndarray, *, target_class: int) -> np.ndarray:
    """Target logit minus the strongest non-target logit for each example."""
    logits = np.asarray(logits)
    if logits.ndim != 2:
        raise ValueError("logits must have shape (examples, classes)")
    if not 0 <= target_class < logits.shape[1]:
        raise ValueError("target_class is outside the logit dimension")
    target = logits[:, target_class]
    competitors = np.delete(logits, target_class, axis=1)
    return target - competitors.max(axis=1)


def visibility_metrics(clean: np.ndarray, triggered: np.ndarray) -> dict[str, float]:
    """Pixel and perceptual visibility metrics for RGB arrays."""
    clean = np.asarray(clean)
    triggered = np.asarray(triggered)
    if clean.shape != triggered.shape:
        raise ValueError("clean and triggered images must have the same shape")

    if np.issubdtype(clean.dtype, np.integer):
        scale = float(np.iinfo(clean.dtype).max)
        clean_float = clean.astype(np.float64) / scale
        triggered_float = triggered.astype(np.float64) / scale
    else:
        clean_float = clean.astype(np.float64)
        triggered_float = triggered.astype(np.float64)

    delta = triggered_float - clean_float
    mse = float(np.mean(delta**2))
    result = {
        "changed_fraction": float(np.mean(np.any(np.abs(delta) > 1e-8, axis=-1))),
        "mean_absolute_delta": float(np.mean(np.abs(delta))),
        "mse": mse,
        "psnr": math.inf if mse == 0.0 else float(10.0 * math.log10(1.0 / mse)),
    }
    try:
        from skimage.color import deltaE_ciede2000, rgb2lab
        from skimage.metrics import structural_similarity

        result["ssim"] = float(
            structural_similarity(clean_float, triggered_float, channel_axis=-1, data_range=1.0)
        )
        result["mean_delta_e"] = float(
            np.mean(deltaE_ciede2000(rgb2lab(clean_float), rgb2lab(triggered_float)))
        )
    except ImportError:
        pass
    return result
