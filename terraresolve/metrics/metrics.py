"""Evaluation metrics -- PSNR, SSIM, CC (spatial/structural fidelity)
plus ERGAS and SAM (spectral fidelity, the pansharpening-literature
standards). Each takes numpy arrays shaped (C, H, W) in [0, 1] and is
registered by name; `evaluate_all` runs whichever names are listed in
a config's `eval.metrics`. Add a metric anywhere, register it, list
its name -- no other wiring needed."""
from __future__ import annotations
from typing import Dict, List

import numpy as np

from terraresolve.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("psnr")
def psnr(pred: np.ndarray, target: np.ndarray, max_val: float = 1.0) -> float:
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10((max_val ** 2) / mse))


@METRIC_REGISTRY.register("ssim")
def ssim(pred: np.ndarray, target: np.ndarray, max_val: float = 1.0) -> float:
    from skimage.metrics import structural_similarity
    pred_hwc = np.moveaxis(pred, 0, -1)
    target_hwc = np.moveaxis(target, 0, -1)
    return float(structural_similarity(target_hwc, pred_hwc, data_range=max_val, channel_axis=-1))


@METRIC_REGISTRY.register("cc")
def correlation_coefficient(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean per-band Pearson correlation coefficient."""
    scores = []
    for c in range(pred.shape[0]):
        p, t = pred[c].ravel(), target[c].ravel()
        if p.std() < 1e-8 or t.std() < 1e-8:
            continue
        scores.append(np.corrcoef(p, t)[0, 1])
    return float(np.mean(scores)) if scores else float("nan")


@METRIC_REGISTRY.register("ergas")
def ergas(pred: np.ndarray, target: np.ndarray, scale: int = 4) -> float:
    """Relative dimensionless global error in synthesis (lower=better,
    0=perfect). Standard pansharpening-quality metric: aggregates
    per-band RMSE normalised by each band's mean, scaled by the
    LR/HR resolution ratio."""
    c = pred.shape[0]
    total = 0.0
    counted = 0
    for band in range(c):
        rmse = np.sqrt(np.mean((pred[band] - target[band]) ** 2))
        mean_ref = np.mean(target[band])
        if mean_ref < 1e-8:
            continue
        total += (rmse / mean_ref) ** 2
        counted += 1
    if counted == 0:
        return float("nan")
    return float(100.0 / scale * np.sqrt(total / counted))


@METRIC_REGISTRY.register("sam")
def spectral_angle_mapper(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean spectral angle across all pixels, in degrees. Lower is
    better; 0 degrees means the spectral shape is perfectly preserved
    even if this function says nothing about absolute brightness."""
    c = pred.shape[0]
    p = pred.reshape(c, -1)
    t = target.reshape(c, -1)
    dot = np.sum(p * t, axis=0)
    denom = np.clip(np.linalg.norm(p, axis=0) * np.linalg.norm(t, axis=0), 1e-8, None)
    cos_angle = np.clip(dot / denom, -1.0, 1.0)
    return float(np.degrees(np.mean(np.arccos(cos_angle))))


def evaluate_all(pred: np.ndarray, target: np.ndarray, metric_names: List[str],
                  **kwargs) -> Dict[str, float]:
    return {
        name: METRIC_REGISTRY.get(name)(pred, target, **kwargs.get(name, {}))
        for name in metric_names
    }
