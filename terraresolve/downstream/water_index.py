"""Downstream 'is the SR output actually useful' tasks -- the piece
of the evaluation story that answers 'did this improve mapping', not
just 'did PSNR go up'. Each plugin takes a (C, H, W) array with bands
ordered [Blue, Green, Red, NIR] (Sentinel-2 B2/B3/B4/B8, which lines
up with NAIP's R/G/B/NIR) and returns a binary mask. Run the same
function on the raw LR input, on your SR output, and on the true HR
reference, then compare with `mask_agreement` -- that comparison is
your real differentiator, not the raw image-quality numbers alone.

Add a new task (building footprint, road extraction, ...) the same
way: write a function, decorate it, name it in the config."""
from __future__ import annotations
import numpy as np

from terraresolve.registry import DOWNSTREAM_REGISTRY


@DOWNSTREAM_REGISTRY.register("ndwi_water")
def ndwi_water_mask(image: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """NDWI = (Green - NIR) / (Green + NIR)."""
    green, nir = image[1], image[3]
    ndwi = (green - nir) / np.clip(green + nir, 1e-6, None)
    return (ndwi > threshold).astype(np.uint8)


@DOWNSTREAM_REGISTRY.register("ndvi_vegetation")
def ndvi_vegetation_mask(image: np.ndarray, threshold: float = 0.2) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red)."""
    red, nir = image[2], image[3]
    ndvi = (nir - red) / np.clip(nir + red, 1e-6, None)
    return (ndvi > threshold).astype(np.uint8)


def mask_agreement(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """IoU between two binary masks, e.g. SR-derived vs HR-derived --
    an accuracy proxy for the downstream-task comparison above."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union > 0 else float("nan")
