"""python scripts/evaluate.py --pred sr_output.tif --ref hr_reference.tif \
    --metrics psnr ssim cc ergas sam"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import terraresolve.metrics  # noqa: F401
from terraresolve.metrics.metrics import evaluate_all

try:
    import rasterio
except ImportError:
    rasterio = None


def _load(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / (65535.0 if arr.max() > 255 else 255.0)
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--metrics", nargs="*", default=["psnr", "ssim", "cc", "ergas", "sam"])
    args = parser.parse_args()

    if rasterio is None:
        raise ImportError("rasterio is required to read GeoTIFFs for evaluation")

    pred, ref = _load(args.pred), _load(args.ref)
    for name, value in evaluate_all(pred, ref, args.metrics).items():
        print(f"{name.upper():>6}: {value:.4f}")


if __name__ == "__main__":
    main()
