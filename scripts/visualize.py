"""Generates human-viewable PNG comparisons from a trained checkpoint:
for each of a few samples, a side-by-side [bicubic-upsampled LR |
model SR output | true HR] panel -- so you can actually look at what
the model produces instead of only reading loss numbers.

    python scripts/visualize.py --config configs/sen2naip.yaml \
        --checkpoint checkpoints/edsr_epoch10.pt --n 6 --out previews

Works on any checkpoint, including ones from earlier, still-training
epochs -- you don't have to wait for all 50 to look at progress.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import terraresolve.data    # noqa: F401  (registers datasets)
import terraresolve.models  # noqa: F401  (registers models)
from terraresolve.registry import DATASET_REGISTRY, MODEL_REGISTRY
from terraresolve.utils.config import load_config


def _to_uint8_rgb(chw: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """(4, H, W) float, band order [Blue, Green, Red, NIR] (matches
    Sentinel-2 B2/B3/B4/B8) -> (H, W, 3) uint8 true-color PNG.

    Applies a display-only contrast stretch (lo/hi -> 0/1) BEFORE
    converting to 8-bit. This does not touch the actual data used for
    training/loss/metrics anywhere else -- real Sentinel-2 reflectance
    is legitimately dim (typical land cover reflects nowhere near the
    sensor's theoretical max), so a raw /10000-style normalization
    looks almost black on screen even though it's scientifically
    correct. Every remote-sensing viewer applies a stretch like this
    purely for human eyes; it's a display convention, not a data fix.
    """
    rgb = chw[[2, 1, 0]]  # reorder to Red, Green, Blue
    stretched = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (np.moveaxis(stretched, 0, -1) * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sen2naip.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n", type=int, default=6, help="how many samples to preview")
    parser.add_argument("--out", default="previews")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    model = MODEL_REGISTRY.build(model_cfg["name"], **model_cfg.get("params", {}))
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    data_cfg = cfg["data"]
    dataset = DATASET_REGISTRY.build(
        data_cfg["type"], **{**data_cfg.get("params", {}), "augment": False}
    )

    checkpoint_tag = Path(args.checkpoint).stem  # e.g. "edsr_epoch2"
    out_dir = Path(args.out) / checkpoint_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    n = min(args.n, len(dataset))
    for i in range(n):
        lr, hr = dataset[i]
        with torch.no_grad():
            sr = model(lr.unsqueeze(0))[0].clamp(0, 1)

        lr_up = torch.nn.functional.interpolate(
            lr.unsqueeze(0), size=hr.shape[-2:], mode="bicubic", align_corners=False
        )[0].clamp(0, 1)

        # Stretch bounds come from the HR reference (the "correctly
        # exposed" image) and are reused for all three panels, so
        # brightness differences you SEE reflect real reconstruction
        # quality, not three independently auto-stretched images.
        lo, hi = np.percentile(hr.numpy(), [1, 99])

        panel = np.concatenate([
            _to_uint8_rgb(lr_up.numpy(), lo, hi),
            _to_uint8_rgb(sr.numpy(), lo, hi),
            _to_uint8_rgb(hr.numpy(), lo, hi),
        ], axis=1)  # left-to-right: naive upsample | model output | ground truth

        Image.fromarray(panel).save(out_dir / f"sample_{i:02d}_LR-vs-SR-vs-HR.png")

    print(f"Wrote {n} comparison PNGs to {out_dir}/")
    print("Each image, left to right: plain bicubic-upsampled LR | model's SR output | true HR reference.")


if __name__ == "__main__":
    main()
