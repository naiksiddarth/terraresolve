"""Generates a handful of synthetic 4-band "HR-like" tiles so the
pipeline can be smoke-tested with zero downloads. Pure numpy + PIL --
no network, no torch needed. NOT real satellite imagery: only useful
for confirming the training loop actually runs end-to-end before you
plug in real SEN2NAIP data.

    python scripts/make_placeholder_data.py --out data/hr_reference --n 12 --size 256
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def low_freq_field(size: int, grid: int, rng: np.random.Generator) -> np.ndarray:
    """Smooth random field: sample a coarse grid, upsample with PIL's
    bicubic filter -- a cheap stand-in for Perlin noise, no scipy
    dependency required."""
    coarse = rng.random((grid, grid)).astype(np.float32)
    img = Image.fromarray((coarse * 255).astype(np.uint8), mode="L")
    img = img.resize((size, size), resample=Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def make_scene(size: int, rng: np.random.Generator) -> np.ndarray:
    """Builds a crude but spectrally-plausible 4-band scene: shared
    low-frequency 'land cover' fields drive correlated Blue/Green/Red/
    NIR values (vegetation -> low R, high NIR; water -> low NIR, higher
    Blue; bare/urban -> flatter, mid-grey across bands), plus a bit of
    fine per-pixel sensor-noise texture on top."""
    veg = low_freq_field(size, 6, rng)       # 1 = vegetation-like
    water = low_freq_field(size, 5, rng) * (low_freq_field(size, 5, rng) < 0.35)
    urban = 1.0 - np.clip(veg + water, 0, 1)

    blue = 0.25 * urban + 0.15 * veg + 0.55 * water
    green = 0.30 * urban + 0.35 * veg + 0.30 * water
    red = 0.35 * urban + 0.20 * veg + 0.20 * water
    nir = 0.30 * urban + 0.75 * veg + 0.05 * water

    bands = np.stack([blue, green, red, nir]).astype(np.float32)
    noise = rng.normal(0, 0.02, bands.shape).astype(np.float32)
    bands = np.clip(bands + noise, 0.0, 1.0)
    return bands


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/hr_reference")
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for i in range(args.n):
        bands = make_scene(args.size, rng)
        rgba = (np.moveaxis(bands, 0, -1) * 255).astype(np.uint8)  # (H, W, 4) B,G,R,NIR packed as RGBA
        Image.fromarray(rgba, mode="RGBA").save(out_dir / f"synthetic_placeholder_{i:02d}.png")

    print(f"Wrote {args.n} placeholder tiles ({args.size}x{args.size}, 4-band) to {out_dir}/")


if __name__ == "__main__":
    main()
