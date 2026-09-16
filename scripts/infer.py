"""python scripts/infer.py --config configs/default.yaml \
    --checkpoint checkpoints/edsr_epoch10.pt --input scene.tif --output scene_sr.tif"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import terraresolve.models  # noqa: F401
from terraresolve.engine.inference import run_inference, write_geotiff
from terraresolve.registry import MODEL_REGISTRY
from terraresolve.utils.config import load_config

try:
    import rasterio
except ImportError:
    rasterio = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if rasterio is None:
        raise ImportError("rasterio is required for GeoTIFF inference")

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    model = MODEL_REGISTRY.build(model_cfg["name"], **model_cfg.get("params", {}))
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    with rasterio.open(args.input) as src:
        image = src.read().astype(np.float32)
        if image.max() > 1.5:
            image = image / (65535.0 if image.max() > 255 else 255.0)
        profile = src.profile

    scale = model_cfg["params"].get("scale", 4)
    inf_cfg = cfg.get("inference", {})
    sr = run_inference(model, image, scale=scale,
                        patch_size=inf_cfg.get("patch_size", 64),
                        overlap=inf_cfg.get("overlap", 8))
    write_geotiff(args.output, sr, profile, scale=scale)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
