"""
Controlled-degradation evaluation: HR reference -> degrade -> LR -> SR -> compare vs HR.
This is the ONLY source of metrics.json — never hand-typed, per project rule.
"""
import json
import rasterio
import numpy as np
import os
from scipy.ndimage import gaussian_filter

# We use scikit-image for metrics
try:
    from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-image"])
    from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

import sys
sys.path.insert(0, '.')
import terraresolve.models  # noqa
from terraresolve.registry import MODEL_REGISTRY
from terraresolve.engine.inference import run_inference

def sam(sr, hr):
    sr_f, hr_f = sr.reshape(sr.shape[0], -1), hr.reshape(hr.shape[0], -1)
    dot = (sr_f * hr_f).sum(axis=0)
    denom = np.linalg.norm(sr_f, axis=0) * np.linalg.norm(hr_f, axis=0) + 1e-8
    return np.degrees(np.arccos(np.clip(dot / denom, -1, 1))).mean()

def ergas(sr, hr, ratio=4):
    rmse_per_band = np.sqrt(((sr - hr) ** 2).mean(axis=(1, 2)))
    mean_per_band = hr.mean(axis=(1, 2)) + 1e-8
    return 100 * (1 / ratio) * np.sqrt(np.mean((rmse_per_band / mean_per_band) ** 2))

def evaluate_scene(scene_id, model, device="cuda"):
    in_path = f"demo/scenes/{scene_id}/input.tif"
    if not os.path.exists(in_path):
        print(f"[{scene_id}] Skipping - {in_path} not found.")
        return None

    with rasterio.open(in_path) as src:
        # NOTE: file has 4 bands ordered [B02, B03, B04, B08].
        # We need [B04, B03, B02, B08] (Red, Green, Blue, NIR), so we use rasterio indices [3, 2, 1, 4]
        hr = src.read([3, 2, 1, 4]).astype("float32") / 10_000

    import cv2
    lr = np.stack([cv2.resize(gaussian_filter(hr[b], sigma=1.0), (hr.shape[2]//4, hr.shape[1]//4), interpolation=cv2.INTER_CUBIC) for b in range(4)])
    sr = run_inference(model, lr, scale=4, device=device, patch_size=128, overlap=32)

    h, w = min(sr.shape[1], hr.shape[1]), min(sr.shape[2], hr.shape[2])
    sr_c, hr_c = sr[:, :h, :w], hr[:, :h, :w]

    metrics = {
        "psnr_db": float(np.mean([psnr(hr_c[b], sr_c[b], data_range=1.0) for b in range(4)])),
        "ssim": float(np.mean([ssim(hr_c[b], sr_c[b], data_range=1.0) for b in range(4)])),
        "sam_degrees": float(sam(sr_c, hr_c)),
        "ergas": float(ergas(sr_c, hr_c)),
    }
    with open(f"demo/scenes/{scene_id}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[{scene_id}] {metrics}")
    return metrics

if __name__ == "__main__":
    model = MODEL_REGISTRY.build("sen2sr", weights_dir="models/pretrained/sen2sr_rgbn_x4", device="cuda")
    model.eval()
    for scene_id in ["yelahanka-lake", "yelahanka-newtown", "yelahanka-afs-buffer", "yelahanka-mixed"]:
        evaluate_scene(scene_id, model)
