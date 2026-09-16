"""
Generates input.tif, sr.tif, previews, and metadata.json for each Yelahanka
demo scene. Run once Siddarth's clipped chips exist under data/raw/{scene_id}/input.tif.
Does NOT compute metrics — that's evaluate_scene.py's job, kept separate on purpose.
"""
import json
import rasterio
from rasterio.transform import Affine
from PIL import Image
import numpy as np
import sys
import os
sys.path.insert(0, '.')
import terraresolve.models  # noqa: registers "sen2sr" in MODEL_REGISTRY
from terraresolve.registry import MODEL_REGISTRY
from terraresolve.engine.inference import run_inference

SCENES = ["yelahanka-lake", "yelahanka-newtown", "yelahanka-afs-buffer", "yelahanka-mixed"]
RAW_DIR = "data/raw"
OUT_DIR = "demo/scenes"
SCALE = 4
BAND_ORDER = [3, 2, 1, 4]  # adjust to Siddarth's actual file band order -> [B04,B03,B02,B08]

def percentile_stretch(arr, low=1, high=99):
    lo, hi = np.percentile(arr, [low, high])
    stretched = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)
    # Gamma correction: brings out shadow detail and tones down glowing whites
    gamma_corrected = np.power(stretched, 0.7)
    return (gamma_corrected * 255).astype("uint8")

def save_preview(rgb_bands_uint16, path):
    rgb = np.stack([percentile_stretch(rgb_bands_uint16[i]) for i in range(3)], axis=-1)
    Image.fromarray(rgb).save(path)

def process_scene(scene_id, model, device="cuda"):
    in_path = f"{RAW_DIR}/{scene_id}/input.tif"
    out_dir = f"{OUT_DIR}/{scene_id}"
    os.makedirs(out_dir, exist_ok=True)

    try:
        with rasterio.open(in_path) as src:
            arr = src.read(BAND_ORDER)  # -> (4, H, W) in B04,B03,B02,B08 order
            profile = src.profile

        # Copy the raw input.tif into the scene folder as-is
        with rasterio.open(f"{out_dir}/input.tif", "w", **profile) as dst:
            with rasterio.open(in_path) as src:
                dst.write(src.read())  # keep original band order/count for reference

        X = (arr / 10_000).astype("float32")
        sr_array = run_inference(model, X, scale=SCALE, device=device, patch_size=128, overlap=32)
        sr_uint16 = (np.clip(sr_array, 0, 1) * 10_000).astype("uint16")

        t = profile["transform"]
        new_transform = Affine(t.a / SCALE, t.b, t.c, t.d, t.e / SCALE, t.f)
        sr_profile = profile.copy()
        sr_profile.update(transform=new_transform, width=sr_uint16.shape[2],
                           height=sr_uint16.shape[1], count=4, dtype="uint16")
        with rasterio.open(f"{out_dir}/sr.tif", "w", **sr_profile) as dst:
            # Reorder from [Red, Green, Blue, NIR] back to [Blue, Green, Red, NIR] for the GeoTIFF
            dst.write(sr_uint16[[2, 1, 0, 3]])

        save_preview(arr[:3], f"{out_dir}/input_preview.png")
        save_preview(sr_uint16[:3], f"{out_dir}/sr_preview.png")

        from rasterio.warp import transform_bounds
        native_bounds = rasterio.open(f"{out_dir}/sr.tif").bounds
        wgs84_bounds = transform_bounds(profile["crs"], "EPSG:4326", *native_bounds)
        
        import cv2
        in_gray = cv2.imread(f"{out_dir}/input_preview.png", cv2.IMREAD_GRAYSCALE)
        sr_gray = cv2.imread(f"{out_dir}/sr_preview.png", cv2.IMREAD_GRAYSCALE)
        in_sh = float(cv2.Laplacian(in_gray, cv2.CV_64F).var())
        in_ct = float(in_gray.std())
        sr_sh = float(cv2.Laplacian(sr_gray, cv2.CV_64F).var())
        sr_ct = float(sr_gray.std())
        sh_imp = float((sr_sh - in_sh) / (in_sh + 1e-6) * 100.0)
        ct_imp = float((sr_ct - in_ct) / (in_ct + 1e-6) * 100.0)
        
        metadata = {
            "id": scene_id,
            "crs": str(profile["crs"]),
            "bounds": {
                "west": wgs84_bounds[0],
                "south": wgs84_bounds[1],
                "east": wgs84_bounds[2],
                "north": wgs84_bounds[3]
            },
            "input_resolution_m": 10,
            "output_resolution_m": 2.5,
            "scale_factor": SCALE,
            "model": "SEN2SR (SPAN, 472K params) — ESAOpenSR",
            "bands_used": ["B04", "B03", "B02", "B08"],
            "quality_stats": {
                "input": {"sharpness": round(in_sh, 2), "contrast": round(in_ct, 2)},
                "sr": {"sharpness": round(sr_sh, 2), "contrast": round(sr_ct, 2)},
                "sharpness_improvement_pct": round(sh_imp, 2),
                "contrast_improvement_pct": round(ct_imp, 2)
            },
            "magnification": {
                "scale_factor": SCALE,
                "input_resolution_m": 10,
                "output_resolution_m": 2.5,
                "input_pixels": [arr.shape[2], arr.shape[1]],
                "output_pixels": [sr_uint16.shape[2], sr_uint16.shape[1]]
            }
        }
        with open(f"{out_dir}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)


        print(f"[{scene_id}] done -> {out_dir}")
    except FileNotFoundError:
        print(f"[{scene_id}] Skipping - {in_path} not found.")


if __name__ == "__main__":
    model = MODEL_REGISTRY.build("sen2sr", weights_dir="models/pretrained/sen2sr_rgbn_x4", device="cuda")
    model.eval()
    for scene_id in SCENES:
        process_scene(scene_id, model)
