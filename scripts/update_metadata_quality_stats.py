"""
Computes and appends real quality_stats and magnification objects to metadata.json
for all 4 scenes, using existing validated methods without touching input.tif,
sr.tif, or any preview PNGs.
"""

import os
import json
import cv2
import rasterio

SCENES = ["yelahanka-lake", "yelahanka-newtown", "yelahanka-afs-buffer", "yelahanka-mixed"]
SCENES_DIR = "demo/scenes"
SCALE = 4

def compute_quality_stats(in_png_path: str, sr_png_path: str) -> dict:
    in_gray = cv2.imread(in_png_path, cv2.IMREAD_GRAYSCALE)
    sr_gray = cv2.imread(sr_png_path, cv2.IMREAD_GRAYSCALE)
    
    in_sharpness = float(cv2.Laplacian(in_gray, cv2.CV_64F).var())
    in_contrast = float(in_gray.std())
    
    sr_sharpness = float(cv2.Laplacian(sr_gray, cv2.CV_64F).var())
    sr_contrast = float(sr_gray.std())
    
    sh_imp = float((sr_sharpness - in_sharpness) / (in_sharpness + 1e-6) * 100.0)
    ct_imp = float((sr_contrast - in_contrast) / (in_contrast + 1e-6) * 100.0)
    
    # Also calculate display-scaled sharpness (interpolated onto canvas at 2.5m resolution)
    in_disp = cv2.resize(in_gray, (sr_gray.shape[1], sr_gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    in_disp_sharpness = float(cv2.Laplacian(in_disp, cv2.CV_64F).var())
    disp_sh_imp = float((sr_sharpness - in_disp_sharpness) / (in_disp_sharpness + 1e-6) * 100.0)
    
    return {
        "input": {
            "sharpness": round(in_sharpness, 2),
            "contrast": round(in_contrast, 2),
            "display_sharpness": round(in_disp_sharpness, 2)
        },
        "sr": {
            "sharpness": round(sr_sharpness, 2),
            "contrast": round(sr_contrast, 2)
        },
        "sharpness_improvement_pct": round(sh_imp, 2),
        "contrast_improvement_pct": round(ct_imp, 2),
        "display_sharpness_improvement_pct": round(disp_sh_imp, 2)
    }

def update_scene_metadata(scene_id: str):
    scene_dir = os.path.join(SCENES_DIR, scene_id)
    meta_path = os.path.join(scene_dir, "metadata.json")
    in_png = os.path.join(scene_dir, "input_preview.png")
    sr_png = os.path.join(scene_dir, "sr_preview.png")
    in_tif = os.path.join(scene_dir, "input.tif")
    sr_tif = os.path.join(scene_dir, "sr.tif")
    
    with open(meta_path, "r") as f:
        meta = json.load(f)
        
    with rasterio.open(in_tif) as src_in:
        in_w, in_h = src_in.width, src_in.height
        
    with rasterio.open(sr_tif) as src_sr:
        sr_w, sr_h = src_sr.width, src_sr.height
        
    quality_stats = compute_quality_stats(in_png, sr_png)
    
    magnification = {
        "scale_factor": SCALE,
        "input_resolution_m": 10,
        "output_resolution_m": 2.5,
        "input_pixels": [in_w, in_h],
        "output_pixels": [sr_w, sr_h]
    }
    
    meta["quality_stats"] = quality_stats
    meta["magnification"] = magnification
    
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
        
    print(f"[{scene_id}] Updated metadata.json successfully.")

def main():
    for scene_id in SCENES:
        update_scene_metadata(scene_id)

if __name__ == "__main__":
    main()
