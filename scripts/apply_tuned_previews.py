"""
Regenerates sr_preview.png for all 4 scenes using tuned display settings:
- Percentile stretch: 1st to 99th percentile
- Gamma correction: 1.0 (natural linear contrast, no washed-out shadow lifting)
- Cartographic display sharpening: ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3)

Reads ONLY sr.tif bands [3, 2, 1] (Red, Green, Blue) to write sr_preview.png.
Leaves sr.tif 100% UNTOUCHED.
"""

import os
import rasterio
import numpy as np
from PIL import Image, ImageFilter

SCENES = ["yelahanka-lake", "yelahanka-newtown", "yelahanka-afs-buffer", "yelahanka-mixed"]
SCENES_DIR = "demo/scenes"

def percentile_stretch(arr, low=1, high=99, gamma=1.0):
    lo, hi = np.percentile(arr, [low, high])
    stretched = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)
    gamma_corrected = np.power(stretched, gamma)
    return (gamma_corrected * 255).astype("uint8")

def main():
    print("=== Regenerating sr_preview.png with Tuned Parameters ===")
    print("Parameters: Gamma=1.0, UnsharpMask(radius=1.5, percent=120, threshold=3)\n")
    
    for scene_id in SCENES:
        sr_tif_path = os.path.join(SCENES_DIR, scene_id, "sr.tif")
        out_png_path = os.path.join(SCENES_DIR, scene_id, "sr_preview.png")
        
        if not os.path.exists(sr_tif_path):
            print(f"Skipping {scene_id} - sr.tif not found.")
            continue
            
        # Read bands [3, 2, 1] -> [Red, Green, Blue] from sr.tif
        with rasterio.open(sr_tif_path, "r") as src:
            rgb_bands = src.read([3, 2, 1])
            
        rgb = np.stack([percentile_stretch(rgb_bands[i], gamma=1.0) for i in range(3)], axis=-1)
        im = Image.fromarray(rgb)
        
        # Apply cartographic unsharp-mask display filter
        im_sharpened = im.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))
        im_sharpened.save(out_png_path)
        
        sz = os.path.getsize(out_png_path)
        print(f"[{scene_id}] Saved {out_png_path} ({sz:,} bytes)")

if __name__ == "__main__":
    main()
