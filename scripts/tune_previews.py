import rasterio
import numpy as np
from PIL import Image, ImageFilter
import os
import itertools

def get_rgb_from_tif(path):
    with rasterio.open(path) as src:
        # File is [B, G, R, NIR], we want [R, G, B]
        arr = src.read([3, 2, 1])
    return arr

def apply_stretch_and_gamma(arr, gamma):
    rgb = np.zeros_like(arr, dtype=np.float32)
    for i in range(3):
        lo, hi = np.percentile(arr[i], [1, 99])
        stretched = np.clip((arr[i] - lo) / (hi - lo + 1e-6), 0, 1)
        rgb[i] = np.power(stretched, gamma)
    
    rgb = np.stack([rgb[0], rgb[1], rgb[2]], axis=-1)
    return (rgb * 255).astype("uint8")

def tune_scene():
    sr_path = "demo/scenes/yelahanka-lake/sr.tif"
    arr = get_rgb_from_tif(sr_path)
    
    gammas = [0.7, 0.85, 1.0, 1.1]
    sharpen_opts = [False, True]
    
    out_dir = "demo/tune_previews"
    os.makedirs(out_dir, exist_ok=True)
    
    results = {}
    
    for gamma in gammas:
        img_arr = apply_stretch_and_gamma(arr, gamma)
        img = Image.fromarray(img_arr)
        
        for sharpen in sharpen_opts:
            final_img = img.copy()
            if sharpen:
                # Radius 2, Percent 150, Threshold 3 is a standard moderate unsharp mask
                final_img = final_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            
            name = f"gamma_{gamma}_sharp_{sharpen}.png"
            final_img.save(f"{out_dir}/{name}")
            results[name] = final_img
            print(f"Saved {name}")

    # Create a grid: rows=gammas, cols=sharpen (False, True)
    w, h = results["gamma_1.0_sharp_False.png"].size
    grid = Image.new('RGB', (w * 2, h * 4))
    
    for i, g in enumerate(gammas):
        for j, s in enumerate(sharpen_opts):
            name = f"gamma_{g}_sharp_{s}.png"
            grid.paste(results[name], (j * w, i * h))
            
    grid.save(f"{out_dir}/grid.png")
    print("Saved grid.png")

if __name__ == "__main__":
    tune_scene()
