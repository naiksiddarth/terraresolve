"""
Debug what run_inference actually returns for yelahanka-lake input.tif.
Compare input visual against SR output visual channel-by-channel.
"""
import sys, os
sys.path.insert(0, '.')
import numpy as np
import rasterio
from PIL import Image

import terraresolve.models  # noqa
from terraresolve.registry import MODEL_REGISTRY
from terraresolve.engine.inference import run_inference

BAND_ORDER = [3, 2, 1, 4]
SCALE = 4

scene_id = "yelahanka-lake"
in_path = f"data/raw/{scene_id}/input.tif"

print("=== Reading input ===")
with rasterio.open(in_path) as src:
    arr = src.read(BAND_ORDER)
    print(f"arr shape: {arr.shape}")
    print(f"arr channel means: {[arr[i].mean() for i in range(arr.shape[0])]}")
    print(f"bounds: {src.bounds}")

# Save what we're feeding into the model as a sanity preview
def percentile_stretch(a, lo=2, hi=98):
    p_lo, p_hi = np.percentile(a, [lo, hi])
    return np.clip(((a - p_lo) / (p_hi - p_lo + 1e-6)) * 255, 0, 255).astype('uint8')

rgb_in = np.stack([percentile_stretch(arr[i]) for i in range(3)], axis=-1)
Image.fromarray(rgb_in).save("scripts/debug_input_fed_to_model.png")
print("Saved: scripts/debug_input_fed_to_model.png (should match input_preview.png)")

print("\n=== Running inference ===")
model = MODEL_REGISTRY.build("sen2sr", weights_dir="models/pretrained/sen2sr_rgbn_x4", device="cuda")
model.eval()
X = (arr / 10_000).astype("float32")
sr_array = run_inference(model, X, scale=SCALE, device="cuda", patch_size=128, overlap=32)

print(f"sr_array shape: {sr_array.shape}")
print(f"sr_array channel means: {[sr_array[i].mean() for i in range(sr_array.shape[0])]}")
print(f"sr_array min/max: {sr_array.min():.4f} / {sr_array.max():.4f}")

sr_uint16 = (np.clip(sr_array, 0, 1) * 10_000).astype("uint16")
rgb_sr = np.stack([percentile_stretch(sr_uint16[i]) for i in range(3)], axis=-1)
Image.fromarray(rgb_sr).save("scripts/debug_sr_output.png")
print("Saved: scripts/debug_sr_output.png (should look like a 4x upscale of input)")
print("DONE - compare the two PNGs to confirm model input/output spatial consistency")
