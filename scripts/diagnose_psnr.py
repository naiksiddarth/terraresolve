import sys, time, pathlib, shutil
sys.path.insert(0, '.')

import numpy as np
import torch
import safetensors.torch
import matplotlib.pyplot as plt

# We'll use scikit-image for SSIM if available
try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-image"])
    from skimage.metrics import structural_similarity as ssim

WEIGHTS_DIR = 'models/pretrained/sen2sr_rgbn_x4'
EXAMPLE_DATA = pathlib.Path(WEIGHTS_DIR) / 'example_data.safetensor'

# Load original data
sample = safetensors.torch.load_file(str(EXAMPLE_DATA))
lr_real = sample['lr'][0].float().numpy()  # (4, 128, 128)
hr_real = sample['hr'][0].float().numpy()  # (4, 512, 512)

# Load previously computed SR
sr_real = np.load('previews/sen2sr_sr_128.npy') # (4, 512, 512)

print("=== Band-wise Means ===")
bands = ['B04 (Red)', 'B03 (Green)', 'B02 (Blue)', 'B08 (NIR)']
for i, band in enumerate(bands):
    lr_mean = lr_real[i].mean()
    hr_mean = hr_real[i].mean()
    sr_mean = sr_real[i].mean()
    print(f"{band:15s} | LR: {lr_mean:.4f} | HR: {hr_mean:.4f} | SR: {sr_mean:.4f}")

# Overall means
print(f"Overall         | LR: {lr_real.mean():.4f} | HR: {hr_real.mean():.4f} | SR: {sr_real.mean():.4f}")
print()

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
# PSNR
mse = np.mean((sr_real - hr_real) ** 2)
psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')

# SSIM (needs channel_axis)
# scikit-image ssim expects (H, W, C) for multichannel
sr_hwc = np.transpose(sr_real, (1, 2, 0))
hr_hwc = np.transpose(hr_real, (1, 2, 0))
# compute SSIM using data_range=1.0 since values are ~[0,1]
ssim_val = ssim(hr_hwc, sr_hwc, data_range=1.0, channel_axis=-1)

print("=== Metrics ===")
print(f"PSNR: {psnr:.2f} dB")
print(f"SSIM: {ssim_val:.4f}")
print()

# ---------------------------------------------------------------------------
# Generate Diagnostic Image
# ---------------------------------------------------------------------------
# We'll save this directly to the conversation's artifact directory so I can embed it
ARTIFACT_DIR = pathlib.Path(r"C:\Users\shawn\.gemini\antigravity-ide\brain\017f8cab-5880-4b4f-896c-25865a7673cf")

def to_rgb(arr, boost=3.0):
    rgb = arr[:3].transpose(1, 2, 0) * boost
    return np.clip(rgb, 0, 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(to_rgb(lr_real));   axes[0].set_title('LR Input (Sentinel-2 10m)')
axes[1].imshow(to_rgb(sr_real));   axes[1].set_title('SEN2SR Output (2.5m)')
axes[2].imshow(to_rgb(hr_real));   axes[2].set_title(f'HR Reference (NAIP 2.5m)\nPSNR={psnr:.1f}dB, SSIM={ssim_val:.3f}')
for ax in axes:
    ax.axis('off')
fig.tight_layout()
out_png = ARTIFACT_DIR / 'sen2sr_diagnostic.png'
fig.savefig(out_png, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved diagnostic image to: {out_png}")
