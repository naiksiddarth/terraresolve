"""Full-scene inference: tiles a georeferenced raster into
overlapping patches (so stitching seams don't show), runs the model
on each, blends the overlaps, mosaics the result back together, and
writes a GeoTIFF whose transform is rescaled to match the finer pixel
size -- the step that makes the output actually usable in GIS rather
than just 'a bigger picture'."""
from __future__ import annotations
from pathlib import Path
from typing import Union

import numpy as np
import torch

try:
    import rasterio
    from rasterio.transform import Affine
except ImportError:
    rasterio = None
    Affine = None


def _blend_weight(size: int, overlap: int) -> np.ndarray:
    ramp = np.ones(size, dtype=np.float32)
    if overlap > 0:
        edge = np.linspace(0, 1, overlap, dtype=np.float32)
        ramp[:overlap] = edge
        ramp[-overlap:] = edge[::-1]
    return np.outer(ramp, ramp)


@torch.no_grad()
def run_inference(model: torch.nn.Module, image: np.ndarray, scale: int,
                   patch_size: int = 64, overlap: int = 8, device: str = "cpu") -> np.ndarray:
    model.eval().to(device)
    c, h, w = image.shape
    step = patch_size - overlap
    out_h, out_w = h * scale, w * scale

    probe = model(torch.zeros(1, c, patch_size, patch_size, device=device))
    out_c = probe.shape[1]

    canvas = np.zeros((out_c, out_h, out_w), dtype=np.float32)
    weight = np.zeros((out_h, out_w), dtype=np.float32)
    blend = _blend_weight(patch_size * scale, overlap * scale)

    ys = list(range(0, max(h - patch_size, 0) + 1, step)) or [0]
    xs = list(range(0, max(w - patch_size, 0) + 1, step)) or [0]
    if ys[-1] + patch_size < h:
        ys.append(h - patch_size)
    if xs[-1] + patch_size < w:
        xs.append(w - patch_size)

    for y in ys:
        for x in xs:
            patch = image[:, y:y + patch_size, x:x + patch_size]
            ph, pw = patch.shape[1], patch.shape[2]
            if ph < patch_size or pw < patch_size:
                patch = np.pad(patch, ((0, 0), (0, patch_size - ph), (0, patch_size - pw)), mode="reflect")
            tensor = torch.from_numpy(patch).unsqueeze(0).float().to(device)
            sr = model(tensor)[0].cpu().numpy()

            oy, ox, oh, ow = y * scale, x * scale, ph * scale, pw * scale
            canvas[:, oy:oy + oh, ox:ox + ow] += sr[:, :oh, :ow] * blend[:oh, :ow]
            weight[oy:oy + oh, ox:ox + ow] += blend[:oh, :ow]

    weight = np.clip(weight, 1e-6, None)
    return canvas / weight[None, ...]


def write_geotiff(out_path: Union[str, Path], array: np.ndarray, ref_profile: dict, scale: int):
    if rasterio is None:
        raise ImportError("rasterio is required to write georeferenced output")
    profile = ref_profile.copy()
    old_t: Affine = profile["transform"]
    new_t = Affine(old_t.a / scale, old_t.b, old_t.c, old_t.d, old_t.e / scale, old_t.f)
    profile.update(height=array.shape[1], width=array.shape[2], count=array.shape[0],
                    transform=new_t, dtype="float32")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(array.astype(np.float32))
