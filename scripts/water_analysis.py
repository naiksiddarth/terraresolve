"""
Downstream Geospatial Application: NDWI Water Body Delineation Analysis.

Evaluates downstream utility on Sentinel-2 input.tif (10m) vs SEN2SR sr.tif (2.5m).
NDWI = (Green - NIR) / (Green + NIR) = (B03 - B08) / (B03 + B08)

Honesty framing (per project specification):
This analysis measures boundary-delineation precision at higher pixel density (4x),
not new spectral information. Super-resolution resolves mixed-pixel boundaries
along land-water transitions rather than creating novel spectral measurements.
"""

import os
import sys
import json
import argparse
import rasterio
import numpy as np
import scipy.ndimage as ndi
from PIL import Image, ImageDraw, ImageFont

HONESTY_STATEMENT = (
    "Downstream boundary-delineation precision at 4x pixel density (10m -> 2.5m). "
    "Note: SR does not create new spectral information; it resolves mixed-pixel boundaries."
)

def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Compute Normalized Difference Water Index (McFeeters 1996)."""
    green_f = green.astype(np.float32)
    nir_f = nir.astype(np.float32)
    denominator = np.clip(green_f + nir_f, 1e-6, None)
    return (green_f - nir_f) / denominator

def extract_shoreline(water_mask: np.ndarray) -> np.ndarray:
    """Extract shoreline pixels (water pixels adjacent to non-water / land)."""
    if water_mask.sum() == 0:
        return np.zeros_like(water_mask, dtype=bool)
    # 4-connectivity erosion of water mask leaves only strictly interior water pixels
    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]], dtype=bool)
    interior = ndi.binary_erosion(water_mask, structure=structure)
    return water_mask ^ interior

def analyze_scene_water(scene_dir: str, threshold: float = 0.0) -> dict:
    """Run NDWI water analysis on input.tif and sr.tif in scene_dir."""
    in_tif = os.path.join(scene_dir, "input.tif")
    sr_tif = os.path.join(scene_dir, "sr.tif")

    if not os.path.exists(in_tif) or not os.path.exists(sr_tif):
        raise FileNotFoundError(f"Missing input.tif or sr.tif in {scene_dir}")

    # Band ordering: Band 1: Blue, Band 2: Green (B03), Band 3: Red, Band 4: NIR (B08)
    with rasterio.open(in_tif) as src_lr:
        green_lr = src_lr.read(2)
        nir_lr = src_lr.read(4)
        rgb_lr = src_lr.read([3, 2, 1])
        lr_profile = src_lr.profile

    with rasterio.open(sr_tif) as src_sr:
        green_sr = src_sr.read(2)
        nir_sr = src_sr.read(4)
        rgb_sr = src_sr.read([3, 2, 1])
        sr_profile = src_sr.profile

    # Compute NDWI
    ndwi_lr = compute_ndwi(green_lr, nir_lr)
    ndwi_sr = compute_ndwi(green_sr, nir_sr)

    # Threshold binary water masks
    mask_lr = (ndwi_lr > threshold)
    mask_sr = (ndwi_sr > threshold)

    # Extract shorelines
    shoreline_lr = extract_shoreline(mask_lr)
    shoreline_sr = extract_shoreline(mask_sr)

    # Metrics
    # 10m Sentinel-2 pixel = 100 m2 = 0.01 ha
    lr_px_count = int(mask_lr.sum())
    lr_area_m2 = float(lr_px_count * 100.0)
    lr_area_ha = float(lr_area_m2 / 10000.0)
    lr_shoreline_px = int(shoreline_lr.sum())
    lr_perimeter_m = float(lr_shoreline_px * 10.0)

    # 2.5m SEN2SR pixel = 6.25 m2 = 0.000625 ha
    sr_px_count = int(mask_sr.sum())
    sr_area_m2 = float(sr_px_count * 6.25)
    sr_area_ha = float(sr_area_m2 / 10000.0)
    sr_shoreline_px = int(shoreline_sr.sum())
    sr_perimeter_m = float(sr_shoreline_px * 2.5)

    area_delta_ha = float(sr_area_ha - lr_area_ha)
    area_change_pct = float((area_delta_ha / (lr_area_ha + 1e-6)) * 100.0) if lr_area_ha > 0 else 0.0
    shoreline_ratio = float(sr_shoreline_px / max(lr_shoreline_px, 1))

    metrics = {
        "scene_id": os.path.basename(os.path.normpath(scene_dir)),
        "task": "NDWI Water-Body Boundary Delineation",
        "threshold": threshold,
        "formula": "NDWI = (Green - NIR) / (Green + NIR)",
        "bands": {
            "green": "B03 (Band 2)",
            "nir": "B08 (Band 4)"
        },
        "input_10m": {
            "resolution_m": 10.0,
            "pixel_area_m2": 100.0,
            "water_pixels": lr_px_count,
            "area_m2": round(lr_area_m2, 2),
            "area_ha": round(lr_area_ha, 3),
            "shoreline_pixels": lr_shoreline_px,
            "estimated_perimeter_m": round(lr_perimeter_m, 2),
            "mean_ndwi_water": float(round(ndwi_lr[mask_lr].mean(), 4)) if lr_px_count > 0 else None
        },
        "sr_2.5m": {
            "resolution_m": 2.5,
            "pixel_area_m2": 6.25,
            "water_pixels": sr_px_count,
            "area_m2": round(sr_area_m2, 2),
            "area_ha": round(sr_area_ha, 3),
            "shoreline_pixels": sr_shoreline_px,
            "estimated_perimeter_m": round(sr_perimeter_m, 2),
            "mean_ndwi_water": float(round(ndwi_sr[mask_sr].mean(), 4)) if sr_px_count > 0 else None
        },
        "comparison": {
            "area_delta_ha": round(area_delta_ha, 3),
            "area_change_pct": round(area_change_pct, 2),
            "shoreline_pixel_density_ratio": round(shoreline_ratio, 2),
            "perimeter_resolved_m_delta": round(sr_perimeter_m - lr_perimeter_m, 2)
        },
        "honesty_framing": HONESTY_STATEMENT
    }

    # Save metrics JSON
    json_path = os.path.join(scene_dir, "downstream_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # 1. Generate transparent GeoTIFF-aligned overlay for Leaflet viewer (804x804 RGBA)
    # Water: translucent cyan (0, 180, 240, 110), Shoreline: vibrant gold/yellow (255, 215, 0, 240)
    overlay_h, overlay_w = mask_sr.shape
    rgba_sr = np.zeros((overlay_h, overlay_w, 4), dtype=np.uint8)
    rgba_sr[mask_sr] = [14, 165, 233, 110]        # Sky-blue / cyan fill
    rgba_sr[shoreline_sr] = [250, 204, 21, 255]   # Amber / gold crisp boundary

    overlay_img_path = os.path.join(scene_dir, "downstream_mask_sr.png")
    Image.fromarray(rgba_sr, "RGBA").save(overlay_img_path)

    # 2. Generate comprehensive visual infographic overlay: downstream.png
    generate_downstream_summary_graphic(
        scene_dir=scene_dir,
        rgb_lr=rgb_lr,
        rgb_sr=rgb_sr,
        mask_lr=mask_lr,
        mask_sr=mask_sr,
        shoreline_lr=shoreline_lr,
        shoreline_sr=shoreline_sr,
        metrics=metrics
    )

    return metrics

def stretch_rgb(rgb_arr: np.ndarray) -> np.ndarray:
    """Standard percentile stretch (1-99%) with linear gamma (1.0)."""
    out = []
    for i in range(3):
        lo, hi = np.percentile(rgb_arr[i], [1, 99])
        s = np.clip((rgb_arr[i] - lo) / (hi - lo + 1e-6), 0, 1)
        out.append((s * 255).astype(np.uint8))
    return np.stack(out, axis=-1)

def generate_downstream_summary_graphic(
    scene_dir: str,
    rgb_lr: np.ndarray,
    rgb_sr: np.ndarray,
    mask_lr: np.ndarray,
    mask_sr: np.ndarray,
    shoreline_lr: np.ndarray,
    shoreline_sr: np.ndarray,
    metrics: dict
):
    """Generate downstream.png showing side-by-side comparison with metrics and honesty banner."""
    panel_size = 500
    
    # Base RGB images stretched to 8-bit
    base_lr = Image.fromarray(stretch_rgb(rgb_lr)).resize((panel_size, panel_size), Image.Resampling.NEAREST)
    base_sr = Image.fromarray(stretch_rgb(rgb_sr)).resize((panel_size, panel_size), Image.Resampling.BILINEAR)

    # Overlay LR mask
    overlay_lr = np.zeros((mask_lr.shape[0], mask_lr.shape[1], 4), dtype=np.uint8)
    overlay_lr[mask_lr] = [14, 165, 233, 110]
    overlay_lr[shoreline_lr] = [239, 68, 68, 255] # Red boundary for coarse LR
    img_ov_lr = Image.fromarray(overlay_lr, "RGBA").resize((panel_size, panel_size), Image.Resampling.NEAREST)
    comp_lr = Image.alpha_composite(base_lr.convert("RGBA"), img_ov_lr)

    # Overlay SR mask
    overlay_sr = np.zeros((mask_sr.shape[0], mask_sr.shape[1], 4), dtype=np.uint8)
    overlay_sr[mask_sr] = [14, 165, 233, 110]
    overlay_sr[shoreline_sr] = [250, 204, 21, 255] # Amber boundary for fine SR
    img_ov_sr = Image.fromarray(overlay_sr, "RGBA").resize((panel_size, panel_size), Image.Resampling.BILINEAR)
    comp_sr = Image.alpha_composite(base_sr.convert("RGBA"), img_ov_sr)

    # Difference / Overlap panel: compare upscale of LR mask vs SR mask
    mask_lr_upscaled = np.array(Image.fromarray(mask_lr.astype(np.uint8)).resize((mask_sr.shape[1], mask_sr.shape[0]), Image.Resampling.NEAREST)) > 0
    diff_rgba = np.zeros((mask_sr.shape[0], mask_sr.shape[1], 4), dtype=np.uint8)
    # Both agree water: deep blue
    both = mask_lr_upscaled & mask_sr
    diff_rgba[both] = [37, 99, 235, 150]
    # Newly resolved water fringe in SR (missed by LR due to mixed-pixel attenuation): emerald green
    sr_only = (~mask_lr_upscaled) & mask_sr
    diff_rgba[sr_only] = [16, 185, 129, 220]
    # LR only (coarse stair-step false positive): orange
    lr_only = mask_lr_upscaled & (~mask_sr)
    diff_rgba[lr_only] = [249, 115, 22, 220]

    img_diff = Image.fromarray(diff_rgba, "RGBA").resize((panel_size, panel_size), Image.Resampling.BILINEAR)
    comp_diff = Image.alpha_composite(base_sr.convert("RGBA"), img_diff)

    # Build composite canvas
    header_h = 100
    footer_h = 70
    canvas_w = panel_size * 3 + 40
    canvas_h = panel_size + header_h + footer_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(15, 23, 42)) # Slate 900
    draw = ImageDraw.Draw(canvas)

    # Header Title & Honesty Callout
    draw.text((20, 16), "TerraResolve Downstream Geospatial Evaluation: NDWI Water-Body Delineation", fill=(255, 255, 255))
    draw.text((20, 42), f"Scene: {metrics['scene_id']} | NDWI = (B03 - B08) / (B03 + B08) > {metrics['threshold']}", fill=(148, 163, 184))
    draw.text((20, 68), f"HONESTY FRAMING: {HONESTY_STATEMENT}", fill=(56, 189, 248))

    # Paste panels
    x0 = 15
    y_panels = header_h
    canvas.paste(comp_lr.convert("RGB"), (x0, y_panels))
    canvas.paste(comp_sr.convert("RGB"), (x0 + panel_size + 10, y_panels))
    canvas.paste(comp_diff.convert("RGB"), (x0 + (panel_size + 10) * 2, y_panels))

    # Panel captions
    draw.text((x0 + 10, y_panels + 10), "Sentinel-2 LR (10m)", fill=(255, 255, 255))
    draw.text((x0 + 10, y_panels + 30), f"Area: {metrics['input_10m']['area_ha']:.2f} ha ({metrics['input_10m']['water_pixels']} px)", fill=(226, 232, 240))
    draw.text((x0 + 10, y_panels + 50), f"Shoreline: {metrics['input_10m']['shoreline_pixels']} px ({metrics['input_10m']['estimated_perimeter_m']:.0f}m)", fill=(248, 113, 113))

    draw.text((x0 + panel_size + 20, y_panels + 10), "SEN2SR Enhanced (2.5m)", fill=(255, 255, 255))
    draw.text((x0 + panel_size + 20, y_panels + 30), f"Area: {metrics['sr_2.5m']['area_ha']:.2f} ha ({metrics['sr_2.5m']['water_pixels']} px)", fill=(226, 232, 240))
    draw.text((x0 + panel_size + 20, y_panels + 50), f"Shoreline: {metrics['sr_2.5m']['shoreline_pixels']} px ({metrics['sr_2.5m']['estimated_perimeter_m']:.0f}m)", fill=(250, 204, 21))

    draw.text((x0 + (panel_size + 10) * 2 + 10, y_panels + 10), "Boundary Resolution Delta", fill=(255, 255, 255))
    draw.text((x0 + (panel_size + 10) * 2 + 10, y_panels + 30), f"Area Delta: +{metrics['comparison']['area_delta_ha']:.2f} ha (+{metrics['comparison']['area_change_pct']:.1f}%)", fill=(52, 211, 153))
    draw.text((x0 + (panel_size + 10) * 2 + 10, y_panels + 50), f"Shoreline Pixel Density: {metrics['comparison']['shoreline_pixel_density_ratio']:.1f}x higher", fill=(250, 204, 21))

    # Footer metrics bar
    footer_y = y_panels + panel_size + 16
    draw.text((20, footer_y), "Legend:", fill=(148, 163, 184))
    draw.text((80, footer_y), "[Blue] Core Water Concordance", fill=(96, 165, 250))
    draw.text((320, footer_y), "[Emerald] Resolved Shoreline Fringes (Unmixed in SR)", fill=(52, 211, 153))
    draw.text((680, footer_y), "[Orange] 10m Coarse Stair-Step Quantization", fill=(251, 146, 60))
    draw.text((1050, footer_y), "[Yellow] 2.5m Delineated Shoreline", fill=(250, 204, 21))

    out_path = os.path.join(scene_dir, "downstream.png")
    canvas.save(out_path)
    print(f"[{metrics['scene_id']}] Saved downstream.png -> {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Run NDWI downstream water analysis on demo scenes.")
    parser.add_argument("--scene", type=str, default="all", help="Scene ID or 'all'")
    parser.add_argument("--scenes-dir", type=str, default="demo/scenes", help="Directory containing demo scenes")
    parser.add_argument("--threshold", type=float, default=0.0, help="NDWI water threshold (default: 0.0)")
    args = parser.parse_args()

    if args.scene == "all":
        scenes = [d for d in os.listdir(args.scenes_dir) if os.path.isdir(os.path.join(args.scenes_dir, d))]
    else:
        scenes = [args.scene]

    for s in scenes:
        s_dir = os.path.join(args.scenes_dir, s)
        if os.path.exists(os.path.join(s_dir, "input.tif")) and os.path.exists(os.path.join(s_dir, "sr.tif")):
            print(f"\nProcessing downstream water analysis for: {s}")
            m = analyze_scene_water(s_dir, threshold=args.threshold)
            print(f" - LR Water Area: {m['input_10m']['area_ha']} ha | Shoreline: {m['input_10m']['shoreline_pixels']} px")
            print(f" - SR Water Area: {m['sr_2.5m']['area_ha']} ha | Shoreline: {m['sr_2.5m']['shoreline_pixels']} px")
            print(f" - Shoreline Density Increase: {m['comparison']['shoreline_pixel_density_ratio']}x")

if __name__ == "__main__":
    main()
