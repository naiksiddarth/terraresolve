"""
TerraResolve viewer backend -- serves the LR / SR / HR imagery as real,
zoomable XYZ map tiles (the same tile scheme Google Maps / Leaflet
use), so you can pan and zoom into a scene and visually compare the
super-resolved output against the original medium-resolution input
and the high-resolution reference.

This is a standalone add-on module: it only *reads* from the existing
data/ and checkpoints/ folders through the registries in
terraresolve/, so it never has to know which model, which dataset
layout, or which checkpoint you're currently using -- change the
config/checkpoint and the viewer follows automatically.

Run it with:

    pip install fastapi uvicorn mercantile rio-tiler
    python webapp/main.py
    # then open http://127.0.0.1:8000 in a browser

Folder layout (kept separate from the core `terraresolve` package on
purpose, so the training/inference code has zero UI dependencies):

    webapp/
        main.py           <- this file: FastAPI app + tile endpoints
        static/
            index.html    <- Leaflet map + comparison slider (single file)
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import rasterio  # noqa: E402
from rasterio.warp import transform_bounds  # noqa: E402

import terraresolve.models  # noqa: F401,E402  (populates MODEL_REGISTRY)
from terraresolve.registry import MODEL_REGISTRY  # noqa: E402
from terraresolve.engine.inference import run_inference  # noqa: E402
from terraresolve.utils.config import load_config  # noqa: E402

try:
    import mercantile
    from rio_tiler.io import Reader
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The viewer needs two extra packages not required by the "
        "training pipeline: pip install mercantile rio-tiler"
    ) from exc

app = FastAPI(title="TerraResolve Viewer")

# ---------------------------------------------------------------------------
# In-memory state: which config/checkpoint is currently loaded, and a small
# cache of super-resolved arrays so zooming/panning doesn't re-run the model
# on every tile request.
# ---------------------------------------------------------------------------
_STATE = {
    "model": None,
    "model_name": None,
    "scale": 4,
    "sr_cache": {},   # tile_id (str) -> (np.ndarray[C,H,W] float32 0..1, rasterio profile)
}


def _lr_dir() -> Path:
    return ROOT / "data" / "sen2naip" / "lr"


def _hr_dir() -> Path:
    return ROOT / "data" / "sen2naip" / "hr"


def _load_model(config_path: str, checkpoint_path: str):
    cfg = load_config(config_path)
    model_cfg = cfg["model"]
    model = MODEL_REGISTRY.build(model_cfg["name"], **model_cfg.get("params", {}))
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    _STATE["model"] = model
    _STATE["model_name"] = f"{model_cfg['name']}::{Path(checkpoint_path).name}"
    _STATE["scale"] = model_cfg.get("params", {}).get("scale", 4)
    _STATE["sr_cache"].clear()
    return _STATE["model_name"]


@app.on_event("startup")
def _startup():
    default_ckpt = ROOT / "checkpoints" / "edsr_epoch50.pt"
    default_cfg = ROOT / "configs" / "sen2naip.yaml"
    if default_ckpt.exists() and default_cfg.exists():
        _load_model(str(default_cfg), str(default_ckpt))
        print(f"[viewer] loaded {_STATE['model_name']}")
    else:
        print("[viewer] no default checkpoint found yet -- call "
              "POST /api/model to load one before requesting SR tiles")


# ---------------------------------------------------------------------------
# Tile listing
# ---------------------------------------------------------------------------
@app.get("/api/tiles")
def list_tiles():
    """List every real LR/HR pair available, with their geographic
    bounding boxes (in WGS84), so the frontend can build a map pin /
    dropdown of real places to jump to.
    """
    lr_dir = _lr_dir()
    if not lr_dir.exists():
        return JSONResponse([])

    out = []
    for lr_path in sorted(lr_dir.glob("*.tif")):
        hr_path = _hr_dir() / lr_path.name
        if not hr_path.exists():
            continue
        try:
            with rasterio.open(lr_path) as src:
                bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        except Exception:
            continue
        out.append({
            "id": lr_path.stem,
            "bbox": bounds,  # [west, south, east, north]
            "center": [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2],
        })
    return JSONResponse(out)


@app.get("/api/model")
def get_model():
    return {"loaded": _STATE["model_name"], "scale": _STATE["scale"]}


@app.post("/api/model")
def set_model(config: str = Query(...), checkpoint: str = Query(...)):
    """Swap the checkpoint/config the viewer uses, without restarting
    the server -- e.g. to compare edsr_epoch10 vs edsr_epoch50, or
    switch to a trained SwinIR checkpoint once you have one.
    """
    try:
        name = _load_model(config, checkpoint)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"loaded": name}


# ---------------------------------------------------------------------------
# Core: run (and cache) SR inference for one tile id
# ---------------------------------------------------------------------------
def _get_sr(tile_id: str):
    if tile_id in _STATE["sr_cache"]:
        return _STATE["sr_cache"][tile_id]

    if _STATE["model"] is None:
        raise HTTPException(status_code=503, detail="No model loaded yet -- POST /api/model first")

    lr_path = _lr_dir() / f"{tile_id}.tif"
    if not lr_path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown tile id '{tile_id}'")

    with rasterio.open(lr_path) as src:
        image = src.read().astype(np.float32)
        profile = src.profile.copy()

    # Same Sentinel-2 L2A reflectance convention used in training.
    image = np.clip(image / 10000.0, 0.0, 1.0)

    scale = _STATE["scale"]
    with torch.no_grad():
        sr = run_inference(_STATE["model"], image, scale=scale, patch_size=64, overlap=8)

    sr_profile = profile.copy()
    sr_profile["width"] = sr.shape[-1]
    sr_profile["height"] = sr.shape[-2]
    transform = profile["transform"]
    sr_profile["transform"] = transform * transform.scale(1 / scale, 1 / scale)

    _STATE["sr_cache"][tile_id] = (sr, sr_profile)
    return sr, sr_profile


def _array_to_png(rgb: np.ndarray, lo: float, hi: float) -> bytes:
    """rgb: (3, H, W) float32 in reflectance units -> stretched 8-bit PNG bytes."""
    from PIL import Image
    stretched = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    img = (np.moveaxis(stretched, 0, -1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# XYZ tile endpoints -- one per source, all sharing the same URL shape so
# Leaflet can treat them as interchangeable basemap layers:
#   /tiles/{source}/{tile_id}/{z}/{x}/{y}.png   source in {lr, sr, hr}
# ---------------------------------------------------------------------------
def _stretch_bounds(hr_path: Path):
    with rasterio.open(hr_path) as src:
        arr = src.read([1, 2, 3]).astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
        lo, hi = np.percentile(arr, [1, 99])
    return float(lo), float(hi)


@app.get("/tiles/{source}/{tile_id}/{z}/{x}/{y}.png")
def get_tile(source: str, tile_id: str, z: int, x: int, y: int):
    if source not in ("lr", "sr", "hr"):
        raise HTTPException(status_code=400, detail="source must be lr, sr, or hr")

    hr_path = _hr_dir() / f"{tile_id}.tif"
    if not hr_path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown tile id '{tile_id}'")
    lo, hi = _stretch_bounds(hr_path)  # shared stretch across all 3 sources for a fair visual comparison

    if source == "hr":
        with Reader(str(hr_path)) as reader:
            img = reader.tile(x, y, z, indexes=(3, 2, 1))  # HR stored as B,G,R,NIR -> RGB
        rgb = img.data.astype(np.float32)
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
        return Response(content=_array_to_png(rgb, lo, hi), media_type="image/png")

    if source == "lr":
        lr_path = _lr_dir() / f"{tile_id}.tif"
        with Reader(str(lr_path)) as reader:
            img = reader.tile(x, y, z, indexes=(3, 2, 1))
        rgb = np.clip(img.data.astype(np.float32) / 10000.0, 0.0, 1.0)
        return Response(content=_array_to_png(rgb, lo, hi), media_type="image/png")

    # source == "sr": rio-tiler needs a real file on disk, so write the
    # (cached) SR array out once per tile_id and read tiles from that.
    sr_path = ROOT / "data" / "_sr_cache" / f"{tile_id}.tif"
    if not sr_path.exists():
        sr, sr_profile = _get_sr(tile_id)
        sr_path.parent.mkdir(parents=True, exist_ok=True)
        sr_profile.update(dtype="float32", driver="GTiff")
        with rasterio.open(sr_path, "w", **sr_profile) as dst:
            dst.write(sr.astype(np.float32))
    with Reader(str(sr_path)) as reader:
        img = reader.tile(x, y, z, indexes=(3, 2, 1))
    rgb = np.clip(img.data.astype(np.float32), 0.0, 1.0)
    return Response(content=_array_to_png(rgb, lo, hi), media_type="image/png")


app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "static"), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
