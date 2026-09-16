from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import json
import os

router = APIRouter()
SCENES_DIR = "demo/scenes"

def get_scene_dir(scene_id: str) -> str:
    path = os.path.join(SCENES_DIR, scene_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Scene not found")
    return path

@router.get("")
async def list_scenes():
    index_path = os.path.join(SCENES_DIR, "index.json")
    if not os.path.exists(index_path):
        return []
    with open(index_path, "r") as f:
        return json.load(f)

@router.get("/{scene_id}")
async def get_scene_metadata(scene_id: str):
    scene_dir = get_scene_dir(scene_id)
    metadata_path = os.path.join(scene_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise HTTPException(status_code=404, detail="Metadata not found")
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    # Try to load metrics as well if available
    metrics_path = os.path.join(scene_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metadata["metrics"] = json.load(f)
            
    return metadata

@router.get("/{scene_id}/input")
async def get_scene_input_preview(scene_id: str):
    scene_dir = get_scene_dir(scene_id)
    preview_path = os.path.join(scene_dir, "input_preview.png")
    if not os.path.exists(preview_path):
        raise HTTPException(status_code=404, detail="Input preview not found")
    return FileResponse(preview_path, media_type="image/png")

@router.get("/{scene_id}/sr")
async def get_scene_sr_preview(scene_id: str):
    scene_dir = get_scene_dir(scene_id)
    preview_path = os.path.join(scene_dir, "sr_preview.png")
    if not os.path.exists(preview_path):
        raise HTTPException(status_code=404, detail="SR preview not found")
    return FileResponse(preview_path, media_type="image/png")

@router.get("/{scene_id}/metrics")
async def get_scene_metrics(scene_id: str):
    scene_dir = get_scene_dir(scene_id)
    metrics_path = os.path.join(scene_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Metrics not found")
    with open(metrics_path, "r") as f:
        return json.load(f)

@router.get("/{scene_id}/downstream")
async def get_scene_downstream(scene_id: str, format: str = None, layer: str = None):
    scene_dir = get_scene_dir(scene_id)
    if format == "json":
        json_path = os.path.join(scene_dir, "downstream_metrics.json")
        if not os.path.exists(json_path):
            raise HTTPException(status_code=404, detail="Downstream metrics not found")
        with open(json_path, "r") as f:
            return json.load(f)
            
    if layer in ("overlay", "mask"):
        overlay_path = os.path.join(scene_dir, "downstream_mask_sr.png")
        if not os.path.exists(overlay_path):
            raise HTTPException(status_code=404, detail="Downstream overlay not found")
        return FileResponse(overlay_path, media_type="image/png")

    downstream_path = os.path.join(scene_dir, "downstream.png")
    if not os.path.exists(downstream_path):
        raise HTTPException(status_code=404, detail="Downstream result not found")
    return FileResponse(downstream_path, media_type="image/png")

@router.get("/{scene_id}/downstream/metrics")
async def get_scene_downstream_metrics(scene_id: str):
    scene_dir = get_scene_dir(scene_id)
    json_path = os.path.join(scene_dir, "downstream_metrics.json")
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Downstream metrics not found")
    with open(json_path, "r") as f:
        return json.load(f)

@router.get("/{scene_id}/downstream/overlay")
async def get_scene_downstream_overlay(scene_id: str):
    scene_dir = get_scene_dir(scene_id)
    overlay_path = os.path.join(scene_dir, "downstream_mask_sr.png")
    if not os.path.exists(overlay_path):
        raise HTTPException(status_code=404, detail="Downstream overlay not found")
    return FileResponse(overlay_path, media_type="image/png")

