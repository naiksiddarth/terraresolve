import threading, time, traceback
from datetime import datetime, timezone
from backend.db.database import get_conn
from backend.config import PATCH_SIZE, OVERLAP, SCALE
from backend.geospatial.validation import validate_output_geotiff

def now():
    return datetime.now(timezone.utc).isoformat()

def _update(conn, job_id, **fields):
    fields["updated_at"] = now()
    cols = ", ".join(f"{k}=?" for k in fields)
    conn.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))
    conn.commit()

def process_job(conn, row, model, device):
    import rasterio
    import numpy as np
    from rasterio.transform import Affine
    from terraresolve.engine.inference import run_inference

    job_id = row["id"]
    _update(conn, job_id, status="RUNNING", started_at=now())

    try:
        _update(conn, job_id, status="PREPROCESSING", stage="reading input", progress_pct=10)
        with rasterio.open(row["input_path"]) as src:
            arr = src.read()
            profile = src.profile
        # NOTE: assumes first 4 bands are already [B04,B03,B02,B08] order.
        # If arbitrary user uploads can have different band order, this needs
        # real band-order detection before it's safe against real user files.
        X = (arr[:4] / 10_000).astype("float32")

        _update(conn, job_id, status="INFERENCE", stage="running SEN2SR", progress_pct=30)
        sr = run_inference(model, X, scale=SCALE, device=device, patch_size=PATCH_SIZE, overlap=OVERLAP)

        _update(conn, job_id, status="POSTPROCESSING", stage="writing GeoTIFF", progress_pct=90)
        sr_uint16 = (np.clip(sr, 0, 1) * 10_000).astype("uint16")
        t = profile["transform"]
        new_transform = Affine(t.a / SCALE, t.b, t.c, t.d, t.e / SCALE, t.f)
        out_profile = profile.copy()
        out_profile.update(transform=new_transform, width=sr_uint16.shape[2],
                            height=sr_uint16.shape[1], count=sr_uint16.shape[0], dtype="uint16")
        output_path = row["input_path"].replace("input.tif", "output.tif")
        with rasterio.open(output_path, "w", **out_profile) as dst:
            dst.write(sr_uint16)

        validate_output_geotiff(output_path)

        _update(conn, job_id, status="COMPLETED", stage="done", progress_pct=100,
                output_path=output_path, completed_at=now())
    except Exception as e:
        _update(conn, job_id, status="FAILED", error_message=f"{e}\n{traceback.format_exc()[-500:]}")

def worker_loop(model, device, poll_interval=1.5):
    conn = get_conn()
    while True:
        row = conn.execute(
            "SELECT * FROM jobs WHERE status='QUEUED' ORDER BY created_at LIMIT 1"
        ).fetchone()
        if row:
            process_job(conn, row, model, device)  # strictly serial by design — do not parallelize
        else:
            time.sleep(poll_interval)

def start_worker(model, device):
    t = threading.Thread(target=worker_loop, args=(model, device), daemon=True)
    t.start()
    return t
