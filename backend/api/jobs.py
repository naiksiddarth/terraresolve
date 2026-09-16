import uuid
import shutil
import os
from datetime import datetime, timezone
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from backend.db.database import get_conn
from backend.geospatial.validation import validate_upload, ValidationError
from backend.config import MAX_UPLOAD_MB, DATA_DIR

router = APIRouter()

def now():
    return datetime.now(timezone.utc).isoformat()

@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(400, "File must be .tif or .tiff")

    job_id = str(uuid.uuid4())
    job_dir = f"{DATA_DIR}/jobs/{job_id}"
    os.makedirs(job_dir, exist_ok=True)
    input_path = f"{job_dir}/input.tif"

    size = 0
    with open(input_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1024 * 1024:
                f.close()
                shutil.rmtree(job_dir)
                raise HTTPException(413, f"File exceeds {MAX_UPLOAD_MB}MB limit")
            f.write(chunk)

    try:
        validate_upload(input_path)
    except ValidationError as e:
        shutil.rmtree(job_dir)
        raise HTTPException(400, str(e))

    conn = get_conn()
    conn.execute(
        "INSERT INTO jobs (id, status, progress_pct, source_type, input_path, created_at, updated_at) "
        "VALUES (?, 'QUEUED', 0, 'upload', ?, ?, ?)",
        (job_id, input_path, now(), now())
    )
    conn.commit()
    return {"job_id": job_id, "status": "QUEUED", "poll_url": f"/api/jobs/{job_id}"}

@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    conn = get_conn()
    row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Job not found")
    return dict(row)

@router.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    conn = get_conn()
    row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Job not found")
    if row["status"] != "COMPLETED":
        raise HTTPException(409, f"Job is {row['status']}, not COMPLETED")
    return FileResponse(row["output_path"], filename=f"{job_id}_sr.tif")
