from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch, sys
sys.path.insert(0, '.')
import terraresolve.models  # noqa: registers "sen2sr" in MODEL_REGISTRY
from terraresolve.registry import MODEL_REGISTRY
from backend.db.database import get_conn, init_db, recover_interrupted_jobs
from backend.worker import start_worker
from backend.config import WEIGHTS_DIR
from backend.api import health, scenes, jobs

app = FastAPI(title="TerraResolve API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict to the deployed frontend origin before demo day
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(scenes.router, prefix="/api/scenes")
app.include_router(jobs.router, prefix="/api")

@app.on_event("startup")
async def startup_event():
    conn = get_conn()
    init_db(conn)
    recover_interrupted_jobs(conn)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MODEL_REGISTRY.build("sen2sr", weights_dir=WEIGHTS_DIR, device=device)
    model.eval()

    app.state.model = model
    app.state.device = device
    app.state.model_ready = True

    start_worker(model, device)
    print(f"TerraResolve backend ready. Device: {device}")
