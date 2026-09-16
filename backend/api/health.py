from fastapi import APIRouter, Request

router = APIRouter()

@router.get("/health")
async def health():
    return {"status": "ok"}

@router.get("/health/ready")
async def ready(request: Request):
    is_ready = getattr(request.app.state, "model_ready", False)
    return {
        "status": "ok" if is_ready else "not_ready",
        "model_loaded": is_ready,
        "device": getattr(request.app.state, "device", None),
    }
