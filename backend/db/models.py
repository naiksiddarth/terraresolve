from pydantic import BaseModel
from typing import Optional

class Job(BaseModel):
    id: str
    status: str
    stage: Optional[str] = None
    progress_pct: int = 0
    source_type: str
    model_version: Optional[str] = None
    input_path: Optional[str] = None
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    updated_at: str
