# backend/app.py

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.core.trainer import (
    TrainConfig,
    build_train_config,
    start_training,
    validate_dataset,
)

app = FastAPI(title="LLM Fine-tuning Backend", version="0.1.0")

# Allow Streamlit frontend on localhost:8501
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Paths / storage
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# In-memory training state
# ----------------------------
training_state: Dict[str, Any] = {
    "status": "idle",      # idle | running | success | failed
    "message": "Ready.",
    "config": None,
    "result": None,
}
state_lock = threading.Lock()


# ----------------------------
# Request models
# ----------------------------
class DatasetValidateRequest(BaseModel):
    dataset_path: str = Field(..., description="Path to the JSONL dataset file")


class TrainRequest(BaseModel):
    base_model: str
    dataset_path: str
    finetune_method: str = "LoRA"
    output_dir: str = "./outputs/finetuned_model"

    learning_rate: float = 2e-4
    batch_size: int = 4
    epochs: int = 3
    lora_rank: int = 8
    max_length: int = 512


# ----------------------------
# State helpers
# ----------------------------
def set_state(**kwargs: Any) -> None:
    with state_lock:
        training_state.update(kwargs)


def get_state() -> Dict[str, Any]:
    with state_lock:
        return dict(training_state)


def run_training_in_background(config: TrainConfig) -> None:
    """
    Background worker for training.
    Updates global training_state after completion.
    """
    try:
        set_state(
            status="running",
            message="Training started.",
            config=config.__dict__,
            result=None,
        )

        result = start_training(config)

        if result.success:
            set_state(
                status="success",
                message=result.message,
                result=result.__dict__,
            )
        else:
            set_state(
                status="failed",
                message=result.message,
                result=result.__dict__,
            )

    except Exception as e:
        set_state(
            status="failed",
            message=f"Unexpected error: {e}",
            result=None,
        )


def _unique_upload_path(original_name: str) -> Path:
    """
    Create a non-colliding upload path under UPLOAD_DIR.
    """
    candidate = UPLOAD_DIR / original_name
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    idx = 1
    while True:
        new_candidate = UPLOAD_DIR / f"{stem}_{idx}{suffix}"
        if not new_candidate.exists():
            return new_candidate
        idx += 1


# ----------------------------
# Routes
# ----------------------------
@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a .jsonl dataset file to the backend machine.
    Returns the saved path so the frontend can use it later.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    if not file.filename.lower().endswith(".jsonl"):
        raise HTTPException(status_code=400, detail="Only .jsonl files are supported.")

    save_path = _unique_upload_path(Path(file.filename).name)

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    save_path.write_bytes(content)

    return {
        "success": True,
        "message": "Dataset uploaded successfully.",
        "filename": file.filename,
        "dataset_path": str(save_path.resolve()),
        "relative_path": str(save_path.relative_to(PROJECT_ROOT)),
        "size_bytes": len(content),
    }


@app.post("/validate-dataset")
def api_validate_dataset(req: DatasetValidateRequest):
    """
    Validate a dataset path before training.
    """
    return validate_dataset(req.dataset_path)


@app.post("/train")
def api_train(req: TrainRequest):
    """
    Start a training job in the background.
    """
    current = get_state()
    if current["status"] == "running":
        raise HTTPException(status_code=409, detail="A training job is already running.")

    try:
        config = build_train_config(req.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Validate dataset before starting
    dataset_check = validate_dataset(config.dataset_path)
    if not dataset_check["valid"]:
        raise HTTPException(status_code=400, detail=dataset_check["message"])

    thread = threading.Thread(
        target=run_training_in_background,
        args=(config,),
        daemon=True,
    )
    thread.start()

    set_state(
        status="running",
        message="Training job submitted.",
        config=config.__dict__,
        result=None,
    )

    return {
        "success": True,
        "message": "Training started.",
        "status": "running",
        "config": config.__dict__,
    }


@app.get("/training-status")
def api_training_status():
    """
    Query current training status.
    """
    return get_state()


@app.get("/health")
def health():
    return {"status": "ok"}