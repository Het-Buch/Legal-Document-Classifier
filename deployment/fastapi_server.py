# deployment/fastapi_server.py
"""
FastAPI server for model inference.
Endpoints:
- GET /health
- POST /predict with JSON: { "model": "baseline"|"mtl"|"auto", "docs": ["..."], "baseline_path": "...", "thresholds": "...", "mtl_ckpt": "...", "mtl_backbone": "..." }
Response:
{ "label_set": [...], "probs": [[...]], "preds": [[...]] }
Run:
    uvicorn deployment.fastapi_server:app --host 0.0.0.0 --port 8000 --workers 1
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)
app = FastAPI(title="Legal ML Inference Server")

class PredictRequest(BaseModel):
    docs: List[str]
    model: Optional[str] = "auto"
    baseline_path: Optional[str] = None
    thresholds: Optional[str] = None
    mtl_ckpt: Optional[str] = None
    mtl_backbone: Optional[str] = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    # Create predictor on the fly (could be cached for production)
    try:
        from src.infer.predictor import UnifiedPredictor
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Predictor import failed: {e}")
    try:
        predictor = UnifiedPredictor(
            baseline_path=req.baseline_path,
            thresholds_path=req.thresholds,
            mtl_checkpoint=req.mtl_ckpt,
            mtl_backbone=req.mtl_backbone,
            device="cpu"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize predictor: {e}")

    try:
        res = predictor.predict_documents(req.docs, model_kind=req.model)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
