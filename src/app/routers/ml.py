from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from src.app.core.dataset import get_dataset
from src.app.core.schemas import APIResponse, Meta, Result
from src.app.services.ml_service import MlService

router = APIRouter(prefix="/ml", tags=["ml"])

class TrainRequest(BaseModel):
    dataset_id: str
    target_col: str = "target"
    model_type: str = "logreg" # logreg, rf
    params: Dict[str, Any] = {}

class PredictRequest(BaseModel):
    dataset_id: str
    model_id: str

@router.post("/train", response_model=APIResponse)
def train_model(request: TrainRequest):
    try:
        df = get_dataset(request.dataset_id)
        result = MlService.train_model(df, request.target_col, request.model_type, request.params)
        return APIResponse(
            meta=Meta(dataset_id=request.dataset_id, params={"model_type": request.model_type, "target": request.target_col}),
            result=Result(model_id=result["model_id"], metrics=result["metrics"])
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/metrics/{model_id}", response_model=APIResponse)
def get_metrics(model_id: str):
    try:
        info = MlService.get_model_info(model_id)
        return APIResponse(
            meta=Meta(),
            result=Result(model_id=model_id, metrics=info["metrics"])
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/predict", response_model=APIResponse)
def predict(request: PredictRequest):
    try:
        df = get_dataset(request.dataset_id)
        preds = MlService.predict(df, request.model_id)
        return APIResponse(
            meta=Meta(dataset_id=request.dataset_id, params={"model_id": request.model_id}),
            result=Result(data=preds)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/model-info/{model_id}", response_model=APIResponse)
def get_model_info(model_id: str):
    try:
        info = MlService.get_model_info(model_id)
        return APIResponse(
            meta=Meta(),
            result=Result(model_id=model_id, data={"type": info["type"], "params": info["params"], "features": info["features"]})
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
