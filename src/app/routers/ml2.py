from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from src.app.core.dataset import get_dataset
from src.app.core.schemas import APIResponse, Meta, Result
from src.app.services.ml_advanced_service import MlAdvancedService

router = APIRouter(prefix="/ml2", tags=["ml_advanced"])

class TuneRequest(BaseModel):
    dataset_id: str
    target_col: str = "target"
    model_type: str = "logreg" 
    param_grid: Dict[str, List[Any]] # e.g. {"C": [0.1, 1, 10]}

class PermutationRequest(BaseModel):
    model_id: str
    dataset_id: str
    target_col: str = "target"

class ExplainRequest(BaseModel):
    model_id: str
    instance: Dict[str, Any]

@router.post("/tune", response_model=APIResponse)
def tune_model(request: TuneRequest):
    try:
        df = get_dataset(request.dataset_id)
        result = MlAdvancedService.tune_model(df, request.target_col, request.model_type, request.param_grid)
        return APIResponse(
            meta=Meta(dataset_id=request.dataset_id, params={"model_type": request.model_type, "grid": request.param_grid}),
            result=Result(
                model_id=result["model_id"], 
                metrics={"best_cv_score": result["best_score"], "test_score": result["test_score"]},
                data={"best_params": result["best_params"]}
            )
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/feature-importance/{model_id}", response_model=APIResponse)
def get_feature_importance(model_id: str):
    try:
        # Note: dataset_id is not strictly needed if we inspect the model object directly
        importance = MlAdvancedService.get_feature_importance(model_id)
        return APIResponse(
            meta=Meta(),
            result=Result(model_id=model_id, data=importance)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/permutation-importance", response_model=APIResponse)
def get_permutation_importance(request: PermutationRequest):
    try:
        df = get_dataset(request.dataset_id)
        importance = MlAdvancedService.get_permutation_importance(request.model_id, df, request.target_col)
        return APIResponse(
            meta=Meta(dataset_id=request.dataset_id, params={"model_id": request.model_id}),
            result=Result(model_id=request.model_id, data=importance)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/explain-instance", response_model=APIResponse)
def explain_instance(request: ExplainRequest):
    try:
        explanation = MlAdvancedService.explain_instance(request.model_id, request.instance)
        return APIResponse(
            meta=Meta(params={"model_id": request.model_id}),
            result=Result(model_id=request.model_id, data=explanation)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
