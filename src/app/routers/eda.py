from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from pydantic import BaseModel
from src.app.core.dataset import get_dataset
from src.app.core.schemas import APIResponse, Meta, Result
from src.app.services.eda_service import EdaService

router = APIRouter(prefix="/eda", tags=["eda"])

class SummaryRequest(BaseModel):
    dataset_id: str

class GroupByRequest(BaseModel):
    dataset_id: str
    group_col: str
    agg: Dict[str, str]

class PlotsRequest(BaseModel):
    dataset_id: str

@router.post("/summary", response_model=APIResponse)
def get_summary(request: SummaryRequest):
    try:
        df = get_dataset(request.dataset_id)
        summary = EdaService.get_summary(df)
        return APIResponse(
            meta=Meta(dataset_id=request.dataset_id),
            result=Result(data=summary)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/groupby", response_model=APIResponse)
def get_groupby(request: GroupByRequest):
    try:
        df = get_dataset(request.dataset_id)
        data = EdaService.get_groupby(df, request.group_col, request.agg)
        return APIResponse(
            meta=Meta(dataset_id=request.dataset_id, params={"group_col": request.group_col, "agg": request.agg}),
            result=Result(data=data)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/plots", response_model=APIResponse)
def get_plots(request: PlotsRequest):
    try:
        df = get_dataset(request.dataset_id)
        artifacts = EdaService.get_plots(df)
        return APIResponse(
            meta=Meta(dataset_id=request.dataset_id),
            artifacts=artifacts
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
