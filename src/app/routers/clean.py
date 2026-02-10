from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from src.app.core.dataset import DATASETS, save_dataset, get_dataset
from src.app.core.schemas import APIResponse, Meta, Result, FitRequest, TransformRequest
from src.app.services.cleaning_service import CleaningService, CLEANERS

router = APIRouter(prefix="/clean", tags=["clean"])

@router.post("/fit", response_model=APIResponse)
def fit_cleaner(request: FitRequest):
    try:
        df = get_dataset(request.dataset_id)
        
        # Calculate quality before fit (optional but good for report)
        quality_before = CleaningService.get_quality_report(df)
        
        # Fit
        cleaner_id = CleaningService.fit_cleaner(df, request.params)
        
        return APIResponse(
            meta=Meta(dataset_id=request.dataset_id, params=request.params),
            result=Result(cleaner_id=cleaner_id, metrics={"quality_before": quality_before})
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/transform", response_model=APIResponse)
def transform_dataset(request: TransformRequest):
    try:
        # Get original data
        df = get_dataset(request.dataset_id)
        
        # Transform
        df_clean = CleaningService.transform_dataset(df, request.cleaner_id)
        
        # Save new dataset
        new_dataset_id = save_dataset(df_clean)
        
        # Report after
        quality_after = CleaningService.get_quality_report(df_clean)
        
        return APIResponse(
            meta=Meta(dataset_id=new_dataset_id, params={"source_dataset_id": request.dataset_id, "cleaner_id": request.cleaner_id}),
            result=Result(data=df_clean.head(20).to_dict(orient="records"), metrics={"quality_after": quality_after})
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/report/{dataset_id}", response_model=APIResponse)
def get_report(dataset_id: str):
    try:
        df = get_dataset(dataset_id)
        report = CleaningService.get_quality_report(df)
        return APIResponse(
            meta=Meta(dataset_id=dataset_id),
            result=Result(data=report)
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
