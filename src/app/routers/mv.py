from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from pydantic import BaseModel
from src.app.core.dataset import get_dataset
from src.app.core.schemas import APIResponse, Meta, Result
from src.app.services.mv_service import MvService

router = APIRouter(prefix="/mv", tags=["mv"])

class PcaRequest(BaseModel):
    dataset_id: str
    n_components: int = 2

class KmeansRequest(BaseModel):
    dataset_id: str
    n_clusters: int = 3

@router.post("/pca/fit_transform", response_model=APIResponse)
def fit_transform_pca(request: PcaRequest):
    try:
        df = get_dataset(request.dataset_id)
        results = MvService.perform_pca(df, request.n_components)
        return APIResponse(
            meta=Meta(dataset_id=request.dataset_id, params={"n_components": request.n_components}),
            result=Result(data=results, metrics={"explained_variance": results["explained_variance_ratio"]})
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/cluster/kmeans", response_model=APIResponse)
def perform_kmeans(request: KmeansRequest):
    try:
        df = get_dataset(request.dataset_id)
        results = MvService.perform_kmeans(df, request.n_clusters)
        return APIResponse(
            meta=Meta(dataset_id=request.dataset_id, params={"n_clusters": request.n_clusters}),
            result=Result(data=results, metrics={"silhouette_score": results["silhouette_score"]})
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/report/{dataset_id}", response_model=APIResponse)
def get_mv_report(dataset_id: str):
    try:
        df = get_dataset(dataset_id)
        report = MvService.get_report(df)
        return APIResponse(
            meta=Meta(dataset_id=dataset_id),
            result=Result(data=report)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
