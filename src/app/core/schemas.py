from typing import Any, Dict, List, Optional
from pydantic import BaseModel

class Meta(BaseModel):
    dataset_id: Optional[str] = None
    schema_version: str = "1.0"
    params: Optional[Dict[str, Any]] = None

class Result(BaseModel):
    data: Optional[Any] = None
    model_id: Optional[str] = None
    cleaner_id: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

class APIResponse(BaseModel):
    meta: Meta
    result: Optional[Any] = None
    report: Optional[Dict[str, Any]] = None
    artifacts: Optional[Dict[str, Any]] = None

class GenerateRequest(BaseModel):
    phase: str
    seed: int = 42
    n: int = 1000

class FitRequest(BaseModel):
    dataset_id: str
    params: Dict[str, Any] = {}

class TransformRequest(BaseModel):
    dataset_id: str
    cleaner_id: str
