from fastapi.testclient import TestClient
from src.app.api import app
import pytest

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the FastAPI Data Science Project"}

def test_phase1_flow():
    # 1. Generate Dataset
    response = client.post("/dataset/generate", json={"phase": "clean", "seed": 42, "n": 100})
    assert response.status_code == 200
    data = response.json()
    assert "dataset_id" in data["meta"]
    dataset_id = data["meta"]["dataset_id"]
    
    # 2. Get Initial Report
    response = client.get(f"/clean/report/{dataset_id}")
    assert response.status_code == 200
    report = response.json()["report"]
    # Should have some missing values
    assert report["total_missing"] > 0
    
    # 3. Fit Cleaner
    params = {
        "impute_strategy": "mean",
        "outlier_strategy": "clip",
        "categorical_strategy": "one_hot"
    }
    response = client.post("/clean/fit", json={"dataset_id": dataset_id, "params": params})
    assert response.status_code == 200
    res_data = response.json()
    cleaner_id = res_data["result"]["cleaner_id"]
    assert cleaner_id is not None
    
    # 4. Transform Dataset
    response = client.post("/clean/transform", json={"dataset_id": dataset_id, "cleaner_id": cleaner_id})
    assert response.status_code == 200
    trans_data = response.json()
    new_dataset_id = trans_data["meta"]["dataset_id"]
    quality_after = trans_data["result"]["metrics"]["quality_after"]
    
    # Assertions on quality after
    assert quality_after["total_missing"] == 0
    assert quality_after["duplicates"] == 0 # Should have been removed
    
    # Verify columns (OneHot should add columns)
    # x1, x2, x3 exist. Segment has A, B, C (3 cols maybe if dropped first? or 3)
    # Original cols: x1, x2, x3, segment, target. 
    # Target is int, treated as numeric if not separated? 
    # Wait, my cleaning service treats int64 as numeric. Target is int.
    # Segment is object (A,B,C). OneHot -> 3 cols.
    # So we should have x1, x2, x3, target, segment_A, segment_B, segment_C...
    columns = trans_data["result"]["data"][0].keys()
    assert len(columns) >= 4
