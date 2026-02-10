from fastapi.testclient import TestClient
from src.app.api import app
import pytest

client = TestClient(app)

def test_full_workflow():
    """
    Simulates a user going through all phases of the project.
    Note: Phases are largely independent in this simplified API implementation,
    so we test them sequentially.
    """
    
    # --- PHASE 1: CLEANING ---
    print("\n--- Testing Phase 1: Cleaning ---")
    # 1. Generate data
    resp = client.post("/dataset/generate", json={"phase": "clean", "seed": 1, "n": 100})
    assert resp.status_code == 200
    ds_clean_id = resp.json()["meta"]["dataset_id"]
    
    # 2. Get Report
    resp = client.get(f"/clean/report/{ds_clean_id}")
    assert resp.status_code == 200
    assert "missing_values" in resp.json()["result"]["data"]
    
    # 3. Fit Cleaner
    resp = client.post("/clean/fit", json={"dataset_id": ds_clean_id, "params": {}})
    assert resp.status_code == 200
    cleaner_id = resp.json()["result"]["cleaner_id"]
    
    # 4. Transform
    resp = client.post("/clean/transform", json={"dataset_id": ds_clean_id, "cleaner_id": cleaner_id})
    assert resp.status_code == 200
    clean_data = resp.json()["result"]["data"]
    assert len(clean_data) > 0

    # --- PHASE 2: EDA ---
    print("\n--- Testing Phase 2: EDA ---")
    # 1. Generate EDA data
    resp = client.post("/dataset/generate", json={"phase": "eda", "seed": 2, "n": 100})
    assert resp.status_code == 200
    ds_eda_id = resp.json()["meta"]["dataset_id"]
    
    # 2. Summary
    resp = client.post("/eda/summary", json={"dataset_id": ds_eda_id})
    assert resp.status_code == 200
    assert "numeric" in resp.json()["result"]["data"]
    
    # 3. Plot
    resp = client.post("/eda/plots", json={"dataset_id": ds_eda_id})
    assert resp.status_code == 200
    assert "hist_age" in resp.json()["artifacts"]

    # --- PHASE 3: MULTIVARIATE ---
    print("\n--- Testing Phase 3: Multivariate ---")
    resp = client.post("/dataset/generate", json={"phase": "mv", "seed": 3, "n": 100})
    ds_mv_id = resp.json()["meta"]["dataset_id"]
    
    # PCA
    resp = client.post("/mv/pca/fit_transform", json={"dataset_id": ds_mv_id, "n_components": 2})
    assert resp.status_code == 200
    assert len(resp.json()["result"]["data"]["projected_data"][0]) == 2

    # --- PHASE 4: ML BASELINE ---
    print("\n--- Testing Phase 4: ML Baseline ---")
    resp = client.post("/dataset/generate", json={"phase": "ml", "seed": 4, "n": 100})
    ds_ml_id = resp.json()["meta"]["dataset_id"]
    
    # Train LogReg
    resp = client.post("/ml/train", json={"dataset_id": ds_ml_id, "target_col": "target", "model_type": "logreg"})
    assert resp.status_code == 200
    model_id = resp.json()["result"]["model_id"]
    
    # Predict
    resp = client.post("/ml/predict", json={"dataset_id": ds_ml_id, "model_id": model_id})
    assert resp.status_code == 200

    # --- PHASE 5: ML ADVANCED ---
    print("\n--- Testing Phase 5: ML Advanced ---")
    resp = client.post("/dataset/generate", json={"phase": "ml_advanced", "seed": 5, "n": 100})
    ds_ml2_id = resp.json()["meta"]["dataset_id"]
    
    # Tune
    resp = client.post("/ml2/tune", json={
        "dataset_id": ds_ml2_id, 
        "target_col": "target", 
        "model_type": "rf",
        "param_grid": {"n_estimators": [5, 10]}
    })
    assert resp.status_code == 200
    best_model_id = resp.json()["result"]["model_id"]
    
    # Feature Importance
    resp = client.get(f"/ml2/feature-importance/{best_model_id}")
    assert resp.status_code == 200
    assert "f0" in resp.json()["result"]["data"]
