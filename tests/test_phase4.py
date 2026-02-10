from fastapi.testclient import TestClient
from src.app.api import app
import pytest

client = TestClient(app)

def test_phase4_flow():
    # 1. Generate Dataset (ML Phase)
    response = client.post("/dataset/generate", json={"phase": "ml", "seed": 42, "n": 200})
    assert response.status_code == 200
    dataset_id = response.json()["meta"]["dataset_id"]
    
    # 2. Train Model (Logistic Regression)
    params = {"C": 1.0, "max_iter": 100}
    response = client.post("/ml/train", json={
        "dataset_id": dataset_id, 
        "target_col": "target", 
        "model_type": "logreg",
        "params": params
    })
    assert response.status_code == 200
    model_id = response.json()["result"]["model_id"]
    metrics = response.json()["result"]["metrics"]
    
    assert "accuracy" in metrics
    assert metrics["accuracy"] >= 0 and metrics["accuracy"] <= 1
    
    # 3. Get Model Info
    response = client.get(f"/ml/model-info/{model_id}")
    assert response.status_code == 200
    data = response.json()["result"]["data"]
    assert data["type"] == "logreg"
    assert data["params"] == params
    
    # 4. Predict
    # Use same dataset for prediction test
    response = client.post("/ml/predict", json={"dataset_id": dataset_id, "model_id": model_id})
    assert response.status_code == 200
    preds = response.json()["result"]["data"]
    
    assert "predictions" in preds
    assert len(preds["predictions"]) == 200
    
    # 5. Train Model (Random Forest)
    response = client.post("/ml/train", json={
        "dataset_id": dataset_id, 
        "target_col": "target", 
        "model_type": "rf",
        "params": {"n_estimators": 10}
    })
    assert response.status_code == 200
    rf_model_id = response.json()["result"]["model_id"]
    
    # Predict with RF
    response = client.post("/ml/predict", json={"dataset_id": dataset_id, "model_id": rf_model_id})
    assert response.status_code == 200
