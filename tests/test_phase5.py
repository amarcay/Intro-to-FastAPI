from fastapi.testclient import TestClient
from src.app.api import app
import pytest

client = TestClient(app)

def test_phase5_flow():
    # 1. Generate Dataset
    response = client.post("/dataset/generate", json={"phase": "ml_advanced", "seed": 42, "n": 200})
    assert response.status_code == 200
    dataset_id = response.json()["meta"]["dataset_id"]
    
    # 2. Tune Model (Logistic Regression)
    param_grid = {"C": [0.1, 1.0, 10.0]}
    response = client.post("/ml2/tune", json={
        "dataset_id": dataset_id, 
        "target_col": "target", 
        "model_type": "logreg",
        "param_grid": param_grid
    })
    
    # It might take a moment, but for 200 rows it's fast
    assert response.status_code == 200
    res = response.json()["result"]
    model_id = res["model_id"]
    best_params = res["data"]["best_params"]
    
    assert "C" in best_params
    assert best_params["C"] in param_grid["C"]
    
    # 3. Feature Importance
    response = client.get(f"/ml2/feature-importance/{model_id}")
    assert response.status_code == 200
    importances = response.json()["result"]["data"]
    # Check if features are present (f0, f1...) 
    # Note: dataset generation logic names them f0..f19 for ml_advanced
    assert len(importances) > 0
    assert "f0" in importances
    
    # 4. Permutation Importance
    response = client.post("/ml2/permutation-importance", json={
        "model_id": model_id,
        "dataset_id": dataset_id,
        "target_col": "target"
    })
    assert response.status_code == 200
    perm_imp = response.json()["result"]["data"]
    assert len(perm_imp) > 0
    assert "f0" in perm_imp
    
    # 5. Explain Instance
    # Create a dummy instance with correct feature names f0..f19
    # We only need a few to test, but let's be safe and provide what might be needed if checked
    instance = {f"f{i}": 0.5 for i in range(20)}
    response = client.post("/ml2/explain-instance", json={
        "model_id": model_id,
        "instance": instance
    })
    assert response.status_code == 200
    explanation = response.json()["result"]["data"]
    
    assert "prediction" in explanation
    assert "contribution" in explanation
    assert "base_value" in explanation["contribution"]
    assert "f0" in explanation["contribution"]
