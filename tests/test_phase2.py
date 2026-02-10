from fastapi.testclient import TestClient
from src.app.api import app
import pytest

client = TestClient(app)

def test_phase2_flow():
    # 1. Generate Dataset
    response = client.post("/dataset/generate", json={"phase": "eda", "seed": 100, "n": 200})
    assert response.status_code == 200
    dataset_id = response.json()["meta"]["dataset_id"]
    
    # 2. Get Summary
    response = client.post("/eda/summary", json={"dataset_id": dataset_id})
    assert response.status_code == 200
    data = response.json()["result"]["data"]
    
    # Check numeric stats (age exists in EDA phase)
    assert "numeric" in data
    assert "age" in data["numeric"]
    assert "mean" in data["numeric"]["age"]
    
    # Check categorical stats (segment exists)
    assert "categorical" in data
    assert "segment" in data["categorical"]
    
    # 3. Group By
    # Group by segment, agg age: mean
    request_data = {
        "dataset_id": dataset_id, 
        "group_col": "segment", 
        "agg": {"age": "mean", "income": "max"}
    }
    response = client.post("/eda/groupby", json=request_data)
    assert response.status_code == 200
    res_data = response.json()["result"]["data"]
    assert len(res_data) > 0
    # Should have 'segment', 'age', 'income'
    assert "segment" in res_data[0]
    assert "age" in res_data[0]
    assert "income" in res_data[0]

    # 4. Plots
    response = client.post("/eda/plots", json={"dataset_id": dataset_id})
    assert response.status_code == 200
    artifacts = response.json()["artifacts"]
    
    # Check for expected plot keys
    # Numeric: age, income, spend, visits
    assert "hist_age" in artifacts
    assert "hist_income" in artifacts
    
    # Categorical: segment, channel
    assert "bar_channel" in artifacts
    
    # Boxplot
    assert any(key.startswith("box_") for key in artifacts.keys())
    
    # Verify content is JSON string
    import json
    plot_json = json.loads(artifacts["hist_age"])
    assert "data" in plot_json
    assert "layout" in plot_json
