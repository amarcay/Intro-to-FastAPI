from fastapi.testclient import TestClient
from src.app.api import app
import pytest

client = TestClient(app)

def test_phase3_flow():
    # 1. Generate Dataset
    response = client.post("/dataset/generate", json={"phase": "mv", "seed": 42, "n": 200})
    assert response.status_code == 200
    dataset_id = response.json()["meta"]["dataset_id"]
    
    # 2. PCA
    response = client.post("/mv/pca/fit_transform", json={"dataset_id": dataset_id, "n_components": 2})
    assert response.status_code == 200
    data = response.json()["result"]["data"]
    metrics = response.json()["result"]["metrics"]
    
    assert "projected_data" in data
    assert len(data["projected_data"]) == 200
    assert len(data["projected_data"][0]) == 2
    assert "explained_variance" in metrics
    assert len(metrics["explained_variance"]) == 2
    
    # 3. KMeans
    response = client.post("/mv/cluster/kmeans", json={"dataset_id": dataset_id, "n_clusters": 3})
    assert response.status_code == 200
    data = response.json()["result"]["data"]
    metrics = response.json()["result"]["metrics"]
    
    assert "labels" in data
    assert len(data["labels"]) == 200
    assert len(set(data["labels"])) == 3
    assert "silhouette_score" in metrics
    assert metrics["silhouette_score"] > 0 # Should be somewhat separable
    
    # 4. Report
    response = client.get(f"/mv/report/{dataset_id}")
    assert response.status_code == 200
    report = response.json()["result"]["data"]
    assert "correlation_matrix" in report
    assert "x1" in report["correlation_matrix"]
