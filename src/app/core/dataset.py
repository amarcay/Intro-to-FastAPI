import pandas as pd
import numpy as np
import uuid
from typing import Dict, Any, Tuple

# In-memory storage for datasets (simulating a database)
DATASETS: Dict[str, pd.DataFrame] = {}

def generate_dataset(phase: str, seed: int, n: int) -> Tuple[str, pd.DataFrame]:
    np.random.seed(seed)
    
    if phase == "clean":
        # TP1: Missing values, outliers, mixed types
        data = {
            "x1": np.random.normal(100, 15, n),
            "x2": np.random.uniform(0, 1, n),
            "x3": np.random.randint(1, 100, n).astype(float),
            "segment": np.random.choice(["A", "B", "C"], n),
            "target": np.random.choice([0, 1], n)
        }
        df = pd.DataFrame(data)
        
        # Inject Missing Values (10-20%)
        for col in ["x1", "x2", "segment"]:
            mask = np.random.rand(n) < np.random.uniform(0.1, 0.2)
            df.loc[mask, col] = np.nan
            
        # Inject Outliers (1-3% in numerics)
        mask_outlier = np.random.rand(n) < 0.02
        df.loc[mask_outlier, "x1"] *= 5  # Example outlier
        
        # Inject Mixed Types (very simple corruption for demo)
        # Note: In a real scenario, mixed types would break pandas float dtype
        # We'll simulate by casting to object if we really want "NaN" string, but let's keep it simple for now as per pandas limitations
        
        # Inject Duplicates (1-5%)
        n_dupes = int(n * 0.03)
        duplicates = df.sample(n_dupes, replace=True)
        df = pd.concat([df, duplicates], ignore_index=True)
        
    elif phase == "eda":
        # TP2: Age, income, spend, visits
        df = pd.DataFrame({
            "age": np.random.randint(18, 70, n),
            "income": np.random.normal(50000, 15000, n),
            "spend": np.random.exponential(1000, n),
            "visits": np.random.poisson(5, n),
            "segment": np.random.choice(["A", "B", "C"], n, p=[0.2, 0.5, 0.3]),
            "channel": np.random.choice(["web", "store", "app"], n)
        })
        
        # Outliers in income
        mask = np.random.rand(n) < 0.01
        df.loc[mask, "income"] += 200000
        
        # Some missing
        mask_na = np.random.rand(n) < 0.05
        df.loc[mask_na, "spend"] = np.nan

    elif phase == "mv":
        # TP3: 8 numeric vars, 3 clusters
        # Cluster centers
        centers = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [5, 5, 2, 2, 0, 0, 0, 0],
            [-5, -5, -2, 2, 10, 0, 0, 0]
        ])
        
        labels = np.random.choice([0, 1, 2], n)
        data = centers[labels] + np.random.normal(0, 1, (n, 8))
        df = pd.DataFrame(data, columns=[f"x{i+1}" for i in range(8)])
        df["cluster_true"] = labels # Hidden truth

    elif phase == "ml" or phase == "ml_baseline":
        # TP4: Classification binary
        # n_features=10, n_informative=5
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=n, n_features=10, n_informative=5, random_state=seed)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
        df["target"] = y

    elif phase == "ml2" or phase == "ml_advanced":
         # TP5: Similar to ML but maybe more complex or same
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=n, n_features=20, n_informative=10, random_state=seed)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(20)])
        df["target"] = y
        
    else:
        raise ValueError(f"Unknown phase: {phase}")

    dataset_id = str(uuid.uuid4())
    DATASETS[dataset_id] = df
    return dataset_id, df

def get_dataset(dataset_id: str) -> pd.DataFrame:
    if dataset_id not in DATASETS:
        raise ValueError(f"Dataset {dataset_id} not found")
    return DATASETS[dataset_id]

def save_dataset(df: pd.DataFrame) -> str:
    dataset_id = str(uuid.uuid4())
    DATASETS[dataset_id] = df
    return dataset_id
