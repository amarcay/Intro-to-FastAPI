from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import uuid
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

MODELS: Dict[str, Any] = {}

class MlService:
    @staticmethod
    def train_model(df: pd.DataFrame, target_col: str, model_type: str, params: Dict[str, Any] = {}) -> Dict[str, Any]:
        """
        Trains a model (logreg or rf).
        Returns model_id and metrics.
        """
        if target_col not in df.columns:
             raise ValueError(f"Target column {target_col} not found")
             
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Simple split for validation metrics
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if model_type == "logreg":
            model = LogisticRegression(**params)
        elif model_type == "rf":
            model = RandomForestClassifier(**params)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'), # weighted for safety if multiclass, though target is 0/1 usually
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1": f1_score(y_test, y_pred, average='weighted')
        }
        
        if y_prob is not None and len(np.unique(y)) == 2:
            try:
                metrics["auc"] = roc_auc_score(y_test, y_prob)
            except:
                metrics["auc"] = None

        model_id = str(uuid.uuid4())
        MODELS[model_id] = {
            "model": model,
            "type": model_type,
            "params": params,
            "metrics": metrics,
            "features": X.columns.tolist()
        }
        
        return {
            "model_id": model_id,
            "metrics": metrics
        }

    @staticmethod
    def predict(df: pd.DataFrame, model_id: str) -> Dict[str, Any]:
        if model_id not in MODELS:
            raise ValueError(f"Model {model_id} not found")
        
        model_data = MODELS[model_id]
        model = model_data["model"]
        expected_features = model_data["features"]
        
        # Align columns
        # If df has extra columns, drop them. If missing, error.
        # Ideally, we should use the same cleaning pipeline logic, but for this TP we assume input is clean/compatible
        # or we just select relevant cols.
        X_pred = df[expected_features]
        
        predictions = model.predict(X_pred)
        probabilities = model.predict_proba(X_pred).tolist() if hasattr(model, "predict_proba") else []
        
        return {
            "predictions": predictions.tolist(),
            "probabilities": probabilities
        }

    @staticmethod
    def get_model_info(model_id: str) -> Dict[str, Any]:
        if model_id not in MODELS:
            raise ValueError(f"Model {model_id} not found")
        return MODELS[model_id]
