from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import uuid
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from src.app.services.ml_service import MODELS

class MlAdvancedService:
    @staticmethod
    def tune_model(df: pd.DataFrame, target_col: str, model_type: str, param_grid: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tunes a model using GridSearchCV.
        Returns best params, best score, and model_id of the best model.
        """
        if target_col not in df.columns:
             raise ValueError(f"Target column {target_col} not found")
             
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if model_type == "logreg":
            base_model = LogisticRegression(solver='liblinear') # robust choice
        elif model_type == "rf":
            base_model = RandomForestClassifier(random_state=42)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
        grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Store in shared MODELS dict so it can be used by /ml/predict or explainability endpoints
        model_id = str(uuid.uuid4())
        
        # Calculate validation metrics for consistency
        score = best_model.score(X_test, y_test)
        
        MODELS[model_id] = {
            "model": best_model,
            "type": model_type,
            "params": grid_search.best_params_,
            "metrics": {"accuracy": score, "best_cv_score": grid_search.best_score_},
            "features": X.columns.tolist()
        }

        return {
            "model_id": model_id,
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "test_score": score
        }

    @staticmethod
    def get_feature_importance(model_id: str) -> Dict[str, Any]:
        if model_id not in MODELS:
            raise ValueError(f"Model {model_id} not found")
        
        model_data = MODELS[model_id]
        model = model_data["model"]
        features = model_data["features"]
        
        importances = {}
        
        if hasattr(model, "feature_importances_"):
            # RF
            for f, imp in zip(features, model.feature_importances_):
                importances[f] = float(imp)
        elif hasattr(model, "coef_"):
            # Logistic Regression
            # Coef shape is (1, n_features) or (n_classes, n_features)
            coefs = model.coef_[0]
            for f, imp in zip(features, coefs):
                importances[f] = float(abs(imp)) # Magnitude implies importance
        else:
            raise ValueError("Model does not support native feature importance")
            
        return dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

    @staticmethod
    def get_permutation_importance(model_id: str, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        if model_id not in MODELS:
            raise ValueError(f"Model {model_id} not found")
        
        model_data = MODELS[model_id]
        model = model_data["model"]
        features = model_data["features"]
        
        if target_col not in df.columns:
            raise ValueError(f"Target {target_col} not found in validation set")
            
        X = df[features] # Align
        y = df[target_col]
        
        # Calculate permutation importance
        perm_imp = permutation_importance(model, X, y, n_repeats=5, random_state=42, n_jobs=-1)
        
        importances = {}
        for i, f in enumerate(features):
            importances[f] = float(perm_imp.importances_mean[i])
            
        return dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

    @staticmethod
    def explain_instance(model_id: str, instance: Dict[str, Any]) -> Dict[str, Any]:
        if model_id not in MODELS:
            raise ValueError(f"Model {model_id} not found")
        
        model_data = MODELS[model_id]
        model = model_data["model"]
        features = model_data["features"]
        
        # Prepare input
        input_data = pd.DataFrame([instance])[features]
        
        prediction = int(model.predict(input_data)[0])
        probas = model.predict_proba(input_data)[0].tolist() if hasattr(model, "predict_proba") else []
        
        explanation = {
            "prediction": prediction,
            "probabilities": probas,
            "contribution": {}
        }
        
        # Simple contribution for LogReg (Feature * Coeff)
        if hasattr(model, "coef_"):
             coefs = model.coef_[0]
             intercept = model.intercept_[0]
             
             contribs = {"base_value": float(intercept)}
             for f, c in zip(features, coefs):
                 val = instance.get(f, 0)
                 contribs[f] = float(val * c)
             explanation["contribution"] = contribs
             
        return explanation
