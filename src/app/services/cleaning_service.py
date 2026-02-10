from typing import Dict, Any, List
import pandas as pd
import numpy as np
import uuid
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

CLEANERS: Dict[str, Any] = {}

class CleaningService:
    @staticmethod
    def get_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyzes dataframe quality: missing values, types, duplicates."""
        missing = df.isnull().sum().to_dict()
        total_missing = df.isnull().sum().sum()
        duplicates = df.duplicated().sum()
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "missing_values": missing,
            "total_missing": int(total_missing),
            "duplicates": int(duplicates),
            "dtypes": dtypes
        }

    @staticmethod
    def fit_cleaner(df: pd.DataFrame, params: Dict[str, Any]) -> str:
        """
        Fits a cleaning pipeline based on params.
        """
        # Identify types
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

        impute_strategy = params.get("impute_strategy", "mean")
        cat_strategy = params.get("categorical_strategy", "one_hot")
        outlier_strategy = params.get("outlier_strategy", "none")

        # 1. Calculate Outlier Bounds (if clip)
        outlier_bounds = {}
        if outlier_strategy == "clip":
            for col in numeric_features:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outlier_bounds[col] = (lower, upper)

        transformers = []

        # Numeric Pipeline
        if numeric_features:
            # For simplicity using SimpleImputer
            num_transformer = SimpleImputer(strategy=impute_strategy)
            transformers.append(("num", num_transformer, numeric_features))

        # Categorical Pipeline
        if categorical_features:
            if cat_strategy == "one_hot":
                cat_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore'))
                ])
            else:
                # Ordinal
                cat_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
                ])
            transformers.append(("cat", cat_transformer, categorical_features))

        preprocessor = ColumnTransformer(transformers=transformers, verbose_feature_names_out=False)
        
        # Fit
        preprocessor.fit(df)
        
        cleaner_id = str(uuid.uuid4())
        CLEANERS[cleaner_id] = {
            "pipeline": preprocessor,
            "params": params,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "outlier_bounds": outlier_bounds
        }
        return cleaner_id

    @staticmethod
    def transform_dataset(df: pd.DataFrame, cleaner_id: str) -> pd.DataFrame:
        if cleaner_id not in CLEANERS:
            raise ValueError(f"Cleaner {cleaner_id} not found")
        
        cleaner = CLEANERS[cleaner_id]
        pipeline = cleaner["pipeline"]
        outlier_bounds = cleaner["outlier_bounds"]
        
        # 1. Handle Duplicates
        df_clean = df.drop_duplicates().copy()
        
        # 2. Handle Outliers (using stored bounds)
        if outlier_bounds:
            for col, (lower, upper) in outlier_bounds.items():
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
        
        # 3. Apply Pipeline
        transformed_data = pipeline.transform(df_clean)
        
        # Get feature names
        try:
             feature_names = pipeline.get_feature_names_out()
        except:
             feature_names = [f"col_{i}" for i in range(transformed_data.shape[1])]

        # If sparse, to dense
        if hasattr(transformed_data, "toarray"):
            transformed_data = transformed_data.toarray()
            
        return pd.DataFrame(transformed_data, columns=feature_names)
