from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class MvService:
    @staticmethod
    def _prepare_numeric_data(df: pd.DataFrame) -> pd.DataFrame:
        """Selects numeric columns and drops na."""
        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        if numeric_df.empty:
            raise ValueError("No numeric data available or all rows contain NaN")
        return numeric_df

    @staticmethod
    def perform_pca(df: pd.DataFrame, n_components: int = 2) -> Dict[str, Any]:
        """
        Performs PCA on numeric columns.
        """
        numeric_df = MvService._prepare_numeric_data(df)
        
        # Standardize
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        # PCA
        n_components = min(n_components, len(numeric_df.columns))
        pca = PCA(n_components=n_components)
        projected_data = pca.fit_transform(scaled_data)
        
        # Components (Loadings)
        components = pd.DataFrame(
            pca.components_, 
            columns=numeric_df.columns, 
            index=[f"PC{i+1}" for i in range(n_components)]
        )
        
        results = {
            "projected_data": projected_data.tolist(),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "loadings": components.to_dict(orient='index'),
            "feature_names": numeric_df.columns.tolist()
        }
        return results

    @staticmethod
    def perform_kmeans(df: pd.DataFrame, n_clusters: int = 3) -> Dict[str, Any]:
        """
        Performs KMeans clustering.
        """
        numeric_df = MvService._prepare_numeric_data(df)
        
        # Standardize? Usually yes for KMeans
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled_data)
        
        # Silhouette Score
        # Requires at least 2 clusters and > 1 sample per cluster
        if n_clusters > 1 and len(numeric_df) > n_clusters:
            score = silhouette_score(scaled_data, labels)
        else:
            score = -1.0
            
        return {
            "labels": labels.tolist(),
            "centers": kmeans.cluster_centers_.tolist(), # On scaled data
            "silhouette_score": score,
            "feature_names": numeric_df.columns.tolist()
        }

    @staticmethod
    def get_report(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Basic correlation matrix report.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        return {
            "correlation_matrix": corr.to_dict(orient="index"),
            "features": numeric_df.columns.tolist()
        }
