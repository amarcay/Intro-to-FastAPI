from typing import Dict, Any, List
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio

class EdaService:
    @staticmethod
    def get_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates descriptive statistics and missing value rates.
        """
        summary = {}
        
        # Numeric Stats
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            stats = numeric_df.describe().T
            stats['missing_rate'] = df[numeric_df.columns].isnull().mean()
            summary['numeric'] = stats.to_dict(orient='index')
            
        # Categorical Stats
        categorical_df = df.select_dtypes(exclude=[np.number])
        if not categorical_df.empty:
            cat_stats = categorical_df.describe().T
            cat_stats['missing_rate'] = df[categorical_df.columns].isnull().mean()
            summary['categorical'] = cat_stats.to_dict(orient='index')
            
        return summary

    @staticmethod
    def get_groupby(df: pd.DataFrame, group_col: str, agg: Dict[str, str]) -> Dict[str, Any]:
        """
        Performs groupby aggregation.
        agg example: {"income": "mean", "age": "max"}
        """
        if group_col not in df.columns:
            raise ValueError(f"Column {group_col} not found")
        
        # Check if agg cols exist
        for col in agg.keys():
            if col not in df.columns:
                raise ValueError(f"Column {col} not found")

        grouped = df.groupby(group_col).agg(agg).reset_index()
        return grouped.to_dict(orient='records')

    @staticmethod
    def get_plots(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generates Plotly artifacts (JSON).
        Standard plots:
        1. Histograms for all numeric variables.
        2. Boxplots for numeric variables vs categorical (if reasonable).
        3. Bar charts for categorical counts.
        """
        artifacts = {}
        
        # 1. Numeric Histograms
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            fig = px.histogram(df, x=col, title=f"Distribution of {col}")
            artifacts[f"hist_{col}"] = pio.to_json(fig)
            
        # 2. Categorical Bar Charts
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in categorical_cols:
            if df[col].nunique() < 50: # Avoid high cardinality
                counts = df[col].value_counts().reset_index()
                counts.columns = [col, 'count']
                fig = px.bar(counts, x=col, y='count', title=f"Count of {col}")
                artifacts[f"bar_{col}"] = pio.to_json(fig)

        # 3. Boxplots (Numeric vs Categorical)
        # Heuristic: Pick first categorical as x, numeric as y
        if len(categorical_cols) > 0:
            cat_col = categorical_cols[0]
            if df[cat_col].nunique() < 20:
                for num_col in numeric_cols:
                    fig = px.box(df, x=cat_col, y=num_col, title=f"{num_col} by {cat_col}")
                    artifacts[f"box_{num_col}_by_{cat_col}"] = pio.to_json(fig)
                    
        return artifacts
