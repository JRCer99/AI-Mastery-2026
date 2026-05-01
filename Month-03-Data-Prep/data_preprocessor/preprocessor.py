import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import List, Optional


class DataPreprocessor:
    """Reusable, dataset-agnostic preprocessing pipeline."""

    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        self.target: Optional[str] = None

    def fit(self, df: pd.DataFrame, target_col: str) -> "DataPreprocessor":
        self.target = target_col
        X = df.drop(columns=[target_col])

        self.numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        self.categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.categorical_features),
            ],
            remainder="passthrough",
        )
        self.pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
        self.pipeline.fit(X)
        print(f"Pipeline fitted | numeric={len(self.numeric_features)} | categorical={len(self.categorical_features)}")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise ValueError("Call fit() before transform().")
        X = df.drop(columns=[self.target], errors="ignore")
        return self.pipeline.transform(X)

    def fit_transform(self, df: pd.DataFrame, target_col: str) -> np.ndarray:
        return self.fit(df, target_col).transform(df)

    def get_feature_names(self) -> List[str]:
        """Return feature names post-transformation for interpretability."""
        if self.pipeline is None:
            raise ValueError("Call fit() first.")
        ct = self.pipeline.named_steps["preprocessor"]
        cat_names = ct.named_transformers_["cat"].get_feature_names_out(self.categorical_features).tolist()
        remainder_idx = ct.transformers_[-1][2]
        remainder_names = (
            [ct.feature_names_in_[i] for i in remainder_idx]
            if ct.remainder == "passthrough" and len(remainder_idx)
            else []
        )
        return self.numeric_features + cat_names + remainder_names

    # Alias so both fit() and fit_preprocessor() work
    fit_preprocessor = fit

    def save(self, filepath: str = "data_preprocessor.pkl") -> None:
        joblib.dump(self, filepath)
        print(f"Preprocessor saved → {filepath}")

    @classmethod
    def load(cls, filepath: str = "data_preprocessor.pkl") -> "DataPreprocessor":
        return joblib.load(filepath)


def quick_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values: median for numeric, mode for categorical."""
    df = df.copy()
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include="object").columns:
        mode = df[col].mode()
        df[col] = df[col].fillna(mode[0] if not mode.empty else "Unknown")
    return df
