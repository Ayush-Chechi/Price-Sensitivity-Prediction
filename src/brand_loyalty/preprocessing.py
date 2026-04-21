"""Preprocessing and feature engineering for leakage-free binary classification."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .config import CFG


class FeatureEngineering(BaseEstimator, TransformerMixin):
    """Simple, interpretable engineered features."""

    usage_map = {"Not likely": 0.0, "Somewhat likely": 1.0, "Very likely": 2.0}
    yes_no_map = {"No": 0.0, "Yes": 1.0}
    social_map = {"Not likely": 0.0, "Somewhat likely": 1.0, "Very likely": 2.0}
    price_map = {"Not important": 0.0, "Somewhat important": 1.0, "Very important": 2.0}
    discount_map = {"Not likely": 0.0, "Somewhat likely": 1.0, "Very likely": 2.0}

    engineered_cols = [
        "Experience Score",
        "Usage Score",
        "Engagement Score",
        "Experience x Usage",
        "Price x Discount",
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        usage = X["Usage Duration"].map(self.usage_map).astype(float)
        exp = X["Experience"].map(self.yes_no_map).astype(float)
        social = X["Social Engagement"].map(self.social_map).astype(float)
        discount = X["Discount Influence"].map(self.discount_map).astype(float)

        # For price-sensitivity mode, "Price Importance" is excluded from X to avoid leakage.
        if "Price Importance" in X.columns:
            price = X["Price Importance"].map(self.price_map).astype(float)
        else:
            price = pd.Series(np.zeros(len(X), dtype=float), index=X.index)

        X["Experience Score"] = exp
        X["Usage Score"] = usage
        X["Engagement Score"] = social
        X["Experience x Usage"] = exp * usage
        X["Price x Discount"] = price * discount
        return X


def build_preprocessor(*, scale_numeric: bool = False, onehot_features: list[str] | None = None) -> ColumnTransformer:
    """Build model-ready preprocessing with imputation and optional scaling."""
    onehot_cols = CFG.onehot_features if onehot_features is None else onehot_features
    ordinal_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=[CFG.ordinal_levels] * len(CFG.ordinal_features),
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )
    binary_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=[["No", "Yes"]] * len(CFG.binary_features),
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )
    onehot_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    engineered_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
        + ([("scaler", StandardScaler())] if scale_numeric else [])
    )

    return ColumnTransformer(
        transformers=[
            ("ordinal", ordinal_pipe, CFG.ordinal_features),
            ("binary", binary_pipe, CFG.binary_features),
            ("onehot", onehot_pipe, onehot_cols),
            ("engineered", engineered_pipe, FeatureEngineering.engineered_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

