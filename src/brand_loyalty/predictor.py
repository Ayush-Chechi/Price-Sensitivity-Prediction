"""Prediction interface for brand loyalty.

Exports:
    predict_loyalty(input_dict)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd

from .config import CFG
from .data import canonicalize_features


def _expected_features(target_mode: str) -> list[str]:
    if target_mode == "price_sensitivity":
        return [c for c in CFG.all_feature_columns if c != "Price Importance"]
    return CFG.all_feature_columns


def _validate_and_build_input_df(input_dict: Dict[str, Any], *, target_mode: str) -> pd.DataFrame:
    feature_columns = _expected_features(target_mode)
    expected = set(feature_columns)
    missing = expected.difference(set(input_dict.keys()))
    extra = set(input_dict.keys()).difference(expected)
    if missing:
        raise ValueError(f"input_dict missing required keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"input_dict has unexpected keys: {sorted(extra)}")
    df = pd.DataFrame([input_dict], columns=feature_columns)
    return df


def predict_loyalty(
    input_dict: Dict[str, Any],
    *,
    target_mode: str = "loyalty",
    model_path: str | Path | None = None,
) -> Tuple[str, float]:
    """Predict for a single college student profile.

    Args:
        input_dict: feature values keyed by the dataset column names:
            - Brand
            - Usage Duration
            - Experience
            - Discount Influence
            - Peer Influence
            - Decision Factor
            - Social Engagement
            - Price Importance
          (Timestamp/Email are not required.)
        target_mode: one of "loyalty" or "price_sensitivity".
        model_path: optional explicit model path.

    Returns:
        Tuple[label, probability_of_predicted_class]
    """
    if target_mode not in ("loyalty", "price_sensitivity"):
        raise ValueError("target_mode must be one of: loyalty, price_sensitivity")

    if model_path is None:
        model_path = Path("artifacts") / "models" / f"{target_mode}_best_model.joblib"
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    pipeline = joblib.load(model_path)
    df_raw = _validate_and_build_input_df(input_dict, target_mode=target_mode)
    df_canonical = canonicalize_features(df_raw)
    feature_columns = _expected_features(target_mode)
    X = df_canonical[feature_columns]

    pred_id = int(pipeline.predict(X)[0])
    proba = float(pipeline.predict_proba(X)[0][pred_id])
    if target_mode == "loyalty":
        return CFG.loyalty_binary_id_to_name[pred_id], proba
    labels = {0: "Not Price Sensitive", 1: "Price Sensitive"}
    return labels[pred_id], proba

