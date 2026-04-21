"""Data loading and strict preprocessing (encoding rules).

The preprocessing is implemented in two layers:
1) *Canonicalization* of raw string values into a fixed label space (strict mapping).
2) *Encoding* via sklearn transformers inside a `ColumnTransformer`.

This separation makes the system reproducible and ensures that interpretation
works reliably (because ordinal/binary semantics are preserved before encoding).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from .config import (
    BINARY_YES_NO_STR_TO_CANONICAL,
    CFG,
    DISCOUNT_INFLUENCE_TO_ORDINAL_STR,
    LOYALTY_STR_TO_ID,
    SOCIAL_ENGAGEMENT_TO_ORDINAL_STR,
    USAGE_DURATION_TO_ORDINAL_STR,
)

TargetMode = Literal["loyalty", "price_sensitivity"]

REPLACEMENT_CHARS = [
    "\ufffd",  # replacement character
    "\u2013",  # EN DASH (–) appears as the malformed separator in the dataset
    "\u2014",  # EM DASH (—) (defensive)
]


def _replace_replacement_char(series: pd.Series) -> pd.Series:
    """Replace Excel-import replacement char with a normal hyphen."""
    out = series.astype(str)
    for ch in REPLACEMENT_CHARS:
        out = out.str.replace(ch, "-", regex=False)
    return out.str.strip()


def _canonicalize_ordinal_value(
    raw_value: object,
    *,
    mapping_raw_to_canonical: Dict[str, str],
    ordinal_levels: List[str],
) -> str:
    """Map a raw ordinal value into one of `ordinal_levels`.

    Accepts either:
    - already-canonical ordinal strings (e.g., "Very likely")
    - integer-like inputs 0/1/2 (mapping onto ordinal_levels)
    - known raw labels present in dataset.xlsx (via `mapping_raw_to_canonical`)
    """
    if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)):
        return np.nan  # type: ignore[return-value]

    # Numeric shortcuts
    if isinstance(raw_value, (int, np.integer)):
        idx = int(raw_value)
        if idx in (0, 1, 2):
            return ordinal_levels[idx]
    if isinstance(raw_value, str):
        s = raw_value.strip()
        # Handle numeric string inputs
        if s.isdigit() and int(s) in (0, 1, 2):
            return ordinal_levels[int(s)]

        s_norm = s
        for ch in REPLACEMENT_CHARS:
            s_norm = s_norm.replace(ch, "-")
        s_norm = s_norm.strip()
        # If the input is already a canonical ordinal value, keep it.
        if s_norm in ordinal_levels:
            return s_norm

        if s_norm in mapping_raw_to_canonical:
            return mapping_raw_to_canonical[s_norm]

        # Try a case-insensitive match as a fallback
        for k, v in mapping_raw_to_canonical.items():
            if s_norm.strip().lower() == k.strip().lower():
                return v

    raise ValueError(f"Unexpected ordinal value: {raw_value!r}")


def _canonicalize_yes_no_value(raw_value: object) -> str:
    """Map raw Yes/No-like inputs into canonical strings: 'Yes' or 'No'."""
    if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)):
        return np.nan  # type: ignore[return-value]

    if isinstance(raw_value, (bool, np.bool_)):
        return "Yes" if bool(raw_value) else "No"

    if isinstance(raw_value, (int, np.integer)):
        if int(raw_value) == 1:
            return "Yes"
        if int(raw_value) == 0:
            return "No"

    if isinstance(raw_value, str):
        s = raw_value.strip()
        s_lower = s.lower()
        if s_lower in BINARY_YES_NO_STR_TO_CANONICAL:
            return BINARY_YES_NO_STR_TO_CANONICAL[s_lower]
        if s in ("Yes", "No"):
            return s

    raise ValueError(f"Unexpected Yes/No value: {raw_value!r}")


def canonicalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize strict-mapping features used by the model.

    Input `df` must contain at least:
    - ordinal_features
    - binary_features
    - onehot_features
    """
    df = df.copy()

    # Fix the dataset's malformed range separator for "Usage Duration".
    if "Usage Duration" in df.columns:
        df["Usage Duration"] = _replace_replacement_char(df["Usage Duration"])

    # Ordinal features -> canonical ordinal strings
    ordinal_levels = CFG.ordinal_levels

    if "Usage Duration" in df.columns:
        df["Usage Duration"] = df["Usage Duration"].apply(
            lambda v: _canonicalize_ordinal_value(
                v,
                mapping_raw_to_canonical=USAGE_DURATION_TO_ORDINAL_STR,
                ordinal_levels=ordinal_levels,
            )
        )
    if "Discount Influence" in df.columns:
        df["Discount Influence"] = df["Discount Influence"].apply(
            lambda v: _canonicalize_ordinal_value(
                v,
                mapping_raw_to_canonical=DISCOUNT_INFLUENCE_TO_ORDINAL_STR,
                ordinal_levels=ordinal_levels,
            )
        )
    if "Social Engagement" in df.columns:
        df["Social Engagement"] = df["Social Engagement"].apply(
            lambda v: _canonicalize_ordinal_value(
                v,
                mapping_raw_to_canonical=SOCIAL_ENGAGEMENT_TO_ORDINAL_STR,
                ordinal_levels=ordinal_levels,
            )
        )
    # Binary features -> canonical 'Yes'/'No'
    for c in ["Experience", "Peer Influence"]:
        if c in df.columns:
            df[c] = df[c].apply(_canonicalize_yes_no_value)

    # One-hot features -> clean strings (no mapping; used as categories)
    for c in CFG.onehot_features:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df


def canonicalize_target(df: pd.DataFrame, *, target_col: str = CFG.target_col) -> pd.Series:
    """Canonicalize target labels into numeric IDs 0..2."""
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col!r}")

    def _map_target_value(v: object) -> int:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan  # type: ignore[return-value]
        if isinstance(v, (int, np.integer)):
            if int(v) in (0, 1, 2):
                return int(v)
        if isinstance(v, str):
            s = v.strip()
            if s.isdigit() and int(s) in (0, 1, 2):
                return int(s)
            # Normalize case and whitespace
            s2 = s
            for ch in REPLACEMENT_CHARS:
                s2 = s2.replace(ch, "-")
            s2 = s2.strip()
            # direct match
            if s2 in LOYALTY_STR_TO_ID:
                return LOYALTY_STR_TO_ID[s2]
            # case-insensitive match
            for k, vid in LOYALTY_STR_TO_ID.items():
                if s2.lower() == k.lower():
                    return vid
        raise ValueError(f"Unexpected target value: {v!r}")

    y = df[target_col].apply(_map_target_value)
    if y.isna().any():
        raise ValueError("Target contains missing/unknown labels after canonicalization.")
    return y.astype(int)


def canonicalize_price_sensitivity_target(df: pd.DataFrame, *, target_col: str = "Price Importance") -> pd.Series:
    """Map price-importance survey responses to binary price sensitivity IDs.

    Mapping:
    - High / Very High -> 1 (Price Sensitive)
    - Low / Medium -> 0 (Not Price Sensitive)
    """
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col!r}")

    sensitive_values = {"high", "very high", "very important", "highly important"}
    not_sensitive_values = {"low", "medium", "not important", "somewhat important", "moderately important"}

    def _map_target_value(v: object) -> int:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan  # type: ignore[return-value]

        if isinstance(v, (int, np.integer)):
            if int(v) in (0, 1):
                return int(v)

        if isinstance(v, str):
            s = v.strip()
            if s.isdigit() and int(s) in (0, 1):
                return int(s)
            s_norm = s.lower()
            if s_norm in sensitive_values:
                return 1
            if s_norm in not_sensitive_values:
                return 0

        raise ValueError(f"Unexpected price-sensitivity target value: {v!r}")

    y = df[target_col].apply(_map_target_value)
    if y.isna().any():
        raise ValueError("Price-sensitivity target contains missing/unknown labels after canonicalization.")
    return y.astype(int)


def get_feature_columns(target_mode: TargetMode) -> List[str]:
    """Return leakage-safe feature columns for the selected target mode."""
    if target_mode == "loyalty":
        return CFG.all_feature_columns
    if target_mode == "price_sensitivity":
        return [c for c in CFG.all_feature_columns if c != "Price Importance"]
    raise ValueError(f"Unsupported target_mode: {target_mode!r}")


def load_dataset(
    dataset_path: str,
    *,
    target_col: str = CFG.target_col,
    target_mode: TargetMode = "loyalty",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load dataset.xlsx and apply strict canonicalization.

    Returns:
    - df_canonical: full canonicalized dataframe (features + target ID)
    - X: features dataframe (strictly model feature columns)
    - y: numeric target IDs (0..2)
    """
    df = pd.read_excel(dataset_path)

    # Support source dataset that uses full question text as column names.
    rename_map = {v: k for k, v in CFG.source_feature_columns.items()}
    rename_map[CFG.source_target_col] = CFG.target_col
    df = df.rename(columns=rename_map)

    # Drop irrelevant features (case-sensitive per dataset; be strict here).
    for c in CFG.drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    required_target_col = target_col if target_mode == "loyalty" else "Price Importance"
    feature_columns = get_feature_columns(target_mode)

    required = set([required_target_col] + CFG.all_feature_columns)
    missing = required.difference(set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in dataset.xlsx: {sorted(missing)}")

    # Canonicalize strict mapping columns
    df_canonical = canonicalize_features(df)
    if target_mode == "loyalty":
        y = canonicalize_target(df_canonical, target_col=target_col)
        final_target_col = target_col
    else:
        y = canonicalize_price_sensitivity_target(df_canonical, target_col="Price Importance")
        final_target_col = "Price Sensitivity"

    # Keep only features for X
    X = df_canonical[feature_columns].copy()
    df_canonical = df_canonical.copy()
    df_canonical[final_target_col] = y

    return df_canonical, X, y


def build_preprocessor() -> ColumnTransformer:
    """Build the exact sklearn ColumnTransformer used for both models."""
    ordinal_encoder = OrdinalEncoder(
        categories=[CFG.ordinal_levels] * len(CFG.ordinal_features),
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    binary_encoder = OrdinalEncoder(
        categories=[["No", "Yes"]] * len(CFG.binary_features),
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal", ordinal_encoder, CFG.ordinal_features),
            ("binary", binary_encoder, CFG.binary_features),
            ("onehot", onehot_encoder, CFG.onehot_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Get the post-encoding feature names aligned to the RF importances array."""
    return list(preprocessor.get_feature_names_out())

