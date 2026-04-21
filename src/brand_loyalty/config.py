"""Central configuration: column names, strict encoding mappings, and hyperparameters.

This file intentionally contains only constants and mapping helpers (no I/O).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Config:
    # Data
    dataset_path: str = "data/new_dataset.xlsx"
    target_col: str = "Loyalty"
    source_target_col: str = "How likely are you to continue using this brand in the future?"
    drop_cols: List[str] = None  # set in __post_init__
    source_feature_columns: Dict[str, str] = None  # set in __post_init__

    # Feature groups (as they appear in dataset.xlsx)
    ordinal_features: List[str] = None  # set in __post_init__
    binary_features: List[str] = None  # set in __post_init__
    onehot_features: List[str] = None  # set in __post_init__

    # Strict output mapping: numeric -> human label
    loyalty_id_to_name: Dict[int, str] = None  # set in __post_init__
    loyalty_binary_id_to_name: Dict[int, str] = None  # set in __post_init__

    # Strict encoding scale for ordinal encoding: Not likely -> 0, Somewhat likely -> 1, Very likely -> 2
    ordinal_levels: List[str] = None  # set in __post_init__

    # Random seed (single source of truth for reproducibility)
    random_state: int = 42

    # Models (per requirements)
    rf_n_estimators: int = 100
    rf_max_depth: int = 5
    logreg_max_iter: int = 2000

    def __post_init__(self) -> None:
        object.__setattr__(self, "drop_cols", ["Timestamp", "Email"])
        object.__setattr__(
            self,
            "source_feature_columns",
            {
                "Brand": "Which smartphone brand do you currently use?",
                "Usage Duration": "How long have you been using this brand?",
                "Experience": "Have you had a good overall experience with this brand?",
                "Discount Influence": "Do discounts or offers influence your smartphone purchase decision?",
                "Peer Influence": "Do your friends mostly use the same smartphone brand?",
                "Decision Factor": "What is the MOST important factor influencing your smartphone choice?",
                "Social Engagement": "How often do you engage with this smartphone brand online (YouTube, Instagram, etc.)?",
                "Price Importance": "How important is price when choosing a smartphone?",
            },
        )
        object.__setattr__(
            self,
            "ordinal_features",
            ["Usage Duration", "Discount Influence", "Social Engagement"],
        )
        object.__setattr__(self, "binary_features", ["Experience", "Peer Influence"])
        object.__setattr__(
            self,
            "onehot_features",
            ["Brand", "Decision Factor", "Price Importance"],
        )

        object.__setattr__(self, "loyalty_id_to_name", {0: "Low", 1: "Medium", 2: "High"})
        object.__setattr__(self, "loyalty_binary_id_to_name", {0: "Not Loyal", 1: "Loyal"})
        object.__setattr__(self, "ordinal_levels", ["Not likely", "Somewhat likely", "Very likely"])

    @property
    def all_feature_columns(self) -> List[str]:
        return self.ordinal_features + self.binary_features + self.onehot_features


CFG = Config()


LOYALTY_STR_TO_ID: Dict[str, int] = {
    "Not likely": 0,
    "Somewhat likely": 1,
    "Very likely": 2,
}


# --- Strict value mappings for ordinal encoding features ---

# Usage Duration -> 0/1/2 on the canonical ordinal scale.
USAGE_DURATION_TO_ORDINAL_STR: Dict[str, str] = {
    "Less than 6 months": "Not likely",
    "6-12 months": "Somewhat likely",
    # Collapse into 3-level: longer usage -> stronger loyalty.
    "1-2 years": "Very likely",
    "More than 2 years": "Very likely",
}

# Discount Influence -> 0/1/2 on canonical ordinal scale.
DISCOUNT_INFLUENCE_TO_ORDINAL_STR: Dict[str, str] = {
    "Rarely": "Not likely",
    "Never": "Not likely",
    "Sometimes": "Somewhat likely",
    "Always": "Very likely",
}

# Social Engagement -> 0/1/2 on canonical ordinal scale.
SOCIAL_ENGAGEMENT_TO_ORDINAL_STR: Dict[str, str] = {
    "Never": "Not likely",
    "Rarely": "Somewhat likely",
    "Occasionally": "Somewhat likely",
    "Frequently": "Very likely",
    # Backward compatibility with prior dataset variants
    "Sometimes": "Somewhat likely",
    "Yes regularly": "Very likely",
}

# --- Strict value mappings for binary encoding features ---

BINARY_YES_NO_STR_TO_CANONICAL: Dict[str, str] = {
    "yes": "Yes",
    "no": "No",
}

