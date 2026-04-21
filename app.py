from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd
import streamlit as st

from src.brand_loyalty.config import CFG
from src.brand_loyalty.data import canonicalize_features
from src.brand_loyalty.train import train

ARTIFACTS_DIR = Path("artifacts")
DATASET_PATH = Path(CFG.dataset_path)

QUESTION_LABELS = {
    "Brand": "Which smartphone brand do you currently use?",
    "Usage Duration": "How long have you been using this brand?",
    "Experience": "Have you had a good overall experience with this brand?",
    "Discount Influence": "Do discounts or offers influence your smartphone purchase decision?",
    "Peer Influence": "Do your friends mostly use the same smartphone brand?",
    "Decision Factor": "What is the MOST important factor influencing your smartphone choice?",
    "Social Engagement": "How often do you engage with this smartphone brand online?",
    "Price Importance": "How important is price when choosing a smartphone?",
}


@st.cache_data(show_spinner=False)
def load_options(dataset_path: Path) -> Dict[str, List[str]]:
    df = pd.read_excel(dataset_path)
    rename_map = {v: k for k, v in CFG.source_feature_columns.items()}
    df = df.rename(columns=rename_map)
    out: Dict[str, List[str]] = {}
    for c in CFG.all_feature_columns:
        vals = (
            df[c]
            .dropna()
            .astype(str)
            .str.replace("\u2013", "-", regex=False)
            .str.strip()
            .unique()
            .tolist()
        )
        out[c] = sorted(vals)
    return out


@st.cache_data(show_spinner=False)
def load_report(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _feature_columns_for_mode(target_mode: str) -> List[str]:
    if target_mode == "price_sensitivity":
        return [c for c in CFG.all_feature_columns if c != "Price Importance"]
    return CFG.all_feature_columns


def _predict_with_confidence(input_dict: Dict[str, Any], *, model_path: Path, target_mode: str) -> Tuple[str, float]:
    pipeline = joblib.load(model_path)
    feature_columns = _feature_columns_for_mode(target_mode)
    X = canonicalize_features(pd.DataFrame([input_dict], columns=feature_columns))[feature_columns]
    proba = pipeline.predict_proba(X)[0]
    pred_id = int(pipeline.predict(X)[0])
    if target_mode == "loyalty":
        label = CFG.loyalty_binary_id_to_name[pred_id]
    else:
        label = {0: "Not Price Sensitive", 1: "Price Sensitive"}[pred_id]
    return label, float(max(proba) * 100.0)


def _resolve_best_cm_plot(report: Dict[str, Any], pdir: Path, *, target_mode: str) -> Path | None:
    prefix = f"{target_mode}_"
    model_name = str(report.get("best_model", "")).split(" (")[0].strip().lower().replace(" ", "_")
    variant = "baseline" if "(Baseline)" in str(report.get("best_model", "")) else "tuned"
    candidate = pdir / f"{prefix}{model_name}_{variant}_cm.png"
    if candidate.exists():
        return candidate
    fallbacks = [
        pdir / f"{prefix}gradient_boosting_baseline_cm.png",
        pdir / f"{prefix}gradient_boosting_tuned_cm.png",
        pdir / f"{prefix}random_forest_baseline_cm.png",
        pdir / f"{prefix}random_forest_tuned_cm.png",
    ]
    for f in fallbacks:
        if f.exists():
            return f
    return None


def _result_card(label: str, confidence: float, *, objective_title: str) -> None:
    color = {"Loyal": "#22c55e", "Not Loyal": "#ef4444", "Price Sensitive": "#f59e0b", "Not Price Sensitive": "#3b82f6"}.get(label, "#64748b")
    st.markdown(
        f"""
        <div style="border:1px solid rgba(148,163,184,.35); border-left:8px solid {color}; border-radius:12px; padding:.9rem 1rem; margin-top:.7rem;">
          <h3 style="margin:0 0 .3rem 0;">Predicted {objective_title} Class</h3>
          <p style="margin:.1rem 0; font-size:1.08rem;"><strong>{label}</strong></p>
          <p style="margin:.1rem 0;">Confidence Score: <strong>{confidence:.1f}%</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_form(options: Dict[str, List[str]], *, target_mode: str) -> Dict[str, Any]:
    st.subheader("Input Survey")
    data: Dict[str, Any] = {}
    feature_columns = _feature_columns_for_mode(target_mode)
    for i, key in enumerate(feature_columns, start=1):
        st.markdown(f"**{i}. {QUESTION_LABELS[key]}**")
        if key in ("Experience", "Peer Influence"):
            data[key] = st.selectbox(" ", ["Yes", "No"], key=f"q_{key}", label_visibility="collapsed")
        else:
            data[key] = st.selectbox(" ", options.get(key, []), key=f"q_{key}", label_visibility="collapsed")
    return data


def main() -> None:
    st.set_page_config(page_title="Brand Loyalty Predictor", layout="wide")
    st.title("Smartphone Brand Loyalty Prediction")
    st.caption("Leakage-free binary ML system: Loyal vs Not Loyal")
    st.markdown(
        """
        <style>
        div.stButton > button:first-child { width:100%; min-height:3rem; font-weight:700; }
        div[data-testid="stSelectbox"] { margin-bottom:.85rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Pipeline Control")
        objective = st.selectbox("Prediction Objective", ["Brand Loyalty", "Price Sensitivity"], index=0)
        target_mode = "loyalty" if objective == "Brand Loyalty" else "price_sensitivity"
        model_path = ARTIFACTS_DIR / "models" / f"{target_mode}_best_model.joblib"
        report_path = ARTIFACTS_DIR / "reports" / f"{target_mode}_project_experiment_metadata.json"
        st.write(f"Dataset: `{DATASET_PATH.as_posix()}`")
        st.write(f"Model: `{model_path.as_posix()}`")
        if st.button("Train / Refresh Artifacts"):
            with st.spinner("Training models and generating plots/report..."):
                train(str(DATASET_PATH), str(ARTIFACTS_DIR), target_mode=target_mode)
            st.success("Training complete.")

    if not DATASET_PATH.exists():
        st.error("`new_dataset.xlsx` not found in `data/`.")
        st.stop()

    options = load_options(DATASET_PATH)
    report = load_report(report_path)
    left, right = st.columns([2, 3], gap="large")

    with left:
        inputs = build_form(options, target_mode=target_mode)
        button_text = "Predict Loyalty" if target_mode == "loyalty" else "Predict Price Sensitivity"
        pred_key = f"pred_{target_mode}"
        if st.button(button_text, type="primary", disabled=not model_path.exists()):
            lbl, conf = _predict_with_confidence(inputs, model_path=model_path, target_mode=target_mode)
            st.session_state[pred_key] = (lbl, conf)
        if st.session_state.get(pred_key):
            lbl, conf = st.session_state[pred_key]
            _result_card(lbl, conf, objective_title=objective)

    with right:
        st.subheader("Experimental Results & Insights")
        if report:
            st.markdown("### Dataset Overview")
            st.write(f"- Samples: **{report.get('dataset_samples', 'N/A')}**")
            st.write(f"- Leakage removed: **{report.get('excluded_feature', 'N/A')}**")

            st.markdown("### Model Performance Table")
            perf = pd.DataFrame(report.get("models_baseline", []))
            if not perf.empty:
                st.caption("Baseline model performance")
                st.dataframe(perf, width="stretch", hide_index=True)

            tuned_perf = pd.DataFrame(report.get("models", []))
            if not tuned_perf.empty:
                st.caption("Top-2 tuned model performance")
                st.dataframe(tuned_perf, width="stretch", hide_index=True)

            st.markdown("### Selected Best Model")
            st.write(f"- Model: **{report.get('best_model', 'N/A')}**")
            st.write(f"- Accuracy: **{report.get('best_accuracy', 'N/A')}**")
            st.write(f"- F1-score: **{report.get('best_f1', 'N/A')}**")
            st.write(f"- Artifact: `{report.get('best_model_artifact', 'N/A')}`")

            st.markdown("### Overfitting Check")
            over = report.get("overfitting_check", {})
            train_acc = over.get("train_accuracy_best")
            test_acc = over.get("test_accuracy_best")
            gap = None if train_acc is None or test_acc is None else float(train_acc) - float(test_acc)
            st.write(f"- Train Accuracy (Best Model): **{train_acc:.4f}**" if train_acc is not None else "- Train Accuracy (Best Model): N/A")
            st.write(f"- Test Accuracy (Best Model): **{test_acc:.4f}**" if test_acc is not None else "- Test Accuracy (Best Model): N/A")
            st.write(f"- Gap: **{gap:.4f}**" if gap is not None else "- Gap: N/A")

            st.markdown("### Feature Importance (Random Forest)")
            for i, row in enumerate(report.get("rf_top_features", [])[:5], start=1):
                st.write(f"{i}. `{row['feature']}` -> **{row['importance']:.4f}**")

            st.markdown("### Behavioral Insights")
            if target_mode == "loyalty":
                st.write("- Stronger **experience** and **usage duration** scores increase loyalty likelihood.")
                st.write("- **Price-discount interactions** influence medium vs low loyalty boundaries.")
                st.write("- **Social engagement + peer context** contributes to loyalty differentiation.")
            else:
                st.write("- Higher discount sensitivity often aligns with **price-sensitive** behavior.")
                st.write("- Brand and decision-factor patterns reveal **economic vs emotional** preference trade-offs.")
                st.write("- Social and peer effects still contribute but with different weight than loyalty target.")
            st.caption("Confidence score is the model's predicted probability for the selected class.")
        else:
            st.info("No project report found yet. Use 'Train / Refresh Artifacts' in the sidebar.")

    st.divider()
    st.subheader("Plots")
    pdir = ARTIFACTS_DIR / "plots"

    row1 = st.columns(3)
    row2 = st.columns(3)

    with row1[0]:
        f = pdir / f"{target_mode}_class_distribution.png"
        if f.exists():
            st.image(str(f), caption="Class Distribution", width="stretch")
    with row1[1]:
        f = pdir / f"{target_mode}_model_comparison_accuracy.png"
        if f.exists():
            st.image(str(f), caption="Baseline Model Accuracy Comparison", width="stretch")
    with row1[2]:
        f = pdir / f"{target_mode}_top2_tuned_comparison.png"
        if f.exists():
            st.image(str(f), caption="Top-2 Tuned Composite Comparison", width="stretch")

    with row2[0]:
        f = _resolve_best_cm_plot(report or {}, pdir, target_mode=target_mode)
        if f and f.exists():
            st.image(str(f), caption="Best Model Confusion Matrix", width="stretch")
    with row2[1]:
        f = pdir / f"{target_mode}_missing_values_by_feature.png"
        if f.exists():
            st.image(str(f), caption="Missing Values by Feature", width="stretch")
    with row2[2]:
        f = pdir / f"{target_mode}_random_forest_feature_importance.png"
        if f.exists():
            st.image(str(f), caption="Random Forest Feature Importance", width="stretch")


if __name__ == "__main__":
    main()

