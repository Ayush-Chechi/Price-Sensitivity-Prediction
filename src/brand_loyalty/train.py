"""Training entrypoint for brand loyalty binary classification."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from .config import CFG
from .data import TargetMode, get_feature_columns, load_dataset
from .preprocessing import FeatureEngineering, build_preprocessor


def evaluate(model, X_train, X_test, y_train, y_test, X_all, y_all) -> Dict[str, Any]:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
    auc = roc_auc_score(y_test, proba)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=CFG.random_state)
    cv_scores = cross_val_score(model, X_all, y_all, cv=cv, scoring="accuracy")
    return {
        "accuracy": float(acc),
        "F1": float(f1),
        "Precision": float(prec),
        "Recall": float(rec),
        "ROC-AUC": float(auc),
        "CV Score": float(cv_scores.mean()),
        "CV Std": float(cv_scores.std(ddof=0)),
        "cm": cm,
    }


def _save_confusion(cm, title: str, path: Path, *, negative_label: str, positive_label: str) -> None:
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[negative_label, positive_label],
        yticklabels=[negative_label, positive_label],
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _save_bar(x, y, title: str, xlabel: str, ylabel: str, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.barplot(x=x, y=y, color="#4f46e5")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _extract_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    keys = ["accuracy", "F1", "Precision", "Recall", "ROC-AUC", "CV Score", "CV Std"]
    out = {k: float(metrics[k]) for k in keys}
    out["composite"] = float(np.mean([out["accuracy"], out["F1"], out["Precision"], out["Recall"]]))
    return out


def _build_pipeline(model, *, scale_numeric: bool, onehot_features: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineering()),
            ("preprocess", build_preprocessor(scale_numeric=scale_numeric, onehot_features=onehot_features)),
            ("model", model),
        ]
    )


def _eda_summary(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    class_counts = y.value_counts().sort_index().to_dict()
    missing_pct = (X.isna().mean() * 100).round(2).sort_values(ascending=False)
    top_brands = X["Brand"].value_counts(dropna=False).head(5).to_dict()
    return {
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "class_distribution": {str(k): int(v) for k, v in class_counts.items()},
        "class_balance_ratio": round(float(class_counts.get(1, 0) / max(class_counts.get(0, 1), 1)), 4),
        "missing_percentage_by_feature": {k: float(v) for k, v in missing_pct.to_dict().items()},
        "top_5_brands": {str(k): int(v) for k, v in top_brands.items()},
    }


def train(dataset_path: str = CFG.dataset_path, output_dir: str = "artifacts", target_mode: TargetMode = "loyalty") -> None:
    out = Path(output_dir)
    models_dir = out / "models"
    plots_dir = out / "plots"
    reports_dir = out / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    _, X, y_raw = load_dataset(dataset_path, target_mode=target_mode)
    y = (y_raw >= 1).astype(int) if target_mode == "loyalty" else y_raw.astype(int)
    onehot_features = [c for c in get_feature_columns(target_mode) if c in CFG.onehot_features]
    positive_label = "Loyal" if target_mode == "loyalty" else "Price Sensitive"
    negative_label = "Not Loyal" if target_mode == "loyalty" else "Not Price Sensitive"
    target_prefix = target_mode
    eda = _eda_summary(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=CFG.random_state)

    lr = _build_pipeline(
        LogisticRegression(max_iter=CFG.logreg_max_iter, class_weight="balanced", random_state=CFG.random_state),
        scale_numeric=True,
        onehot_features=onehot_features,
    )
    rf = _build_pipeline(
        RandomForestClassifier(n_estimators=350, class_weight="balanced", random_state=CFG.random_state, n_jobs=-1),
        scale_numeric=False,
        onehot_features=onehot_features,
    )
    et = _build_pipeline(
        ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=CFG.random_state, n_jobs=-1),
        scale_numeric=False,
        onehot_features=onehot_features,
    )
    gb = _build_pipeline(
        GradientBoostingClassifier(random_state=CFG.random_state),
        scale_numeric=False,
        onehot_features=onehot_features,
    )

    baseline_models = [
        ("Logistic Regression", lr),
        ("Random Forest", rf),
        ("Gradient Boosting", gb),
        ("Extra Trees", et),
    ]
    baseline_model_map = {name: model for name, model in baseline_models}
    baseline_eval: Dict[str, Dict[str, Any]] = {}
    for name, model in baseline_models:
        baseline_eval[name] = evaluate(model, X_train, X_test, y_train, y_test, X, y)

    baseline_table = (
        pd.DataFrame([{"Model": n, **_extract_metrics(m)} for n, m in baseline_eval.items()])
        .rename(columns={"accuracy": "Accuracy"})
        .sort_values(by=["composite", "Accuracy"], ascending=False)
        .reset_index(drop=True)
    )
    top2_names = baseline_table.head(2)["Model"].tolist()

    tuned_models: Dict[str, Pipeline] = {}
    tuned_eval: Dict[str, Dict[str, Any]] = {}
    tuning_params: Dict[str, Dict[str, Any]] = {}

    if "Random Forest" in top2_names:
        rf_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions={
                "model__n_estimators": [250, 350, 500],
                "model__max_depth": [None, 8, 12, 16],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", None],
            },
            n_iter=12,
            cv=5,
            scoring="accuracy",
            random_state=CFG.random_state,
            n_jobs=-1,
        )
        rf_search.fit(X_train, y_train)
        tuned_models["Random Forest"] = rf_search.best_estimator_
        tuned_eval["Random Forest"] = evaluate(rf_search.best_estimator_, X_train, X_test, y_train, y_test, X, y)
        tuning_params["Random Forest"] = rf_search.best_params_

    if "Gradient Boosting" in top2_names:
        gb_search = RandomizedSearchCV(
            estimator=gb,
            param_distributions={
                "model__n_estimators": [150, 250, 350],
                "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
                "model__max_depth": [2, 3, 4],
                "model__subsample": [0.7, 0.85, 1.0],
                "model__min_samples_leaf": [1, 2, 4],
            },
            n_iter=12,
            cv=5,
            scoring="accuracy",
            random_state=CFG.random_state,
            n_jobs=-1,
        )
        gb_search.fit(X_train, y_train)
        tuned_models["Gradient Boosting"] = gb_search.best_estimator_
        tuned_eval["Gradient Boosting"] = evaluate(gb_search.best_estimator_, X_train, X_test, y_train, y_test, X, y)
        tuning_params["Gradient Boosting"] = gb_search.best_params_

    if "Extra Trees" in top2_names:
        et_search = RandomizedSearchCV(
            estimator=et,
            param_distributions={
                "model__n_estimators": [300, 500, 700, 900],
                "model__max_depth": [None, 8, 12, 16, 24],
                "model__min_samples_split": [2, 4, 6, 8],
                "model__min_samples_leaf": [1, 2, 3, 4],
                "model__max_features": ["sqrt", "log2", None],
            },
            n_iter=25,
            cv=5,
            scoring="accuracy",
            random_state=CFG.random_state,
            n_jobs=-1,
        )
        et_search.fit(X_train, y_train)
        tuned_models["Extra Trees"] = et_search.best_estimator_
        tuned_eval["Extra Trees"] = evaluate(et_search.best_estimator_, X_train, X_test, y_train, y_test, X, y)
        tuning_params["Extra Trees"] = et_search.best_params_

    if "Logistic Regression" in top2_names:
        lr_search = RandomizedSearchCV(
            estimator=lr,
            param_distributions={"model__C": np.logspace(-2, 2, 15), "model__solver": ["lbfgs", "liblinear"]},
            n_iter=10,
            cv=5,
            scoring="accuracy",
            random_state=CFG.random_state,
            n_jobs=-1,
        )
        lr_search.fit(X_train, y_train)
        tuned_models["Logistic Regression"] = lr_search.best_estimator_
        tuned_eval["Logistic Regression"] = evaluate(lr_search.best_estimator_, X_train, X_test, y_train, y_test, X, y)
        tuning_params["Logistic Regression"] = lr_search.best_params_

    tuned_table = (
        pd.DataFrame([{"Model": n, **_extract_metrics(m)} for n, m in tuned_eval.items()])
        .rename(columns={"accuracy": "Accuracy"})
        .sort_values(by=["composite", "Accuracy"], ascending=False)
        .reset_index(drop=True)
    )

    comparison_rows: List[Dict[str, Any]] = []
    for model_name in top2_names:
        before = _extract_metrics(baseline_eval[model_name])
        after = _extract_metrics(tuned_eval[model_name])
        comparison_rows.append(
            {
                "Model": model_name,
                "Accuracy Before": before["accuracy"],
                "Accuracy After": after["accuracy"],
                "F1 Before": before["F1"],
                "F1 After": after["F1"],
                "Precision Before": before["Precision"],
                "Precision After": after["Precision"],
                "Recall Before": before["Recall"],
                "Recall After": after["Recall"],
                "Composite Before": before["composite"],
                "Composite After": after["composite"],
            }
        )
    comparison_table = pd.DataFrame(comparison_rows).sort_values(by="Composite After", ascending=False).reset_index(drop=True)

    model_candidates: List[Dict[str, Any]] = []
    for model_name, metrics in baseline_eval.items():
        extracted = _extract_metrics(metrics)
        model_candidates.append(
            {
                "Model": model_name,
                "Variant": "Baseline",
                "Accuracy": extracted["accuracy"],
                "Composite": extracted["composite"],
            }
        )
    for model_name, metrics in tuned_eval.items():
        extracted = _extract_metrics(metrics)
        model_candidates.append(
            {
                "Model": model_name,
                "Variant": "Tuned",
                "Accuracy": extracted["accuracy"],
                "Composite": extracted["composite"],
            }
        )
    candidates_df = pd.DataFrame(model_candidates).sort_values(by=["Accuracy", "Composite"], ascending=False).reset_index(drop=True)
    best_model_name = str(candidates_df.iloc[0]["Model"])
    best_model_variant = str(candidates_df.iloc[0]["Variant"])
    if best_model_variant == "Tuned":
        best_model = tuned_models[best_model_name]
        best_metrics = tuned_eval[best_model_name]
        model_filename = f"{best_model_name.lower().replace(' ', '_')}_tuned"
        best_params = tuning_params.get(best_model_name, {})
    else:
        best_model = baseline_model_map[best_model_name]
        best_metrics = baseline_eval[best_model_name]
        model_filename = f"{best_model_name.lower().replace(' ', '_')}_baseline"
        best_params = {}

    # Save models
    for model_name, model in baseline_models:
        model.fit(X_train, y_train)
        joblib.dump(model, models_dir / f"{model_name.lower().replace(' ', '_')}_baseline.joblib")
    for model_name, model in tuned_models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, models_dir / f"{model_name.lower().replace(' ', '_')}_tuned.joblib")
    best_model.fit(X_train, y_train)
    joblib.dump(best_model, models_dir / f"{target_prefix}_best_model.joblib")

    # Plots
    for name, metrics in baseline_eval.items():
        _save_confusion(
            metrics["cm"],
            f"Confusion Matrix - {name} (Baseline)",
            plots_dir / f"{target_prefix}_{name.lower().replace(' ', '_')}_baseline_cm.png",
            negative_label=negative_label,
            positive_label=positive_label,
        )
    for name, metrics in tuned_eval.items():
        _save_confusion(
            metrics["cm"],
            f"Confusion Matrix - {name} (Tuned)",
            plots_dir / f"{target_prefix}_{name.lower().replace(' ', '_')}_tuned_cm.png",
            negative_label=negative_label,
            positive_label=positive_label,
        )

    counts = y.value_counts().sort_index()
    _save_bar(
        x=[negative_label, positive_label],
        y=[int(counts.get(0, 0)), int(counts.get(1, 0))],
        title="Class Distribution",
        xlabel="Class",
        ylabel="Count",
        path=plots_dir / f"{target_prefix}_class_distribution.png",
    )

    _save_bar(
        x=baseline_table["Model"].tolist(),
        y=baseline_table["Accuracy"].tolist(),
        title="Baseline Model Comparison (Accuracy)",
        xlabel="Model",
        ylabel="Accuracy",
        path=plots_dir / f"{target_prefix}_model_comparison_accuracy.png",
    )
    _save_bar(
        x=comparison_table["Model"].tolist(),
        y=comparison_table["Composite After"].tolist(),
        title="Top 2 Tuned Model Comparison (Composite)",
        xlabel="Model",
        ylabel="Composite Score",
        path=plots_dir / f"{target_prefix}_top2_tuned_comparison.png",
    )

    missing_vals = pd.Series(eda["missing_percentage_by_feature"]).sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=missing_vals.values, y=missing_vals.index, color="#64748b")
    plt.title("Missing Values by Feature (%)")
    plt.xlabel("Missing Percentage")
    plt.tight_layout()
    plt.savefig(plots_dir / f"{target_prefix}_missing_values_by_feature.png", dpi=180)
    plt.close()

    top_names: List[str] = []
    top_vals: List[float] = []
    if best_model_name == "Random Forest":
        feature_names = list(best_model.named_steps["preprocess"].get_feature_names_out())
        importances = best_model.named_steps["model"].feature_importances_
        idx = np.argsort(importances)[::-1][:10]
        top_names = [feature_names[i] for i in idx]
        top_vals = [float(importances[i]) for i in idx]
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_vals, y=top_names, color="#0ea5e9")
        plt.title("Random Forest Feature Importance (Top 10)")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(plots_dir / f"{target_prefix}_random_forest_feature_importance.png", dpi=180)
        plt.close()

    # Reports
    baseline_table.to_csv(reports_dir / f"{target_prefix}_baseline_model_performance_table.csv", index=False)
    tuned_table.to_csv(reports_dir / f"{target_prefix}_tuned_top_models_performance_table.csv", index=False)
    comparison_table.to_csv(reports_dir / f"{target_prefix}_top2_model_comparison.csv", index=False)
    tuned_table.to_csv(reports_dir / f"{target_prefix}_model_performance_table.csv", index=False)
    (reports_dir / f"{target_prefix}_eda_summary.json").write_text(json.dumps(eda, indent=2), encoding="utf-8")

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_path,
        "dataset_samples": eda["n_samples"],
        "dataset_features": eda["n_features"],
        "leakage_free": True,
        "excluded_feature": "Next Purchase Decision",
        "target_mode": target_mode,
        "target": "Loyal (1) vs Not Loyal (0)" if target_mode == "loyalty" else "Price Sensitive (1) vs Not Price Sensitive (0)",
        "models_baseline": baseline_table.to_dict(orient="records"),
        "top_2_models": top2_names,
        "models": tuned_table.to_dict(orient="records"),
        "top2_comparison": comparison_table.to_dict(orient="records"),
        "best_model": f"{best_model_name} ({best_model_variant})",
        "best_model_artifact": f"models/{model_filename}.joblib",
        "best_model_target_artifact": f"models/{target_prefix}_best_model.joblib",
        "best_accuracy": best_metrics["accuracy"],
        "best_f1": best_metrics["F1"],
        "best_params": best_params,
        "tuning_params": tuning_params,
        "rf_top_features": [{"feature": n, "importance": v} for n, v in zip(top_names, top_vals)],
        "eda_summary": eda,
        "overfitting_check": {
            "train_accuracy_best": float(accuracy_score(y_train, best_model.predict(X_train))),
            "test_accuracy_best": float(best_metrics["accuracy"]),
        },
        "model_selection_justification": (
            f"{best_model_name} ({best_model_variant}) achieved the highest held-out test accuracy; "
            "composite score was used as a tie-breaker."
        ),
    }
    (reports_dir / f"{target_prefix}_project_experiment_metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Backward-compatible default report path for loyalty mode.
    if target_mode == "loyalty":
        (reports_dir / "project_experiment_metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logging.info("Training complete. Best model: %s | Accuracy: %.4f", best_model_name, best_metrics["accuracy"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train final brand loyalty binary classifier")
    parser.add_argument("--dataset", type=str, default=CFG.dataset_path)
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--target-mode", type=str, choices=["loyalty", "price_sensitivity"], default="loyalty")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    train(args.dataset, args.output_dir, args.target_mode)


if __name__ == "__main__":
    main()

