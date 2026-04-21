# Smartphone Brand Loyalty Prediction

End-to-end, leakage-free binary classification pipeline for predicting whether a student is likely to remain loyal to their smartphone brand.

## 1) Dataset Update

- Active dataset: `data/new_dataset.xlsx`
- Records used by latest run: **500**
- Modeling features: **8**
- Dropped columns: `Timestamp`, `Email`
- Leakage-protected exclusion: `Next Purchase Decision`

Target definition:
- `Very likely` and `Somewhat likely` -> `Loyal (1)`
- `Not likely` -> `Not Loyal (0)`

## 2) Data Handling and Preprocessing

The training system uses a modular `sklearn` pipeline with strict canonicalization + preprocessing:

- **Missing values**
  - Categorical fields: `SimpleImputer(strategy="most_frequent")`
  - Engineered numeric fields: `SimpleImputer(strategy="median")`
- **Encoding**
  - Ordinal encoding: `Usage Duration`, `Discount Influence`, `Social Engagement`
  - Binary encoding: `Experience`, `Peer Influence`
  - One-hot encoding: `Brand`, `Decision Factor`, `Price Importance`
- **Scaling**
  - `StandardScaler` is applied for Logistic Regression numeric inputs
- **Feature engineering**
  - `Experience Score`, `Usage Score`, `Engagement Score`
  - `Experience x Usage`, `Price x Discount`

## 3) EDA Summary (Latest Run)

Generated artifacts:
- `artifacts/reports/eda_summary.json`
- `artifacts/plots/missing_values_by_feature.png`
- `artifacts/plots/class_distribution.png`

Key insights from `eda_summary.json`:
- Class distribution: `Not Loyal=213`, `Loyal=287` (mild positive-class skew)
- Missing values: **0% across all 8 features** in the current file
- Top brands by frequency: `Poco`, `Google Pixel`, `Realme`, `Vivo`, `Samsung`

## 4) Models Trained

The updated training pipeline now trains:

1. **Logistic Regression** (baseline)
2. **Random Forest**
3. **Gradient Boosting**
4. **Extra Trees**

Baseline performance table:
- `artifacts/reports/baseline_model_performance_table.csv`

## 5) Evaluation Metrics

Each model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- 5-fold CV Accuracy

Confusion matrix plots are generated for baseline and tuned models in:
- `artifacts/plots/*_baseline_cm.png`
- `artifacts/plots/*_tuned_cm.png`

## 6) Top 2 Model Selection + Hyperparameter Tuning

Top 2 baseline models selected by composite score:
- **Gradient Boosting**
- **Random Forest**

Tuning method:
- `RandomizedSearchCV` (5-fold CV, scoring by `accuracy`)

Tuning comparison artifact:
- `artifacts/reports/top2_model_comparison.csv`

### Before vs After Tuning (Top 2)

| Model | Accuracy Before | Accuracy After | F1 Before | F1 After | Composite Before | Composite After |
|---|---:|---:|---:|---:|---:|---:|
| Gradient Boosting | 0.9000 | 0.8800 | 0.9180 | 0.9048 | 0.9155 | 0.9027 |
| Random Forest | 0.8900 | 0.8800 | 0.9120 | 0.9048 | 0.9101 | 0.9027 |

## 7) Final Selected Model

From `artifacts/reports/project_experiment_metadata.json`:

- **Selected model:** `Gradient Boosting (Baseline)`
- **Best Accuracy:** `0.90`
- **Best F1:** `0.9180`
- **Saved artifact:** `artifacts/models/best_model.joblib`
- **Model-specific artifact:** `artifacts/models/gradient_boosting_baseline.joblib`

Justification:
- Gradient Boosting baseline achieved the highest held-out test accuracy in the latest run, so it is selected as deployment model.

## 8) Reports and Artifacts

### Models
- `artifacts/models/best_model.joblib`
- `artifacts/models/logistic_regression_baseline.joblib`
- `artifacts/models/extra_trees_baseline.joblib`
- `artifacts/models/random_forest_baseline.joblib`
- `artifacts/models/gradient_boosting_baseline.joblib`
- `artifacts/models/random_forest_tuned.joblib`
- `artifacts/models/gradient_boosting_tuned.joblib`

### Reports
- `artifacts/reports/project_experiment_metadata.json`
- `artifacts/reports/model_performance_table.csv`
- `artifacts/reports/baseline_model_performance_table.csv`
- `artifacts/reports/tuned_top_models_performance_table.csv`
- `artifacts/reports/top2_model_comparison.csv`
- `artifacts/reports/eda_summary.json`

### Plots
- `artifacts/plots/class_distribution.png`
- `artifacts/plots/model_comparison_accuracy.png`
- `artifacts/plots/top2_tuned_comparison.png`
- `artifacts/plots/missing_values_by_feature.png`
- `artifacts/plots/*_baseline_cm.png`
- `artifacts/plots/*_tuned_cm.png`

## 9) How to Run

```bash
python -m src.brand_loyalty.train --dataset "data/new_dataset.xlsx" --output-dir "artifacts"
python -m src.brand_loyalty.train --dataset "data/new_dataset.xlsx" --output-dir "artifacts" --target-mode price_sensitivity
streamlit run app.py
```

Web app updates:
- Sidebar retraining now refreshes all baseline + tuned reports and model artifacts.
- Results panel now shows baseline performance, tuned top-2 performance, and the selected best model details.
- Best-model confusion matrix is selected dynamically based on the current report metadata.

## 10) Project Structure

```text
BrandLoyality/
├── data/
│   ├── project_dataset.xlsx
│   └── new_dataset.xlsx
├── src/brand_loyalty/
│   ├── config.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── train.py
│   └── predictor.py
├── artifacts/
│   ├── models/
│   ├── plots/
│   └── reports/
├── app.py
├── README.md
└── requirements.txt
```

## 11) Research Extension: Price Sensitivity Prediction

Motivation:
- This extension studies whether users prioritize price in purchase behavior, complementing loyalty-focused modeling.

Target definition:
- New target is derived from `Price Importance`.
- `High` / `Very High` -> `Price Sensitive (1)`
- `Low` / `Medium` -> `Not Price Sensitive (0)`

Key difference from loyalty:
- Loyalty captures emotional or brand attachment behavior.
- Price sensitivity captures economic decision behavior.

Experimental consistency:
- Same dataset, same preprocessing pipeline, and same model family are reused.
- This keeps the comparison between loyalty and price sensitivity objectives fair and methodologically consistent.

