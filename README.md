# Telecom Customer Churn Prediction

An end-to-end machine learning pipeline to predict customer churn for a telecom provider. The project covers data preprocessing, feature engineering, training 13 models (classical ML, ensemble methods, and deep learning), Bayesian hyperparameter optimisation, probability calibration, and threshold tuning — all within a modular, reproducible pipeline.

---

## Project Structure

```
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Original IBM Telco dataset
├── src/
│   ├── preprocess.py      # Data cleaning, feature engineering, SMOTE, scaling
│   ├── train.py           # Training all 13 models + Optuna optimisation
│   ├── evaluate.py        # Evaluation metrics, visualisations, threshold tuning
│   └── predict.py         # Single-customer inference
├── results/               # Output charts and comparison CSV (auto-generated)
├── saved_models/          # Serialised model files (auto-generated)
├── run_all.py             # Orchestrator — runs the full pipeline end-to-end
├── Points-for-thesis.txt  # Chronological development notes for thesis writing
└── README.md
```

---

## Dataset

**IBM Telco Customer Churn** — 7,043 customers, 20 features, binary target (`Churn: Yes/No`).

- Source: [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Class imbalance: ~73% No Churn / ~27% Churn
- Key features: tenure, contract type, monthly charges, internet service, payment method

---

## Pipeline Overview

### 1. Preprocessing (`src/preprocess.py`)
- Handles missing values in `TotalCharges` (blank strings → median imputation)
- Encodes binary and multi-class categorical columns
- Applies **7 engineered features**:
  - `charges_per_tenure` — monthly cost burden relative to loyalty
  - `high_monthly_charges` — binary flag for above-median charges
  - `is_senior_no_support` — senior citizen without tech support
  - `total_services` — count of active add-on services
  - `log(TotalCharges)` — normalises right-skewed financial data
  - `tenure_contract_ratio` — tenure relative to contract length
  - `long_tenure_churner` — long-term customer still on month-to-month
  - `high_pay_no_support` — high charges with no technical support
  - `no_sticky_services` — no security, backup, or device protection
  - `risky_payment` — payment via Electronic check (highest churn segment)
- Applies **SMOTE** to balance the training set
- Scales numerical features with `StandardScaler`

### 2. Training (`src/train.py`)
Trains 13 models:

| # | Model | Notes |
|---|---|---|
| 1 | Logistic Regression | Baseline linear model |
| 2 | Decision Tree | Baseline tree model |
| 3 | Random Forest | Bagging ensemble |
| 4 | SVM | Kernel-based classifier |
| 5 | XGBoost | Gradient boosting |
| 6 | LightGBM | Fast gradient boosting |
| 7 | XGBoost (Optimized) | Optuna, 50 trials, 5-fold CV |
| 8 | LightGBM (Optimized) | Optuna, 50 trials, 5-fold CV |
| 9 | CatBoost | Auto class-weight balancing |
| 10 | ANN | 4-layer deep network, class_weight, EarlyStopping |
| 11 | Stacking Ensemble | RF+XGB+LGB+LR+SVM, passthrough=True |
| 12 | Stacking Calibrated | Stacking + isotonic probability calibration |
| 13 | Stacking Optimized | Full-stack Optuna tuning (30 trials) |

### 3. Evaluation (`src/evaluate.py`)
- Accuracy, Precision, Recall, F1, AUC-ROC for all 13 models
- Threshold tuning on Stacking Ensemble (Precision-Recall curve)
- Brier Score on Calibrated Stacking (probability reliability)
- Outputs: ROC curves, Confusion matrices, Feature importance, PR curves

### 4. Prediction (`src/predict.py`)
- Loads the best model (ranked by F1 from `comparison_results.csv`)
- Accepts a single customer dict and returns: `churn_prediction`, `churn_probability`, `risk_level`

---

## Results Summary (Run 4)

| Model | Accuracy | F1 | AUC-ROC | Precision |
|---|---|---|---|---|
| **XGBoost** | **77.1%** | **0.777** | 0.821 | 78.8% |
| Stacking_Calibrated | **77.8%** | 0.776 | 0.821 | 77.5% |
| XGBoost_Optimized | 76.9% | 0.775 | **0.828** | 78.5% |
| CatBoost | 76.9% | 0.775 | 0.828 | 78.5% |
| ANN | 71.9% | 0.735 | 0.826 | 79.1% |
| Logistic Regression | 75.9% | 0.767 | **0.833** | 78.3% |

> Logistic Regression achieved the highest AUC-ROC (0.833), confirming the engineered features are strongly linearly separable.

---

## Key Findings

1. **Data engineering > algorithm choice** — switching from SMOTEENN back to SMOTE improved F1 from 0.759 → 0.777, a larger gain than any algorithm swap.
2. **RFECV can hurt** — automated feature removal eliminated features that tree models relied on. Manual domain-knowledge features outperformed automated selection.
3. **ANN ranks last** — with ~7,000 rows, gradient boosting methods consistently outperform deep learning. ANNs require far larger datasets to surpass tree-based ensembles on tabular data.
4. **Probability calibration** — `Stacking_Calibrated` achieved the best accuracy (77.8%) alongside reliable, well-calibrated probabilities (lowest Brier Score), making it the best choice for real-world deployment.

---

## Setup

```powershell
# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install pandas numpy scikit-learn imbalanced-learn xgboost lightgbm catboost tensorflow optuna matplotlib seaborn joblib
```

## Run

```powershell
# Full pipeline (preprocess → train → evaluate)
python run_all.py

# Single prediction
python src/predict.py
```

---

## Dependencies

- Python 3.12
- scikit-learn, imbalanced-learn
- XGBoost, LightGBM, CatBoost
- TensorFlow / Keras 3
- Optuna
- pandas, numpy, matplotlib, seaborn, joblib

---

## Academic Context

This project is part of a Final Year Project (FYP) investigating machine learning techniques for telecom customer churn prediction. Development notes and experiment logs are maintained in [`Points-for-thesis.txt`](Points-for-thesis.txt).
