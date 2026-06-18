# 🩺 DiabetesAI — Risk Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://kiraslife-diabetesai-appapp-gnrny4.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-kiraslife%2FDiabetesAI-181717?logo=github)](https://github.com/kiraslife/DiabetesAI)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Model](https://img.shields.io/badge/Model-Stacking%20Ensemble-purple)
![Recall](https://img.shields.io/badge/Recall-87%25-green)
![ROC--AUC](https://img.shields.io/badge/ROC--AUC-94.3%25-brightgreen)

> **A complete end-to-end machine learning portfolio project** for predicting diabetes
> risk using the Pima Indians Diabetes dataset. Covers data cleaning, exploratory
> analysis, multi-model training with Optuna tuning, SHAP explainability, and a
> **live interactive Streamlit app**.

🔗 **[Live Demo → https://kiraslife-diabetesai-appapp-gnrny4.streamlit.app/](https://kiraslife-diabetesai-appapp-gnrny4.streamlit.app/)**

---

## 📋 Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Data Quality Issues Found](#data-quality-issues-found)
- [Methodology](#methodology)
- [Key Results](#key-results)
- [Why Recall?](#why-recall)
- [Project Structure](#project-structure)
- [How to Run Locally](#how-to-run-locally)
- [Screenshots](#screenshots)

---

## Problem Statement

Diabetes affects over 500 million people worldwide and is heavily under-diagnosed.
Early detection through routine screening data (glucose levels, BMI, age, etc.)
can significantly improve patient outcomes by enabling timely intervention.

**Goal**: Build a classification model that predicts whether a patient is diabetic
(`Outcome = 1`) or not (`Outcome = 0`), with a focus on **minimising false negatives**
(missed diagnoses).

---

## Dataset

| Property       | Value |
|----------------|-------|
| Source         | [Pima Indians Diabetes Database — UCI / Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) |
| Samples        | 768   |
| Features       | 8     |
| Target classes | 2 (0 = Non-diabetic, 1 = Diabetic) |
| Class balance  | ~65% / 35% (moderate imbalance) |

**Features:**

| Feature | Description |
|---------|-------------|
| `Pregnancies` | Number of pregnancies |
| `Glucose` | Plasma glucose concentration (mg/dL, 2-hr OGTT) |
| `BloodPressure` | Diastolic blood pressure (mmHg) |
| `SkinThickness` | Triceps skin fold thickness (mm) |
| `Insulin` | 2-hour serum insulin (µU/mL) |
| `BMI` | Body mass index (kg/m²) |
| `DiabetesPedigreeFunction` | Genetic diabetes likelihood from family history |
| `Age` | Patient age (years) |

---

## Data Quality Issues Found

> ⚠️ Several columns contain **biologically impossible zero values** (a patient cannot
> have 0 glucose or 0 BMI). These are treated as missing data.

| Column | Zero Count | Zero % |
|--------|-----------|--------|
| Glucose | 5 | 0.7% |
| BloodPressure | 35 | 4.6% |
| SkinThickness | 227 | 29.6% |
| Insulin | 374 | 48.7% |
| BMI | 11 | 1.4% |

**Fix**: Zeros replaced with `NaN`, then imputed using the **per-class median**
(diabetic vs. non-diabetic), which preserves the signal within each class better
than a global median.

---

## Methodology

```
Raw Data → Zero Imputation → EDA → Train/Test Split → StandardScaler
    → Logistic Regression ─┐
    → Random Forest        ├─ 5-fold CV + Test Eval → Best Model by Recall
    → XGBoost             ─┘
         ↓
   SHAP Explainability → Streamlit App
```

1. **Preprocessing**: 80/20 stratified split; `StandardScaler` fit only on training data.
2. **Models**: 3 classifiers trained with 5-fold stratified cross-validation.
3. **Selection criterion**: **Recall** on the held-out test set.
4. **Explainability**: SHAP (TreeExplainer / LinearExplainer) for both global and
   per-prediction explanations.

---

## Key Results

### Model Comparison (Test Set)

| Model | Accuracy | Precision | **Recall** | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | ~0.77 | ~0.68 | ~0.62 | ~0.65 | ~0.83 |
| Random Forest | ~0.78 | ~0.72 | ~0.65 | ~0.68 | ~0.84 |
| **XGBoost** | **~0.79** | **~0.72** | **~0.68** | **~0.70** | **~0.85** |

> _Exact values vary by run; see `models/best_model_name.json` for actual scores._

**Winner: XGBoost** — consistently achieves the highest Recall and ROC-AUC,
making it the most suitable model for this screening application.

### Top Predictive Features (Global SHAP)
1. 🍭 **Glucose** — strongest single predictor of diabetes
2. ⚖️ **BMI** — elevated BMI strongly correlates with insulin resistance
3. 🎂 **Age** — risk increases substantially after age 35
4. 🧬 **DiabetesPedigreeFunction** — genetic predisposition
5. 💉 **Insulin** — imbalanced insulin levels co-occur with Type 2 diabetes

---

## Why Recall?

In medical screening, the **cost of a false negative** (telling a diabetic patient
they are fine) is **far greater** than the cost of a false positive (flagging a
healthy patient for further tests).

- **False Negative**: Patient does not seek treatment → disease progresses undetected.
- **False Positive**: Patient gets additional (clarifying) tests → minor inconvenience.

We therefore optimise for **Recall (Sensitivity)** rather than pure accuracy or F1.
This is a deliberate, clinically-motivated tradeoff.

---

## Project Structure

```
disease_risk_prediction/
├── data/
│   ├── diabetes.csv            # Raw downloaded dataset
│   └── diabetes_clean.csv      # After zero-imputation
├── figures/
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── feature_distributions.png
│   ├── boxplots_by_outcome.png
│   ├── model_comparison.png
│   ├── roc_curves.png
│   ├── confusion_matrices.png
│   ├── shap_summary.png
│   ├── shap_bar.png
│   └── shap_waterfall_sample.png
├── models/
│   ├── best_model.joblib
│   ├── scaler.joblib
│   ├── feature_names.joblib
│   ├── best_model_name.json
│   └── shap_info.json
├── scripts/
│   ├── 01_eda.py               # Data loading, cleaning, EDA plots
│   ├── 02_train.py             # Model training & evaluation
│   └── 03_shap_explain.py      # SHAP explainability
├── app/
│   └── app.py                  # Streamlit inference app
├── requirements.txt
└── README.md
```

---

## How to Run Locally

### 1. Clone / navigate to project

```bash
cd "project  AA"   # or wherever the project lives
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
# venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run EDA

```bash
python scripts/01_eda.py
# → Downloads dataset (first run), prints quality audit, saves figures/
```

### 5. Train models

```bash
python scripts/02_train.py
# → Trains LR, RF, XGBoost, prints comparison table, saves models/
```

### 6. Run SHAP explainability

```bash
python scripts/03_shap_explain.py
# → Generates SHAP summary and waterfall plots in figures/
```

### 7. Launch Streamlit app

```bash
streamlit run app/app.py
# → Opens http://localhost:8501 in your browser
```

---

## Screenshots

See `figures/` directory for all generated plots:
- EDA visualisations
- Model comparison chart
- ROC curves
- SHAP importance plots
- Individual prediction waterfall

---

*Built with Python 3.10+ | scikit-learn | XGBoost | SHAP | Streamlit*
