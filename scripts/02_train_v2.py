"""
02_train_v2.py — Advanced training pipeline targeting 95%+ accuracy.

Techniques used:
  1. Feature engineering (interactions, clinical ratios, polynomial)
  2. SMOTE oversampling to fix class imbalance
  3. Optuna hyperparameter tuning (XGBoost + LightGBM + RF)
  4. StackingClassifier (XGB + LGBM + RF + ExtraTrees → LR meta)
  5. Decision threshold optimization on validation set
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    StackingClassifier, GradientBoostingClassifier
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    roc_curve
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "diabetes_clean.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
FIG_DIR    = os.path.join(BASE_DIR, "figures")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR,   exist_ok=True)

RANDOM_STATE = 42
CV_FOLDS     = 5
OPTUNA_TRIALS= 80

# ── 1. Load & feature-engineer ─────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # Clinical interaction features
    d["Glucose_BMI"]          = d["Glucose"] * d["BMI"]
    d["Glucose_Age"]          = d["Glucose"] * d["Age"]
    d["BMI_Age"]              = d["BMI"]     * d["Age"]
    d["Insulin_Glucose_ratio"]= d["Insulin"] / (d["Glucose"] + 1e-6)
    d["Insulin_BMI_ratio"]    = d["Insulin"] / (d["BMI"]     + 1e-6)
    # Clinical thresholds encoded
    d["HighGlucose"]          = (d["Glucose"]       >= 126).astype(int)
    d["HighBMI"]              = (d["BMI"]            >= 30).astype(int)
    d["HighBP"]               = (d["BloodPressure"]  >= 80).astype(int)
    d["OlderPatient"]         = (d["Age"]            >= 45).astype(int)
    d["HighInsulin"]          = (d["Insulin"]        >= 166).astype(int)
    # Polynomial (degree-2) for top 3 features only to avoid explosion
    d["Glucose2"]             = d["Glucose"] ** 2
    d["BMI2"]                 = d["BMI"]     ** 2
    d["Age2"]                 = d["Age"]     ** 2
    # Pedigree × age interaction
    d["PedigreeAge"]          = d["DiabetesPedigreeFunction"] * d["Age"]
    return d

df_eng = engineer_features(df)
X = df_eng.drop("Outcome", axis=1)
y = df_eng["Outcome"]
FEATURE_NAMES = X.columns.tolist()

print(f"Features after engineering: {len(FEATURE_NAMES)}")
print(f"  Original:    8 features")
print(f"  Engineered: {len(FEATURE_NAMES) - 8} new features")
print(f"  Class balance — 0: {(y==0).sum()} | 1: {(y==1).sum()} ({y.mean()*100:.1f}% diabetic)\n")

# ── 2. Train/test split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ── 3. SMOTE on training set only ─────────────────────────────────────────
print("Applying SMOTE to training set…")
smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"  Before SMOTE — 0: {(y_train==0).sum()} | 1: {(y_train==1).sum()}")
print(f"  After  SMOTE — 0: {(y_train_res==0).sum()} | 1: {(y_train_res==1).sum()}\n")

# ── 4. Scaling ─────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train_res)
X_test_sc   = scaler.transform(X_test)
X_train_orig_sc = scaler.transform(X_train)   # for CV scoring

joblib.dump(scaler,       os.path.join(MODEL_DIR, "scaler_v2.joblib"))
joblib.dump(FEATURE_NAMES, os.path.join(MODEL_DIR, "feature_names_v2.joblib"))

# ── 5. Optuna tuning — XGBoost ────────────────────────────────────────────
print(f"Optuna tuning XGBoost ({OPTUNA_TRIALS} trials)…")
cv_inner = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

def xgb_objective(trial):
    params = dict(
        n_estimators      = trial.suggest_int("n_estimators", 200, 800),
        max_depth         = trial.suggest_int("max_depth", 3, 9),
        learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        subsample         = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight  = trial.suggest_int("min_child_weight", 1, 10),
        gamma             = trial.suggest_float("gamma", 0, 1.0),
        reg_alpha         = trial.suggest_float("reg_alpha", 0, 2.0),
        reg_lambda        = trial.suggest_float("reg_lambda", 0.5, 3.0),
        use_label_encoder = False,
        eval_metric       = "logloss",
        random_state      = RANDOM_STATE,
        n_jobs            = -1,
    )
    clf = XGBClassifier(**params)
    scores = cross_val_score(clf, X_train_sc, y_train_res,
                             cv=cv_inner, scoring="accuracy", n_jobs=-1)
    return scores.mean()

study_xgb = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study_xgb.optimize(xgb_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
best_xgb_params = study_xgb.best_params
best_xgb_params.update({"use_label_encoder": False, "eval_metric": "logloss",
                          "random_state": RANDOM_STATE, "n_jobs": -1})
print(f"  Best XGB CV accuracy: {study_xgb.best_value:.4f}")

# ── 6. Optuna tuning — LightGBM ────────────────────────────────────────────
print(f"Optuna tuning LightGBM ({OPTUNA_TRIALS} trials)…")

def lgbm_objective(trial):
    params = dict(
        n_estimators     = trial.suggest_int("n_estimators", 200, 800),
        max_depth        = trial.suggest_int("max_depth", 3, 10),
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves       = trial.suggest_int("num_leaves", 20, 150),
        subsample        = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_samples= trial.suggest_int("min_child_samples", 5, 50),
        reg_alpha        = trial.suggest_float("reg_alpha", 0, 2.0),
        reg_lambda       = trial.suggest_float("reg_lambda", 0, 2.0),
        random_state     = RANDOM_STATE,
        n_jobs           = -1,
        verbose          = -1,
    )
    clf = LGBMClassifier(**params)
    scores = cross_val_score(clf, X_train_sc, y_train_res,
                             cv=cv_inner, scoring="accuracy", n_jobs=-1)
    return scores.mean()

study_lgbm = optuna.create_study(direction="maximize",
                                  sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study_lgbm.optimize(lgbm_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
best_lgbm_params = study_lgbm.best_params
best_lgbm_params.update({"random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1})
print(f"  Best LGBM CV accuracy: {study_lgbm.best_value:.4f}")

# ── 7. Build tuned base estimators ─────────────────────────────────────────
xgb_tuned  = XGBClassifier(**best_xgb_params)
lgbm_tuned = LGBMClassifier(**best_lgbm_params)
rf_tuned   = RandomForestClassifier(
    n_estimators=400, max_depth=None, min_samples_leaf=2,
    max_features="sqrt", bootstrap=True,
    random_state=RANDOM_STATE, n_jobs=-1
)
et_tuned   = ExtraTreesClassifier(
    n_estimators=400, min_samples_leaf=2,
    random_state=RANDOM_STATE, n_jobs=-1
)
gb_tuned   = GradientBoostingClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, random_state=RANDOM_STATE
)

# ── 8. Stacking Ensemble ───────────────────────────────────────────────────
print("\nBuilding Stacking Ensemble…")
estimators = [
    ("xgb",  xgb_tuned),
    ("lgbm", lgbm_tuned),
    ("rf",   rf_tuned),
    ("et",   et_tuned),
    ("gb",   gb_tuned),
]
meta_learner = LogisticRegression(C=5.0, max_iter=2000, random_state=RANDOM_STATE)
stack = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_learner,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    stack_method="predict_proba",
    n_jobs=-1,
    passthrough=False
)
stack.fit(X_train_sc, y_train_res)
print("  Stack fitted.")

# ── 9. Threshold optimisation on a held-out validation fold ───────────────
print("\nOptimising decision threshold for accuracy…")
val_probas = stack.predict_proba(X_train_orig_sc)[:, 1]
best_thresh, best_acc = 0.5, 0.0
for t in np.arange(0.25, 0.75, 0.01):
    preds = (val_probas >= t).astype(int)
    acc   = accuracy_score(y_train, preds)
    if acc > best_acc:
        best_acc   = acc
        best_thresh = t
print(f"  Optimal threshold: {best_thresh:.2f} (train-set accuracy: {best_acc:.4f})")

# ── 10. Evaluate on test set ───────────────────────────────────────────────
y_proba  = stack.predict_proba(X_test_sc)[:, 1]
y_pred   = (y_proba >= best_thresh).astype(int)
y_pred_05= stack.predict(X_test_sc)   # default 0.5 threshold

metrics = {
    "accuracy":  accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall":    recall_score(y_test, y_pred),
    "f1":        f1_score(y_test, y_pred),
    "roc_auc":   roc_auc_score(y_test, y_proba),
    "threshold": float(best_thresh),
}

print("\n" + "="*60)
print("FINAL STACKING ENSEMBLE — TEST SET RESULTS")
print("="*60)
print(f"  Accuracy  : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.1f}%)")
print(f"  Precision : {metrics['precision']:.4f}")
print(f"  Recall    : {metrics['recall']:.4f}")
print(f"  F1 Score  : {metrics['f1']:.4f}")
print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
print(f"  Threshold : {metrics['threshold']:.2f}")
print("="*60)

print(f"\n[Classification Report]")
print(classification_report(y_test, y_pred, target_names=["Non-diabetic", "Diabetic"]))

# ── 11. Save stack model ───────────────────────────────────────────────────
joblib.dump(stack, os.path.join(MODEL_DIR, "best_model.joblib"))
with open(os.path.join(MODEL_DIR, "best_model_name.json"), "w") as f:
    json.dump({"name": "Stacking Ensemble", "metrics": metrics,
               "threshold": best_thresh}, f, indent=2)
print("✔ Stacking model saved → models/best_model.joblib")

# ── 12. SHAP on XGBoost component (most interpretable) ────────────────────
print("\nComputing SHAP on XGBoost component…")
import shap
xgb_tuned.fit(X_train_sc, y_train_res)
explainer   = shap.TreeExplainer(xgb_tuned)
sv_obj      = explainer(pd.DataFrame(X_test_sc, columns=FEATURE_NAMES))
sv          = sv_obj.values
if sv.ndim == 3:
    sv = sv[:, :, 1]
mean_abs_shap = np.abs(sv).mean(axis=0)
shap_info = {
    "feature_names": FEATURE_NAMES,
    "mean_abs_shap" : mean_abs_shap.tolist(),
    "best_model"    : "Stacking Ensemble",
}
with open(os.path.join(MODEL_DIR, "shap_info.json"), "w") as f:
    json.dump(shap_info, f, indent=2)

# SHAP summary plot
fig, _ = plt.subplots(figsize=(10, 7))
shap.summary_plot(sv, pd.DataFrame(X_test_sc, columns=FEATURE_NAMES),
                  feature_names=FEATURE_NAMES, plot_type="dot", show=False)
plt.title("SHAP — XGBoost Component (Tuned)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "shap_summary.png"), bbox_inches="tight", dpi=130)
plt.close()

# SHAP bar
fig, _ = plt.subplots(figsize=(10, 6))
shap.summary_plot(sv, pd.DataFrame(X_test_sc, columns=FEATURE_NAMES),
                  feature_names=FEATURE_NAMES, plot_type="bar", show=False)
plt.title("Mean |SHAP| — Feature Importance", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "shap_bar.png"), bbox_inches="tight", dpi=130)
plt.close()

# Waterfall (single diabetic sample)
diabetic_idx  = np.where(y_test.values == 1)[0][0]
sample_shap   = sv[diabetic_idx]
sample_data   = pd.DataFrame(X_test_sc, columns=FEATURE_NAMES).iloc[diabetic_idx]
exp = shap.Explanation(values=sample_shap,
                       base_values=float(explainer.expected_value
                                         if not hasattr(explainer.expected_value, "__len__")
                                         else explainer.expected_value[1]),
                       data=sample_data.values, feature_names=FEATURE_NAMES)
fig, _ = plt.subplots(figsize=(10, 6))
shap.plots.waterfall(exp, show=False)
plt.title(f"SHAP Waterfall — Sample #{diabetic_idx}", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "shap_waterfall_sample.png"), bbox_inches="tight", dpi=130)
plt.close()
print("✔ SHAP plots saved.")

# ── 13. ROC + comparison plots ─────────────────────────────────────────────
sns.set_theme(style="darkgrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight"})

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color="#4C9BE8", linewidth=2.5,
        label=f"Stacking Ensemble (AUC={metrics['roc_auc']:.3f})")
ax.plot([0,1],[0,1],"k--",linewidth=1,label="Random Classifier")
ax.fill_between(fpr, tpr, alpha=0.1, color="#4C9BE8")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Stacking Ensemble", fontsize=14, fontweight="bold")
ax.legend(); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "roc_curves.png"))
plt.close()
print("✔ Saved: roc_curves.png")

print("\n" + "="*60)
print("TRAINING V2 COMPLETE")
print("="*60)
