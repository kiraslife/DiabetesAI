"""
02_train.py — Preprocessing, model training, cross-validation, and evaluation.
Models: Logistic Regression, Random Forest, XGBoost
Priority metric: Recall (minimise false negatives / missed diagnoses)
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

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    RocCurveDisplay, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "diabetes_clean.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
FIG_DIR    = os.path.join(BASE_DIR, "figures")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

RANDOM_STATE = 42
CV_FOLDS     = 5

# ── 1. Load data ───────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
X  = df.drop("Outcome", axis=1)
y  = df["Outcome"]
FEATURE_NAMES = X.columns.tolist()
print(f"Dataset: {X.shape[0]} samples | {X.shape[1]} features")
print(f"Class balance — 0: {(y==0).sum()} | 1: {(y==1).sum()} "
      f"({y.mean()*100:.1f}% diabetic)\n")

# ── 2. Train / test split ──────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train set: {X_train.shape[0]} samples")
print(f"Test  set: {X_test.shape[0]} samples\n")

# ── 3. Feature scaling ─────────────────────────────────────────────────────
scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler,       os.path.join(MODEL_DIR, "scaler.joblib"))
joblib.dump(FEATURE_NAMES, os.path.join(MODEL_DIR, "feature_names.joblib"))
print("✔ Scaler saved\n")

# ── 4. Model definitions ───────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(
        C=1.0, max_iter=1000, random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=None,
        min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="logloss",
        random_state=RANDOM_STATE, n_jobs=-1
    ),
}

# ── 5. Cross-validation + test evaluation ─────────────────────────────────
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

results   = {}
cv_results= {}

print("="*68)
print(f"{'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
print("="*68)

for name, model in models.items():
    # Cross-validation
    cv_res = cross_validate(model, X_train_sc, y_train,
                            cv=cv, scoring=scoring, return_train_score=False)
    cv_results[name] = {k: v.mean() for k, v in cv_res.items()
                        if k.startswith("test_")}

    # Fit on full train set → evaluate on test set
    model.fit(X_train_sc, y_train)
    y_pred  = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:, 1]

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_proba),
    }
    results[name] = metrics
    models[name]  = model   # keep fitted model

    print(f"{name:<22} "
          f"{metrics['accuracy']:>6.3f} "
          f"{metrics['precision']:>6.3f} "
          f"{metrics['recall']:>6.3f} "
          f"{metrics['f1']:>6.3f} "
          f"{metrics['roc_auc']:>6.3f}")

print("="*68)

# ── 6. Cross-validation summary ────────────────────────────────────────────
print("\n[5-Fold CV means on training set]")
print(f"{'Model':<22} {'CV Acc':>7} {'CV Rec':>7} {'CV AUC':>7}")
print("-"*46)
for name, cv_m in cv_results.items():
    print(f"{name:<22} "
          f"{cv_m['test_accuracy']:>7.3f} "
          f"{cv_m['test_recall']:>7.3f} "
          f"{cv_m['test_roc_auc']:>7.3f}")

# ── 7. Identify best model (by Recall on test set) ─────────────────────────
best_name = max(results, key=lambda n: results[n]["recall"])
best_model = models[best_name]
print(f"\n★  Best model by Recall: {best_name} "
      f"(Recall={results[best_name]['recall']:.3f})")

# ── 8. Save best model ─────────────────────────────────────────────────────
joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.joblib"))
with open(os.path.join(MODEL_DIR, "best_model_name.json"), "w") as f:
    json.dump({"name": best_name, "metrics": results[best_name]}, f, indent=2)
print(f"✔ Best model saved → models/best_model.joblib")

# Save all models too
for name, model in models.items():
    safe_name = name.lower().replace(" ", "_")
    joblib.dump(model, os.path.join(MODEL_DIR, f"{safe_name}.joblib"))

# ── 9. Classification report for best model ────────────────────────────────
print(f"\n[Classification Report — {best_name}]")
y_pred_best = best_model.predict(X_test_sc)
print(classification_report(y_test, y_pred_best,
                             target_names=["Non-diabetic", "Diabetic"]))

# ── 10. Plots ──────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight"})
PALETTE = {0: "#4C9BE8", 1: "#E85D75"}

# 10a. Model comparison bar chart
metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc"]
res_df = pd.DataFrame(results).T[metrics_to_plot]

fig, ax = plt.subplots(figsize=(11, 5))
x      = np.arange(len(metrics_to_plot))
width  = 0.22
colors = ["#4C9BE8", "#2ECC71", "#F39C12"]
for i, (model_name, row) in enumerate(res_df.iterrows()):
    offset = (i - 1) * width
    bars   = ax.bar(x + offset, row.values, width,
                    label=model_name, color=colors[i], alpha=0.9, edgecolor="white")
    for bar, val in zip(bars, row.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005, f"{val:.2f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([m.replace("_", " ").title() for m in metrics_to_plot])
ax.set_ylim(0.5, 1.05)
ax.set_ylabel("Score")
ax.set_title("Model Comparison — All Metrics (Test Set)", fontsize=14, fontweight="bold")
ax.legend(loc="lower right")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "model_comparison.png"))
plt.close()
print("\n✔ Saved: model_comparison.png")

# 10b. ROC curves
fig, ax = plt.subplots(figsize=(7, 6))
line_styles = ["-", "--", "-."]
for i, (name, model) in enumerate(models.items()):
    y_proba = model.predict_proba(X_test_sc)[:, 1]
    RocCurveDisplay.from_predictions(
        y_test, y_proba, name=f"{name} (AUC={roc_auc_score(y_test, y_proba):.3f})",
        ax=ax, linestyle=line_styles[i], linewidth=2
    )
ax.plot([0,1],[0,1], "k--", linewidth=1, label="Random Classifier")
ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
ax.legend(loc="lower right")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "roc_curves.png"))
plt.close()
print("✔ Saved: roc_curves.png")

# 10c. Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, model) in zip(axes, models.items()):
    y_pred = model.predict(X_test_sc)
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["Non-diabetic", "Diabetic"],
        cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(f"{name}", fontsize=11, fontweight="bold")
plt.suptitle("Confusion Matrices (Test Set)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "confusion_matrices.png"))
plt.close()
print("✔ Saved: confusion_matrices.png")

print("\n" + "="*68)
print("TRAINING COMPLETE")
print("="*68)
