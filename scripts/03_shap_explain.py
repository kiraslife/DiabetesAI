"""
03_shap_explain.py — SHAP global feature importance and individual prediction
explanation using the best-performing model.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import shap

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "diabetes_clean.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
FIG_DIR   = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

from sklearn.model_selection import train_test_split

# ── Load data ──────────────────────────────────────────────────────────────
df      = pd.read_csv(DATA_PATH)
X       = df.drop("Outcome", axis=1)
y       = df["Outcome"]
FEATURE_NAMES = X.columns.tolist()

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Load best model + scaler ───────────────────────────────────────────────
scaler     = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
best_model = joblib.load(os.path.join(MODEL_DIR, "best_model.joblib"))

with open(os.path.join(MODEL_DIR, "best_model_name.json")) as f:
    meta = json.load(f)
best_name = meta["name"]
print(f"Using best model: {best_name}")

X_test_sc = scaler.transform(X_test)
X_test_df = pd.DataFrame(X_test_sc, columns=FEATURE_NAMES)

# ── SHAP explainer ─────────────────────────────────────────────────────────
print("Computing SHAP values…")

model_type = best_name.lower()
if "logistic" in model_type:
    explainer   = shap.LinearExplainer(best_model, X_test_df)
    shap_values = explainer.shap_values(X_test_df)
    # LinearExplainer returns 1D array for binary classification
    if isinstance(shap_values, list):
        shap_vals_pos = shap_values[1]
    else:
        shap_vals_pos = shap_values
    expected_value = (explainer.expected_value if not hasattr(explainer.expected_value, "__len__")
                      else explainer.expected_value[1])
else:
    # Tree-based: use TreeExplainer
    explainer = shap.TreeExplainer(best_model)
    shap_vals_obj = explainer(X_test_df)
    # Handle multi-output shape (n_samples, n_features, n_classes)
    if shap_vals_obj.values.ndim == 3:
        shap_vals_pos     = shap_vals_obj.values[:, :, 1]
        expected_value    = explainer.expected_value[1]
    else:
        shap_vals_pos     = shap_vals_obj.values
        expected_value    = explainer.expected_value

print(f"SHAP values shape: {shap_vals_pos.shape}")

# ── Plot 1: Beeswarm (global summary) ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
shap.summary_plot(
    shap_vals_pos, X_test_df,
    feature_names=FEATURE_NAMES,
    plot_type="dot",
    show=False
)
plt.title(f"SHAP Global Feature Importance — {best_name}",
          fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "shap_summary.png"), bbox_inches="tight", dpi=130)
plt.close()
print("✔ Saved: shap_summary.png")

# ── Plot 2: Bar summary ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
shap.summary_plot(
    shap_vals_pos, X_test_df,
    feature_names=FEATURE_NAMES,
    plot_type="bar",
    show=False
)
plt.title(f"Mean |SHAP| Feature Importance — {best_name}",
          fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "shap_bar.png"), bbox_inches="tight", dpi=130)
plt.close()
print("✔ Saved: shap_bar.png")

# ── Plot 3: Waterfall for a single prediction ──────────────────────────────
# Pick the first diabetic sample in the test set for a meaningful example
diabetic_idx = np.where(y_test.values == 1)[0][0]
sample_idx   = diabetic_idx

print(f"\nIndividual explanation for test sample #{sample_idx} "
      f"(True label: {y_test.values[sample_idx]})")

y_pred_sample = best_model.predict(X_test_sc[sample_idx:sample_idx+1])[0]
y_prob_sample = best_model.predict_proba(X_test_sc[sample_idx:sample_idx+1])[0, 1]
print(f"  Predicted label: {y_pred_sample} | Probability: {y_prob_sample:.3f}")

# Build Explanation object for waterfall
sample_shap = shap_vals_pos[sample_idx]
sample_data = X_test_df.iloc[sample_idx]

exp = shap.Explanation(
    values=sample_shap,
    base_values=float(expected_value),
    data=sample_data.values,
    feature_names=FEATURE_NAMES
)

fig, ax = plt.subplots(figsize=(9, 6))
shap.plots.waterfall(exp, show=False)
plt.title(
    f"SHAP Waterfall — Sample #{sample_idx} "
    f"(Prob={y_prob_sample:.2f}, True={y_test.values[sample_idx]})",
    fontsize=12, fontweight="bold", pad=10
)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "shap_waterfall_sample.png"), bbox_inches="tight", dpi=130)
plt.close()
print("✔ Saved: shap_waterfall_sample.png")

# ── Print top drivers ──────────────────────────────────────────────────────
top_features = pd.Series(np.abs(sample_shap), index=FEATURE_NAMES).nlargest(3)
print("\n[Top 3 SHAP drivers for this prediction]")
for feat, val in top_features.items():
    direction = "↑ increases" if sample_shap[FEATURE_NAMES.index(feat)] > 0 else "↓ decreases"
    print(f"  {feat:30s} SHAP={sample_shap[FEATURE_NAMES.index(feat)]:+.4f}  {direction} risk")

# Save top features as JSON for the Streamlit app
shap_info = {
    "feature_names": FEATURE_NAMES,
    "mean_abs_shap": np.abs(shap_vals_pos).mean(axis=0).tolist(),
    "best_model": best_name
}
with open(os.path.join(MODEL_DIR, "shap_info.json"), "w") as f:
    json.dump(shap_info, f, indent=2)
print("\n✔ SHAP info saved → models/shap_info.json")
print("\nSHAP EXPLANATION COMPLETE")
