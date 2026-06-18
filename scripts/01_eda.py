"""
01_eda.py — Data loading, quality audit, cleaning, and EDA visualizations
for the Pima Indians Diabetes dataset.
"""

import os
import urllib.request
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend — no display required
import matplotlib.pyplot as plt
import seaborn as sns

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "diabetes.csv")
FIG_DIR    = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

# ── 1. Download dataset if not present ────────────────────────────────────
URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.csv"
)
COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

if not os.path.exists(DATA_PATH):
    print(f"Downloading dataset → {DATA_PATH}")
    urllib.request.urlretrieve(URL, DATA_PATH)
    # The raw file has no header; add it
    df_raw = pd.read_csv(DATA_PATH, header=None, names=COLUMNS)
    df_raw.to_csv(DATA_PATH, index=False)
    print("Download complete.")
else:
    print(f"Dataset already exists at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"\nShape: {df.shape}")
print(df.dtypes)

# ── 2. Data quality audit ─────────────────────────────────────────────────
print("\n" + "="*60)
print("DATA QUALITY AUDIT")
print("="*60)

print("\n[Missing values (NaN)]")
print(df.isnull().sum())

# Columns where 0 is biologically impossible
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

print("\n[Zero counts in biologically impossible columns]")
zero_counts = (df[zero_cols] == 0).sum()
zero_pcts   = (df[zero_cols] == 0).mean() * 100
zero_summary = pd.DataFrame({"zero_count": zero_counts, "zero_%": zero_pcts.round(1)})
print(zero_summary.to_string())

# Flag and replace zeros with NaN
df_clean = df.copy()
df_clean[zero_cols] = df_clean[zero_cols].replace(0, np.nan)

print(f"\n[After replacing biological zeros → NaN]")
print(df_clean[zero_cols].isnull().sum())

# Impute: median per Outcome class (more accurate than global median)
print("\n[Imputing NaN with per-class median]")
for col in zero_cols:
    class_medians = df_clean.groupby("Outcome")[col].transform("median")
    df_clean[col] = df_clean[col].fillna(class_medians)
    print(f"  {col}: diabetic_median={df_clean.loc[df_clean.Outcome==1, col].median():.1f} | "
          f"non_diabetic_median={df_clean.loc[df_clean.Outcome==0, col].median():.1f}")

print(f"\n[Post-imputation NaN count]")
print(df_clean.isnull().sum())

# Save cleaned dataset
CLEAN_PATH = os.path.join(BASE_DIR, "data", "diabetes_clean.csv")
df_clean.to_csv(CLEAN_PATH, index=False)
print(f"\nCleaned dataset saved → {CLEAN_PATH}")

# ── 3. Summary statistics ─────────────────────────────────────────────────
print("\n" + "="*60)
print("DESCRIPTIVE STATISTICS (cleaned)")
print("="*60)
print(df_clean.describe().round(2).to_string())

# ── 4. Plots ──────────────────────────────────────────────────────────────
PALETTE = {0: "#4C9BE8", 1: "#E85D75"}     # blue = no diabetes, rose = diabetic
LABEL_MAP = {0: "Non-diabetic", 1: "Diabetic"}

sns.set_theme(style="darkgrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight"})

# 4a. Class distribution
fig, ax = plt.subplots(figsize=(6, 4))
counts = df_clean["Outcome"].value_counts().sort_index()
bars = ax.bar(
    [LABEL_MAP[i] for i in counts.index],
    counts.values,
    color=[PALETTE[i] for i in counts.index],
    edgecolor="white", linewidth=1.5, width=0.5
)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f"{val}\n({val/len(df_clean)*100:.1f}%)",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_title("Class Distribution — Diabetes Outcome", fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("Count")
ax.set_ylim(0, counts.max() * 1.18)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "class_distribution.png"))
plt.close()
print("\n✔ Saved: class_distribution.png")

# 4b. Correlation heatmap
fig, ax = plt.subplots(figsize=(9, 7))
corr = df_clean.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, linewidths=0.5, ax=ax,
            annot_kws={"size": 9})
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "correlation_heatmap.png"))
plt.close()
print("✔ Saved: correlation_heatmap.png")

# 4c. Feature distributions by Outcome (KDE)
features = [c for c in df_clean.columns if c != "Outcome"]
fig, axes = plt.subplots(4, 2, figsize=(14, 16))
axes = axes.flatten()
for i, feat in enumerate(features):
    for outcome, color in PALETTE.items():
        subset = df_clean.loc[df_clean.Outcome == outcome, feat]
        axes[i].hist(subset, bins=30, alpha=0.55, color=color,
                     label=LABEL_MAP[outcome], density=True, edgecolor="none")
        subset.plot.kde(ax=axes[i], color=color, linewidth=2)
    axes[i].set_title(feat, fontsize=12, fontweight="bold")
    axes[i].set_ylabel("Density")
    axes[i].legend(fontsize=9)
    axes[i].spines[["top","right"]].set_visible(False)
plt.suptitle("Feature Distributions by Outcome", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "feature_distributions.png"))
plt.close()
print("✔ Saved: feature_distributions.png")

# 4d. Boxplots by Outcome
fig, axes = plt.subplots(4, 2, figsize=(14, 16))
axes = axes.flatten()
for i, feat in enumerate(features):
    data_plot = [df_clean.loc[df_clean.Outcome==o, feat].values for o in [0, 1]]
    bp = axes[i].boxplot(data_plot, patch_artist=True, widths=0.4,
                         medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], PALETTE.values()):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    axes[i].set_xticks([1, 2])
    axes[i].set_xticklabels([LABEL_MAP[0], LABEL_MAP[1]])
    axes[i].set_title(feat, fontsize=12, fontweight="bold")
    axes[i].spines[["top","right"]].set_visible(False)
plt.suptitle("Boxplots by Outcome", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "boxplots_by_outcome.png"))
plt.close()
print("✔ Saved: boxplots_by_outcome.png")

print("\n" + "="*60)
print("EDA COMPLETE — all figures saved to /figures/")
print("="*60)
