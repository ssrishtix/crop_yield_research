# -*- coding: utf-8 -*-
"""
yield_model_rf.py
=================
Train per-crop Random Forest models to predict crop yield (yield_kg_ha)
from the sliding-window feature matrix.  Handles small datasets gracefully
with leave-one-out CV fallback and prints clean diagnostic tables.

Outputs:
    rf_feature_importance.png   – bar chart of top feature importances
    rf_yield_predictions.csv    – predicted vs actual yield for all horizons
"""

# ── Install dependencies quietly ────────────────────────────────────────
import subprocess, sys
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q",
     "scikit-learn", "pandas", "numpy", "matplotlib"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)

import warnings, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # headless backend
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── Formatting helpers ──────────────────────────────────────────────────
def header(step: str, title: str):
    """Print a prominent step header."""
    print(f"\n{'='*70}")
    print(f"  {step} — {title}")
    print(f"{'='*70}\n")


def safe_mape(y_true, y_pred):
    """Mean Absolute Percentage Error, guarded against zero actuals."""
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


# =====================================================================
#  STEP 1 — Load and Prepare
# =====================================================================
header("Step 1", "Load and Prepare")

df = pd.read_csv("feature_matrix_weekly.csv")
print(f"Loaded feature_matrix_weekly.csv  →  shape {df.shape}")
print(f"\nSplit counts per crop:")
print(df.groupby(["crop", "split"]).size().unstack(fill_value=0).to_string())

# ── Identify feature columns ────────────────────────────────────────────
EXCLUDE_COLS = {"crop", "district", "year", "horizon_weeks",
                "yield_kg_ha", "split", "window_start", "window_end", "n_days"}
feature_cols = [c for c in df.columns
                if c not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(df[c])]

print(f"\nFeature columns ({len(feature_cols)}):")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i:>2}. {col}")

# ── Drop features that are entirely NaN (weather cols may be empty) ──────
non_null_counts = df[feature_cols].notna().sum()
valid_features = [c for c in feature_cols if non_null_counts[c] > 0]
dropped = set(feature_cols) - set(valid_features)
if dropped:
    print(f"\n[!] Dropped {len(dropped)} all-NaN feature(s): {sorted(dropped)}")
feature_cols = valid_features

# ── Fill remaining NaN with column median (robust to outliers) ──────────
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

# ── Split ────────────────────────────────────────────────────────────────
train_mask = df["split"].isin(["train", "val"])
test_mask  = df["split"] == "test"

X_train_full = df.loc[train_mask, feature_cols]
y_train_full = df.loc[train_mask, "yield_kg_ha"]
X_test_full  = df.loc[test_mask, feature_cols]
y_test_full  = df.loc[test_mask, "yield_kg_ha"]

print(f"\nTrain+Val rows: {len(X_train_full)}   |   Test rows: {len(X_test_full)}")

# =====================================================================
#  STEP 2 — Train Separate Models per Crop
# =====================================================================
header("Step 2", "Train Separate Models per Crop")

crops = df["crop"].unique()
models = {}   # crop_name → fitted model

for crop in crops:
    crop_train = df[(df["crop"] == crop) & train_mask]
    n_train = len(crop_train)

    if n_train < 5:
        print(f"[!] {crop}: only {n_train} training row(s) -- "
              f"{'skipping model training' if n_train < 2 else 'training with warning (very small dataset)'}")
        if n_train < 2:
            continue
        # Still train if we have at least 2 rows, but warn

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
    )
    X_tr = crop_train[feature_cols]
    y_tr = crop_train["yield_kg_ha"]
    rf.fit(X_tr, y_tr)
    models[crop] = rf
    print(f"[OK] {crop:12s}  ->  trained on {n_train} rows")

if not models:
    print("\n[X] No models could be trained -- exiting.")
    sys.exit(1)

# =====================================================================
#  STEP 3 — Evaluate on Test Set
# =====================================================================
header("Step 3", "Evaluate on Test Set")

eval_rows = []

for crop, model in models.items():
    crop_test = df[(df["crop"] == crop) & test_mask]
    crop_all  = df[df["crop"] == crop]

    if len(crop_test) >= 2:
        # Normal test-set evaluation
        y_true = crop_test["yield_kg_ha"].values
        y_pred = model.predict(crop_test[feature_cols])
        mae  = mean_absolute_error(y_true, y_pred)
        mape = safe_mape(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
        method = "test set"
    else:
        # Fallback: Leave-One-Out CV on full dataset
        print(f"  [i] {crop}: test set has <2 rows -> using Leave-One-Out CV on "
              f"full dataset ({len(crop_all)} rows)")
        loo = LeaveOneOut()
        X_all = crop_all[feature_cols].values
        y_all = crop_all["yield_kg_ha"].values
        if len(crop_all) < 2:
            print(f"  [!] {crop}: not enough data even for LOO-CV, skipping evaluation")
            continue
        y_pred = cross_val_predict(
            RandomForestRegressor(
                n_estimators=200, max_depth=4,
                min_samples_leaf=2, random_state=RANDOM_STATE,
            ),
            X_all, y_all, cv=loo,
        )
        y_true = y_all
        mae  = mean_absolute_error(y_true, y_pred)
        mape = safe_mape(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)
        method = "LOO-CV"

    eval_rows.append({"Crop": crop, "Method": method,
                       "MAE (kg/ha)": f"{mae:.1f}",
                       "MAPE (%)": f"{mape:.2f}",
                       "RMSE (kg/ha)": f"{rmse:.1f}",
                       "R²": f"{r2:.4f}" if not np.isnan(r2) else "N/A"})

eval_df = pd.DataFrame(eval_rows)
print(eval_df.to_string(index=False))

# =====================================================================
#  STEP 4 — Feature Importance
# =====================================================================
header("Step 4", "Feature Importance")

fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 6),
                         squeeze=False)

for idx, (crop, model) in enumerate(models.items()):
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    top_n = min(10, len(feat_imp))
    top = feat_imp.head(top_n)

    print(f"Top {top_n} features for {crop}:")
    for rank, (feat, score) in enumerate(top.items(), 1):
        print(f"  {rank:>2}. {feat:<30s}  {score:.4f}")
    print()

    ax = axes[0, idx]
    top.sort_values().plot.barh(ax=ax, color="#4C72B0", edgecolor="white")
    ax.set_title(f"{crop} — Feature Importance", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.invert_yaxis()

plt.tight_layout()
plt.savefig("rf_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved ->  rf_feature_importance.png")

# =====================================================================
#  STEP 5 — Yield Prediction for All Horizons
# =====================================================================
header("Step 5", "Yield Prediction for All Horizons")

pred_rows = []

for crop, model in models.items():
    crop_df = df[df["crop"] == crop]
    for horizon in [4, 8, 12]:
        horizon_rows = crop_df[crop_df["horizon_weeks"] == horizon]
        if horizon_rows.empty:
            continue
        X_h = horizon_rows[feature_cols]
        y_pred = model.predict(X_h)
        y_actual = horizon_rows["yield_kg_ha"].values
        for i in range(len(horizon_rows)):
            pred_rows.append({
                "crop": crop,
                "district": horizon_rows.iloc[i]["district"],
                "year": int(horizon_rows.iloc[i]["year"]),
                "horizon_weeks": horizon,
                "predicted_yield_kg_ha": round(y_pred[i], 1),
                "actual_yield_kg_ha": y_actual[i],
                "error_kg_ha": round(y_pred[i] - y_actual[i], 1),
            })

pred_df = pd.DataFrame(pred_rows)

# Print a concise table (last entry per crop-horizon for brevity, or all)
print(pred_df.to_string(index=False))
pred_df.to_csv("rf_yield_predictions.csv", index=False)
print(f"\nSaved ->  rf_yield_predictions.csv  ({len(pred_df)} rows)")

# =====================================================================
#  STEP 6 — Cross-Region Prediction (Turmeric model on Tapioca)
# =====================================================================
header("Step 6", "Cross-Region Prediction (Turmeric model → Tapioca)")

if "Turmeric" in models and "Tapioca" in df["crop"].values:
    turmeric_model = models["Turmeric"]
    tapioca_rows = df[df["crop"] == "Tapioca"]

    # Use test rows if available, otherwise all available rows
    tapioca_test = tapioca_rows[tapioca_rows["split"] == "test"]
    if len(tapioca_test) < 1:
        print("  [i] No Tapioca test rows -- using all Tapioca rows for cross-region test")
        tapioca_test = tapioca_rows

    if len(tapioca_test) > 0:
        X_cross = tapioca_test[feature_cols]
        y_true_cross = tapioca_test["yield_kg_ha"].values
        y_pred_cross = turmeric_model.predict(X_cross)

        cross_mae  = mean_absolute_error(y_true_cross, y_pred_cross)
        cross_mape = safe_mape(y_true_cross, y_pred_cross)

        print(f"  Turmeric model applied to {len(tapioca_test)} Tapioca row(s):")
        print(f"    MAE  : {cross_mae:,.1f} kg/ha")
        print(f"    MAPE : {cross_mape:.2f} %")
        print()
        print(textwrap.fill(
            "NOTE -- This cross-region / cross-crop generalisation test applies "
            "a model trained on Turmeric (Erode district, yield ~7000-7700 kg/ha) "
            "to Tapioca (Salem district, yield ~26000 kg/ha). The two crops have "
            "very different yield scales and growth characteristics, so high MAE/MAPE "
            "is expected. This test measures whether the learned price-signal patterns "
            "transfer across crops and regions; a large error confirms that "
            "crop-specific models are necessary for accurate prediction.",
            width=72, initial_indent="  ", subsequent_indent="  ",
        ))
    else:
        print("  [!] No Tapioca rows available for cross-region test.")
else:
    missing = []
    if "Turmeric" not in models:
        missing.append("Turmeric model")
    if "Tapioca" not in df["crop"].values:
        missing.append("Tapioca data")
    print(f"  [!] Cannot run cross-region test -- missing: {', '.join(missing)}")

print(f"\n{'='*70}")
print("  Done -- all steps completed successfully.")
print(f"{'='*70}\n")
