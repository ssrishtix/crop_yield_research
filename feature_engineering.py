"""
MODULE 2 — feature_engineering.py
====================================
Crop Yield Research: A Robust Spatio-Temporal Multi-Modal Framework
for Early and Generalizable Crop Yield Prediction (Tamil Nadu Agriculture)

Purpose:
    Loads the cleaned price CSVs produced by data_pipeline.py and
    engineers predictive features including rolling averages, lag
    variables, seasonal indicators, and price-volatility measures.
    Outputs feature-enriched CSVs for model training.

Author : [Your Name]
Date   : 2026-02-27
"""

# ─────────────────────────────────────────────
# 0.  IMPORTS
# ─────────────────────────────────────────────
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1.  SEASON CLASSIFICATION
# ─────────────────────────────────────────────
def assign_season(month: int) -> str:
    """
    Returns the Indian agricultural season name for a given calendar month.

    Season definitions (Tamil Nadu context):
        Kharif : June – October   (south-west monsoon sowing season)
        Rabi   : November – February (winter / retreating-monsoon season)
        Summer : March – May      (dry / summer season)

    Parameters
    ----------
    month : int   Calendar month number (1–12).

    Returns
    -------
    str  One of 'Kharif', 'Rabi', or 'Summer'.
    """
    if month in (6, 7, 8, 9, 10):
        return "Kharif"
    elif month in (11, 12, 1, 2):
        return "Rabi"
    else:                           # 3, 4, 5
        return "Summer"


# ─────────────────────────────────────────────
# 2.  CORE FEATURE ENGINEERING FUNCTION
# ─────────────────────────────────────────────
def engineer_features(filepath: str, label: str) -> pd.DataFrame:
    """
    Reads a cleaned price CSV and adds the following feature columns:

        rolling_avg_7d   : 7-day rolling mean of modal_price
        rolling_avg_30d  : 30-day rolling mean of modal_price
        month            : integer month extracted from arrival_date
        season           : categorical season label
        price_lag_7      : modal_price shifted backward by 7 rows (days)
        price_lag_30     : modal_price shifted backward by 30 rows (days)
        price_volatility : 30-day rolling standard deviation of modal_price

    Note: Rolling and lag features produce NaN for early rows.
    These rows are retained so the dataset remains aligned with the
    original date index; downstream models should handle NaN accordingly.

    Parameters
    ----------
    filepath : str   Path to the cleaned CSV (output of data_pipeline.py).
    label    : str   Crop label for log messages.

    Returns
    -------
    pd.DataFrame  Feature-enriched DataFrame.
    """
    print(f"\n{'='*60}")
    print(f"  Engineering features for  :  {label}")
    print(f"{'='*60}")

    # ── 2a. Load cleaned data ──────────────────────────────────────
    df = pd.read_csv(filepath, parse_dates=["arrival_date"])
    df.sort_values("arrival_date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  Loaded {len(df):,} rows from  {filepath}")

    # ── 2b. Rolling averages ──────────────────────────────────────
    df["rolling_avg_7d"]  = (
        df["modal_price"].rolling(window=7, min_periods=1).mean()
    )
    df["rolling_avg_30d"] = (
        df["modal_price"].rolling(window=30, min_periods=1).mean()
    )
    print("  ✓  rolling_avg_7d, rolling_avg_30d  computed")

    # ── 2c. Temporal features ─────────────────────────────────────
    df["month"]  = df["arrival_date"].dt.month
    df["season"] = df["month"].apply(assign_season)
    print("  ✓  month, season  computed")

    # ── 2d. Lag features ──────────────────────────────────────────
    df["price_lag_7"]  = df["modal_price"].shift(7)
    df["price_lag_30"] = df["modal_price"].shift(30)
    print("  ✓  price_lag_7, price_lag_30  computed")

    # ── 2e. Price volatility ──────────────────────────────────────
    df["price_volatility"] = (
        df["modal_price"].rolling(window=30, min_periods=2).std()
    )
    print("  ✓  price_volatility (30-day rolling std)  computed")

    return df


# ─────────────────────────────────────────────
# 3.  MAIN
# ─────────────────────────────────────────────
def main():
    # ── File paths ────────────────────────────────────────────────
    IN_TURMERIC  = "turmeric_clean.csv"
    IN_TAPIOCA   = "tapioca_clean.csv"
    OUT_TURMERIC = "turmeric_features.csv"
    OUT_TAPIOCA  = "tapioca_features.csv"

    # ── Engineer features for each crop ───────────────────────────
    df_turmeric = engineer_features(IN_TURMERIC, "Turmeric — Erode")
    df_tapioca  = engineer_features(IN_TAPIOCA,  "Tapioca  — Salem")

    # ── Preview first 5 rows ──────────────────────────────────────
    PREVIEW_COLS = [
        "arrival_date", "modal_price",
        "rolling_avg_7d", "rolling_avg_30d",
        "month", "season",
        "price_lag_7", "price_lag_30",
        "price_volatility",
    ]

    print("\n\n  ── Turmeric : First 5 Rows ──────────────────────────")
    print(df_turmeric[PREVIEW_COLS].head(5).to_string(index=False))

    print("\n  ── Tapioca  : First 5 Rows ──────────────────────────")
    print(df_tapioca[PREVIEW_COLS].head(5).to_string(index=False))

    # ── Save feature-engineered CSVs ─────────────────────────────
    df_turmeric.to_csv(OUT_TURMERIC, index=False)
    df_tapioca.to_csv(OUT_TAPIOCA,   index=False)
    print(f"\n  Saved →  {OUT_TURMERIC}")
    print(f"  Saved →  {OUT_TAPIOCA}")
    print("\n  ✓  feature_engineering.py complete.\n")


if __name__ == "__main__":
    main()
