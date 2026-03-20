"""
MODULE 1 — data_pipeline.py
============================
Crop Yield Research: A Robust Spatio-Temporal Multi-Modal Framework
for Early and Generalizable Crop Yield Prediction (Tamil Nadu Agriculture)

Purpose:
    Loads raw AGMARKNET mandi price data for Turmeric (Erode) and
    Tapioca (Salem), cleans and standardises the data, adds a derived
    price_spread feature, prints summary statistics, plots modal price
    trends, and saves cleaned outputs for downstream modules.

Author : [Your Name]
Date   : 2026-02-27
"""

# ─────────────────────────────────────────────
# 0.  IMPORTS
# ─────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1.  COLUMN RENAMING MAP  (raw → snake_case)
# ─────────────────────────────────────────────
COLUMN_MAP = {
    "State"          : "state",
    "District"       : "district",
    "Market"         : "market",
    "Commodity"      : "commodity",
    "Variety"        : "variety",
    "Grade"          : "grade",
    "Arrival_Date"   : "arrival_date",
    "Min_Price"      : "min_price",
    "Max_Price"      : "max_price",
    "Modal_Price"    : "modal_price",
    "Commodity_Code" : "commodity_code",
}


# ─────────────────────────────────────────────
# 2.  HELPER — LOAD & CLEAN ONE CSV
# ─────────────────────────────────────────────
def load_and_clean(filepath: str, label: str) -> pd.DataFrame:
    """
    Reads a raw AGMARKNET CSV, applies cleaning steps, and returns a
    tidy DataFrame ready for feature engineering.

    Steps applied:
        1. Read CSV and rename columns to snake_case.
        2. Parse arrival_date as datetime; drop rows with invalid dates.
        3. Drop rows with any null values.
        4. Drop exact duplicate rows.
        5. Sort by arrival_date ascending.
        6. Reset the integer index.
        7. Derive price_spread = modal_price − min_price.

    Parameters
    ----------
    filepath : str   Path to the raw CSV file.
    label    : str   Short crop label used in log messages (e.g. 'Turmeric').

    Returns
    -------
    pd.DataFrame  Clean, sorted DataFrame.
    """
    print(f"\n{'='*60}")
    print(f"  Loading  :  {label}")
    print(f"{'='*60}")

    # ── 2a. Read raw file ──────────────────────────────────────────
    df = pd.read_csv(filepath)

    # ── 2b. Rename columns ────────────────────────────────────────
    df.rename(columns=COLUMN_MAP, inplace=True)

    # ── 2c. Parse dates ───────────────────────────────────────────
    df["arrival_date"] = pd.to_datetime(
        df["arrival_date"], dayfirst=True, errors="coerce"
    )
    before = len(df)
    df.dropna(subset=["arrival_date"], inplace=True)
    print(f"  Rows with un-parseable dates removed : {before - len(df)}")

    # ── 2d. Drop nulls ────────────────────────────────────────────
    before = len(df)
    df.dropna(inplace=True)
    print(f"  Rows with null values removed        : {before - len(df)}")

    # ── 2e. Drop duplicates ───────────────────────────────────────
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"  Duplicate rows removed               : {before - len(df)}")

    # ── 2f. Sort & re-index ───────────────────────────────────────
    df.sort_values("arrival_date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── 2g. Derive price_spread ───────────────────────────────────
    df["price_spread"] = df["modal_price"] - df["min_price"]

    return df


# ─────────────────────────────────────────────
# 3.  HELPER — PRINT SUMMARY STATISTICS
# ─────────────────────────────────────────────
def print_summary(df: pd.DataFrame, label: str) -> None:
    """
    Prints shape, date range, and descriptive statistics for a cleaned
    DataFrame.

    Parameters
    ----------
    df    : pd.DataFrame  The cleaned crop price DataFrame.
    label : str           Crop label for display headers.
    """
    print(f"\n  ── {label} Summary ──────────────────────────────────")
    print(f"  Shape      : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Date range : {df['arrival_date'].min().date()}  →  "
          f"{df['arrival_date'].max().date()}")

    numeric_cols = ["min_price", "max_price", "modal_price", "price_spread"]
    print(f"\n  Descriptive Statistics (prices in ₹/quintal):")
    print(df[numeric_cols].describe().round(2).to_string())


# ─────────────────────────────────────────────
# 4.  HELPER — PLOT BOTH CROPS ON ONE FIGURE
# ─────────────────────────────────────────────
def plot_modal_prices(df_turmeric: pd.DataFrame,
                      df_tapioca: pd.DataFrame) -> None:
    """
    Plots the modal price time-series for Turmeric (Erode) and
    Tapioca (Salem) on a single figure using dual y-axes, because
    the two crops trade at very different price levels.

    The figure is saved as 'modal_price_comparison.png' and displayed.

    Parameters
    ----------
    df_turmeric : pd.DataFrame  Cleaned Turmeric DataFrame.
    df_tapioca  : pd.DataFrame  Cleaned Tapioca DataFrame.
    """
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # ── Turmeric on left axis ─────────────────────────────────────
    color_turmeric = "#E8A838"   # golden-amber for turmeric
    ax1.plot(
        df_turmeric["arrival_date"],
        df_turmeric["modal_price"],
        color=color_turmeric,
        linewidth=0.9,
        alpha=0.85,
        label="Turmeric – Erode (left axis)",
    )
    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel("Modal Price — Turmeric (₹/quintal)", color=color_turmeric,
                   fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_turmeric)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45, ha="right")

    # ── Tapioca on right axis ─────────────────────────────────────
    ax2 = ax1.twinx()
    color_tapioca = "#3A7CA5"    # steel-blue for tapioca
    ax2.plot(
        df_tapioca["arrival_date"],
        df_tapioca["modal_price"],
        color=color_tapioca,
        linewidth=0.9,
        alpha=0.85,
        label="Tapioca – Salem (right axis)",
    )
    ax2.set_ylabel("Modal Price — Tapioca (₹/quintal)", color=color_tapioca,
                   fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_tapioca)

    # ── Legend (combined from both axes) ──────────────────────────
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper left", fontsize=10)

    plt.title(
        "Modal Price Over Time — Turmeric (Erode) vs Tapioca (Salem)\n"
        "Source: AGMARKNET, data.gov.in  |  2014–2026",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    plt.savefig("modal_price_comparison.png", dpi=150)
    plt.show()
    print("\n  Plot saved →  modal_price_comparison.png")


# ─────────────────────────────────────────────
# 5.  MAIN PIPELINE
# ─────────────────────────────────────────────
def main():
    # ── File paths ────────────────────────────────────────────────
    RAW_TURMERIC = "turmeric_erode_prices.csv"
    RAW_TAPIOCA  = "tapioca_salem_prices.csv"
    OUT_TURMERIC = "turmeric_clean.csv"
    OUT_TAPIOCA  = "tapioca_clean.csv"

    # ── Load & clean ─────────────────────────────────────────────
    df_turmeric = load_and_clean(RAW_TURMERIC, "Turmeric — Erode")
    df_tapioca  = load_and_clean(RAW_TAPIOCA,  "Tapioca  — Salem")

    # ── Print summaries ───────────────────────────────────────────
    print_summary(df_turmeric, "Turmeric — Erode")
    print_summary(df_tapioca,  "Tapioca  — Salem")

    # ── Plot ──────────────────────────────────────────────────────
    plot_modal_prices(df_turmeric, df_tapioca)

    # ── Save cleaned outputs ──────────────────────────────────────
    df_turmeric.to_csv(OUT_TURMERIC, index=False)
    df_tapioca.to_csv(OUT_TAPIOCA,  index=False)
    print(f"\n  Saved →  {OUT_TURMERIC}")
    print(f"  Saved →  {OUT_TAPIOCA}")
    print("\n  ✓  data_pipeline.py complete.\n")


if __name__ == "__main__":
    main()
