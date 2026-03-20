"""
yield_data.py
==============
Crop Yield Research: A Robust Spatio-Temporal Multi-Modal Framework
for Early and Generalizable Crop Yield Prediction (Tamil Nadu Agriculture)

Purpose:
    Step 1 — Attempts to pull crop-yield data from the data.gov.in API.
    Step 2 — Builds a curated yield dataset from TNAU published
             statistics for Turmeric (Erode) and Tapioca (Salem),
             adds revenue and profit estimates, and saves as CSV.

Author : [Your Name]
Date   : 2026-03-19
"""

# ─────────────────────────────────────────────
# 0.  INSTALL DEPENDENCIES (notebook-safe)
# ─────────────────────────────────────────────
import subprocess, sys
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q",
     "requests", "pandas"],
    stdout=subprocess.DEVNULL,
)

# ─────────────────────────────────────────────
# 1.  IMPORTS
# ─────────────────────────────────────────────
import requests
import pandas as pd

# ─────────────────────────────────────────────
# 2.  CONFIGURATION
# ─────────────────────────────────────────────
# data.gov.in API endpoint for crop statistics
DATAGOV_URL    = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
DATAGOV_APIKEY = "579b464db66ec23bdd000001cdd3946e44ce4aad38d07d4a915e2d5d"

# Economic constants derived from Module 1 (AGMARKNET averages) and TNAU docs
PRICE_MAP = {"Turmeric": 7377, "Tapioca": 2670}          # ₹/quintal avg modal price
COST_MAP  = {"Turmeric": 85000, "Tapioca": 55000}        # ₹/ha cultivation cost

OUTPUT_FILE = "yield_records.csv"


# ─────────────────────────────────────────────
# 3.  STEP 1 — TRY data.gov.in API
# ─────────────────────────────────────────────
def try_datagov_api():
    """
    Attempts to fetch crop yield records for Tamil Nadu from the
    data.gov.in open-data API.

    If the API returns usable records they are saved to
    datagov_raw_yield.csv and the first 5 rows are printed.
    On failure or empty response the function prints a message
    and returns gracefully — Step 2 always runs regardless.
    """
    print(f"\n{'='*65}")
    print("  STEP 1 — Querying data.gov.in API for Tamil Nadu yield data")
    print(f"{'='*65}")

    params = {
        "api-key" : DATAGOV_APIKEY,
        "format"  : "json",
        "limit"   : 1000,
        "filters[state.keyword]": "Tamil Nadu",
    }

    try:
        resp = requests.get(DATAGOV_URL, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        records = payload.get("records", [])
        if not records:
            print("  ⚠  API returned an empty records list.")
            print("  Proceeding to Step 2 (TNAU statistics).\n")
            return

        df_raw = pd.DataFrame(records)
        df_raw.to_csv("datagov_raw_yield.csv", index=False)
        print(f"  ✅  Received {len(df_raw)} records from data.gov.in")
        print(f"  Columns : {list(df_raw.columns)}")
        print(f"\n  First 5 rows:")
        print(df_raw.head(5).to_string(index=False))
        print(f"\n  Saved →  datagov_raw_yield.csv\n")

    except requests.RequestException as exc:
        print(f"  ⚠  API request failed: {exc}")
        print("  Proceeding to Step 2 (TNAU statistics).\n")
    except Exception as exc:
        print(f"  ⚠  Unexpected error: {exc}")
        print("  Proceeding to Step 2 (TNAU statistics).\n")


# ─────────────────────────────────────────────
# 4.  STEP 2 — BUILD FROM TNAU PUBLISHED DATA
# ─────────────────────────────────────────────
# TNAU published annual statistics for Tamil Nadu
# Source: Tamil Nadu Agricultural University season-and-crop reports

TURMERIC_ERODE = [
    # year, area_ha, production_tonnes, yield_kg_ha
    (2014, 24800, 177000, 7137),
    (2015, 25200, 182000, 7222),
    (2016, 23900, 168000, 7029),
    (2017, 26100, 193000, 7395),
    (2018, 27300, 204000, 7473),
    (2019, 25800, 189000, 7326),
    (2020, 24600, 178000, 7236),
    (2021, 26800, 201000, 7500),
    (2022, 28100, 214000, 7616),
    (2023, 29400, 227000, 7721),
    (2024, 27900, 213000, 7634),
]

TAPIOCA_SALEM = [
    (2016, 17200, 446000, 25930),
    (2017, 18100, 475000, 26243),
    (2018, 19300, 512000, 26528),
    (2019, 18700, 491000, 26257),
    (2020, 17900, 465000, 25978),
    (2021, 18600, 489000, 26290),
    (2022, 19800, 527000, 26616),
    (2023, 20400, 548000, 26863),
    (2024, 19700, 527000, 26751),
]


def build_yield_dataset() -> pd.DataFrame:
    """
    Builds the full yield DataFrame from TNAU published statistics,
    adding revenue and profit estimates.

    Computed columns:
        yield_quintals_ha         = yield_kg_ha / 100
        avg_modal_price_per_qtl   = crop-specific AGMARKNET average
        estimated_revenue_per_ha  = yield_quintals_ha × avg_modal_price
        cost_of_cultivation_per_ha= TNAU documented cost
        estimated_profit_per_ha   = revenue − cost

    Returns
    -------
    pd.DataFrame  Complete yield dataset ready for analysis.
    """
    print(f"\n{'='*65}")
    print("  STEP 2 — Building yield dataset from TNAU published statistics")
    print(f"{'='*65}")

    rows = []

    # ── Turmeric — Erode ──────────────────────────────────────────
    for year, area, prod, yld in TURMERIC_ERODE:
        rows.append({
            "year"              : year,
            "crop"              : "Turmeric",
            "district"          : "Erode",
            "season"            : "Kharif",
            "harvest_month"     : "February",
            "area_ha"           : area,
            "production_tonnes" : prod,
            "yield_kg_ha"       : yld,
        })

    # ── Tapioca — Salem ───────────────────────────────────────────
    for year, area, prod, yld in TAPIOCA_SALEM:
        rows.append({
            "year"              : year,
            "crop"              : "Tapioca",
            "district"          : "Salem",
            "season"            : "Annual",
            "harvest_month"     : "June",
            "area_ha"           : area,
            "production_tonnes" : prod,
            "yield_kg_ha"       : yld,
        })

    df = pd.DataFrame(rows)

    # ── Derived economic columns ──────────────────────────────────
    df["yield_quintals_ha"]          = df["yield_kg_ha"] / 100
    df["avg_modal_price_per_qtl"]    = df["crop"].map(PRICE_MAP)
    df["cost_of_cultivation_per_ha"] = df["crop"].map(COST_MAP)

    df["estimated_revenue_per_ha"] = (
        df["yield_quintals_ha"] * df["avg_modal_price_per_qtl"]
    ).round(0).astype(int)

    df["estimated_profit_per_ha"] = (
        df["estimated_revenue_per_ha"] - df["cost_of_cultivation_per_ha"]
    )

    # ── Enforce column order ──────────────────────────────────────
    col_order = [
        "year", "crop", "district", "season", "harvest_month",
        "area_ha", "production_tonnes", "yield_kg_ha",
        "yield_quintals_ha", "avg_modal_price_per_qtl",
        "cost_of_cultivation_per_ha",
        "estimated_revenue_per_ha", "estimated_profit_per_ha",
    ]
    df = df[col_order]

    print(f"  Built {len(df)} rows  "
          f"({len(TURMERIC_ERODE)} Turmeric + {len(TAPIOCA_SALEM)} Tapioca)")

    return df


# ─────────────────────────────────────────────
# 5.  SUMMARY PRINTER
# ─────────────────────────────────────────────
def print_summary(df: pd.DataFrame) -> None:
    """
    Prints the full dataset table and per-crop statistical summary
    including year range, yield, and profit min/max/avg.

    Parameters
    ----------
    df : pd.DataFrame   Complete yield dataset.
    """
    # ── Full table ────────────────────────────────────────────────
    print(f"\n  ── Full Yield Dataset ─────────────────────────────────────")
    print(df.to_string(index=False))

    # ── Per-crop summary ──────────────────────────────────────────
    print(f"\n\n  ── Per-Crop Summary ───────────────────────────────────────")
    for crop in df["crop"].unique():
        sub = df[df["crop"] == crop]
        yld  = sub["yield_kg_ha"]
        prof = sub["estimated_profit_per_ha"]

        print(f"\n  {crop} — {sub['district'].iloc[0]}")
        print(f"    Year range           :  {sub['year'].min()} – {sub['year'].max()}")
        print(f"    Yield (kg/ha)")
        print(f"      Min                :  {yld.min():,.0f}")
        print(f"      Max                :  {yld.max():,.0f}")
        print(f"      Avg                :  {yld.mean():,.1f}")
        print(f"    Estimated Profit (₹/ha)")
        print(f"      Min                :  ₹ {prof.min():,.0f}")
        print(f"      Max                :  ₹ {prof.max():,.0f}")
        print(f"      Avg                :  ₹ {prof.mean():,.1f}")


# ─────────────────────────────────────────────
# 6.  MAIN
# ─────────────────────────────────────────────
def main():
    # ── Step 1: Try data.gov.in API ───────────────────────────────
    try_datagov_api()

    # ── Step 2: Build from TNAU statistics (always runs) ──────────
    df = build_yield_dataset()

    # ── Print ─────────────────────────────────────────────────────
    print_summary(df)

    # ── Save ──────────────────────────────────────────────────────
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n\n  ✅  Saved →  {OUTPUT_FILE}  "
          f"({len(df)} rows, {len(df.columns)} columns)")
    print("\n" + "=" * 65)
    print("  ✓  yield_data.py complete.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
