"""
soil_download.py
=================
Crop Yield Research: A Robust Spatio-Temporal Multi-Modal Framework
for Early and Generalizable Crop Yield Prediction (Tamil Nadu Agriculture)

Purpose:
    Downloads soil property data for Erode and Salem districts from
    the ISRIC SoilGrids REST API (free, no API key required).
    Retrieves 8 properties across 3 depth layers, applies unit
    conversion factors, computes weighted 0–30 cm averages, and
    saves a single CSV with one row per location.

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
import time
import requests
import pandas as pd

# ─────────────────────────────────────────────
# 2.  CONFIGURATION
# ─────────────────────────────────────────────
API_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"

# Soil properties to download
PROPERTIES = ["phh2o", "soc", "clay", "sand", "silt", "nitrogen", "bdod", "cec"]

# Depth layers and their thickness in cm (used for weighted averaging)
DEPTHS = [
    {"label": "0-5cm",   "top": 0,  "bottom": 5,  "thickness": 5},
    {"label": "5-15cm",  "top": 5,  "bottom": 15, "thickness": 10},
    {"label": "15-30cm", "top": 15, "bottom": 30, "thickness": 15},
]
TOTAL_THICKNESS = sum(d["thickness"] for d in DEPTHS)   # 30 cm

# SoilGrids stores integer-scaled values; divide by these factors
# to get standard units (pH, g/kg, g/cm³, cmol(c)/kg, etc.)
CONVERSION_FACTORS = {
    "phh2o"   : 10,     # pH × 10      → pH
    "soc"     : 10,     # dg/kg        → g/kg
    "clay"    : 10,     # g/kg × 10    → g/kg  (‰ → %)
    "sand"    : 10,
    "silt"    : 10,
    "nitrogen": 10,     # cg/kg        → g/kg
    "bdod"    : 100,    # cg/cm³       → g/cm³
    "cec"     : 10,     # mmol(c)/kg   → cmol(c)/kg
}

# Location definitions
LOCATIONS = [
    {"label": "Erode",  "crop": "Turmeric", "lat": 11.341, "lon": 77.717},
    {"label": "Salem",  "crop": "Tapioca",  "lat": 11.658, "lon": 78.146},
]


# ─────────────────────────────────────────────
# 3.  DOWNLOAD FUNCTION
# ─────────────────────────────────────────────
def download_soil(location: dict) -> dict:
    """
    Queries the SoilGrids API for a single location and returns a
    flat dictionary containing:
        - location metadata (label, crop, lat, lon)
        - weighted 0–30 cm averages for each property
        - individual depth-layer values for each property

    Weighted average formula:
        avg = Σ (value_i × thickness_i) / total_thickness

    Parameters
    ----------
    location : dict   Keys: label, crop, lat, lon.

    Returns
    -------
    dict  One-row record with all columns.
    """
    label = location["label"]
    print(f"\n{'='*60}")
    print(f"  Downloading soil data for  :  {label} ({location['crop']})")
    print(f"  Coordinates                :  {location['lat']}°N, "
          f"{location['lon']}°E")
    print(f"{'='*60}")

    # ── 3a. API request (with retry for transient errors) ──────────
    params = {
        "lon"       : location["lon"],
        "lat"       : location["lat"],
        "property"  : PROPERTIES,
        "depth"     : [d["label"] for d in DEPTHS],
        "value"     : "mean",
    }

    MAX_RETRIES = 3
    for attempt in range(1, MAX_RETRIES + 1):
        response = requests.get(API_URL, params=params, timeout=30)
        if response.status_code in (502, 503, 429):
            wait = 5 * attempt
            print(f"  ⚠  Server returned {response.status_code}, "
                  f"retrying in {wait}s (attempt {attempt}/{MAX_RETRIES})…")
            time.sleep(wait)
            continue
        response.raise_for_status()
        break
    else:
        raise RuntimeError(
            f"SoilGrids API unavailable after {MAX_RETRIES} attempts "
            f"(HTTP {response.status_code}). Try again later."
        )

    data = response.json()
    print("  API response received  ✓")

    # ── 3b. Parse response into flat record ───────────────────────
    record = {
        "location": label,
        "crop"    : location["crop"],
        "lat"     : location["lat"],
        "lon"     : location["lon"],
    }

    # Temporary storage for weighted average computation
    depth_values = {}   # {property_name: {depth_label: converted_value}}

    for layer in data["properties"]["layers"]:
        prop_name  = layer["name"]
        conv       = CONVERSION_FACTORS[prop_name]

        depth_values[prop_name] = {}

        for depth_entry in layer["depths"]:
            depth_label = depth_entry["label"]
            raw_value   = depth_entry["values"].get("mean")

            if raw_value is not None:
                converted = raw_value / conv
            else:
                converted = None

            depth_values[prop_name][depth_label] = converted

            # Store individual depth column
            col_name = f"{prop_name}_{depth_label}"
            record[col_name] = converted

        # ── 3c. Weighted 0–30 cm average ─────────────────────────
        weighted_sum   = 0.0
        weight_total   = 0
        for d in DEPTHS:
            val = depth_values[prop_name].get(d["label"])
            if val is not None:
                weighted_sum += val * d["thickness"]
                weight_total += d["thickness"]

        if weight_total > 0:
            avg = weighted_sum / weight_total
        else:
            avg = None

        record[f"{prop_name}_avg_0_30cm"] = avg

    # ── 3d. Print per-property summary ────────────────────────────
    print(f"\n  ── {label} Soil Properties (mean, converted units) ──")
    for prop in PROPERTIES:
        avg_val = record.get(f"{prop}_avg_0_30cm")
        parts = []
        for d in DEPTHS:
            dlabel = d["label"]
            col_key = f"{prop}_{dlabel}"
            val = record.get(col_key)
            if val is None:
                parts.append(f"{dlabel}: N/A")
            else:
                parts.append(f"{dlabel}: {val:.2f}")
        depth_str = "  |  ".join(parts)
        avg_str = f"{avg_val:.2f}" if avg_val is not None else "N/A"
        print(f"    {prop:<12s}  avg={avg_str:<8s}  ({depth_str})")

    return record


# ─────────────────────────────────────────────
# 4.  MAIN
# ─────────────────────────────────────────────
def main():
    OUTPUT_FILE = "soil_properties.csv"

    records = []
    for i, loc in enumerate(LOCATIONS):
        record = download_soil(loc)
        records.append(record)

        # Rate-limit courtesy: sleep between calls
        if i < len(LOCATIONS) - 1:
            print("\n  ⏳  Sleeping 2 s to respect rate limits …")
            time.sleep(2)

    # ── Build DataFrame with correct column order ─────────────────
    df = pd.DataFrame(records)

    # Desired column order: metadata → avg columns → depth columns
    meta_cols  = ["location", "crop", "lat", "lon"]
    avg_cols   = [f"{p}_avg_0_30cm" for p in PROPERTIES]
    depth_cols = [
        f"{p}_{d['label']}" for p in PROPERTIES for d in DEPTHS
    ]
    ordered = meta_cols + avg_cols + depth_cols
    # Keep only columns that actually exist (safety check)
    ordered = [c for c in ordered if c in df.columns]
    df = df[ordered]

    # ── Print summary table ───────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("  SUMMARY — Weighted 0–30 cm Averages")
    print(f"{'='*60}")
    summary_cols = meta_cols[:2] + avg_cols
    summary = df[summary_cols].copy()
    # Round numeric columns for display
    for c in avg_cols:
        summary[c] = summary[c].round(3)
    print(summary.to_string(index=False))

    # ── Save CSV ──────────────────────────────────────────────────
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  ✅  Saved →  {OUTPUT_FILE}  ({len(df)} rows, "
          f"{len(df.columns)} columns)\n")
    print("=" * 60)
    print("  ✓  soil_download.py complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
