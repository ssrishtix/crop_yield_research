import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict, Any


# Install minimal dependencies (quietly).
# This is safe to keep even if they're already installed.
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pandas", "numpy"])
except Exception:
    # If pip install fails (e.g., offline), continue assuming deps exist.
    pass


def _warn(msg: str) -> None:
    print(f"WARNING: {msg}", file=sys.stderr)


def _validate_required_columns(df: pd.DataFrame, required_cols: List[str], source_name: str) -> bool:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        _warn(f"{source_name} is missing required columns: {missing}. Skipping this source.")
        return False
    return True


def load_csv_if_exists(path: Path, source_name: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        _warn(f"Missing file: {path}. Skipping {source_name}.")
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        _warn(f"Failed to read {source_name} at {path}: {e}. Skipping.")
        return None
    return df


def print_shape_and_range(df: pd.DataFrame, source_name: str, date_range_kind: str) -> None:
    # date_range_kind is only for display: "arrival_date", "date", "year", or "none".
    if date_range_kind == "arrival_date":
        dcol = "arrival_date"
        dates = pd.to_datetime(df[dcol], errors="coerce")
        if dates.notna().any():
            print(
                f"{source_name}: shape={df.shape}, arrival_date range="
                f"{dates.min().date()} to {dates.max().date()}"
            )
        else:
            print(f"{source_name}: shape={df.shape}, arrival_date range=invalid/empty")
    elif date_range_kind == "date":
        dcol = "date"
        dates = pd.to_datetime(df[dcol], errors="coerce")
        if dates.notna().any():
            print(f"{source_name}: shape={df.shape}, date range={dates.min().date()} to {dates.max().date()}")
        else:
            print(f"{source_name}: shape={df.shape}, date range=invalid/empty")
    elif date_range_kind == "year":
        ycol = "year"
        years = pd.to_numeric(df[ycol], errors="coerce").dropna()
        if len(years) > 0:
            print(f"{source_name}: shape={df.shape}, year range={int(years.min())} to {int(years.max())}")
        else:
            print(f"{source_name}: shape={df.shape}, year range=invalid/empty")
    else:
        print(f"{source_name}: shape={df.shape}, date range=N/A")


def align_daily(
    price_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    *,
    crop: str,
    district: str,
    soil_df: Optional[pd.DataFrame],
    soil_columns: Optional[List[str]],
    harvest_month_day: Tuple[int, int],
    print_prefix: str,
) -> Optional[pd.DataFrame]:
    def _to_datetime_no_tz(s: pd.Series) -> pd.Series:
        dt = pd.to_datetime(s, errors="coerce")
        # Ensure tz-naive DatetimeIndex for safe joins.
        try:
            if getattr(dt.dt, "tz", None) is not None:
                dt = dt.dt.tz_localize(None)
        except Exception:
            pass
        return dt

    # Prepare price spine
    price_required = [
        "arrival_date",
        "modal_price",
        "rolling_avg_7d",
        "rolling_avg_30d",
        "month",
        "season",
        "price_lag_7",
        "price_lag_30",
        "price_volatility",
    ]
    if not _validate_required_columns(price_df, price_required, f"{print_prefix} price"):
        return None

    weather_required = [
        "date",
        "temp_max_c",
        "temp_min_c",
        "rainfall_mm",
        "humidity_max_pct",
        "humidity_min_pct",
        "evapotranspiration_mm",
        "windspeed_max_kmh",
        "temp_range_c",
        "humidity_avg_pct",
    ]
    if not _validate_required_columns(weather_df, weather_required, f"{print_prefix} weather"):
        return None

    # Keep only required columns to avoid aggregating unrelated non-numeric strings.
    df_price = price_df[price_required].copy()
    df_price["date"] = _to_datetime_no_tz(df_price["arrival_date"])
    df_price = df_price.dropna(subset=["date"]).sort_values("date")
    if df_price.empty:
        _warn(f"{print_prefix}: price data empty after date parsing. Skipping.")
        return None
    # De-duplicate in case of repeated arrival_date.
    # Coerce numeric columns for safe aggregation; keep season as first value.
    for c in price_required:
        if c in ["arrival_date", "season"]:
            continue
        df_price[c] = pd.to_numeric(df_price[c], errors="coerce")

    agg_dict = {}
    for c in price_required:
        if c in ["arrival_date"]:
            continue
        if c == "season":
            agg_dict[c] = "first"
        else:
            agg_dict[c] = "mean"

    df_price = df_price.groupby("date", as_index=False).agg(agg_dict)

    df_weather = weather_df.copy()
    df_weather["date"] = _to_datetime_no_tz(df_weather["date"])
    df_weather = df_weather.dropna(subset=["date"]).sort_values("date")
    if df_weather.empty:
        _warn(f"{print_prefix}: weather data empty after date parsing. Skipping.")
        return None
    # Keep only weather columns needed downstream + all columns for completeness.
    weather_cols = [
        "temp_max_c",
        "temp_min_c",
        "rainfall_mm",
        "humidity_max_pct",
        "humidity_min_pct",
        "evapotranspiration_mm",
        "windspeed_max_kmh",
        "temp_range_c",
        "humidity_avg_pct",
    ]

    df_price = df_price.set_index("date")
    df_weather = df_weather.set_index("date")

    aligned = df_price.join(df_weather[weather_cols], how="left")

    # Forward-fill weather gaps up to 3 "daily spine" steps.
    aligned[weather_cols] = aligned[weather_cols].ffill(limit=3)

    # Add crop/district identifiers
    aligned["crop"] = crop
    aligned["district"] = district

    # Add soil as static repeated values (if available)
    if soil_df is not None and soil_columns:
        soil_match = soil_df.copy()
        if "location" in soil_match.columns:
            soil_match = soil_match[soil_match["location"].astype(str).str.strip() == district]
        if "crop" in soil_match.columns:
            soil_match = soil_match[soil_match["crop"].astype(str).str.strip() == crop]

        if not soil_match.empty:
            row = soil_match.iloc[0]
            for c in soil_columns:
                aligned[c] = row[c]
        else:
            _warn(f"{print_prefix}: no matching soil row found for crop={crop}, district={district}. Skipping soil columns.")
            # Ensure soil columns are absent (do not crash).

    # Add yield and days_to_harvest
    aligned = aligned.reset_index().rename(columns={"date": "date"})
    aligned["year"] = aligned["date"].dt.year.astype(int)

    # Harvest date = fixed day/month in that year
    h_month, h_day = harvest_month_day
    aligned["harvest_date"] = pd.to_datetime(aligned["year"].astype(str) + f"-{h_month:02d}-{h_day:02d}", errors="coerce")
    aligned["days_to_harvest"] = (aligned["harvest_date"] - aligned["date"]).dt.days.astype("float")
    aligned["days_to_harvest"] = aligned["days_to_harvest"].fillna(0)
    aligned.loc[aligned["days_to_harvest"] < 0, "days_to_harvest"] = 0
    aligned["days_to_harvest"] = aligned["days_to_harvest"].astype(int)

    # Compute yield_kg_ha later (needs yield_records). Placeholder if missing.
    # Caller will merge yield; we keep structure consistent.
    aligned["yield_kg_ha"] = np.nan

    return aligned


def attach_yield(aligned_df: pd.DataFrame, yield_df: pd.DataFrame, crop: str, district: str) -> pd.DataFrame:
    required = ["year", "crop", "district", "yield_kg_ha"]
    if not _validate_required_columns(yield_df, required, "yield_records"):
        aligned_df["yield_kg_ha"] = np.nan
        return aligned_df

    yd = yield_df.copy()
    yd["year"] = pd.to_numeric(yd["year"], errors="coerce").astype("Int64")
    yd["yield_kg_ha"] = pd.to_numeric(yd["yield_kg_ha"], errors="coerce")
    yd["crop"] = yd["crop"].astype(str).str.strip()
    yd["district"] = yd["district"].astype(str).str.strip()

    match = yd[(yd["crop"] == crop) & (yd["district"] == district)]
    if match.empty:
        _warn(f"No yield_records matches for crop={crop}, district={district}. yield_kg_ha will be null.")
        aligned_df["yield_kg_ha"] = np.nan
        return aligned_df

    mapping = match.dropna(subset=["year"]).set_index("year")["yield_kg_ha"].to_dict()
    aligned_df["yield_kg_ha"] = aligned_df["year"].map(mapping).astype(float)
    return aligned_df


def build_feature_matrix(
    aligned_df: pd.DataFrame,
    *,
    crop: str,
    district: str,
    soil_columns: List[str],
    horizons: List[int] = [28, 56, 84],
) -> pd.DataFrame:
    if aligned_df is None or aligned_df.empty:
        return pd.DataFrame()

    # Ensure required columns exist (feature requirements)
    required = [
        "year",
        "days_to_harvest",
        "modal_price",
        "temp_max_c",
        "temp_min_c",
        "rainfall_mm",
        "humidity_avg_pct",
        "evapotranspiration_mm",
        "price_volatility",
        "yield_kg_ha",
    ]
    if not all(c in aligned_df.columns for c in required):
        missing = [c for c in required if c not in aligned_df.columns]
        _warn(f"Aligned df missing required columns for feature matrix: {missing}. Returning empty.")
        return pd.DataFrame()

    # Normalize soil columns presence
    available_soil_cols = [c for c in soil_columns if c in aligned_df.columns]

    records: List[Dict[str, Any]] = []
    horizon_to_weeks = {28: 4, 56: 8, 84: 12}
    aligned_df = aligned_df.copy()
    aligned_df["days_to_harvest"] = pd.to_numeric(aligned_df["days_to_harvest"], errors="coerce").fillna(0).astype(int)

    # One feature row per (crop,district,year,horizon)
    for year, g in aligned_df.groupby("year", dropna=False):
        g = g.sort_values("date")
        for horizon in horizons:
            lower = horizon
            upper = horizon + 30  # as requested; observation window includes this range
            win = g[(g["days_to_harvest"] >= lower) & (g["days_to_harvest"] <= upper)]
            if win.empty:
                continue

            rec: dict = {
                "crop": crop,
                "district": district,
                "year": int(year),
                "horizon_weeks": int(horizon_to_weeks[horizon]),
                # modal_price stats
                "modal_price_mean": float(win["modal_price"].mean(skipna=True)),
                "modal_price_std": float(win["modal_price"].std(skipna=True, ddof=0)),
                "modal_price_min": float(win["modal_price"].min(skipna=True)),
                "modal_price_max": float(win["modal_price"].max(skipna=True)),
                # temperature stats
                "temp_max_c_mean": float(win["temp_max_c"].mean(skipna=True)),
                "temp_max_c_std": float(win["temp_max_c"].std(skipna=True, ddof=0)),
                "temp_min_c_mean": float(win["temp_min_c"].mean(skipna=True)),
                "temp_min_c_std": float(win["temp_min_c"].std(skipna=True, ddof=0)),
                # rainfall/humidity/evap
                "rainfall_mm_mean": float(win["rainfall_mm"].mean(skipna=True)),
                "rainfall_mm_std": float(win["rainfall_mm"].std(skipna=True, ddof=0)),
                "rainfall_mm_sum": float(win["rainfall_mm"].sum(skipna=True)),
                "humidity_avg_pct_mean": float(win["humidity_avg_pct"].mean(skipna=True)),
                "humidity_avg_pct_std": float(win["humidity_avg_pct"].std(skipna=True, ddof=0)),
                "evapotranspiration_mm_mean": float(win["evapotranspiration_mm"].mean(skipna=True)),
                "evapotranspiration_mm_std": float(win["evapotranspiration_mm"].std(skipna=True, ddof=0)),
                # volatility stats
                "price_volatility_mean": float(win["price_volatility"].mean(skipna=True)),
                "price_volatility_std": float(win["price_volatility"].std(skipna=True, ddof=0)),
                # target
                "yield_kg_ha": float(win["yield_kg_ha"].dropna().iloc[0]) if win["yield_kg_ha"].notna().any() else np.nan,
            }

            for sc in available_soil_cols:
                # Soil columns are static: all values should match within a year, but take first non-null.
                series = win[sc].dropna()
                rec[sc] = float(series.iloc[0]) if len(series) else np.nan

            records.append(rec)

    if not records:
        return pd.DataFrame()
    feat = pd.DataFrame.from_records(records)
    # Drop rows without target
    feat = feat.dropna(subset=["yield_kg_ha"])
    return feat


def assign_split_by_year(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        # Pandas doesn't allow assigning a plain empty list to an empty frame column
        # (it can lead to dtype/shape issues). Use an explicit empty Series instead.
        df["split"] = pd.Series([], dtype="object")
        return df
    df = df.copy()
    df["split"] = np.where(df["year"].isin([2023, 2024]), "test", np.where(df["year"].isin([2021, 2022]), "val", "train"))
    return df


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    # STEP 1 - Load and validate all inputs
    print("## STEP 1 - Load and validate inputs")
    files = {
        "erode_turmeric_price": base_dir / "turmeric_features.csv",
        "salem_tapioca_price": base_dir / "tapioca_features.csv",
        "erode_weather": base_dir / "erode_weather.csv",
        "salem_weather": base_dir / "salem_weather.csv",
        "yield_records": base_dir / "yield_records.csv",
        "soil_properties": base_dir / "soil_properties.csv",
    }

    turmeric_features = load_csv_if_exists(files["erode_turmeric_price"], "turmeric_features.csv")
    tapioca_features = load_csv_if_exists(files["salem_tapioca_price"], "tapioca_features.csv (Salem tapioca features)")
    erode_weather = load_csv_if_exists(files["erode_weather"], "erode_weather.csv")
    salem_weather = load_csv_if_exists(files["salem_weather"], "salem_weather.csv")
    yield_records = load_csv_if_exists(files["yield_records"], "yield_records.csv")
    soil_properties = load_csv_if_exists(files["soil_properties"], "soil_properties.csv")

    if turmeric_features is not None:
        print_shape_and_range(turmeric_features, "turmeric_features.csv", "arrival_date")
    if tapioca_features is not None:
        print_shape_and_range(tapioca_features, "tapioca_features.csv", "arrival_date")
    if erode_weather is not None:
        print_shape_and_range(erode_weather, "erode_weather.csv", "date")
    if salem_weather is not None:
        print_shape_and_range(salem_weather, "salem_weather.csv", "date")
    if yield_records is not None:
        print_shape_and_range(yield_records, "yield_records.csv", "year")
    if soil_properties is not None:
        # Static: no date range.
        print_shape_and_range(soil_properties, "soil_properties.csv", "none")
    else:
        _warn("soil_properties.csv is missing (or API was down). Soil columns will be skipped.")

    # Soil columns selection
    soil_columns: List[str] = []
    if soil_properties is not None:
        soil_columns = [c for c in soil_properties.columns if c.endswith("_avg_0_30cm")]

    # STEP 2 - Daily alignment
    print("\n## STEP 2 - Daily alignment (price spine + weather + static soil)")

    erode_turmeric_aligned = None
    salem_tapioca_aligned = None

    # Turmeric/Erode
    if turmeric_features is not None and erode_weather is not None:
        erode_turmeric_aligned = align_daily(
            turmeric_features,
            erode_weather,
            crop="Turmeric",
            district="Erode",
            soil_df=soil_properties,
            soil_columns=soil_columns if soil_columns else None,
            harvest_month_day=(2, 15),
            print_prefix="Turmeric/Erode",
        )
        if erode_turmeric_aligned is not None:
            if yield_records is not None:
                erode_turmeric_aligned = attach_yield(erode_turmeric_aligned, yield_records, "Turmeric", "Erode")
            else:
                _warn("yield_records.csv is missing. yield_kg_ha will be null for Erode/Turmeric.")
            print(f"Built aligned daily df: Erode/Turmeric rows={len(erode_turmeric_aligned)}")
    else:
        _warn("Missing price and/or weather for Turmeric/Erode. Skipping this aligned dataset.")

    # Tapioca/Salem
    if tapioca_features is not None and salem_weather is not None:
        salem_tapioca_aligned = align_daily(
            tapioca_features,
            salem_weather,
            crop="Tapioca",
            district="Salem",
            soil_df=soil_properties,
            soil_columns=soil_columns if soil_columns else None,
            harvest_month_day=(6, 30),
            print_prefix="Tapioca/Salem",
        )
        if salem_tapioca_aligned is not None:
            if yield_records is not None:
                salem_tapioca_aligned = attach_yield(salem_tapioca_aligned, yield_records, "Tapioca", "Salem")
            else:
                _warn("yield_records.csv is missing. yield_kg_ha will be null for Salem/Tapioca.")
            print(f"Built aligned daily df: Salem/Tapioca rows={len(salem_tapioca_aligned)}")
    else:
        _warn("Missing price and/or weather for Tapioca/Salem. Skipping this aligned dataset.")

    # STEP 3 - Add yield as annual target variable
    print("\n## STEP 3 - Add annual yield target and days_to_harvest")
    # (Already computed in STEP 2 alignment + attach_yield)
    # Sanity prints for yield coverage.
    for name, df in [("Erode/Turmeric", erode_turmeric_aligned), ("Salem/Tapioca", salem_tapioca_aligned)]:
        if df is None or df.empty:
            continue
        n_total = len(df)
        n_nonnull = int(pd.notna(df["yield_kg_ha"]).sum()) if "yield_kg_ha" in df.columns else 0
        print(f"{name}: yield_kg_ha non-null rows={n_nonnull}/{n_total}")

    # STEP 4 - Build sliding window feature matrix
    print("\n## STEP 4 - Sliding window feature matrix")
    horizons: List[int] = [28, 56, 84]
    all_feat = []

    if erode_turmeric_aligned is not None:
        feat_erode = build_feature_matrix(
            erode_turmeric_aligned,
            crop="Turmeric",
            district="Erode",
            soil_columns=soil_columns,
            horizons=horizons,
        )
        print(f"Erode/Turmeric feature rows={len(feat_erode)}")
        all_feat.append(feat_erode)
    if salem_tapioca_aligned is not None:
        feat_salem = build_feature_matrix(
            salem_tapioca_aligned,
            crop="Tapioca",
            district="Salem",
            soil_columns=soil_columns,
            horizons=horizons,
        )
        print(f"Salem/Tapioca feature rows={len(feat_salem)}")
        all_feat.append(feat_salem)

    if all_feat:
        feature_matrix = pd.concat(all_feat, ignore_index=True)
    else:
        feature_matrix = pd.DataFrame()

    # STEP 5 - Train / validation / test split by season (year-based)
    print("\n## STEP 5 - Train/val/test split by year")
    feature_matrix = assign_split_by_year(feature_matrix)
    if not feature_matrix.empty:
        counts = feature_matrix.groupby(["crop", "split"]).size().reset_index(name="n_rows")
        for crop in sorted(feature_matrix["crop"].unique()):
            sub = counts[counts["crop"] == crop].sort_values("split")
            print(f"{crop} split counts: {sub.to_dict(orient='records')}")
    else:
        _warn("Feature matrix is empty; skipping split count printing.")

    # STEP 6 - Save outputs
    print("\n## STEP 6 - Save outputs")
    out_erode = base_dir / "erode_turmeric_aligned.csv"
    out_salem = base_dir / "salem_tapioca_aligned.csv"
    out_feat = base_dir / "feature_matrix.csv"

    if erode_turmeric_aligned is not None:
        erode_turmeric_aligned.to_csv(out_erode, index=False)
        print(f"Saved: {out_erode}")
    else:
        _warn(f"Not saving {out_erode} (aligned df missing).")

    if salem_tapioca_aligned is not None:
        salem_tapioca_aligned.to_csv(out_salem, index=False)
        print(f"Saved: {out_salem}")
    else:
        _warn(f"Not saving {out_salem} (aligned df missing).")

    if feature_matrix is not None:
        feature_matrix.to_csv(out_feat, index=False)
        print(f"Saved: {out_feat}")
    else:
        _warn(f"Not saving {out_feat} (feature matrix missing).")

    # Final summary
    print("\n## Final Summary")
    if feature_matrix.empty:
        print("Feature matrix is empty.")
        return

    total_rows = len(feature_matrix)
    print(f"Total feature matrix rows: {total_rows}")

    summary = (
        feature_matrix.groupby(["crop", "horizon_weeks", "split"]).size().reset_index(name="n_rows").sort_values(["crop", "horizon_weeks", "split"])
    )
    # Print as readable lines to avoid nested bullets.
    for _, r in summary.iterrows():
        print(f"{r['crop']} | horizon_weeks={int(r['horizon_weeks'])} | split={r['split']} -> {int(r['n_rows'])} rows")


if __name__ == "__main__":
    main()

