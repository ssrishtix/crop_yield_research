import subprocess
import sys
from pathlib import Path

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "pandas", "numpy"],
    check=False,
)

import numpy as np
import pandas as pd


def _warn(msg: str) -> None:
    print(f"WARNING: {msg}")


def _print_date_range(df: pd.DataFrame, col: str, label: str) -> None:
    if col not in df.columns:
        print(f"{label}: shape={df.shape}, {col}=missing")
        return
    if col == "year":
        y = pd.to_numeric(df[col], errors="coerce").dropna()
        if y.notna().any():
            print(f"{label}: shape={df.shape}, year range={int(y.min())} to {int(y.max())}")
        else:
            print(f"{label}: shape={df.shape}, year range=invalid/empty")
        return
    s = pd.to_datetime(df[col], errors="coerce")
    if s.notna().any():
        print(f"{label}: shape={df.shape}, {col} range={s.min().date()} to {s.max().date()}")
    else:
        print(f"{label}: shape={df.shape}, {col} range=invalid/empty")


def _normalize_key(x) -> str:
    return str(x).strip().lower()


def _to_datetime_no_tz(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    # Ensure tz-naive DatetimeIndex/Series for safe merges.
    try:
        if getattr(dt.dt, "tz", None) is not None:
            dt = dt.dt.tz_localize(None)
    except Exception:
        pass
    # Normalize to day boundary so joins work even if source timestamps differ (e.g., weather at 18:30).
    try:
        dt = dt.dt.normalize()
    except Exception:
        pass
    return dt


def load_if_exists(path: Path, label: str):
    if not path.exists():
        _warn(f"Missing file: {path}. Skipping {label}.")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        _warn(f"Failed to read {label} ({path}): {e}. Skipping {label}.")
        return None


def build_daily_alignment(
    *,
    price_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    crop: str,
    district: str,
    harvest_month_day: tuple[int, int],
    soil_df: pd.DataFrame | None,
    soil_columns: list[str],
    yield_df: pd.DataFrame,
    print_prefix: str,
) -> pd.DataFrame | None:
    # Validate required columns (skip if missing).
    required_price_cols = [
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
    missing_price = [c for c in required_price_cols if c not in price_df.columns]
    if missing_price:
        _warn(f"{print_prefix} price missing columns {missing_price}. Skipping crop/district.")
        return None

    required_weather_cols = [
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
    missing_weather = [c for c in required_weather_cols if c not in weather_df.columns]
    if missing_weather:
        _warn(f"{print_prefix} weather missing columns {missing_weather}. Skipping crop/district.")
        return None

    # ---- Daily spine from price arrival_date range ----
    price_df = price_df.copy()
    price_df["arrival_date"] = _to_datetime_no_tz(price_df["arrival_date"])
    price_df = price_df.dropna(subset=["arrival_date"]).copy()
    if price_df.empty:
        _warn(f"{print_prefix} price data empty after date parsing. Skipping.")
        return None
    price_df = price_df.sort_values("arrival_date")
    start_date = price_df["arrival_date"].min()
    end_date = price_df["arrival_date"].max()
    spine = pd.date_range(start=start_date, end=end_date, freq="D")
    master = pd.DataFrame({"arrival_date": spine})

    # ---- Aggregate price per day (if duplicates) ----
    price_keep = required_price_cols
    price_daily = price_df[price_keep].copy()
    for c in price_keep:
        if c not in ["arrival_date", "month", "season"]:
            price_daily[c] = pd.to_numeric(price_daily[c], errors="coerce")
    # season/month: take first; numeric: mean
    agg = {}
    for c in price_keep:
        if c in ["season", "month"]:
            agg[c] = "first"
        elif c == "arrival_date":
            continue
        else:
            agg[c] = "mean"
    price_daily = (
        price_daily.groupby("arrival_date", as_index=False)
        .agg(agg)
        .sort_values("arrival_date")
    )

    master = master.merge(price_daily, on="arrival_date", how="left")

    # ---- Join weather on date + forward-fill up to 3 days ----
    weather_df = weather_df.copy()
    weather_df["date"] = _to_datetime_no_tz(weather_df["date"])
    weather_df = weather_df.dropna(subset=["date"]).copy()
    if weather_df.empty:
        _warn(f"{print_prefix} weather data empty after date parsing. Skipping.")
        return None
    weather_keep = required_weather_cols
    for c in weather_keep:
        if c != "date":
            weather_df[c] = pd.to_numeric(weather_df[c], errors="coerce")
    weather_daily = weather_df[weather_keep].groupby("date", as_index=False).mean(numeric_only=True)

    weather_daily = weather_daily.rename(columns={"date": "arrival_date"})
    master = master.merge(weather_daily, on="arrival_date", how="left")

    weather_fill_cols = [
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
    master[weather_fill_cols] = master[weather_fill_cols].ffill(limit=3)

    # ---- Static identifiers ----
    master["crop"] = crop
    master["district"] = district

    # ---- Soil static columns ----
    if soil_df is not None and soil_columns:
        soil_df = soil_df.copy()
        if "location" in soil_df.columns:
            soil_df["location_norm"] = soil_df["location"].map(_normalize_key)
        else:
            soil_df["location_norm"] = np.nan
        if "crop" in soil_df.columns:
            soil_df["crop_norm"] = soil_df["crop"].map(_normalize_key)
        else:
            soil_df["crop_norm"] = np.nan

        target_loc = _normalize_key(district)
        target_crop = _normalize_key(crop)
        match = soil_df[(soil_df["location_norm"] == target_loc) & (soil_df["crop_norm"] == target_crop)]
        if match.empty:
            _warn(f"{print_prefix}: no soil row match for crop={crop}, district={district}. Skipping soil columns.")
        else:
            row = match.iloc[0]
            for c in soil_columns:
                if c in row.index:
                    master[c] = row[c]
    else:
        if soil_df is None:
            _warn(f"{print_prefix}: soil_properties.csv missing, soil columns skipped.")
        else:
            _warn(f"{print_prefix}: soil columns not found, soil columns skipped.")

    # ---- Yield + days_to_harvest ----
    master["year"] = master["arrival_date"].dt.year.astype(int)

    ycol = "yield_kg_ha"
    if ycol not in yield_df.columns:
        master["yield_kg_ha"] = np.nan
    else:
        yd = yield_df.copy()
        yd["year"] = pd.to_numeric(yd["year"], errors="coerce")
        yd = yd.dropna(subset=["year"]).copy()
        yd["year"] = yd["year"].astype(int)
        yd["crop_norm"] = yd["crop"].map(_normalize_key) if "crop" in yd.columns else np.nan
        yd["district_norm"] = yd["district"].map(_normalize_key) if "district" in yd.columns else np.nan
        target_crop = _normalize_key(crop)
        target_district = _normalize_key(district)
        yd_match = yd[(yd["crop_norm"] == target_crop) & (yd["district_norm"] == target_district)]
        mapping = yd_match.dropna(subset=[ycol]).set_index("year")[ycol].to_dict()
        master["yield_kg_ha"] = master["year"].map(mapping).astype(float)

    h_month, h_day = harvest_month_day
    harvest_dates = pd.to_datetime(
        master["year"].astype(str) + f"-{h_month:02d}-{h_day:02d}",
        errors="coerce",
    )
    master["days_to_harvest"] = (harvest_dates - master["arrival_date"]).dt.days.astype(float)
    master["days_to_harvest"] = master["days_to_harvest"].fillna(0).clip(lower=0).astype(int)

    # Requirement output uses arrival_date, modal_price, weather, soil, yield_kg_ha, days_to_harvest.
    return master


def build_feature_matrix(
    aligned_df: pd.DataFrame,
    *,
    crop: str,
    district: str,
    soil_columns: list[str],
    yield_col: str = "yield_kg_ha",
    horizons: list[int] = [28, 56, 84],
) -> pd.DataFrame:
    if aligned_df is None or aligned_df.empty:
        return pd.DataFrame()

    required_cols = [
        "year",
        "days_to_harvest",
        "modal_price",
        "temp_max_c",
        "temp_min_c",
        "rainfall_mm",
        "humidity_avg_pct",
        "evapotranspiration_mm",
        "price_volatility",
        yield_col,
    ]
    missing = [c for c in required_cols if c not in aligned_df.columns]
    if missing:
        _warn(f"Feature matrix build missing columns for {crop}/{district}: {missing}.")
        return pd.DataFrame()

    records = []
    horizon_to_weeks = {28: 4, 56: 8, 84: 12}

    for year, g in aligned_df.groupby("year", dropna=False):
        g = g.copy()
        for horizon in horizons:
            win = g[
                (g["days_to_harvest"] >= horizon) & (g["days_to_harvest"] <= horizon + 30)
            ]
            if win.empty:
                continue

            rec = {
                "crop": crop,
                "district": district,
                "year": int(year) if not pd.isna(year) else np.nan,
                "horizon_weeks": horizon_to_weeks[horizon],
                # modal_price stats
                "modal_price_mean": float(win["modal_price"].mean(skipna=True)),
                "modal_price_std": float(win["modal_price"].std(skipna=True, ddof=0)),
                "modal_price_min": float(win["modal_price"].min(skipna=True)),
                "modal_price_max": float(win["modal_price"].max(skipna=True)),
                # temp stats
                "temp_max_c_mean": float(win["temp_max_c"].mean(skipna=True)),
                "temp_max_c_std": float(win["temp_max_c"].std(skipna=True, ddof=0)),
                "temp_min_c_mean": float(win["temp_min_c"].mean(skipna=True)),
                "temp_min_c_std": float(win["temp_min_c"].std(skipna=True, ddof=0)),
                # rainfall stats
                "rainfall_mm_mean": float(win["rainfall_mm"].mean(skipna=True)),
                "rainfall_mm_std": float(win["rainfall_mm"].std(skipna=True, ddof=0)),
                "rainfall_mm_sum": float(win["rainfall_mm"].sum(skipna=True)),
                # humidity stats
                "humidity_avg_pct_mean": float(win["humidity_avg_pct"].mean(skipna=True)),
                "humidity_avg_pct_std": float(win["humidity_avg_pct"].std(skipna=True, ddof=0)),
                # evap stats
                "evapotranspiration_mm_mean": float(win["evapotranspiration_mm"].mean(skipna=True)),
                "evapotranspiration_mm_std": float(win["evapotranspiration_mm"].std(skipna=True, ddof=0)),
                # volatility stats
                "price_volatility_mean": float(win["price_volatility"].mean(skipna=True)),
                "price_volatility_std": float(win["price_volatility"].std(skipna=True, ddof=0)),
                # target
                yield_col: float(win[yield_col].dropna().iloc[0]) if win[yield_col].notna().any() else np.nan,
            }

            for sc in soil_columns:
                if sc in win.columns and win[sc].notna().any():
                    rec[sc] = float(win[sc].dropna().iloc[0])
                else:
                    rec[sc] = np.nan

            records.append(rec)

    feat = pd.DataFrame.from_records(records)
    feat = feat.dropna(subset=[yield_col]).copy()
    return feat


def assign_split_by_year(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        df["split"] = pd.Series([], dtype="object")
        return df
    df = df.copy()
    df["split"] = np.where(
        df["year"].isin([2023, 2024]),
        "test",
        np.where(df["year"].isin([2021, 2022]), "val", "train"),
    )
    return df


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    print("\n## STEP 1 - Load and validate all inputs")
    turmeric_path = base_dir / "turmeric_features.csv"
    salem_features_path = base_dir / "tapioca_features.csv"
    erode_weather_path = base_dir / "erode_weather.csv"
    salem_weather_path = base_dir / "salem_weather.csv"
    yield_path = base_dir / "yield_records.csv"
    soil_path = base_dir / "soil_properties.csv"

    turmeric_df = load_if_exists(turmeric_path, "turmeric_features.csv")
    salem_features_df = load_if_exists(salem_features_path, "tapioca_features.csv")
    erode_weather_df = load_if_exists(erode_weather_path, "erode_weather.csv")
    salem_weather_df = load_if_exists(salem_weather_path, "salem_weather.csv")
    yield_df = load_if_exists(yield_path, "yield_records.csv")
    soil_df = load_if_exists(soil_path, "soil_properties.csv")

    if turmeric_df is not None:
        _print_date_range(turmeric_df, "arrival_date", "turmeric_features.csv")
    if salem_features_df is not None:
        _print_date_range(salem_features_df, "arrival_date", "tapioca_features.csv")
    if erode_weather_df is not None:
        _print_date_range(erode_weather_df, "date", "erode_weather.csv")
    if salem_weather_df is not None:
        _print_date_range(salem_weather_df, "date", "salem_weather.csv")
    if yield_df is not None:
        _print_date_range(yield_df, "year", "yield_records.csv")
    if soil_df is not None:
        print(f"soil_properties.csv: shape={soil_df.shape} (static, no date range)")

    if yield_df is None:
        _warn("yield_records.csv missing; cannot add target. Exiting.")
        return

    # Soil columns to carry forward (all *_avg_0_30cm).
    soil_columns: list[str] = []
    if soil_df is not None:
        soil_columns = [c for c in soil_df.columns if c.endswith("_avg_0_30cm")]
        if not soil_columns:
            _warn("soil_properties.csv loaded but no columns ending with *_avg_0_30cm found.")

    # ---- STEP 2 - Daily alignment ----
    print("\n## STEP 2 - Daily alignment (price spine + weather + static soil)")
    erode_turmeric_aligned = None
    salem_tapioca_aligned = None

    if turmeric_df is not None and erode_weather_df is not None:
        erode_turmeric_aligned = build_daily_alignment(
            price_df=turmeric_df,
            weather_df=erode_weather_df,
            crop="Turmeric",
            district="Erode",
            harvest_month_day=(2, 15),
            soil_df=soil_df,
            soil_columns=soil_columns,
            yield_df=yield_df,
            print_prefix="Turmeric/Erode",
        )
    else:
        _warn("Missing turmeric price and/or erode weather; skipping Turmeric/Erode alignment.")

    if salem_features_df is not None and salem_weather_df is not None:
        salem_tapioca_aligned = build_daily_alignment(
            price_df=salem_features_df,
            weather_df=salem_weather_df,
            crop="Tapioca",
            district="Salem",
            harvest_month_day=(6, 30),
            soil_df=soil_df,
            soil_columns=soil_columns,
            yield_df=yield_df,
            print_prefix="Tapioca/Salem",
        )
    else:
        _warn("Missing tapioca price and/or salem weather; skipping Tapioca/Salem alignment.")

    # ---- STEP 3 - Yield already attached in alignment, plus days_to_harvest ----
    print("\n## STEP 3 - Add yield as annual target variable")
    for name, df in [
        ("erode_turmeric_aligned.csv", erode_turmeric_aligned),
        ("salem_tapioca_aligned.csv", salem_tapioca_aligned),
    ]:
        if df is None:
            continue
        nonnull = int(df["yield_kg_ha"].notna().sum()) if "yield_kg_ha" in df.columns else 0
        print(f"{name}: rows={len(df)}, yield_kg_ha non-null={nonnull}")

    # ---- STEP 4 - Sliding window feature matrix ----
    print("\n## STEP 4 - Sliding window feature matrix")
    horizons = [28, 56, 84]
    feature_parts = []

    if erode_turmeric_aligned is not None and not erode_turmeric_aligned.empty:
        feat_erode = build_feature_matrix(
            erode_turmeric_aligned,
            crop="Turmeric",
            district="Erode",
            soil_columns=soil_columns,
            horizons=horizons,
        )
        print(f"Feature rows (Turmeric/Erode): {len(feat_erode)}")
        feature_parts.append(feat_erode)

    if salem_tapioca_aligned is not None and not salem_tapioca_aligned.empty:
        feat_salem = build_feature_matrix(
            salem_tapioca_aligned,
            crop="Tapioca",
            district="Salem",
            soil_columns=soil_columns,
            horizons=horizons,
        )
        print(f"Feature rows (Tapioca/Salem): {len(feat_salem)}")
        feature_parts.append(feat_salem)

    feature_matrix = pd.concat(feature_parts, ignore_index=True) if feature_parts else pd.DataFrame()

    # ---- STEP 5 - Train/val/test split by year ----
    print("\n## STEP 5 - Train/validation/test split by year (not random)")
    feature_matrix = assign_split_by_year(feature_matrix)
    if not feature_matrix.empty:
        for crop in sorted(feature_matrix["crop"].unique()):
            sub = feature_matrix[feature_matrix["crop"] == crop]
            counts = sub["split"].value_counts().to_dict()
            print(f"{crop} split counts: {counts}")
    else:
        _warn("feature_matrix.csv is empty after building features.")

    # ---- STEP 6 - Save outputs ----
    print("\n## STEP 6 - Save outputs")
    out_erode = base_dir / "erode_turmeric_aligned.csv"
    out_salem = base_dir / "salem_tapioca_aligned.csv"
    out_feat = base_dir / "feature_matrix.csv"

    if erode_turmeric_aligned is not None:
        erode_turmeric_aligned.to_csv(out_erode, index=False)
        print(f"Saved: {out_erode}")
    else:
        _warn("Skipping save: erode_turmeric_aligned.csv (missing input data).")

    if salem_tapioca_aligned is not None:
        salem_tapioca_aligned.to_csv(out_salem, index=False)
        print(f"Saved: {out_salem}")
    else:
        _warn("Skipping save: salem_tapioca_aligned.csv (missing input data).")

    if not feature_matrix.empty:
        feature_matrix.to_csv(out_feat, index=False)
        print(f"Saved: {out_feat}")
    else:
        _warn("Skipping save: feature_matrix.csv (empty).")

    print("\n## Final Summary")
    if feature_matrix.empty:
        print("Feature matrix is empty.")
        return

    total_rows = len(feature_matrix)
    print(f"Total rows in feature matrix: {total_rows}")

    rows_summary = (
        feature_matrix.groupby(["crop", "horizon_weeks", "split"])
        .size()
        .reset_index(name="n_rows")
        .sort_values(["crop", "horizon_weeks", "split"])
    )
    for _, r in rows_summary.iterrows():
        print(
            f"{r['crop']} | horizon_weeks={int(r['horizon_weeks'])} | split={r['split']} -> {int(r['n_rows'])} rows"
        )


if __name__ == "__main__":
    main()

