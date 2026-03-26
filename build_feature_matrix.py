import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pandas", "numpy"], check=False)

import numpy as np
import pandas as pd


def print_header(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")


def normalize_text(value: str) -> str:
    return str(value).strip().lower()


def get_yield_value(yield_df: pd.DataFrame, year: int, crop: str, district: str):
    mask = (
        (yield_df["year"] == year)
        & (yield_df["crop_norm"] == normalize_text(crop))
        & (yield_df["district_norm"] == normalize_text(district))
    )
    matches = yield_df.loc[mask, "yield_kg_ha"]
    if matches.empty:
        return None
    return float(matches.iloc[0])


def build_features_for_dataset(
    aligned_df: pd.DataFrame,
    yield_df: pd.DataFrame,
    crop: str,
    district: str,
    harvest_month: int,
    harvest_day: int,
):
    rows = []
    horizons_days = [28, 56, 84]

    aligned_df = aligned_df.sort_values("arrival_date").copy()
    years = sorted(aligned_df["arrival_date"].dt.year.dropna().unique())

    for year in years:
        harvest_date = pd.Timestamp(year=int(year), month=harvest_month, day=harvest_day)
        target_yield = get_yield_value(yield_df, int(year), crop, district)

        if target_yield is None:
            print(f"Skipping {crop}-{district} year {year}: no yield_kg_ha in yield_records.csv")
            continue

        for horizon_days in horizons_days:
            window_end = harvest_date - pd.Timedelta(days=horizon_days)
            window_start = window_end - pd.Timedelta(days=56)

            window_df = aligned_df[
                (aligned_df["arrival_date"] >= window_start)
                & (aligned_df["arrival_date"] <= window_end)
            ].copy()

            n_days = len(window_df)
            if n_days < 20:
                continue

            prices = window_df["modal_price"].to_numpy(dtype=float)
            x = np.arange(n_days, dtype=float)
            price_trend = float(np.polyfit(x, prices, 1)[0]) if n_days >= 2 else np.nan

            row = {
                "crop": crop,
                "district": district,
                "year": int(year),
                "horizon_weeks": int(horizon_days // 7),
                "window_start": window_start.date().isoformat(),
                "window_end": window_end.date().isoformat(),
                "n_days": int(n_days),
                "price_mean": float(window_df["modal_price"].mean()),
                "price_std": float(window_df["modal_price"].std()),
                "price_min": float(window_df["modal_price"].min()),
                "price_max": float(window_df["modal_price"].max()),
                "price_trend": price_trend,
                "rolling_avg_7d_mean": float(window_df["rolling_avg_7d"].mean()),
                "rolling_avg_30d_mean": float(window_df["rolling_avg_30d"].mean()),
                "price_volatility_mean": float(window_df["price_volatility"].mean()),
                "temp_max_mean": float(window_df["temp_max_c"].mean()),
                "temp_min_mean": float(window_df["temp_min_c"].mean()),
                "temp_range_mean": float(window_df["temp_range_c"].mean()),
                "rainfall_total": float(window_df["rainfall_mm"].sum()),
                "rainfall_mean": float(window_df["rainfall_mm"].mean()),
                "humidity_mean": float(window_df["humidity_avg_pct"].mean()),
                "evapotranspiration_total": float(window_df["evapotranspiration_mm"].sum()),
                "windspeed_mean": float(window_df["windspeed_max_kmh"].mean()),
                "yield_kg_ha": float(target_yield),
            }
            rows.append(row)

    return rows


def assign_split(year: int) -> str:
    if year in {2023, 2024}:
        return "test"
    if year in {2021, 2022}:
        return "val"
    return "train"


def load_aligned_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = "arrival_date" if "arrival_date" in df.columns else "date"
    if date_col not in df.columns:
        raise ValueError(f"{path} must contain 'arrival_date' or 'date' column.")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.rename(columns={date_col: "arrival_date"})
    df = df.dropna(subset=["arrival_date"]).copy()
    return df


def main() -> None:
    turmeric_path = "erode_turmeric_aligned.csv"
    tapioca_path = "salem_tapioca_aligned.csv"
    yield_path = "yield_records.csv"

    print_header("Step 1 - Load Aligned CSVs")
    turmeric_df = load_aligned_csv(turmeric_path)
    tapioca_df = load_aligned_csv(tapioca_path)
    yield_df = pd.read_csv(yield_path)

    print(f"Loaded {turmeric_path}: shape={turmeric_df.shape}")
    print(f"Loaded {tapioca_path}: shape={tapioca_df.shape}")
    print(f"Loaded {yield_path}: shape={yield_df.shape}")

    yield_df["crop_norm"] = yield_df["crop"].map(normalize_text)
    yield_df["district_norm"] = yield_df["district"].map(normalize_text)
    yield_df["year"] = pd.to_numeric(yield_df["year"], errors="coerce").astype("Int64")
    yield_df = yield_df.dropna(subset=["year", "yield_kg_ha"]).copy()
    yield_df["year"] = yield_df["year"].astype(int)

    print_header("Step 2 - Build Weekly Sequence Windows")
    all_rows = []
    all_rows.extend(
        build_features_for_dataset(
            aligned_df=turmeric_df,
            yield_df=yield_df,
            crop="Turmeric",
            district="Erode",
            harvest_month=2,
            harvest_day=15,
        )
    )
    all_rows.extend(
        build_features_for_dataset(
            aligned_df=tapioca_df,
            yield_df=yield_df,
            crop="Tapioca",
            district="Salem",
            harvest_month=6,
            harvest_day=30,
        )
    )

    feature_df = pd.DataFrame(all_rows)
    if feature_df.empty:
        print("No feature rows generated. Check date coverage and yield records.")
        return

    print(f"Generated rows before split assignment: {len(feature_df)}")

    print_header("Step 3 - Train/Val/Test Split By Year")
    feature_df["split"] = feature_df["year"].apply(assign_split)
    print(feature_df["split"].value_counts(dropna=False).sort_index())

    print_header("Step 4 - Summary")
    print(f"Total rows in feature matrix: {len(feature_df)}")
    print("\nRows per crop per horizon per split:")
    summary_counts = (
        feature_df.groupby(["crop", "horizon_weeks", "split"])
        .size()
        .reset_index(name="rows")
        .sort_values(["crop", "horizon_weeks", "split"])
    )
    print(summary_counts.to_string(index=False))

    print("\nYield statistics (min/max/mean) per crop:")
    yield_stats = feature_df.groupby("crop")["yield_kg_ha"].agg(["min", "max", "mean"]).reset_index()
    print(yield_stats.to_string(index=False))

    feature_cols = [
        "price_mean",
        "price_std",
        "price_min",
        "price_max",
        "price_trend",
        "rolling_avg_7d_mean",
        "rolling_avg_30d_mean",
        "price_volatility_mean",
        "temp_max_mean",
        "temp_min_mean",
        "temp_range_mean",
        "rainfall_total",
        "rainfall_mean",
        "humidity_mean",
        "evapotranspiration_total",
        "windspeed_mean",
    ]
    null_counts = feature_df[feature_cols].isnull().sum()
    total_nulls = int(null_counts.sum())
    print("\nNull value count in feature columns:")
    print(null_counts.to_string())
    print(f"\nTotal nulls across feature columns: {total_nulls}")

    print_header("Step 5 - Save Feature Matrix")
    output_path = "feature_matrix_weekly.csv"
    feature_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    print(f"Output shape: {feature_df.shape}")
    print("\nFirst 3 rows:")
    print(feature_df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
