"""
weather_download.py
====================
Crop Yield Research: A Robust Spatio-Temporal Multi-Modal Framework
for Early and Generalizable Crop Yield Prediction (Tamil Nadu Agriculture)

Purpose:
    Downloads historical daily weather data for Erode and Salem districts
    from the Open-Meteo Archive API (free, no API key required).
    Cleans column names, adds derived features, prints summaries, and
    saves the output as CSV files for downstream feature engineering.

Author : [Your Name]
Date   : 2026-03-19
"""

# ─────────────────────────────────────────────
# 0.  INSTALL DEPENDENCIES (notebook-safe)
# ─────────────────────────────────────────────
import subprocess, sys
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q",
     "openmeteo-requests", "requests-cache", "retry-requests", "pandas"],
    stdout=subprocess.DEVNULL,
)

# ─────────────────────────────────────────────
# 1.  IMPORTS
# ─────────────────────────────────────────────
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry

# ─────────────────────────────────────────────
# 2.  CONFIGURATION
# ─────────────────────────────────────────────
# Open-Meteo daily weather variables to request
DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "relative_humidity_2m_max",
    "relative_humidity_2m_min",
    "et0_fao_evapotranspiration",
    "windspeed_10m_max",
]

# Mapping from Open-Meteo raw names → clean research column names
COLUMN_RENAME = {
    "temperature_2m_max"         : "temp_max_c",
    "temperature_2m_min"         : "temp_min_c",
    "precipitation_sum"          : "rainfall_mm",
    "relative_humidity_2m_max"   : "humidity_max_pct",
    "relative_humidity_2m_min"   : "humidity_min_pct",
    "et0_fao_evapotranspiration" : "evapotranspiration_mm",
    "windspeed_10m_max"          : "windspeed_max_kmh",
}

# Location definitions: (label, lat, lon, start_date, end_date, output_file)
LOCATIONS = [
    {
        "label"      : "Erode, Tamil Nadu",
        "latitude"   : 11.341,
        "longitude"  : 77.717,
        "start_date" : "2015-01-01",
        "end_date"   : "2024-06-30",
        "output_file": "erode_weather.csv",
    },
    {
        "label"      : "Salem, Tamil Nadu",
        "latitude"   : 11.658,
        "longitude"  : 78.146,
        "start_date" : "2016-05-01",
        "end_date"   : "2026-01-01",
        "output_file": "salem_weather.csv",
    },
]

TIMEZONE = "Asia/Kolkata"
API_URL  = "https://archive-api.open-meteo.com/v1/archive"


# ─────────────────────────────────────────────
# 3.  SESSION SETUP (caching + retry)
# ─────────────────────────────────────────────
# Cache responses for 1 hour so repeated runs during development
# don't hit the API unnecessarily.
cache_session = requests_cache.CachedSession(
    ".openmeteo_cache", expire_after=3600
)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo     = openmeteo_requests.Client(session=retry_session)


# ─────────────────────────────────────────────
# 4.  DOWNLOAD & PROCESS FUNCTION
# ─────────────────────────────────────────────
def download_weather(location: dict) -> pd.DataFrame:
    """
    Fetches daily weather data for a single location from the
    Open-Meteo Archive API, renames columns, adds derived features,
    and returns a clean DataFrame.

    Parameters
    ----------
    location : dict
        Must contain keys: label, latitude, longitude,
        start_date, end_date, output_file.

    Returns
    -------
    pd.DataFrame with columns:
        date, temp_max_c, temp_min_c, rainfall_mm,
        humidity_max_pct, humidity_min_pct,
        evapotranspiration_mm, windspeed_max_kmh,
        temp_range_c, humidity_avg_pct
    """
    label = location["label"]
    print(f"\n{'='*60}")
    print(f"  Downloading weather data for  :  {label}")
    print(f"  Coordinates                   :  {location['latitude']}°N, "
          f"{location['longitude']}°E")
    print(f"  Date range                    :  {location['start_date']} → "
          f"{location['end_date']}")
    print(f"{'='*60}")

    # ── 4a. API request ───────────────────────────────────────────
    params = {
        "latitude"   : location["latitude"],
        "longitude"  : location["longitude"],
        "start_date" : location["start_date"],
        "end_date"   : location["end_date"],
        "daily"      : DAILY_VARIABLES,
        "timezone"   : TIMEZONE,
    }

    responses = openmeteo.weather_api(API_URL, params=params)
    response  = responses[0]   # single location → single response

    print(f"  API response received")
    print(f"  Elevation            : {response.Elevation()} m")

    # ── 4b. Parse daily data into DataFrame ───────────────────────
    daily = response.Daily()

    data = {"date": pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.DateOffset(seconds=daily.Interval()),
        inclusive="left",
    )}

    # Map each requested variable into the data dict
    for idx, var_name in enumerate(DAILY_VARIABLES):
        clean_name = COLUMN_RENAME[var_name]
        data[clean_name] = daily.Variables(idx).ValuesAsNumpy()

    df = pd.DataFrame(data)

    # ── 4c. Derived columns ───────────────────────────────────────
    df["temp_range_c"]    = df["temp_max_c"] - df["temp_min_c"]
    df["humidity_avg_pct"] = (df["humidity_max_pct"] + df["humidity_min_pct"]) / 2

    # ── 4d. Print summary ─────────────────────────────────────────
    print(f"\n  ── {label} Summary ──────────────────────────────────")
    print(f"  Row count    : {len(df):,}")
    print(f"  Date range   : {df['date'].min().date()}  →  "
          f"{df['date'].max().date()}")

    print(f"\n  Null counts per column:")
    null_counts = df.isnull().sum()
    for col, cnt in null_counts.items():
        marker = "  ⚠" if cnt > 0 else ""
        print(f"    {col:<25s} : {cnt}{marker}")

    print(f"\n  First 3 rows:")
    print(df.head(3).to_string(index=False))

    return df


# ─────────────────────────────────────────────
# 5.  MAIN
# ─────────────────────────────────────────────
def main():
    for loc in LOCATIONS:
        df = download_weather(loc)

        # ── Save CSV ──────────────────────────────────────────────
        out = loc["output_file"]
        df.to_csv(out, index=False)
        print(f"\n  ✅  Saved →  {out}  ({len(df):,} rows)\n")

    print("=" * 60)
    print("  ✓  weather_download.py complete — both files saved.")
    print("=" * 60)


if __name__ == "__main__":
    main()
