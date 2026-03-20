"""
MODULE 3 — price_forecast_model.py
=====================================
Crop Yield Research: A Robust Spatio-Temporal Multi-Modal Framework
for Early and Generalizable Crop Yield Prediction (Tamil Nadu Agriculture)

Purpose:
    Loads the Turmeric feature-engineered dataset, fits an ARIMA(5,1,0)
    model on the modal_price series, evaluates forecast accuracy on the
    2024 hold-out set (MAE & MAPE), generates a 30-day future forecast,
    plots the results, and saves forecast output to CSV.

Dependencies:
    pip install statsmodels pandas matplotlib

Author : [Your Name]
Date   : 2026-02-27
"""

# ─────────────────────────────────────────────
# 0.  IMPORTS
# ─────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA


# ─────────────────────────────────────────────
# 1.  DATA LOADING & PREPARATION
# ─────────────────────────────────────────────
def load_price_series(filepath: str) -> pd.Series:
    """
    Reads the feature-engineered Turmeric CSV, aggregates to a clean
    daily modal_price time series (taking the mean when multiple
    arrival records share the same date), and returns it as a
    DatetimeIndex pd.Series.

    Aggregating to a uniform daily series is necessary because ARIMA
    requires an evenly-spaced time index.

    Parameters
    ----------
    filepath : str   Path to turmeric_features.csv.

    Returns
    -------
    pd.Series  Daily modal_price series with DatetimeIndex.
    """
    df = pd.read_csv(filepath, parse_dates=["arrival_date"])
    df.sort_values("arrival_date", inplace=True)

    # Aggregate multiple records per day → take mean modal_price
    series = (
        df.groupby("arrival_date")["modal_price"]
        .mean()
        .asfreq("D")           # enforce daily frequency; gaps → NaN
        .interpolate(method="time")   # fill sparse gaps linearly
    )
    print(f"  Price series loaded  :  {len(series):,} daily observations")
    print(f"  Date range           :  {series.index.min().date()}  →  "
          f"{series.index.max().date()}")
    return series


# ─────────────────────────────────────────────
# 2.  TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
def split_train_test(series: pd.Series):
    """
    Splits the daily price series into:
        - Train : all data up to and including 2023-12-31
        - Test  : all data from 2024-01-01 onward

    Parameters
    ----------
    series : pd.Series   Full daily price series.

    Returns
    -------
    tuple[pd.Series, pd.Series]  (train, test)
    """
    train = series[series.index.year <= 2023]
    test  = series[series.index.year >= 2024]
    print(f"\n  Train observations  :  {len(train):,}  "
          f"({train.index.min().date()} → {train.index.max().date()})")
    print(f"  Test  observations  :  {len(test):,}  "
          f"({test.index.min().date()} → {test.index.max().date()})")
    return train, test


# ─────────────────────────────────────────────
# 3.  ARIMA FIT & IN-SAMPLE TEST FORECAST
# ─────────────────────────────────────────────
def fit_and_evaluate(train: pd.Series, test: pd.Series):
    """
    Fits ARIMA(5,1,0) on the training set and produces a one-step-ahead
    rolling forecast for the test period (walk-forward validation).

    ARIMA order chosen:
        p=5  (5 autoregressive lags capture weekly seasonality patterns)
        d=1  (one differencing step makes the price series stationary)
        q=0  (no moving-average term; AR alone is sufficient here)

    Walk-forward validation re-fits the model at each test step using
    all available history, which gives a realistic out-of-sample
    evaluation without look-ahead bias.

    Parameters
    ----------
    train : pd.Series   Training price series.
    test  : pd.Series   Hold-out test series.

    Returns
    -------
    tuple[ARIMA fit result, pd.Series]
        Final fitted model (on full data), and per-day test predictions.
    """
    print("\n  Fitting ARIMA(5,1,0) on training data …")

    # ── Walk-forward test evaluation ──────────────────────────────
    history   = list(train.values)
    test_preds = []

    for i, actual in enumerate(test.values):
        model  = ARIMA(history, order=(5, 1, 0))
        fitted = model.fit()
        yhat   = fitted.forecast(steps=1)[0]
        test_preds.append(yhat)
        history.append(actual)       # expand window with true value
        if (i + 1) % 50 == 0:
            print(f"    … {i+1}/{len(test)} test steps evaluated")

    test_pred_series = pd.Series(test_preds, index=test.index)

    # ── Metrics ───────────────────────────────────────────────────
    mae  = np.mean(np.abs(test.values - test_preds))
    mape = np.mean(
        np.abs((test.values - test_preds) / np.where(test.values == 0, 1, test.values))
    ) * 100

    print(f"\n  ── Test-Set Evaluation (2024) ──────────────────────")
    print(f"  MAE  (Mean Absolute Error)        :  ₹ {mae:,.2f} / quintal")
    print(f"  MAPE (Mean Absolute % Error)      :  {mape:.2f} %")

    # ── Refit on all available data for forecasting ───────────────
    print("\n  Refitting ARIMA on full dataset for future forecasting …")
    full_series = pd.concat([train, test])
    final_model  = ARIMA(full_series.values, order=(5, 1, 0)).fit()

    return final_model, test_pred_series, mae, mape


# ─────────────────────────────────────────────
# 4.  FUTURE 30-DAY FORECAST
# ─────────────────────────────────────────────
def forecast_future(model, series: pd.Series,
                    horizon: int = 30) -> pd.DataFrame:
    """
    Generates a point forecast and 95% confidence interval for the
    next `horizon` calendar days beyond the last observed date.

    Uses summary_frame() instead of conf_int() for robust compatibility
    across statsmodels versions (conf_int may return ndarray or DataFrame
    depending on version, causing AttributeError with .iloc).

    Parameters
    ----------
    model   : fitted ARIMA result object
    series  : pd.Series   Full price series (used to derive the last date).
    horizon : int         Number of future days to forecast (default 30).

    Returns
    -------
    pd.DataFrame  Columns: date, forecasted_price, lower_ci, upper_ci.
    """
    forecast_result = model.get_forecast(steps=horizon)
    # summary_frame() reliably returns a DataFrame with mean/lower/upper cols
    sf = forecast_result.summary_frame(alpha=0.05)

    last_date    = series.index.max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D"
    )

    df_forecast = pd.DataFrame({
        "date"            : future_dates,
        "forecasted_price": sf["mean"].values,
        "lower_ci"        : sf["mean_ci_lower"].values,
        "upper_ci"        : sf["mean_ci_upper"].values,
    })
    # Clip negative price predictions (prices cannot be negative)
    for col in ["forecasted_price", "lower_ci"]:
        df_forecast[col] = df_forecast[col].clip(lower=0)

    return df_forecast


# ─────────────────────────────────────────────
# 5.  VISUALISATION
# ─────────────────────────────────────────────
def plot_forecast(train: pd.Series,
                  test: pd.Series,
                  test_preds: pd.Series,
                  df_forecast: pd.DataFrame) -> None:
    """
    Produces a two-panel figure:
        Left  : Actual vs ARIMA-predicted modal price on the 2024 test set.
        Right : 30-day future forecast with 95% confidence band,
                anchored on the last 6 months of observed data.

    Saved as 'turmeric_price_forecast.png'.

    Parameters
    ----------
    train        : pd.Series   Training series.
    test         : pd.Series   Test series.
    test_preds   : pd.Series   Walk-forward predictions on test.
    df_forecast  : pd.DataFrame  Future 30-day forecast table.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ── Left panel: Test-set evaluation ───────────────────────────
    ax1.plot(test.index, test.values,
             color="#2D6A4F", linewidth=1.2, label="Actual")
    ax1.plot(test_preds.index, test_preds.values,
             color="#E76F51", linewidth=1.2, linestyle="--", label="ARIMA Forecast")
    ax1.set_title("ARIMA(5,1,0) — Test Set (2024)\nActual vs Predicted Modal Price",
                  fontsize=12, fontweight="bold")
    ax1.set_xlabel("Date", fontsize=10)
    ax1.set_ylabel("Modal Price (₹/quintal)", fontsize=10)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # ── Right panel: Future 30-day forecast ───────────────────────
    # Show last 6 months of actual data as context
    context_start = test.index.max() - pd.DateOffset(months=6)
    context = pd.concat([train, test])[context_start:]

    ax2.plot(context.index, context.values,
             color="#2D6A4F", linewidth=1.2, label="Historical (last 6 months)")
    ax2.plot(df_forecast["date"], df_forecast["forecasted_price"],
             color="#E76F51", linewidth=1.5, linestyle="--", label="30-Day Forecast")
    ax2.fill_between(
        df_forecast["date"],
        df_forecast["lower_ci"],
        df_forecast["upper_ci"],
        color="#E76F51", alpha=0.2, label="95% Confidence Interval",
    )
    ax2.axvline(x=test.index.max(), color="grey",
                linestyle=":", linewidth=1.2, label="Last Observed Date")
    ax2.set_title("30-Day Future Price Forecast\nTurmeric — Erode (ARIMA)",
                  fontsize=12, fontweight="bold")
    ax2.set_xlabel("Date", fontsize=10)
    ax2.set_ylabel("Modal Price (₹/quintal)", fontsize=10)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.suptitle(
        "Turmeric (Erode) — ARIMA Price Forecast  |  Source: AGMARKNET",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    plt.savefig("turmeric_price_forecast.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n  Plot saved →  turmeric_price_forecast.png")


# ─────────────────────────────────────────────
# 6.  MAIN
# ─────────────────────────────────────────────
def main():
    INPUT_FILE  = "turmeric_features.csv"
    OUTPUT_FILE = "turmeric_price_forecast.csv"

    print(f"\n{'='*60}")
    print("  Turmeric Price Forecast  —  ARIMA(5,1,0)")
    print(f"{'='*60}")

    # ── Load & prepare series ────────────────────────────────────
    series = load_price_series(INPUT_FILE)

    # ── Train/test split ─────────────────────────────────────────
    train, test = split_train_test(series)

    # ── Fit ARIMA & evaluate on test set ─────────────────────────
    final_model, test_preds, mae, mape = fit_and_evaluate(train, test)

    # ── 30-day future forecast ───────────────────────────────────
    df_forecast = forecast_future(final_model, series, horizon=30)
    print(f"\n  ── 30-Day Future Forecast ──────────────────────────")
    print(df_forecast[["date", "forecasted_price"]].to_string(index=False))

    # ── Plot ─────────────────────────────────────────────────────
    plot_forecast(train, test, test_preds, df_forecast)

    # ── Save forecast CSV ────────────────────────────────────────
    df_forecast.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  Saved →  {OUTPUT_FILE}")
    print("\n  ✓  price_forecast_model.py complete.\n")


if __name__ == "__main__":
    main()
