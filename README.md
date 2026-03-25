# 🌾 Crop Yield Research

**A Robust Spatio-Temporal Multi-Modal Framework for Early and Generalizable Crop Yield Prediction — Tamil Nadu Agriculture**

This project builds an end-to-end data science pipeline for predicting crop yield, forecasting market prices, and estimating farmer profitability for **Turmeric (Erode)** and **Tapioca (Salem)** in Tamil Nadu, India.

---

## 📂 Project Structure

| File                       | Description                                                        |
|----------------------------|--------------------------------------------------------------------|
| `data_pipeline.py`        | Module 1 — Data loading, cleaning, and exploratory analysis        |
| `feature_engineering.py`  | Module 2 — Feature engineering (rolling stats, lags, seasonality)  |
| `price_forecast_model.py` | Module 3 — ARIMA time-series price forecasting                     |
| `profit_estimator.py`     | Module 4 — Farmer profit estimation from forecasted prices         |
| `yield_data.py`           | Yield dataset builder from data.gov.in API + TNAU published stats  |
| `weather_download.py`     | Historical weather data downloader (Open-Meteo Archive API)        |
| `soil_download.py`        | Soil property data downloader (ISRIC SoilGrids REST API)           |

---

## 🔬 Module Details

### Module 1 — `data_pipeline.py`

Loads raw **AGMARKNET** mandi price data for Turmeric (Erode) and Tapioca (Salem), applies a multi-step cleaning pipeline, and saves standardised outputs.

**Key operations:**
- Renames columns to `snake_case`
- Parses `arrival_date` as datetime; drops unparseable dates
- Removes null values and exact duplicates
- Derives `price_spread = modal_price − min_price`
- Plots a dual-axis modal price comparison chart
- Prints descriptive statistics (shape, date range, price distributions)

**Outputs:** `turmeric_clean.csv`, `tapioca_clean.csv`, `modal_price_comparison.png`

---

### Module 2 — `feature_engineering.py`

Reads the cleaned price CSVs and engineers predictive features for model training.

**Features created:**
| Feature            | Description                                      |
|--------------------|--------------------------------------------------|
| `rolling_avg_7d`   | 7-day rolling mean of modal price                |
| `rolling_avg_30d`  | 30-day rolling mean of modal price               |
| `month`            | Calendar month (1–12)                            |
| `season`           | Indian agricultural season (Kharif/Rabi/Summer)  |
| `price_lag_7`      | Modal price shifted back by 7 days               |
| `price_lag_30`     | Modal price shifted back by 30 days              |
| `price_volatility` | 30-day rolling standard deviation of modal price |

**Outputs:** `turmeric_features.csv`, `tapioca_features.csv`

---

### Module 3 — `price_forecast_model.py`

Fits an **ARIMA(5,1,0)** model on the Turmeric modal price series and generates a 30-day future price forecast.

**Pipeline:**
1. Loads feature-engineered data and aggregates to a clean daily series
2. Train/test split: train ≤ 2023 | test ≥ 2024
3. Walk-forward validation (re-fits at each test step — no look-ahead bias)
4. Evaluation metrics: **MAE** and **MAPE** on the 2024 hold-out set
5. Refits on full data and produces a 30-day forecast with 95% confidence intervals
6. Generates a two-panel visualisation (test evaluation + future forecast)

**Outputs:** `turmeric_price_forecast.csv`, `turmeric_price_forecast.png`

---

### Module 4 — `profit_estimator.py`

Takes the 30-day Turmeric price forecast and estimates the net profit a farmer in Erode would earn on each day.

**Assumptions:**
- Yield: 2,500 kg/ha (typical Erode turmeric)
- Farm size: 1.0 ha
- Cultivation cost: ₹85,000/ha

**Outputs calculated per day:**
- Forecasted revenue = price × yield (in quintals) × area
- Estimated profit = revenue − cultivation cost
- Pessimistic / optimistic profit (from 95% CI bounds)
- Best and worst selling days identified

**Outputs:** `turmeric_profit_forecast.png` (visualisation with break-even line and CI band)

---

### `yield_data.py`

Builds a curated crop yield dataset from two sources:

1. **data.gov.in API** — attempts to pull Tamil Nadu crop statistics
2. **TNAU published statistics** — hard-coded annual records for Turmeric (Erode, 2014–2024) and Tapioca (Salem, 2016–2024)

**Derived columns:** `yield_quintals_ha`, `estimated_revenue_per_ha`, `estimated_profit_per_ha`

**Output:** `yield_records.csv`

---

### `weather_download.py`

Downloads historical daily weather data from the **Open-Meteo Archive API** (free, no key required).

**Variables downloaded:**
- `temp_max_c`, `temp_min_c`, `rainfall_mm`
- `humidity_max_pct`, `humidity_min_pct`
- `evapotranspiration_mm`, `windspeed_max_kmh`

**Derived features:** `temp_range_c`, `humidity_avg_pct`

**Locations & periods:**
| Location | Period              | Output File          |
|----------|---------------------|----------------------|
| Erode    | 2015-01-01 → 2024-06-30 | `erode_weather.csv`  |
| Salem    | 2016-05-01 → 2026-01-01 | `salem_weather.csv`  |

---

### `soil_download.py`

Downloads soil property data from the **ISRIC SoilGrids REST API** (free, no key required).

**Properties retrieved:** `phh2o`, `soc`, `clay`, `sand`, `silt`, `nitrogen`, `bdod`, `cec`

**Depth layers:** 0–5 cm, 5–15 cm, 15–30 cm (with weighted 0–30 cm averages)

**Locations:** Erode (11.341°N, 77.717°E) and Salem (11.658°N, 78.146°E)

**Output:** `soil_properties.csv`

---

## 📊 Data Sources

| Source          | Data Type           | Access                        |
|-----------------|---------------------|-------------------------------|
| **AGMARKNET** (data.gov.in)  | Mandi commodity prices | Raw CSV download      |
| **data.gov.in API**          | Crop yield statistics  | REST API (key required) |
| **TNAU**                     | Published yield stats  | Hard-coded from reports |
| **Open-Meteo Archive API**   | Historical weather     | Free REST API           |
| **ISRIC SoilGrids API**      | Soil properties        | Free REST API           |

---

## 🚀 How to Run

### Prerequisites

```bash
pip install pandas matplotlib statsmodels requests openmeteo-requests requests-cache retry-requests numpy
```

### Execution Order

Run the modules in this order (each module depends on outputs from the previous):

```bash
# Step 1 — Clean raw AGMARKNET price data
python data_pipeline.py

# Step 2 — Engineer features from cleaned data
python feature_engineering.py

# Step 3 — Fit ARIMA model and generate 30-day price forecast
python price_forecast_model.py

# Step 4 — Estimate farmer profitability from forecasted prices
python profit_estimator.py

# Independent — Download yield, weather, and soil data (can run anytime)
python yield_data.py
python weather_download.py
python soil_download.py

# Independent — Build CNN-LSTM-ready aligned dataset + sliding-window features
python temporal_alignment.py
```

---

## 📁 Generated Outputs

| File                           | Description                                   |
|--------------------------------|-----------------------------------------------|
| `turmeric_clean.csv`          | Cleaned Turmeric (Erode) price data           |
| `tapioca_clean.csv`           | Cleaned Tapioca (Salem) price data            |
| `turmeric_features.csv`      | Feature-engineered Turmeric dataset           |
| `tapioca_features.csv`       | Feature-engineered Tapioca dataset            |
| `turmeric_price_forecast.csv`| 30-day ARIMA price forecast                   |
| `yield_records.csv`          | TNAU yield + revenue + profit dataset         |
| `erode_weather.csv`          | Historical daily weather for Erode            |
| `salem_weather.csv`          | Historical daily weather for Salem            |
| `erode_turmeric_aligned.csv`| Full daily aligned data for Erode/Turmeric  |
| `salem_tapioca_aligned.csv`| Full daily aligned data for Salem/Tapioca   |
| `feature_matrix.csv`        | 30-day sliding-window feature matrix + `split`|
| `modal_price_comparison.png` | Dual-axis price comparison chart              |
| `turmeric_price_forecast.png`| ARIMA forecast visualisation (2-panel)        |
| `turmeric_profit_forecast.png`| Profit estimation plot with CI band          |

---

## 🛠️ Tech Stack

- **Python 3.x**
- **pandas** — data manipulation and cleaning
- **matplotlib** — visualisation and charts
- **statsmodels** — ARIMA time-series modelling
- **requests** — API data downloads
- **openmeteo-requests** — Open-Meteo API client
- **numpy** — numerical computations

---

## 📝 Notes

- The ARIMA model is fitted with order **(5, 1, 0)** — 5 autoregressive lags, 1 differencing step, 0 moving-average terms.
- Walk-forward validation is used for realistic out-of-sample evaluation without look-ahead bias.
- Indian agricultural seasons are classified as: **Kharif** (Jun–Oct), **Rabi** (Nov–Feb), **Summer** (Mar–May).
- All prices are in **₹/quintal** as per AGMARKNET convention.
- Cultivation cost assumptions are based on Tamil Nadu Department of Agriculture averages.
