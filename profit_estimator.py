"""
MODULE 4 — profit_estimator.py
=================================
Crop Yield Research: A Robust Spatio-Temporal Multi-Modal Framework
for Early and Generalizable Crop Yield Prediction (Tamil Nadu Agriculture)

Purpose:
    Loads the 30-day Turmeric price forecast produced by
    price_forecast_model.py and estimates the net profit a farmer
    in Erode district would earn if they sold their produce on each
    of those 30 days, given typical cultivated-area and cost inputs.
    Identifies the optimal (best) and worst selling windows, prints
    a per-day profit table, and produces a profit-over-time plot.

Assumptions / Constants:
    assumed_yield_kg_per_ha   = 2,500 kg/ha  (typical Erode turmeric)
    area_ha                   = 1.0 ha        (unit farm size)
    cost_of_cultivation_per_ha= ₹ 85,000      (based on state averages)

Revenue formula:
    forecasted_revenue = forecasted_price (₹/quintal)
                         × (yield_kg_per_ha / 100)   [converts kg → quintal]
                         × area_ha

Profit formula:
    estimated_profit = forecasted_revenue − cost_of_cultivation_per_ha

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
# 1.  CULTIVATION CONSTANTS
# ─────────────────────────────────────────────
# These values reflect Tamil Nadu Department of Agriculture averages
# for Erode district Turmeric cultivation.
ASSUMED_YIELD_KG_PER_HA       = 2_500    # kg / hectare
AREA_HA                       = 1.0      # farm size in hectares
COST_OF_CULTIVATION_PER_HA    = 85_000   # ₹ / hectare

# 1 quintal = 100 kg  →  yield in quintals
YIELD_QUINTALS = (ASSUMED_YIELD_KG_PER_HA / 100) * AREA_HA


# ─────────────────────────────────────────────
# 2.  PROFIT COMPUTATION
# ─────────────────────────────────────────────
def compute_profit(df_forecast: pd.DataFrame) -> pd.DataFrame:
    """
    Adds revenue and profit columns to the forecast DataFrame.

    Parameters
    ----------
    df_forecast : pd.DataFrame
        Must contain columns: date, forecasted_price, lower_ci, upper_ci.

    Returns
    -------
    pd.DataFrame  With additional columns:
        yield_quintals          : fixed quintal yield per hectare
        forecasted_revenue      : ₹ revenue at forecasted price
        estimated_profit        : ₹ profit after cultivation cost
        pessimistic_profit      : ₹ profit at lower 95% CI price
        optimistic_profit       : ₹ profit at upper 95% CI price
    """
    df = df_forecast.copy()

    df["yield_quintals"]     = YIELD_QUINTALS
    df["forecasted_revenue"] = df["forecasted_price"] * YIELD_QUINTALS
    df["estimated_profit"]   = df["forecasted_revenue"] - COST_OF_CULTIVATION_PER_HA

    # Pessimistic (lower CI) and optimistic (upper CI) scenarios
    df["pessimistic_profit"] = (
        df["lower_ci"] * YIELD_QUINTALS - COST_OF_CULTIVATION_PER_HA
    )
    df["optimistic_profit"]  = (
        df["upper_ci"] * YIELD_QUINTALS - COST_OF_CULTIVATION_PER_HA
    )

    return df


# ─────────────────────────────────────────────
# 3.  PRINT PER-DAY TABLE
# ─────────────────────────────────────────────
def print_profit_table(df: pd.DataFrame) -> None:
    """
    Prints a formatted per-day profit table and highlights the best
    and worst selling days.

    Parameters
    ----------
    df : pd.DataFrame   Profit DataFrame returned by compute_profit().
    """
    display_cols = {
        "date"             : "Date",
        "forecasted_price" : "Price (₹/qtl)",
        "forecasted_revenue": "Revenue (₹)",
        "estimated_profit" : "Profit (₹)",
    }
    table = df[list(display_cols.keys())].copy()
    table.rename(columns=display_cols, inplace=True)
    table["Date"] = table["Date"].dt.strftime("%d %b %Y")
    table["Price (₹/qtl)"]  = table["Price (₹/qtl)"].round(2)
    table["Revenue (₹)"]    = table["Revenue (₹)"].round(2)
    table["Profit (₹)"]     = table["Profit (₹)"].round(2)

    print("\n  ── 30-Day Profit Forecast ─────────────────────────────────")
    print(f"  Yield Assumption   :  {ASSUMED_YIELD_KG_PER_HA:,} kg/ha  "
          f"= {YIELD_QUINTALS:.1f} quintals (for {AREA_HA} ha)")
    print(f"  Cultivation Cost   :  ₹ {COST_OF_CULTIVATION_PER_HA:,} / ha")
    print()
    print(table.to_string(index=False))

    # ── Best & worst days ─────────────────────────────────────────
    best_idx  = df["estimated_profit"].idxmax()
    worst_idx = df["estimated_profit"].idxmin()

    best_day  = df.loc[best_idx]
    worst_day = df.loc[worst_idx]

    print("\n  ─────────────────────────────────────────────────────────────")
    print("  📈  BEST  DAY  TO  SELL")
    print(f"      Date         :  {best_day['date'].strftime('%d %b %Y')}")
    print(f"      Price        :  ₹ {best_day['forecasted_price']:,.2f} / quintal")
    print(f"      Revenue      :  ₹ {best_day['forecasted_revenue']:,.2f}")
    print(f"      Net Profit   :  ₹ {best_day['estimated_profit']:,.2f}")

    print("\n  📉  WORST DAY  TO  SELL")
    print(f"      Date         :  {worst_day['date'].strftime('%d %b %Y')}")
    print(f"      Price        :  ₹ {worst_day['forecasted_price']:,.2f} / quintal")
    print(f"      Revenue      :  ₹ {worst_day['forecasted_revenue']:,.2f}")
    print(f"      Net Profit   :  ₹ {worst_day['estimated_profit']:,.2f}")
    print("  ─────────────────────────────────────────────────────────────")


# ─────────────────────────────────────────────
# 4.  VISUALISATION
# ─────────────────────────────────────────────
def plot_profit(df: pd.DataFrame) -> None:
    """
    Plots the estimated net profit over the 30-day forecast window,
    including pessimistic / optimistic scenario bands derived from
    the 95% ARIMA confidence interval.

    A horizontal break-even line (profit = 0) helps identify which
    days cover cultivation costs.

    Saved as 'turmeric_profit_forecast.png'.

    Parameters
    ----------
    df : pd.DataFrame   Profit DataFrame returned by compute_profit().
    """
    fig, ax = plt.subplots(figsize=(13, 6))

    # ── Point estimate ────────────────────────────────────────────
    ax.plot(
        df["date"], df["estimated_profit"],
        color="#2D6A4F", linewidth=2, marker="o", markersize=4,
        label="Estimated Net Profit (base forecast)",
    )

    # ── Uncertainty band (optimistic / pessimistic) ───────────────
    ax.fill_between(
        df["date"],
        df["pessimistic_profit"],
        df["optimistic_profit"],
        color="#52B788", alpha=0.2,
        label="Profit Range (95% CI)",
    )

    # ── Break-even line ───────────────────────────────────────────
    ax.axhline(y=0, color="red", linewidth=1.2, linestyle="--", label="Break-Even (₹0)")

    # ── Annotate best & worst days ────────────────────────────────
    best_row  = df.loc[df["estimated_profit"].idxmax()]
    worst_row = df.loc[df["estimated_profit"].idxmin()]

    ax.annotate(
        f"Best\n₹{best_row['estimated_profit']:,.0f}",
        xy=(best_row["date"], best_row["estimated_profit"]),
        xytext=(10, 15), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="#2D6A4F"),
        fontsize=9, color="#2D6A4F", fontweight="bold",
    )
    ax.annotate(
        f"Worst\n₹{worst_row['estimated_profit']:,.0f}",
        xy=(worst_row["date"], worst_row["estimated_profit"]),
        xytext=(10, -30), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="#E76F51"),
        fontsize=9, color="#E76F51", fontweight="bold",
    )

    # ── Formatting ────────────────────────────────────────────────
    ax.set_title(
        f"Estimated Farmer Net Profit — Next 30 Days  |  "
        f"Turmeric, Erode\n"
        f"Yield: {ASSUMED_YIELD_KG_PER_HA:,} kg/ha  |  "
        f"Area: {AREA_HA} ha  |  "
        f"Cultivation Cost: ₹{COST_OF_CULTIVATION_PER_HA:,}",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Estimated Net Profit (₹)", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.xticks(rotation=40, ha="right")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Add a subtle watermark / source note
    fig.text(0.99, 0.01, "Source: AGMARKNET | ARIMA(5,1,0) Forecast",
             ha="right", va="bottom", fontsize=8, color="grey", style="italic")

    fig.tight_layout()
    plt.savefig("turmeric_profit_forecast.png", dpi=150)
    plt.show()
    print("\n  Plot saved →  turmeric_profit_forecast.png")


# ─────────────────────────────────────────────
# 5.  MAIN
# ─────────────────────────────────────────────
def main():
    INPUT_FILE = "turmeric_price_forecast.csv"

    print(f"\n{'='*60}")
    print("  Turmeric Profit Estimator  —  30-Day Forecast Window")
    print(f"{'='*60}")

    # ── Load 30-day price forecast ────────────────────────────────
    df_forecast = pd.read_csv(INPUT_FILE, parse_dates=["date"])
    print(f"  Loaded {len(df_forecast)} forecast rows from  {INPUT_FILE}")

    # ── Compute profit ────────────────────────────────────────────
    df_profit = compute_profit(df_forecast)

    # ── Print detailed table + best/worst summary ─────────────────
    print_profit_table(df_profit)

    # ── Plot ──────────────────────────────────────────────────────
    plot_profit(df_profit)

    print("\n  ✓  profit_estimator.py complete.\n")


if __name__ == "__main__":
    main()
