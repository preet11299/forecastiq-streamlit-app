"""
ForecastIQ - Core Engine (v0.4)
================================
Reusable demand forecasting logic for SKU-level planning.

Features:
- CSV loading and data quality checks
- Granularity detection
- Pareto analysis
- Demand pattern classification
- Five forecasting methods: Moving Average, Holt-Winters,
  Trend+Seasonal, Croston, and TSB (Teunter-Syntetos-Babai)
- Rolling backtesting with model confidence scoring
- Density filter (skip unforecastable SKUs)
- Auto weekly aggregation (daily → weekly)
- Location aggregation (sum → forecast → disaggregate)
- Industry-aware seasonality settings
- Business explanation layer
- Forecast generation for one SKU or many SKUs
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")


INDUSTRY_PROFILES: Dict[str, Dict[str, str]] = {
    "General": {
        "seasonality_type": "additive",
        "preferred_model": "Holt-Winters",
        "description": "Neutral profile for general demand patterns.",
    },
    "Retail": {
        "seasonality_type": "multiplicative",
        "preferred_model": "Holt-Winters",
        "description": "Retail demand often scales with the base level and reacts to promotions.",
    },
    "Healthcare": {
        "seasonality_type": "additive",
        "preferred_model": "Holt-Winters",
        "description": "Healthcare demand is often steadier with recurring seasonal effects.",
    },
    "Auto": {
        "seasonality_type": "additive",
        "preferred_model": "Trend+Seasonal",
        "description": "Auto demand can be trend-led with periodic market effects.",
    },
    "Semi": {
        "seasonality_type": "multiplicative",
        "preferred_model": "Trend+Seasonal",
        "description": "Semiconductor demand is cyclical and can move with strong amplitude changes.",
    },
}


# -----------------------------------------------------------------------------
# DATA HELPERS
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# COLUMN ALIAS MAPS — name-based detection, not positional
# -----------------------------------------------------------------------------
_DATE_ALIASES     = {"date", "week", "order_date", "trans_date", "period", "time", "day", "month"}
_SKU_ALIASES      = {"sku", "item", "product", "product_code", "meal_id", "item_id", "product_id", "article"}
_QTY_ALIASES      = {"quantity", "qty", "sales", "num_orders", "order_demand", "demand", "units", "volume", "orders"}
_CATEGORY_ALIASES = {"category", "product_category", "cat", "segment", "family", "group", "type", "class"}
_LOCATION_ALIASES = {"location", "store", "warehouse", "site", "center_id", "region", "branch", "whse", "facility"}
_PRICE_ALIASES    = {"price", "unit_price", "checkout_price", "base_price", "cost", "rate"}


def _detect_column(cols_lower: List[str], aliases: set) -> Optional[int]:
    """Return index of first column whose lowercase name is in aliases, or None."""
    for i, c in enumerate(cols_lower):
        if c in aliases:
            return i
    return None


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map raw CSV columns to standard names using alias detection.
    Required: Date, SKU, Quantity.
    Optional: Category, Location, Price.
    Raises clear ValueError messages when required columns are missing.
    """
    df = df.copy()
    cols_lower = [c.strip().lower() for c in df.columns]

    # ── Detect required columns ──────────────────────────────────
    date_idx = _detect_column(cols_lower, _DATE_ALIASES)
    sku_idx  = _detect_column(cols_lower, _SKU_ALIASES)
    qty_idx  = _detect_column(cols_lower, _QTY_ALIASES)

    if date_idx is None:
        raise ValueError(
            "Could not find a Date column. "
            "Common names we look for: date, week, order_date, trans_date, period. "
            "Please rename your column to one of these, or use the column mapper."
        )
    if sku_idx is None:
        raise ValueError(
            "Could not find a Product/SKU column. "
            "Common names we look for: sku, item, product_code, meal_id, product. "
            "Please rename your column to one of these, or use the column mapper."
        )
    if qty_idx is None:
        raise ValueError(
            "Could not find a Quantity column. "
            "Common names we look for: quantity, qty, sales, num_orders, order_demand, demand. "
            "Please rename your column to one of these, or use the column mapper."
        )

    # ── Detect optional columns (skip if same index as required) ─
    used = {date_idx, sku_idx, qty_idx}

    cat_idx   = _detect_column(cols_lower, _CATEGORY_ALIASES)
    loc_idx   = _detect_column(cols_lower, _LOCATION_ALIASES)
    price_idx = _detect_column(cols_lower, _PRICE_ALIASES)

    if cat_idx   in used: cat_idx   = None
    if loc_idx   in used: loc_idx   = None
    if price_idx in used: price_idx = None

    # ── Build clean dataframe ────────────────────────────────────
    out = pd.DataFrame()
    out["Date"]     = df.iloc[:, date_idx]
    out["SKU"]      = df.iloc[:, sku_idx]
    out["Quantity"] = df.iloc[:, qty_idx]

    if cat_idx is not None:
        out["Category"] = df.iloc[:, cat_idx].astype(str).str.strip()
    if loc_idx is not None:
        out["Location"] = df.iloc[:, loc_idx].astype(str).str.strip()
    if price_idx is not None:
        out["Price"] = df.iloc[:, price_idx]

    # ── Type coercion ────────────────────────────────────────────
    before = len(out)

    # Date: try ISO parse first, then dayfirst fallback for DD/MM/YYYY
    # infer_datetime_format is deprecated in pandas 2.0+ and is now the default — omit it
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    ambiguous_mask = out["Date"].isna() & out.iloc[:, 0].notna() if False else (
        out["Date"].isna() & out["Date"].astype(str).ne("NaT")
    )
    # Re-attempt with dayfirst=True on rows that failed ISO parse
    raw_dates = out["Date"].copy()
    failed_mask = raw_dates.isna()
    if failed_mask.any():
        # Pull original string values for failed rows from the input df
        original_col = df.iloc[:len(out), date_idx] if date_idx < len(df.columns) else None
        if original_col is not None:
            fallback = pd.to_datetime(
                original_col.iloc[out.index[failed_mask]],
                errors="coerce", dayfirst=True,
            )
            out.loc[failed_mask, "Date"] = fallback.values

    out["SKU"]      = out["SKU"].astype(str).str.strip()

    # Strip common formatting from numeric columns before coercion
    for _col in ["Quantity"]:
        out[_col] = (
            out[_col].astype(str)
            .str.replace(r"[\$€£,\s]", "", regex=True)
            .str.strip()
        )
    out["Quantity"] = pd.to_numeric(out["Quantity"], errors="coerce")

    if "Price" in out.columns:
        out["Price"] = (
            out["Price"].astype(str)
            .str.replace(r"[\$€£,\s]", "", regex=True)
            .str.strip()
        )
        out["Price"] = pd.to_numeric(out["Price"], errors="coerce")

    out = out.dropna(subset=["Date", "SKU", "Quantity"]).reset_index(drop=True)
    dropped = before - len(out)

    if dropped > 0:
        warnings.warn(
            f"{dropped} rows were removed during parsing "
            f"(unreadable dates or non-numeric Quantity values).",
            UserWarning,
        )
    if out.empty:
        raise ValueError(
            "No valid rows remain after parsing. "
            "Check that your Date column is formatted correctly (e.g. 2024-01-07) "
            "and your Quantity column contains numeric values."
        )

    return out


def generate_demo_data(
    num_skus: int = 3,
    periods: int = 104,
    granularity: str = "Weekly",
    seed: int = 7,
) -> pd.DataFrame:
    """
    Create a synthetic dataset with Date, SKU, Quantity, Category, Location
    so the built-in sample exercises all dashboard features including filters.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2023-01-01")

    if granularity == "Daily":
        dates = pd.date_range(start=start, periods=periods, freq="D")
    elif granularity == "Monthly":
        dates = pd.date_range(start=start, periods=periods, freq="MS")
    else:
        dates = pd.date_range(start=start, periods=periods, freq="W")

    categories = ["Electronics", "Apparel", "Home & Garden", "Sporting Goods", "Automotive"]
    locations  = ["Store_A", "Store_B", "Store_C", "Warehouse_1", "Warehouse_2"]

    rows = []
    for i in range(num_skus):
        sku      = f"SKU-{i+1:03d}"
        category = categories[i % len(categories)]
        location = locations[i % len(locations)]
        base     = 80 + i * 35
        trend    = 0.8 + i * 0.35
        seasonal_strength = 12 + i * 4

        for t, date in enumerate(dates):
            seasonal = seasonal_strength * np.sin(2 * np.pi * t / max(12, min(len(dates), 52)))
            noise    = rng.normal(0, 5 + i * 1.5)
            qty      = max(base + trend * t + seasonal + noise, 0)
            rows.append((date, sku, round(qty, 0), category, location))

    return pd.DataFrame(rows, columns=["Date", "SKU", "Quantity", "Category", "Location"])


def load_data(filepath: str, silent: bool = False) -> pd.DataFrame:
    """Load a CSV file with Date, SKU, Quantity columns."""
    if not silent:
        print("=" * 60)
        print("STEP 1: LOADING DATA")
        print("=" * 60)

    df = pd.read_csv(filepath)
    df = _normalize_columns(df)

    if not silent:
        print(f"  Loaded {len(df)} rows")
        print(f"  Found {df['SKU'].nunique()} unique SKUs")
        print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        print()

    return df


# -----------------------------------------------------------------------------
# BASIC ANALYTICS
# -----------------------------------------------------------------------------

def detect_granularity(df: pd.DataFrame, silent: bool = False) -> str:
    """Infer daily / weekly / monthly cadence from date gaps across SKUs."""
    if not silent:
        print("=" * 60)
        print("STEP 2: DETECTING GRANULARITY")
        print("=" * 60)

    if df.empty or {"SKU", "Date"}.difference(df.columns):
        return "Weekly"

    sample_skus = df["SKU"].value_counts().head(5).index.tolist()
    medians: List[float] = []

    for sku in sample_skus:
        sku_dates = df.loc[df["SKU"] == sku, "Date"].sort_values()
        diffs = sku_dates.diff().dropna().dt.days
        if len(diffs):
            medians.append(float(diffs.median()))

    median_diff = float(np.median(medians)) if medians else 7.0

    if median_diff <= 2:
        granularity = "Daily"
    elif median_diff <= 10:
        granularity = "Weekly"
    else:
        granularity = "Monthly"

    if not silent:
        print(f"  Median gap between dates: {median_diff:.1f} days")
        print(f"  Detected granularity: {granularity}")
        print()

    return granularity


def data_quality_check(df: pd.DataFrame, silent: bool = False) -> Dict[str, object]:
    """Run basic quality checks and return a report dictionary."""
    if not silent:
        print("=" * 60)
        print("STEP 3: DATA QUALITY CHECK")
        print("=" * 60)

    report: Dict[str, object] = {}
    report["total_rows"] = int(len(df))
    report["unique_skus"] = int(df["SKU"].nunique()) if not df.empty else 0
    report["date_range"] = (
        f"{df['Date'].min().date()} to {df['Date'].max().date()}" if not df.empty else "n/a"
    )
    report["missing_values"] = int(df.isnull().sum().sum())
    report["zero_qty_rows"] = int((df["Quantity"] == 0).sum()) if "Quantity" in df else 0
    report["negative_qty"] = int((df["Quantity"] < 0).sum()) if "Quantity" in df else 0
    report["duplicate_rows"] = int(df.duplicated().sum())

    if not df.empty:
        history_days = (df["Date"].max() - df["Date"].min()).days
        report["history_months"] = round(history_days / 30.44, 1)
    else:
        report["history_months"] = 0.0

    score = 100
    if report["missing_values"] > 0:
        score -= 15
    if report["negative_qty"] > 0:
        score -= 20
    if report["duplicate_rows"] > 0:
        score -= 10
    if report["zero_qty_rows"] > len(df) * 0.1:
        score -= 10
    if report["history_months"] < 12:
        score -= 15
    report["quality_score"] = max(int(score), 0)

    if report["quality_score"] >= 80:
        status = "GOOD"
    elif report["quality_score"] >= 60:
        status = "FAIR"
    else:
        status = "POOR"
    report["status"] = status

    if not silent:
        print(f"  Total rows:        {report['total_rows']}")
        print(f"  Unique SKUs:       {report['unique_skus']}")
        print(f"  Date range:        {report['date_range']}")
        print(f"  History depth:     {report['history_months']} months")
        print(f"  Missing values:    {report['missing_values']}")
        print(f"  Zero-quantity:     {report['zero_qty_rows']}")
        print(f"  Negative quantity: {report['negative_qty']}")
        print(f"  Duplicate rows:    {report['duplicate_rows']}")
        print("  -----------------------------")
        print(f"  Quality Score:     {report['quality_score']}% ({status})")
        print()

    return report


def pareto_analysis(df: pd.DataFrame, top_n: int = 5, silent: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    """Rank SKUs by revenue (if Price column exists) or by total quantity."""
    has_price = "Price" in df.columns and df["Price"].notna().any()
    rank_label = "Revenue" if has_price else "Volume"

    if not silent:
        print("=" * 60)
        print(f"STEP 4: PARETO ANALYSIS (Top SKUs by {rank_label})")
        print("=" * 60)

    if has_price:
        tmp = df.copy()
        tmp["_Revenue"] = tmp["Quantity"] * tmp["Price"]
        sku_totals = (
            tmp.groupby("SKU", as_index=False)["_Revenue"]
            .sum()
            .sort_values("_Revenue", ascending=False)
        )
        sku_totals.columns = ["SKU", "Total_Quantity"]
    else:
        sku_totals = (
            df.groupby("SKU", as_index=False)["Quantity"].sum().sort_values("Quantity", ascending=False)
        )
        sku_totals.columns = ["SKU", "Total_Quantity"]

    grand_total = float(sku_totals["Total_Quantity"].sum()) if len(sku_totals) else 1.0
    sku_totals["Percentage"] = (sku_totals["Total_Quantity"] / grand_total * 100).round(1)
    sku_totals["Cumulative_%"] = sku_totals["Percentage"].cumsum().round(1)
    sku_totals["Rank"] = range(1, len(sku_totals) + 1)
    sku_totals["Pareto_Mode"] = rank_label

    if not silent:
        print(f"  {'Rank':<6} {'SKU':<12} {'Total':<12} {'%':<8} {'Cumulative %':<14}")
        print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*8} {'-'*14}")
        for _, row in sku_totals.iterrows():
            marker = " <- 80%" if row["Cumulative_%"] >= 80 and (row["Cumulative_%"] - row["Percentage"]) < 80 else ""
            print(
                f"  {int(row['Rank']):<6} {row['SKU']:<12} {int(row['Total_Quantity']):<12} "
                f"{row['Percentage']:<8} {row['Cumulative_%']:<14}{marker}"
            )
        print()
        print(f"  Top {top_n} SKUs selected for forecasting: {', '.join(sku_totals.head(top_n)['SKU'].tolist())}")
        print()

    top_skus = sku_totals.head(top_n)["SKU"].tolist()
    return sku_totals, top_skus


def classify_demand_pattern(
    series: Iterable[float],
    zero_threshold: float = 0.5,
    cv_threshold: float = 0.8,
) -> Dict[str, object]:
    """Return a simple demand pattern label for a SKU."""
    s = pd.Series(series).dropna().astype(float)
    if len(s) == 0:
        return {
            "pattern": "Unknown",
            "zero_share": 0.0,
            "cv": 0.0,
            "non_zero_share": 0.0,
            "is_intermittent": False,
        }

    zero_share = float((s == 0).mean())
    non_zero = s[s > 0]
    non_zero_share = float(len(non_zero) / len(s))
    mean_val = float(s.mean())
    std_val = float(s.std(ddof=0))
    cv = float(std_val / mean_val) if mean_val > 0 else float("inf")

    if zero_share >= zero_threshold or non_zero_share <= 0.5:
        pattern = "Intermittent"
    elif cv >= cv_threshold:
        pattern = "Volatile"
    elif cv <= 0.4:
        pattern = "Stable"
    else:
        pattern = "Mixed"

    return {
        "pattern": pattern,
        "zero_share": round(zero_share, 3),
        "cv": round(cv, 3) if np.isfinite(cv) else cv,
        "non_zero_share": round(non_zero_share, 3),
        "is_intermittent": pattern == "Intermittent",
    }


def get_industry_profile(industry: str) -> Dict[str, str]:
    return INDUSTRY_PROFILES.get(industry, INDUSTRY_PROFILES["General"])


# -----------------------------------------------------------------------------
# FORECASTING HELPERS
# -----------------------------------------------------------------------------

def _coerce_series(series: Iterable[float]) -> pd.Series:
    return pd.Series(series).dropna().astype(float).reset_index(drop=True)


def seasonal_period_from_granularity(granularity: str) -> int:
    if granularity == "Weekly":
        return 52
    if granularity == "Monthly":
        return 12
    return 7


def horizon_unit_label(granularity: str) -> str:
    return {"Daily": "days", "Weekly": "weeks", "Monthly": "months"}.get(granularity, "periods")


def future_dates(last_date: pd.Timestamp, periods: int, granularity: str) -> pd.DatetimeIndex:
    last_date = pd.Timestamp(last_date)
    dates: List[pd.Timestamp] = []
    for i in range(1, periods + 1):
        if granularity == "Weekly":
            dates.append(last_date + pd.DateOffset(weeks=i))
        elif granularity == "Monthly":
            dates.append(last_date + pd.DateOffset(months=i))
        else:
            dates.append(last_date + pd.DateOffset(days=i))
    return pd.DatetimeIndex(dates)


# -----------------------------------------------------------------------------
# FORECASTING METHODS
# -----------------------------------------------------------------------------

def moving_average_forecast(series: Iterable[float], periods: int = 12, window: int = 4) -> np.ndarray:
    """Trend-adjusted moving average with a small safety fallback."""
    s = _coerce_series(series)
    if len(s) == 0:
        return np.zeros(periods)
    if len(s) == 1:
        return np.repeat(float(s.iloc[-1]), periods)

    w = min(window, len(s))
    last_ma = float(s.tail(w).mean())
    trend = (float(s.iloc[-1]) - float(s.iloc[-w])) / max(w, 1) if len(s) > w else 0.0
    forecast = [max(last_ma + trend * (i + 1), 0) for i in range(periods)]
    return np.array(forecast)


def holt_winters_forecast(
    series: Iterable[float],
    periods: int = 12,
    seasonal_periods: Optional[int] = None,
    seasonality_type: str = "additive",
) -> np.ndarray:
    """Holt-Winters exponential smoothing with fallback."""
    s = _coerce_series(series)
    if len(s) < 3:
        return moving_average_forecast(s, periods)

    seasonal_type = "mul" if seasonality_type == "multiplicative" else "add"
    if seasonal_type == "mul" and (s <= 0).any():
        seasonal_type = "add"

    try:
        if seasonal_periods and len(s) >= max(2 * seasonal_periods, 6):
            model = ExponentialSmoothing(
                s,
                trend="add",
                seasonal=seasonal_type,
                seasonal_periods=seasonal_periods,
                initialization_method="estimated",
            )
        else:
            model = ExponentialSmoothing(
                s,
                trend="add",
                seasonal=None,
                initialization_method="estimated",
            )

        fitted = model.fit(optimized=True)
        forecast = fitted.forecast(periods)
        return np.maximum(np.asarray(forecast), 0)
    except Exception:
        return moving_average_forecast(s, periods)


def trend_seasonal_forecast(series: Iterable[float], periods: int = 12, granularity: str = "Weekly") -> np.ndarray:
    """Simple quadratic trend plus repeating seasonal pattern."""
    s = _coerce_series(series)
    if len(s) < 4:
        return moving_average_forecast(s, periods)

    try:
        x = np.arange(len(s))
        degree = 2 if len(s) >= 5 else 1
        coefficients = np.polyfit(x, s.values, degree)
        trend_function = np.poly1d(coefficients)
        trend_values = trend_function(x)
        residual = s.values - trend_values

        if granularity == "Weekly" and len(s) >= 52:
            seasonal_period = 52
        elif granularity == "Monthly" and len(s) >= 12:
            seasonal_period = 12
        elif granularity == "Daily" and len(s) >= 7:
            seasonal_period = 7
        else:
            seasonal_period = max(min(len(s) // 2, 12), 1)

        seasonal = np.zeros(seasonal_period)
        counts = np.zeros(seasonal_period)
        for i, value in enumerate(residual):
            idx = i % seasonal_period
            seasonal[idx] += value
            counts[idx] += 1
        seasonal = seasonal / np.maximum(counts, 1)

        future_x = np.arange(len(s), len(s) + periods)
        future_trend = trend_function(future_x)
        future_seasonal = np.array([
            seasonal[i % seasonal_period] for i in range(len(s), len(s) + periods)
        ])
        forecast = future_trend + future_seasonal
        return np.maximum(forecast, 0)
    except Exception:
        return moving_average_forecast(s, periods)



def croston_forecast(
    series: Iterable[float],
    periods: int = 12,
    alpha: float = 0.1,
) -> np.ndarray:
    """Croston forecast for intermittent demand."""
    s = _coerce_series(series)
    if len(s) == 0:
        return np.zeros(periods)
    if len(s) == 1:
        return np.repeat(float(s.iloc[-1]), periods)

    values = s.values.astype(float)
    non_zero_idx = np.flatnonzero(values > 0)
    if len(non_zero_idx) == 0:
        return np.zeros(periods)
    if len(non_zero_idx) == 1:
        return np.repeat(float(values[non_zero_idx[0]]), periods)

    first = int(non_zero_idx[0])
    z = float(values[first])
    p = float(first + 1)
    last_non_zero = first

    for idx in non_zero_idx[1:]:
        x = float(values[idx])
        interval = float(idx - last_non_zero)
        z = z + alpha * (x - z)
        p = p + alpha * (interval - p)
        p = max(p, 1e-6)
        last_non_zero = int(idx)

    demand_rate = max(z / p, 0.0)
    return np.repeat(demand_rate, periods)


def tsb_forecast(
    series: Iterable[float],
    periods: int = 12,
    alpha: float = 0.1,
    beta: float = 0.1,
) -> np.ndarray:
    """
    Teunter-Syntetos-Babai (TSB) forecast for intermittent demand.
    Smooths demand size and demand *probability* (not interval),
    adapting faster when a dormant SKU reactivates or an active one dies.
    """
    s = _coerce_series(series)
    if len(s) == 0:
        return np.zeros(periods)
    if len(s) == 1:
        return np.repeat(float(s.iloc[-1]), periods)

    values = s.values.astype(float)
    non_zero_idx = np.flatnonzero(values > 0)
    if len(non_zero_idx) == 0:
        return np.zeros(periods)
    if len(non_zero_idx) == 1:
        return np.repeat(float(values[non_zero_idx[0]]), periods)

    # Initialize
    z = float(values[non_zero_idx[0]])          # demand size estimate
    p = float(len(non_zero_idx)) / len(values)  # demand probability estimate

    for t in range(1, len(values)):
        if values[t] > 0:
            z = z + alpha * (values[t] - z)
            p = p + beta * (1.0 - p)
        else:
            p = p * (1.0 - beta)

    demand_rate = max(z * p, 0.0)
    return np.repeat(demand_rate, periods)


# --- Density filter helper ---------------------------------------------------

MIN_DENSITY_RECORDS = 30  # SKUs below this are skipped as unforecastable


def _check_density(series_len: int, sku: str) -> Optional[Dict[str, object]]:
    """Return a 'skipped' result dict if series is too short, else None."""
    if series_len < MIN_DENSITY_RECORDS:
        return {
            "sku": sku,
            "skipped": True,
            "skip_reason": f"Insufficient history ({series_len} records, need {MIN_DENSITY_RECORDS}+)",
        }
    return None


# --- Model confidence scoring ------------------------------------------------

def model_confidence(accuracy_results: List[Dict[str, object]], best_idx: int) -> Dict[str, object]:
    """
    Score confidence of the best model based on MAPE level and fold consistency.
    Returns {'score': 0-100, 'label': High/Medium/Low, 'detail': str}.
    """
    if not accuracy_results:
        return {"score": 0, "label": "Unknown", "detail": "No backtest data"}

    best = accuracy_results[best_idx]
    mape = best.get("MAPE_%", float("inf"))
    folds = best.get("Folds", 1)

    # Base score from MAPE level
    if mape <= 15:
        base = 95
    elif mape <= 30:
        base = 80
    elif mape <= 50:
        base = 65
    elif mape <= 100:
        base = 45
    else:
        base = max(20, int(100 - mape))

    # Bonus for multi-fold stability
    fold_bonus = min(10, (folds - 1) * 3) if folds > 1 else 0
    score = min(100, max(0, base + fold_bonus))

    if score >= 75:
        label = "High"
    elif score >= 50:
        label = "Medium"
    else:
        label = "Low"

    detail = f"{best['Method']}: {score}% confidence ({folds} fold{'s' if folds != 1 else ''})"
    return {"score": score, "label": label, "detail": detail}


# --- Auto weekly aggregation helper ------------------------------------------

def _auto_aggregate_weekly(sku_df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily data to weekly sums. Expects Date + Quantity columns."""
    agg = (
        sku_df.set_index("Date")
        .resample("W")["Quantity"]
        .sum()
        .reset_index()
    )
    # Carry forward non-numeric columns from first row
    for col in sku_df.columns:
        if col not in ("Date", "Quantity") and col not in agg.columns:
            agg[col] = sku_df[col].iloc[0]
    return agg


# -----------------------------------------------------------------------------
# ACCURACY + BACKTESTING
# -----------------------------------------------------------------------------

def calculate_accuracy(actual: Iterable[float], predicted: Iterable[float], method_name: str) -> Dict[str, object]:
    """
    Calculate MAE, MAPE, WAPE, and RMSE.

    MAPE_% is now computed as WAPE (Weighted Absolute Percentage Error)
    which is robust to zero-actual periods: sum(|error|) / sum(|actual|).
    Traditional MAPE is kept as MAPE_traditional_% for reference.
    """
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)

    n = min(len(actual), len(predicted))
    if n == 0:
        return {"Method": method_name, "MAE": np.nan, "MAPE_%": np.nan, "RMSE": np.nan}

    actual = actual[-n:]
    predicted = predicted[-n:]

    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    # WAPE: robust to zeros, industry-standard for intermittent demand
    sum_actual = np.sum(np.abs(actual))
    wape = (np.sum(np.abs(actual - predicted)) / max(sum_actual, 1.0)) * 100

    return {
        "Method": method_name,
        "MAE": round(float(mae), 2),
        "MAPE_%": round(float(wape), 2),
        "RMSE": round(float(rmse), 2),
    }


def _aggregate_accuracy(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    if not records:
        return []

    by_method: Dict[str, List[Dict[str, object]]] = {}
    for rec in records:
        by_method.setdefault(rec["Method"], []).append(rec)

    averaged: List[Dict[str, object]] = []
    for method, rows in by_method.items():
        averaged.append(
            {
                "Method": method,
                "MAE": round(float(np.mean([r["MAE"] for r in rows])), 2),
                "MAPE_%": round(float(np.mean([r["MAPE_%"] for r in rows])), 2),
                "RMSE": round(float(np.mean([r["RMSE"] for r in rows])), 2),
                "Folds": len(rows),
            }
        )
    return averaged



# -----------------------------------------------------------------------------
# ACCURACY + BACKTESTING
# -----------------------------------------------------------------------------

def rolling_backtest(
    series: Iterable[float],
    granularity: str,
    forecast_horizon: int = 12,
    n_splits: int = 4,
    seasonality_type: str = "additive",
) -> List[Dict[str, object]]:
    """Evaluate all methods across multiple expanding windows."""
    s = _coerce_series(series)
    if len(s) < 4:
        return []

    holdout = max(2, min(forecast_horizon, max(2, len(s) // 5)))
    seasonal_p = seasonal_period_from_granularity(granularity)

    if len(s) <= holdout * 2:
        train = s.iloc[:-holdout]
        test = s.iloc[-holdout:]
        return [
            calculate_accuracy(test.values, moving_average_forecast(train, holdout), "Moving Average"),
            calculate_accuracy(
                test.values,
                holt_winters_forecast(
                    train,
                    holdout,
                    seasonal_periods=min(seasonal_p, max(2, len(train) // 2)),
                    seasonality_type=seasonality_type,
                ),
                "Holt-Winters",
            ),
            calculate_accuracy(test.values, trend_seasonal_forecast(train, holdout, granularity), "Trend+Seasonal"),
            calculate_accuracy(test.values, croston_forecast(train, holdout), "Croston"),
            calculate_accuracy(test.values, tsb_forecast(train, holdout), "TSB"),
        ]

    max_splits = min(n_splits, max(1, (len(s) // holdout) - 1))
    start = max(3, len(s) - holdout * (max_splits + 1))
    raw_records: List[Dict[str, object]] = []

    for split_idx in range(max_splits):
        train_end = start + split_idx * holdout
        test_start = train_end
        test_end = min(test_start + holdout, len(s))

        train = s.iloc[:train_end]
        test = s.iloc[test_start:test_end]
        if len(train) < 3 or len(test) < 2:
            continue

        bt_ma = moving_average_forecast(train, len(test))
        bt_hw = holt_winters_forecast(
            train,
            len(test),
            seasonal_periods=min(seasonal_p, max(2, len(train) // 2)),
            seasonality_type=seasonality_type,
        )
        bt_ts = trend_seasonal_forecast(train, len(test), granularity)
        bt_cr = croston_forecast(train, len(test))
        bt_tsb = tsb_forecast(train, len(test))

        raw_records.extend(
            [
                calculate_accuracy(test.values, bt_ma, "Moving Average"),
                calculate_accuracy(test.values, bt_hw, "Holt-Winters"),
                calculate_accuracy(test.values, bt_ts, "Trend+Seasonal"),
                calculate_accuracy(test.values, bt_cr, "Croston"),
                calculate_accuracy(test.values, bt_tsb, "TSB"),
            ]
        )

    return _aggregate_accuracy(raw_records)


# -----------------------------------------------------------------------------
# FORECAST PIPELINE
# -----------------------------------------------------------------------------

def generate_business_explanation(
    demand_profile: Dict[str, object],
    best_method: str,
    accuracy_results: List[Dict[str, object]],
    industry_profile: Dict[str, str],
    granularity: str,
    forecast_horizon: int,
) -> Dict[str, object]:
    """Create a short business-friendly explanation for the forecast."""
    pattern = demand_profile.get("pattern", "Unknown")
    zero_share = float(demand_profile.get("zero_share", 0.0))
    cv = demand_profile.get("cv", 0.0)

    if accuracy_results:
        best_row = min(accuracy_results, key=lambda r: r.get("MAPE_%", float("inf")))
        best_mape = best_row.get("MAPE_%", float("nan"))
    else:
        best_mape = float("nan")

    if pattern == "Intermittent":
        headline = "Intermittent demand detected"
        recommendation = (
            f"{best_method} is the best statistical fit, but this SKU has many zero periods "
            f"({zero_share:.0%} zero demand). Use the forecast as a directional signal, "
            "and keep inventory policy conservative."
        )
        planning_action = "Plan smaller replenishment batches and monitor stockouts closely."
        risk_note = "Intermittent demand can swing suddenly, so forecast error may stay elevated."
        confidence = "Low to Medium"
    elif pattern == "Stable":
        headline = "Stable demand profile"
        recommendation = (
            f"Demand is relatively steady (CV {cv}). {best_method} fits well, so the forecast "
            "is suitable for routine replenishment planning."
        )
        planning_action = "Use the forecast directly for baseline ordering and service-level planning."
        risk_note = "Watch for rare spikes from promotions or supply disruptions."
        confidence = "High"
    elif pattern == "Volatile":
        headline = "Volatile demand profile"
        recommendation = (
            f"Demand is more variable than average, so {best_method} is a useful guide but not a "
            "perfect prediction. Keep some planner judgment in the loop."
        )
        planning_action = "Use the forecast with a buffer and review exceptions frequently."
        risk_note = "Large swings can make the model less stable than the MAPE suggests."
        confidence = "Medium"
    else:
        headline = "Mixed demand pattern"
        recommendation = (
            f"{best_method} performed best on history, but the SKU has mixed behavior. "
            "Treat the forecast as an operational input, not a fixed answer."
        )
        planning_action = "Combine the forecast with planner review and inventory rules."
        risk_note = "Mixed patterns can switch between stable and volatile periods."
        confidence = "Medium"

    industry_note = (
        f"Industry profile: {industry_profile.get('description', 'No industry note available.')}"
    )
    cadence_note = f"Forecast horizon is {forecast_horizon} {horizon_unit_label(granularity)} ahead."
    fit_note = f"Backtesting favoured {best_method} with the lowest MAPE among the tested methods."

    return {
        "headline": headline,
        "recommendation": recommendation,
        "planning_action": planning_action,
        "risk_note": risk_note,
        "confidence": confidence,
        "industry_note": industry_note,
        "cadence_note": cadence_note,
        "fit_note": fit_note,
        "best_mape": best_mape,
    }


def forecast_sku(
    df: pd.DataFrame,
    sku: str,
    granularity: str,
    forecast_horizon: int = 12,
    industry: str = "General",
    n_splits: int = 4,
    silent: bool = False,
    location: Optional[str] = None,
) -> Dict[str, object]:
    """
    Forecast a single SKU (optionally scoped to a specific Location).
    When location is specified, only that location's data is used.

    v0.4 additions:
      - Density filter: skip SKUs with < MIN_DENSITY_RECORDS data points
      - Auto weekly aggregation: daily data is resampled to weekly before forecasting
      - TSB model: 5th forecasting method alongside the original 4
      - Model confidence scoring attached to results
    """
    mask = df["SKU"] == sku
    if location is not None and "Location" in df.columns:
        mask = mask & (df["Location"] == location)
    sku_df = df[mask].sort_values("Date").reset_index(drop=True)

    if len(sku_df) == 0:
        loc_str = f" / {location}" if location else ""
        raise ValueError(f"No data available for SKU {sku}{loc_str}")

    # ── Density filter ────────────────────────────────────────────
    skip = _check_density(len(sku_df), sku)
    if skip is not None:
        return skip

    # ── Pre-aggregate: collapse multiple rows per date ────────────
    # When no location filter is set, a SKU may have multiple rows per
    # date (one per store).  Sum them so the time series has one value
    # per date before any further resampling.
    if sku_df["Date"].duplicated().any():
        sku_df = (
            sku_df.groupby("Date", as_index=False)["Quantity"]
            .sum()
            .sort_values("Date")
            .reset_index(drop=True)
        )
        # Carry forward metadata columns from original
        for col in df.columns:
            if col not in ("Date", "Quantity") and col not in sku_df.columns:
                orig = df.loc[df["SKU"] == sku, col]
                if not orig.empty:
                    sku_df[col] = orig.iloc[0]

    # ── Auto weekly aggregation (daily → weekly) ──────────────────
    effective_granularity = granularity
    if granularity == "Daily":
        sku_df = _auto_aggregate_weekly(sku_df)
        effective_granularity = "Weekly"

    series = sku_df["Quantity"].astype(float)
    dates = sku_df["Date"]

    demand_profile = classify_demand_pattern(series)
    industry_profile = get_industry_profile(industry)
    seasonality_type = industry_profile["seasonality_type"]
    seasonal_p = seasonal_period_from_granularity(effective_granularity)

    if not silent:
        print("=" * 60)
        print(f"FORECASTING - {sku}")
        print("=" * 60)
        print(f"  Data points: {len(series)}")
        if granularity == "Daily":
            print(f"  Auto-aggregated: Daily → Weekly ({len(series)} weeks)")
        print(f"  Date range:  {dates.iloc[0].date()} to {dates.iloc[-1].date()}")
        print(f"  Avg quantity: {series.mean():.0f}")
        print(f"  Min/Max:     {series.min():.0f} / {series.max():.0f}")
        print(f"  Demand pattern: {demand_profile['pattern']}")
        print(f"  Industry profile: {industry} ({seasonality_type})")
        print()

    results = rolling_backtest(
        series,
        effective_granularity,
        forecast_horizon=forecast_horizon,
        n_splits=n_splits,
        seasonality_type=seasonality_type,
    )

    if not results:
        holdout = max(2, min(forecast_horizon, max(2, len(series) // 4)))
        train = series.iloc[:-holdout]
        test = series.iloc[-holdout:]
        results = [
            calculate_accuracy(test.values, moving_average_forecast(train, holdout), "Moving Average"),
            calculate_accuracy(
                test.values,
                holt_winters_forecast(
                    train,
                    holdout,
                    seasonal_periods=min(seasonal_p, max(2, len(train) // 2)),
                    seasonality_type=seasonality_type,
                ),
                "Holt-Winters",
            ),
            calculate_accuracy(test.values, trend_seasonal_forecast(train, holdout, effective_granularity), "Trend+Seasonal"),
            calculate_accuracy(test.values, croston_forecast(train, holdout), "Croston"),
            calculate_accuracy(test.values, tsb_forecast(train, holdout), "TSB"),
        ]

    best_model_hint = industry_profile.get("preferred_model", "Holt-Winters")
    best_idx = min(
        range(len(results)),
        key=lambda i: (results[i]["MAPE_%"], 0 if results[i]["Method"] == best_model_hint else 0.001),
    )
    best_method = results[best_idx]["Method"]

    if not silent:
        print("  ACCURACY COMPARISON")
        print(f"  {'Method':<20} {'MAE':<10} {'MAPE %':<10} {'RMSE':<10} {'Folds':<8}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
        for acc in results:
            folds = acc.get("Folds", 1)
            print(f"  {acc['Method']:<20} {acc['MAE']:<10} {acc['MAPE_%']:<10} {acc['RMSE']:<10} {folds:<8}")
        print()
        print(f"  BEST MODEL: {best_method} (MAPE: {results[best_idx]['MAPE_%']}%)")
        print()

    # ── Horizon is already in weeks when daily data auto-aggregates ──
    # The dashboard converts the slider to weeks before calling forecast_sku,
    # so we must NOT divide by 7 here — that was the cause of the 4-row export bug.
    effective_horizon = forecast_horizon

    fc_ma = moving_average_forecast(series, effective_horizon)
    fc_hw = holt_winters_forecast(
        series,
        effective_horizon,
        seasonal_periods=min(seasonal_p, max(2, len(series) // 2)),
        seasonality_type=seasonality_type,
    )
    fc_ts = trend_seasonal_forecast(series, effective_horizon, effective_granularity)
    fc_cr = croston_forecast(series, effective_horizon)
    fc_tsb = tsb_forecast(series, effective_horizon)

    future = future_dates(dates.iloc[-1], effective_horizon, effective_granularity)
    # Attach optional dimension labels
    category_label = (
        df.loc[df["SKU"] == sku, "Category"].iloc[0]
        if "Category" in df.columns and not df.loc[df["SKU"] == sku, "Category"].empty
        else None
    )

    forecast_df = pd.DataFrame(
        {
            "Date": future,
            "SKU": sku,
            "Moving_Average": np.round(fc_ma, 0).astype(int),
            "Holt_Winters": np.round(fc_hw, 0).astype(int),
            "Trend_Seasonal": np.round(fc_ts, 0).astype(int),
            "Croston": np.round(fc_cr, 0).astype(int),
            "TSB": np.round(fc_tsb, 0).astype(int),
            "Demand_Pattern": demand_profile["pattern"],
            "Industry": industry,
            "Seasonality_Type": seasonality_type,
        }
    )
    if location is not None:
        forecast_df.insert(2, "Location", location)
    if category_label is not None:
        forecast_df.insert(2, "Category", category_label)

    forecast_df["Best_Model"] = np.round(
        {
            "Moving Average": fc_ma,
            "Holt-Winters": fc_hw,
            "Trend+Seasonal": fc_ts,
            "Croston": fc_cr,
            "TSB": fc_tsb,
        }[best_method],
        0,
    ).astype(int)
    forecast_df["Best_Method"] = best_method

    # ── Confidence scoring ────────────────────────────────────────
    confidence = model_confidence(results, best_idx)
    forecast_df["Confidence"] = confidence["label"]
    forecast_df["Confidence_Score"] = confidence["score"]

    business_insight = generate_business_explanation(
        demand_profile=demand_profile,
        best_method=best_method,
        accuracy_results=results,
        industry_profile=industry_profile,
        granularity=effective_granularity,
        forecast_horizon=effective_horizon,
    )

    return {
        "sku": sku,
        "skipped": False,
        "location": location,
        "category": category_label,
        "granularity": effective_granularity,
        "original_granularity": granularity,
        "industry": industry,
        "industry_profile": industry_profile,
        "demand_profile": demand_profile,
        "accuracy_results": results,
        "best_method": best_method,
        "best_index": best_idx,
        "confidence": confidence,
        "forecast_df": forecast_df,
        "historical_df": sku_df,
        "series": series,
        "dates": dates,
        "forecast_series": {
            "Moving Average": fc_ma,
            "Holt-Winters": fc_hw,
            "Trend+Seasonal": fc_ts,
            "Croston": fc_cr,
            "TSB": fc_tsb,
        },
        "business_insight": business_insight,
    }

# -----------------------------------------------------------------------------
# DYNAMIC HORIZON + CATEGORY PIPELINE
# -----------------------------------------------------------------------------

def max_forecast_horizon(df: pd.DataFrame, granularity: str) -> int:
    """
    Calculate a sensible maximum forecast horizon based on data history length.
    Rule: horizon <= history_length / 2, capped at 1 year in the detected cadence.
    Returns periods in the detected granularity unit.
    """
    if df.empty or "Date" not in df.columns:
        return 52  # safe default

    history_days = (df["Date"].max() - df["Date"].min()).days
    caps = {"Daily": 365, "Weekly": 52, "Monthly": 24}
    cap = caps.get(granularity, 52)

    if granularity == "Daily":
        history_periods = history_days
    elif granularity == "Weekly":
        history_periods = history_days // 7
    else:
        history_periods = history_days // 30

    return max(4, min(cap, history_periods // 2))


def forecast_sku_location_agg(
    df: pd.DataFrame,
    sku: str,
    granularity: str,
    forecast_horizon: int = 12,
    industry: str = "General",
    n_splits: int = 4,
    silent: bool = False,
) -> Dict[str, object]:
    """
    Location-aggregated forecast for a single SKU.
    Stage 1: Sum across all locations → forecast the total (smoother signal).
    Stage 2: Disaggregate back to locations by historical demand share.
    Returns the same structure as forecast_sku, but with aggregated accuracy.
    """
    if "Location" not in df.columns:
        return forecast_sku(df, sku, granularity, forecast_horizon, industry, n_splits, silent)

    sku_df = df[df["SKU"] == sku].copy()
    locations = sku_df["Location"].dropna().unique().tolist()
    if len(locations) <= 1:
        return forecast_sku(df, sku, granularity, forecast_horizon, industry, n_splits, silent)

    # Stage 1: aggregate across locations
    result = forecast_sku(
        df=df, sku=sku, granularity=granularity,
        forecast_horizon=forecast_horizon, industry=industry,
        n_splits=n_splits, silent=silent, location=None,
    )
    if result.get("skipped"):
        return result

    # Stage 2: compute location shares and disaggregate
    loc_totals = sku_df.groupby("Location")["Quantity"].sum()
    grand_total = float(loc_totals.sum()) if float(loc_totals.sum()) > 0 else 1.0
    loc_shares = (loc_totals / grand_total).to_dict()

    best_fc = result["forecast_df"]["Best_Model"].values.astype(float)
    loc_forecasts = {
        loc: np.round(best_fc * share, 0).astype(int)
        for loc, share in loc_shares.items()
    }

    result["location_shares"] = loc_shares
    result["location_forecasts"] = loc_forecasts
    result["aggregation_mode"] = "location"
    return result





def run_forecast_pipeline(
    df: pd.DataFrame,
    sku: str,
    granularity: str,
    forecast_horizon: int = 12,
    industry: str = "General",
):
    """CLI-friendly wrapper that returns the same tuple as the older version."""
    result = forecast_sku(
        df=df,
        sku=sku,
        granularity=granularity,
        forecast_horizon=forecast_horizon,
        industry=industry,
        silent=False,
    )
    return result["forecast_df"], result["accuracy_results"], result["best_method"]


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          ForecastIQ - Core Engine v0.3                  ║")
    print("║          Demand Forecasting Prototype                   ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    data_file = os.path.join("sample_data", "retail_sample.csv")
    forecast_horizon = 12
    top_n_skus = 3

    if os.path.exists(data_file):
        df = load_data(data_file)
    else:
        print(f"  Data file not found: {data_file}")
        print("  Using a synthetic demo dataset so the pipeline can still run.")
        df = generate_demo_data(num_skus=3, periods=104, granularity="Weekly")

    granularity = detect_granularity(df)
    quality = data_quality_check(df)
    pareto_table, top_skus = pareto_analysis(df, top_n=top_n_skus)

    forecast_df, accuracy_results, best = run_forecast_pipeline(
        df,
        sku=top_skus[0],
        granularity=granularity,
        forecast_horizon=forecast_horizon,
        industry="General",
    )

    output_file = "forecast_output.csv"
    export_df = forecast_df.copy()
    export_df["Date"] = export_df["Date"].dt.strftime("%Y-%m-%d")
    export_df.to_csv(output_file, index=False)

    print("=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"  Forecast saved to: {output_file}")
    print(f"  Best model: {best}")
    print("  Open forecast_output.csv in Excel to review the numbers")
    print()
    print("  ---------------------------------------------")
    print("  Privacy: No data was stored or transmitted.")
    print("  This tool is for reference and prototyping only.")
    print("  ---------------------------------------------")
    print()
