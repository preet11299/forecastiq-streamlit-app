"""
ForecastIQ — Dashboard (v1.0)
Redesigned UI: planner-first layout, reactive SKU/model switching,
confidence band chart, forward planning table, collapsed model details.

Run:
    streamlit run dashboard.py
"""

from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core_engine import (
    INDUSTRY_PROFILES,
    MIN_DENSITY_RECORDS,
    data_quality_check,
    detect_granularity,
    forecast_sku,
    generate_demo_data,
    get_industry_profile,
    horizon_unit_label,
    max_forecast_horizon,
    pareto_analysis,
    _normalize_columns,
)

warnings.filterwarnings("ignore")

# ── Analytics ─────────────────────────────────────────────────────────────────
_ANALYTICS_URL = "https://script.google.com/macros/s/AKfycbx_1HTCDP_GuzqScLcM9au-CsOhioWn5xOoWCKwwX0n0dv5jcSm6IFMJx3SgdLJgY_NQQ/exec"

def _log_run(skus_count: int, horizon: int, industry: str, granularity: str, sku_mode: str) -> None:
    """Fire-and-forget usage log. Contains zero user data — counts and settings only."""
    import threading, urllib.request, json as _json
    from datetime import datetime, timezone
    payload = _json.dumps({
        "timestamp":   datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "skus_count":  skus_count,
        "horizon":     horizon,
        "industry":    industry,
        "granularity": granularity,
        "sku_mode":    sku_mode,
    }).encode()
    def _post():
        try:
            req = urllib.request.Request(
                _ANALYTICS_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=4)
        except Exception:
            pass  # never surface analytics errors to the user
    threading.Thread(target=_post, daemon=True).start()

# ── Palette ───────────────────────────────────────────────────────────────────
BG      = "#0d1117"
BG2     = "#161b22"
CARD    = "#1f2937"
BORDER  = "#30363d"
BORDER2 = "#374151"
ACCENT  = "#2563eb"
BLUE    = "#60a5fa"
PINK    = "#f472b6"
TEXT    = "#e6edf3"
TEXT2   = "#9ca3af"
TEXT3   = "#7d8590"
TEXT4   = "#4b5563"
GREEN   = "#3fb950"
AMBER   = "#f59e0b"
RED     = "#f87171"
TEAL    = "#7dd3fc"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ForecastIQ",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, sans-serif;
    color: {TEXT};
}}
.stApp {{ background: {BG}; }}
#MainMenu, footer, header {{ visibility: hidden; }}

[data-testid="stSidebar"] {{
    background: {BG2} !important;
    border-right: 0.5px solid {BORDER} !important;
}}
[data-testid="stSidebar"] * {{ color: {TEXT2}; }}

.block-container {{
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
    max-width: 100% !important;
}}

[data-testid="stMetric"] {{
    background: {BG2};
    border-radius: 8px;
    padding: 10px 14px !important;
}}
[data-testid="stMetric"] label {{
    color: {TEXT3} !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
[data-testid="stMetric"] [data-testid="stMetricValue"] {{
    color: {TEXT} !important;
    font-size: 1.1rem !important;
    font-weight: 500 !important;
}}
[data-testid="stMetric"] [data-testid="stMetricDelta"] {{
    font-size: 0.8rem !important;
}}

.stButton > button {{
    background: {ACCENT} !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    padding: 10px 16px !important;
    width: 100% !important;
}}
.stButton > button:hover {{ opacity: 0.88 !important; }}

.stDownloadButton > button {{
    background: {CARD} !important;
    color: {TEXT2} !important;
    border: 0.5px solid {BORDER2} !important;
    border-radius: 6px !important;
    font-size: 12px !important;
    padding: 5px 12px !important;
}}
.stDownloadButton > button:hover {{
    border-color: {ACCENT} !important;
    color: {BLUE} !important;
}}

div[data-testid="stSelectbox"] > label p,
div[data-testid="stMultiSelect"] > label p,
div[data-testid="stRadio"] > label p,
div[data-testid="stSlider"] > label p,
div[data-testid="stFileUploader"] > label p,
.stCheckbox > label span {{
    color: {TEXT3} !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

div[data-testid="stMultiSelect"] span[data-baseweb="tag"] {{
    background: {CARD} !important;
    border: 0.5px solid {BORDER2} !important;
    border-radius: 4px !important;
    color: {TEXT2} !important;
    font-size: 11px !important;
}}

div[data-testid="stSlider"] [data-baseweb="slider"] div[role="progressbar"] {{
    background: {ACCENT} !important;
}}
div[data-testid="stSlider"] [role="slider"] {{
    background: {ACCENT} !important;
    border: 2px solid {BLUE} !important;
}}
div[data-testid="stSlider"] [data-testid="stThumbValue"] {{
    color: {BLUE} !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}}

div[data-testid="stFileUploader"] section {{
    border-color: {BORDER} !important;
    background: {BG2} !important;
    border-style: dashed !important;
    border-radius: 8px !important;
}}

.stDataFrame {{ border-radius: 8px; overflow: hidden; }}
hr {{ border-color: {BORDER} !important; }}
.stCaption p {{ color: {TEXT4} !important; font-size: 0.75rem !important; }}
.streamlit-expanderHeader {{ color: {TEXT3} !important; font-size: 0.8rem !important; }}
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
for key, default in [
    ("df_original", None),
    ("forecast_results", {}),
    ("last_run_key", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt(n) -> str:
    try:
        n = int(n)
    except (TypeError, ValueError):
        return str(n)
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return f"{n:,}"


def _read_csv_silent(uploaded_file) -> pd.DataFrame:
    """Load and normalise CSV — suppress all warnings from the parser."""
    df = pd.read_csv(uploaded_file)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _normalize_columns(df)
    return df


@st.cache_data(show_spinner=False)
def _sample_xlsx_bytes() -> bytes:
    """Build a two-sheet Excel: Read Me guide + Sample Data."""
    import openpyxl
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    rng = np.random.default_rng(7)
    dates = pd.date_range("2023-01-02", periods=104, freq="W")
    cats  = ["Electronics","Apparel","Home & Garden","Sporting Goods","Automotive"]
    locs  = ["Store_A","Store_B","Store_C","Warehouse_1","Warehouse_2"]
    rows  = []
    for i in range(10):
        sku = f"SKU-{i+1:03d}"
        base, trend = 80+i*20, 0.5+i*0.2
        for t, d in enumerate(dates):
            s = (10+i*3)*np.sin(2*np.pi*t/52)
            qty = max(base+trend*t+s+rng.normal(0,6+i),0)
            rows.append([d.strftime("%Y-%m-%d"),sku,round(qty,0),cats[i%5],locs[i%5],round(9.99+i*2.5+rng.uniform(-1,1),2)])
    sample_df = pd.DataFrame(rows, columns=["Date","SKU","Quantity","Category","Location","Price"])

    wb = Workbook()
    thin   = Side(style="thin", color="CBD5E1")
    bdr    = Border(left=thin,right=thin,top=thin,bottom=thin)
    DARK   = PatternFill("solid", start_color="1F3864")
    MID    = PatternFill("solid", start_color="2563EB")
    ALT    = PatternFill("solid", start_color="F8FAFC")
    REQ    = PatternFill("solid", start_color="FEF9C3")
    OPT    = PatternFill("solid", start_color="DCFCE7")
    CENT   = Alignment(horizontal="center",vertical="center",wrap_text=True)
    LEFT   = Alignment(horizontal="left",  vertical="center",wrap_text=True)

    def _h(ws,r,c,v,fill=None,bold=True,color="FFFFFF",align=CENT):
        cell=ws.cell(row=r,column=c,value=v)
        cell.font=Font(name="Arial",bold=bold,size=10,color=color)
        cell.fill=fill or DARK; cell.alignment=align; cell.border=bdr
        return cell
    def _c(ws,r,c,v,fill=None,bold=False,align=LEFT):
        cell=ws.cell(row=r,column=c,value=v)
        cell.font=Font(name="Arial",bold=bold,size=10)
        if fill: cell.fill=fill
        cell.alignment=align; cell.border=bdr
        return cell

    # ── Sheet 1: Read Me ──────────────────────────────────────
    ws1=wb.active; ws1.title="Read Me"
    ws1.merge_cells("A1:G1")
    t=ws1["A1"]; t.value="ForecastIQ — Data Format & Output Guide"
    t.font=Font(name="Arial",bold=True,size=13,color="FFFFFF")
    t.fill=DARK; t.alignment=CENT; ws1.row_dimensions[1].height=30

    # Column reference
    ws1.merge_cells("A3:G3")
    s=ws1["A3"]; s.value="REQUIRED & OPTIONAL COLUMNS"
    s.font=Font(name="Arial",bold=True,size=10,color="FFFFFF"); s.fill=MID; s.alignment=CENT
    for ci,h in enumerate(["Column","Required?","Data Type","Format / Example","Accepted Aliases","Notes","Status"],1):
        _h(ws1,4,ci,h)
    col_data=[
        ["Date","Yes","Date (text)","2024-01-07","date, week, order_date, trans_date, period","ISO format preferred. Day-first fallback supported.","Required"],
        ["SKU","Yes","Text","SKU-001 or 1025","sku, item, product, product_code, article","Any alphanumeric ID. Spaces trimmed.","Required"],
        ["Quantity","Yes","Number","120 or 34.5","quantity, qty, sales, demand, units, orders","Must be numeric. $ symbols stripped. Negatives dropped.","Required"],
        ["Category","No","Text","Electronics","category, cat, segment, family, group","Enables category filter in sidebar.","Optional"],
        ["Location","No","Text","Store_A","location, store, warehouse, site, region","Enables location filter.","Optional"],
        ["Price","No","Number","9.99","price, unit_price, checkout_price, cost","Enables revenue-based Pareto ranking.","Optional"],
    ]
    sf={"Required":REQ,"Optional":OPT}
    for ri,row in enumerate(col_data,5):
        bg=ALT if ri%2==0 else None
        for ci,val in enumerate(row,1):
            f=sf.get(val,bg); c=_c(ws1,ri,ci,val,fill=f)
            if ci==2: c.alignment=CENT
            if ci==7:
                c.font=Font(name="Arial",size=10,bold=True,
                            color="92400E" if val=="Required" else "14532D")

    # Limitations
    r=13; ws1.merge_cells(f"A{r}:G{r}")
    s2=ws1[f"A{r}"]; s2.value="DATA REQUIREMENTS & LIMITATIONS"
    s2.font=Font(name="Arial",bold=True,size=10,color="FFFFFF"); s2.fill=MID; s2.alignment=CENT
    lims=[
        ["Min rows per SKU","30 rows minimum. SKUs below this are skipped automatically."],
        ["Recommended history","52+ weeks (1 year) for seasonality. 2+ years is ideal."],
        ["File format","CSV only (.csv). Comma-delimited. UTF-8 encoding."],
        ["Max file size","200 MB per upload."],
        ["Duplicate rows","Duplicate Date+SKU rows are summed before forecasting."],
        ["Zero demand","Allowed. Handled by Croston & TSB models."],
        ["Negative quantity","Dropped during parsing."],
        ["Column order","Does not matter — detection is by name, not position."],
    ]
    ws1.merge_cells(f"B{r+1}:G{r+1}")
    _h(ws1,r+1,1,"Parameter"); _h(ws1,r+1,2,"Detail")
    for ri2,(p,d) in enumerate(lims,r+2):
        bg=ALT if ri2%2==0 else None
        _c(ws1,ri2,1,p,fill=bg,bold=True)
        ws1.merge_cells(f"B{ri2}:G{ri2}"); _c(ws1,ri2,2,d,fill=bg)

    # Output guide
    r2=r+len(lims)+3; ws1.merge_cells(f"A{r2}:G{r2}")
    s3=ws1[f"A{r2}"]; s3.value="WHAT TO EXPECT FROM THE OUTPUT"
    s3.font=Font(name="Arial",bold=True,size=10,color="FFFFFF"); s3.fill=MID; s3.alignment=CENT
    outs=[
        ["Forecast CSV columns","Date, SKU, Forecasted_Qty, Lower_Bound, Upper_Bound, Best_Model, MAPE_pct, Confidence, Confidence_Score, Demand_Pattern"],
        ["Models compared","5 models per SKU: Moving Average, Holt-Winters, Trend+Seasonal, Croston, TSB. Best = lowest WAPE."],
        ["Confidence band","±15% around the active forecast line. Shown as shaded area on the chart."],
        ["Demand patterns","Stable (CV≤0.4), Mixed (0.4–0.8), Volatile (CV>0.8), Intermittent (>50% zero periods)."],
        ["Confidence label","High (score≥75), Medium (50–74), Low (<50). Based on MAPE + cross-validation stability."],
        ["Forward table","3-period quantities per SKU + period total + YoY % vs same period prior year."],
        ["Skipped SKUs","SKUs with <30 data points are skipped and listed with a reason."],
    ]
    ws1.merge_cells(f"B{r2+1}:G{r2+1}")
    _h(ws1,r2+1,1,"Output"); _h(ws1,r2+1,2,"Description")
    for ri3,(o,d) in enumerate(outs,r2+2):
        bg=ALT if ri3%2==0 else None
        _c(ws1,ri3,1,o,fill=bg,bold=True)
        ws1.merge_cells(f"B{ri3}:G{ri3}"); _c(ws1,ri3,2,d,fill=bg)

    for col,w in zip("ABCDEFG",[22,12,18,22,30,36,11]):
        ws1.column_dimensions[col].width=w
    ws1.freeze_panes="A5"

    # ── Sheet 2: Sample Data ──────────────────────────────────
    ws2=wb.create_sheet("Sample Data")
    for ci,h in enumerate(sample_df.columns,1):
        _h(ws2,1,ci,h)
    for ri,row in enumerate(sample_df.itertuples(index=False),2):
        bg=ALT if ri%2==0 else None
        for ci,val in enumerate(row,1):
            c=_c(ws2,ri,ci,val,fill=bg)
            if ci==1: c.alignment=CENT
            if ci==3: c.number_format="#,##0"
            if ci==6: c.number_format="#,##0.00"
    for col,w in zip("ABCDEF",[14,12,12,16,14,10]):
        ws2.column_dimensions[col].width=w
    ws2.freeze_panes="A2"
    ws2.auto_filter.ref=f"A1:F{len(sample_df)+1}"

    buf=io.BytesIO(); wb.save(buf); return buf.getvalue()


def _starter_kit_zip() -> bytes:
    """Zip containing the Excel guide + a sample CSV."""
    import zipfile, io as _io
    xlsx_bytes = _sample_xlsx_bytes()
    rng = np.random.default_rng(7)
    dates = pd.date_range("2023-01-02", periods=104, freq="W")
    cats = ["Electronics","Apparel","Home & Garden","Sporting Goods","Automotive"]
    locs = ["Store_A","Store_B","Store_C","Warehouse_1","Warehouse_2"]
    rows = []
    for i in range(10):
        sku = f"SKU-{i+1:03d}"
        base, trend = 80+i*20, 0.5+i*0.2
        for t, d in enumerate(dates):
            s = (10+i*3)*np.sin(2*np.pi*t/52)
            qty = max(base+trend*t+s+rng.normal(0,6+i), 0)
            rows.append([d.strftime("%Y-%m-%d"), sku, round(qty,0), cats[i%5], locs[i%5]])
    csv_bytes = pd.DataFrame(rows, columns=["Date","SKU","Quantity","Category","Location"]
                             ).to_csv(index=False).encode()
    buf = _io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("forecastiq_guide.xlsx", xlsx_bytes)
        zf.writestr("forecastiq_sample.csv",  csv_bytes)
    return buf.getvalue()


def _run_key(skus, horizon, industry, location) -> str:
    return f"{sorted(skus)}|{horizon}|{industry}|{location}"


def _build_chart(result: dict, active_model: Optional[str] = None) -> go.Figure:
    hist = result["historical_df"]
    fdf  = result["forecast_df"]
    best = result["best_method"]
    show = active_model or best

    fc_len = len(fdf)
    trail  = min(len(hist), max(fc_len * 3, 52))
    h_tail = hist.tail(trail)

    model_cols = {
        "Moving Average":  "Moving_Average",
        "Holt-Winters":    "Holt_Winters",
        "Trend+Seasonal":  "Trend_Seasonal",
        "Croston":         "Croston",
        "TSB":             "TSB",
    }

    # Confidence band: ±15% around the active model forecast
    active_col = model_cols.get(show, "Best_Model")
    if active_col not in fdf.columns:
        active_col = "Best_Model"
    fc_vals = fdf[active_col].values.astype(float)
    lo = np.maximum(fc_vals * 0.85, 0)
    hi = fc_vals * 1.15

    fig = go.Figure()

    # Confidence band (filled area)
    fig.add_trace(go.Scatter(
        x=pd.concat([fdf["Date"], fdf["Date"][::-1]]),
        y=np.concatenate([hi, lo[::-1]]),
        fill="toself",
        fillcolor="rgba(37,99,235,0.12)",
        line=dict(width=0),
        name="Confidence band",
        hoverinfo="skip",
    ))

    # Historical
    fig.add_trace(go.Scatter(
        x=h_tail["Date"], y=h_tail["Quantity"],
        name="Historical",
        mode="lines",
        line=dict(color=TEXT2, width=1.5),
    ))

    # Alt model styles — each visually distinct
    ALT_STYLES = {
        "Moving Average":  dict(color="#9ca3af", dash="solid",    width=1.2),
        "Holt-Winters":    dict(color="#2dd4bf", dash="dash",     width=1.2),
        "Trend+Seasonal":  dict(color="#a78bfa", dash="dot",      width=1.2),
        "Croston":         dict(color=PINK,      dash="dashdot",  width=1.2),
        "TSB":             dict(color="#fb923c", dash="longdash", width=1.2),
    }

    # All non-active models — distinct colors, reduced opacity
    for label, col in model_cols.items():
        if col not in fdf.columns:
            continue
        if label == show:
            continue
        style = ALT_STYLES.get(label, dict(color=BLUE, dash="dot", width=1))
        fig.add_trace(go.Scatter(
            x=fdf["Date"], y=fdf[col],
            name=label,
            mode="lines",
            line=dict(color=style["color"], width=style["width"], dash=style["dash"]),
            opacity=0.45,
        ))

    # Active model — prominent
    fig.add_trace(go.Scatter(
        x=fdf["Date"], y=fdf[active_col],
        name=f"{show} {'★' if show == best else ''}".strip(),
        mode="lines+markers",
        line=dict(color=PINK, width=2.5, dash="dash"),
        marker=dict(size=5, color=PINK),
    ))

    # Forecast starts line
    last_hist = hist["Date"].iloc[-1]
    fig.add_vline(x=last_hist, line_dash="dash", line_color=TEXT4, line_width=1)
    fig.add_annotation(
        x=last_hist, y=1.04, yref="paper",
        text="Forecast starts", showarrow=False,
        font=dict(size=10, color=TEXT3),
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(l=40, r=10, t=20, b=35),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.06,
            xanchor="right", x=1,
            font=dict(size=10, color=TEXT3),
        ),
        font=dict(family="Inter, sans-serif", color=TEXT3, size=11),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=10, color=TEXT4)),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            tickfont=dict(size=10, color=TEXT4),
            tickformat=",",
            title=dict(text="Quantity", font=dict(size=11, color=TEXT3)),
        ),
    )
    return fig


def _build_export(results: dict) -> pd.DataFrame:
    """
    Export CSV: one row per forecast period per SKU.
    Columns: Date, SKU, Forecasted_Qty, Lower_Bound, Upper_Bound,
             Best_Model, MAPE_pct, Confidence, Demand_Pattern
    Quantities formatted with commas, no None values.
    """
    rows = []
    col_map = {
        "Moving Average": "Moving_Average",
        "Holt-Winters":   "Holt_Winters",
        "Trend+Seasonal": "Trend_Seasonal",
        "Croston":        "Croston",
        "TSB":            "TSB",
    }
    for sku, r in results.items():
        if r.get("skipped"):
            continue
        fdf         = r["forecast_df"].copy()
        best_method = r["best_method"]
        conf        = r.get("confidence", {})
        best_col    = col_map.get(best_method, "Best_Model")
        if best_col not in fdf.columns:
            best_col = "Best_Model"
        fc_vals = fdf[best_col].fillna(0).values.astype(float)
        lo_vals = np.maximum(fc_vals * 0.85, 0).astype(int)
        hi_vals = (fc_vals * 1.15).astype(int)
        export = pd.DataFrame({
            "Date":           pd.to_datetime(fdf["Date"]).dt.strftime("%Y-%m-%d"),
            "SKU":            sku,
            "Forecasted_Qty": fc_vals.astype(int),
            "Lower_Bound":    lo_vals,
            "Upper_Bound":    hi_vals,
            "Best_Model":     best_method,
            "MAPE_pct":       r["accuracy_results"][r["best_index"]]["MAPE_%"],
            "Confidence":     conf.get("label", ""),
            "Demand_Pattern": r["demand_profile"]["pattern"],
        })
        rows.append(export)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    # Comma-format quantity columns as strings for readability
    for col in ["Forecasted_Qty", "Lower_Bound", "Upper_Bound"]:
        out[col] = out[col].apply(lambda x: f"{int(x):,}")
    return out


def _forward_table(results: dict, granularity: str) -> pd.DataFrame:
    """
    3-period forward planning table. Only shows periods that exist.
    Quantities comma-formatted. No None values.
    """
    col_map = {
        "Moving Average": "Moving_Average",
        "Holt-Winters":   "Holt_Winters",
        "Trend+Seasonal": "Trend_Seasonal",
        "Croston":        "Croston",
        "TSB":            "TSB",
    }
    rows = []
    # Collect period labels from first non-skipped result
    period_labels = []
    for r in results.values():
        if not r.get("skipped"):
            fdf = r["forecast_df"]
            dates = pd.to_datetime(fdf["Date"].values[:3])
            if granularity == "Monthly":
                period_labels = [d.strftime("%b %Y") for d in dates]
            else:
                period_labels = [d.strftime("%d %b") for d in dates]
            break

    # Pad to exactly 3 labels
    while len(period_labels) < 3:
        period_labels.append(f"P{len(period_labels)+1}")
    period_labels = period_labels[:3]

    for sku, r in results.items():
        if r.get("skipped"):
            continue
        fdf      = r["forecast_df"]
        conf     = r.get("confidence", {})
        best_col = col_map.get(r["best_method"], "Best_Model")
        if best_col not in fdf.columns:
            best_col = "Best_Model"

        # Only take up to 3 real forecast values
        fc_series = fdf[best_col].fillna(0).values[:3].astype(float)
        n_periods = len(fc_series)

        # YoY vs same window in history
        hist = r["historical_df"]["Quantity"].values.astype(float)
        yr_back = max(0, len(hist) - (52 if granularity == "Weekly" else 12))
        hist_same = hist[yr_back: yr_back + n_periods]
        if len(hist_same) == n_periods and hist_same.sum() > 0:
            yoy = ((fc_series.sum() - hist_same.sum()) / hist_same.sum()) * 100
            yoy_str = f"+{yoy:.1f}%" if yoy >= 0 else f"{yoy:.1f}%"
        else:
            yoy_str = "n/a"

        row = {
            "SKU":            sku,
            "Pattern":        r["demand_profile"]["pattern"],
            period_labels[0]: f"{int(fc_series[0]):,}" if n_periods > 0 else "—",
            period_labels[1]: f"{int(fc_series[1]):,}" if n_periods > 1 else "—",
            period_labels[2]: f"{int(fc_series[2]):,}" if n_periods > 2 else "—",
            "3-period total": f"{int(fc_series.sum()):,}",
            "YoY %":          yoy_str,
            "Confidence":     conf.get("label", "?"),
        }
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:9px;padding:4px 0 12px;">
        <div style="width:26px;height:26px;border-radius:6px;background:{ACCENT};
                    display:flex;align-items:center;justify-content:center;
                    font-size:13px;font-weight:600;color:#fff;">F</div>
        <span style="font-size:15px;font-weight:500;color:{TEXT};">ForecastIQ</span>
    </div>
    <hr style="margin:0 0 12px;"/>
    """, unsafe_allow_html=True)

    # ── Step 1: Data ──────────────────────────────────────────────
    st.markdown(f'<p style="font-size:10px;color:{TEXT3};text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;">① Data</p>', unsafe_allow_html=True)

    st.download_button(
        "↓ Get started (.zip)",
        data=_starter_kit_zip(),
        file_name="forecastiq_starter_kit.zip",
        mime="application/zip",
        use_container_width=True,
        help="Includes the format guide (.xlsx) and a sample dataset for reference.",
    )

    st.markdown('<div style="margin-top:8px;"></div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    use_sample = st.checkbox("Use built-in sample data", value=False)

    # Load data into session state
    if uploaded_file is not None:
        try:
            st.session_state.df_original = _read_csv_silent(uploaded_file)
            st.success("File loaded.")
        except Exception as exc:
            st.error(f"Could not read file: {exc}")
            st.session_state.df_original = None
    elif use_sample and st.session_state.df_original is None:
        st.session_state.df_original = generate_demo_data(num_skus=10, periods=104, granularity="Weekly")
        st.success("Sample data loaded.")

    df = st.session_state.df_original

    if df is None or df.empty:
        st.info("Upload a CSV or load the sample to begin.")
        st.stop()

    # Compute once after data load
    granularity    = detect_granularity(df, silent=True)
    quality        = data_quality_check(df, silent=True)
    effective_gran = "Weekly" if granularity == "Daily" else granularity
    effective_unit = horizon_unit_label(effective_gran)
    pareto_table, all_skus = pareto_analysis(df, top_n=quality["unique_skus"], silent=True)

    # ── Data quality cards ────────────────────────────────────────
    st.markdown('<hr style="margin:8px 0;"/>', unsafe_allow_html=True)
    q_cols = st.columns(2)
    q_cols[0].metric("Rows",    _fmt(quality["total_rows"]))
    q_cols[1].metric("SKUs",    _fmt(quality["unique_skus"]))
    q_cols2 = st.columns(2)
    q_cols2[0].metric("History", f"{quality['history_months']} mo")
    q_score = quality["quality_score"]
    q_cols2[1].metric("Quality",  f"{q_score}%",
                       delta="Good" if q_score >= 80 else ("Fair" if q_score >= 60 else "Poor"),
                       delta_color="normal" if q_score >= 80 else "inverse")

    if granularity == "Daily":
        st.caption("Daily data — forecasts auto-aggregate to weekly.")

    st.markdown('<hr style="margin:8px 0;"/>', unsafe_allow_html=True)

    # ── Step 2: Filters ───────────────────────────────────────────
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:6px;margin-bottom:6px;">
        <p style="font-size:10px;color:{TEXT3};text-transform:uppercase;letter-spacing:.06em;margin:0;">② Filters</p>
        <span style="font-size:9px;background:#14532d;color:#4ade80;border-radius:3px;padding:1px 5px;">RE-RUN TO APPLY</span>
    </div>
    """, unsafe_allow_html=True)

    industry_profile_preview = get_industry_profile(
        list(INDUSTRY_PROFILES.keys())[0]
    )
    industry = st.selectbox(
        "Industry",
        list(INDUSTRY_PROFILES.keys()),
        index=0,
        help=(
            "Sets the seasonality model used by Holt-Winters and Trend+Seasonal.\n\n"
            "• Retail / Semi → Multiplicative seasonality: demand swings scale with volume (bigger peaks in high months).\n"
            "• General / Healthcare / Auto → Additive seasonality: swings stay constant regardless of volume level.\n\n"
            "Also acts as a tiebreaker when two models score equally — each industry has a preferred model. "
            "You'll see the active setting reflected in the chart legend and MAPE line."
        ),
        label_visibility="visible",
    )
    ip = get_industry_profile(industry)
    st.caption(f"Seasonality: {ip['seasonality_type']} · Preferred: {ip['preferred_model']}")

    has_category = "Category" in df.columns
    has_location = "Location" in df.columns
    selected_category = None
    selected_location = None
    df_filtered = df.copy()

    if has_category:
        cats = ["All categories"] + sorted(df["Category"].dropna().unique().tolist())
        sel_cat = st.selectbox("Category", cats, label_visibility="visible")
        if sel_cat != "All categories":
            selected_category = sel_cat
            df_filtered = df_filtered[df_filtered["Category"] == sel_cat].copy()

    if has_location:
        locs = ["All locations"] + sorted(df["Location"].dropna().unique().tolist())
        sel_loc = st.selectbox("Location", locs, label_visibility="visible")
        if sel_loc != "All locations":
            selected_location = sel_loc
            df_filtered = df_filtered[df_filtered["Location"] == sel_loc].copy()

    # Recompute pareto after filter
    pareto_table, all_skus = pareto_analysis(df_filtered, top_n=df_filtered["SKU"].nunique(), silent=True)

    st.markdown('<hr style="margin:8px 0;"/>', unsafe_allow_html=True)

    # ── Step 3: SKU mode ──────────────────────────────────────────
    st.markdown(f'<p style="font-size:10px;color:{TEXT3};text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;">③ SKU mode</p>', unsafe_allow_html=True)

    sku_mode = st.radio(
        "SKU selection method",
        ["Manual pick", "Pareto top N"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )

    if sku_mode == "Manual pick":
        select_all = st.checkbox("Select all", value=False, key="select_all_skus")
        default_skus = all_skus if select_all else all_skus[:min(3, len(all_skus))]
        selected_skus = st.multiselect(
            "SKUs to forecast",
            options=all_skus,
            default=default_skus,
            label_visibility="collapsed",
        )
    else:
        pareto_mode = pareto_table["Pareto_Mode"].iloc[0] if len(pareto_table) else "volume"
        pareto_n = st.slider(
            f"Top N SKUs by {pareto_mode.lower()}",
            min_value=1, max_value=min(20, len(all_skus)),
            value=min(3, len(all_skus)), step=1,
        )
        selected_skus = all_skus[:pareto_n]
        st.caption(f"Auto-selected top {pareto_n} SKUs by Pareto.")

    st.markdown('<hr style="margin:8px 0;"/>', unsafe_allow_html=True)

    # ── Step 4: Horizon ───────────────────────────────────────────
    st.markdown(f'<p style="font-size:10px;color:{TEXT3};text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;">④ Horizon</p>', unsafe_allow_html=True)
    _max_h    = max_forecast_horizon(df_filtered, effective_gran)
    _default  = min(12, _max_h)
    horizon   = st.slider(
        f"Forecast horizon ({effective_unit})",
        min_value=4, max_value=_max_h, value=_default, step=1,
        label_visibility="visible",
    )
    st.caption(f"Max {_max_h} {effective_unit} based on history · cadence: {effective_gran.lower()}")

    if horizon > _max_h // 2:
        st.warning(f"Horizon over {_max_h // 2} {effective_unit} may reduce accuracy.")

    # ── Run button ────────────────────────────────────────────────
    st.markdown('<div style="margin-top:8px;"></div>', unsafe_allow_html=True)
    run_clicked = st.button("Run forecast", use_container_width=True)

    st.markdown(f'<p style="font-size:10px;color:{TEXT4};text-align:center;margin-top:8px;">Your data never leaves your browser. Anonymous usage stats logged.</p>', unsafe_allow_html=True)


# ── Mobile CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@media (max-width: 768px) {{
    [data-testid="stSidebar"] {{
        width: 85vw !important;
        min-width: unset !important;
    }}
    .block-container {{
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
        padding-top: 1rem !important;
    }}
    [data-testid="stMetric"] {{
        padding: 8px 10px !important;
    }}
    [data-testid="stMetric"] [data-testid="stMetricValue"] {{
        font-size: 0.95rem !important;
    }}
}}
</style>
""", unsafe_allow_html=True)

# ── Main area ─────────────────────────────────────────────────────────────────
if df is None or df.empty:
    st.info("Load data to get started.")
    st.stop()

if not selected_skus:
    st.warning("Select at least one SKU in the sidebar, then click Run forecast.")
    st.stop()

# ── Run engine only when inputs change ───────────────────────────────────────
run_key = _run_key(selected_skus, horizon, industry, selected_location)

if run_clicked or (run_key != st.session_state.last_run_key and not st.session_state.forecast_results):
    with st.spinner("Running backtests and generating forecasts…"):
        results: Dict[str, dict] = {}
        for sku in selected_skus:
            try:
                results[sku] = forecast_sku(
                    df=df_filtered, sku=sku, granularity=granularity,
                    forecast_horizon=horizon, industry=industry,
                    n_splits=4, silent=True,
                    location=selected_location,
                )
            except Exception as exc:
                st.warning(f"Could not forecast SKU {sku}: {exc}")
        st.session_state.forecast_results = results
        st.session_state.last_run_key = run_key
        # Log usage — no file data, counts and settings only
        _log_run(
            skus_count  = len(selected_skus),
            horizon     = horizon,
            industry    = industry,
            granularity = granularity,
            sku_mode    = sku_mode,
        )

results = st.session_state.forecast_results

if not results:
    st.info("Click **Run forecast** in the sidebar to generate forecasts.")
    st.stop()

skipped    = {k: v for k, v in results.items() if v.get("skipped")}
forecasted = {k: v for k, v in results.items() if not v.get("skipped")}

if skipped:
    skipped_names = ", ".join(skipped.keys())
    st.caption(f"Skipped ({len(skipped)}): {skipped_names} — fewer than {MIN_DENSITY_RECORDS} data points.")

if not forecasted:
    st.warning("All selected SKUs were skipped due to insufficient data.")
    st.stop()

# ── KPI strip ────────────────────────────────────────────────────────────────
first_result  = next(iter(forecasted.values()))
first_fdf     = first_result["forecast_df"]
forecast_start = pd.to_datetime(first_fdf["Date"].iloc[0]).strftime("%b %Y")
forecast_end   = pd.to_datetime(first_fdf["Date"].iloc[-1]).strftime("%b %Y")

all_hist_vals = np.concatenate([
    v["historical_df"]["Quantity"].values for v in forecasted.values()
])
all_fc_vals = np.concatenate([
    v["forecast_df"]["Best_Model"].values for v in forecasted.values()
])
avg_monthly_demand = round(float(np.mean(all_fc_vals)), 1)

# YoY: compare avg forecast vs avg of same-length history window
avg_hist = float(np.mean(all_hist_vals[-len(all_fc_vals):]))
yoy_pct  = ((avg_monthly_demand - avg_hist) / max(avg_hist, 1)) * 100

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Forecast period",     f"{forecast_start} – {forecast_end}")
kpi2.metric("Avg forecast demand", f"{avg_monthly_demand:.0f} units")
kpi3.metric("YoY trend",           f"{yoy_pct:+.1f}%")
kpi4.metric("SKUs forecasted",     f"{len(forecasted)} of {quality['unique_skus']}")

st.markdown('<div style="margin-bottom:4px;"></div>', unsafe_allow_html=True)

# ── Chart area ────────────────────────────────────────────────────────────────
sku_list = list(forecasted.keys())

chart_col, _ = st.columns([1, 0.001])
with chart_col:
    with st.container():
        # Chart header row
        h_left, h_right = st.columns([2, 1])
        with h_left:
            focus_sku = st.selectbox(
                "SKU",
                options=sku_list,
                index=0,
                label_visibility="collapsed",
                key="sku_selector",
            )
        with h_right:
            focus = forecasted[focus_sku]
            model_options = [r["Method"] for r in focus["accuracy_results"]]
            best_method   = focus["best_method"]
            default_idx   = model_options.index(best_method) if best_method in model_options else 0

            active_model = st.selectbox(
                "Model",
                options=model_options,
                index=default_idx,
                format_func=lambda m: f"{m} ★" if m == best_method else m,
                label_visibility="collapsed",
                key="model_selector",
            )

        # Star explanation — brief, single line
        st.caption("★ = lowest MAPE across 4-fold cross-validation")

        # Warn only if user picks a worse model
        if active_model != best_method:
            best_mape  = focus["accuracy_results"][focus["best_index"]]["MAPE_%"]
            active_row = next((r for r in focus["accuracy_results"] if r["Method"] == active_model), None)
            if active_row and active_row["MAPE_%"] > best_mape:
                st.warning(f"{active_model} MAPE {active_row['MAPE_%']}% vs {best_method} {best_mape}% — recommended model has lower error.")

        # Chart
        st.plotly_chart(_build_chart(focus, active_model), use_container_width=True)

        # Export button — inline after chart
        export_df = _build_export(forecasted)
        if not export_df.empty:
            buf = io.StringIO()
            export_df.to_csv(buf, index=False)
            st.download_button(
                "↓ Export forecast CSV",
                data=buf.getvalue(),
                file_name="forecastiq_forecast.csv",
                mime="text/csv",
            )

st.markdown('<div style="margin-bottom:4px;"></div>', unsafe_allow_html=True)

# ── Forward planning table ────────────────────────────────────────────────────
fwd_df = _forward_table(forecasted, effective_gran)

if not fwd_df.empty:
    st.markdown(f"""
    <div style="background:{BG2};border-radius:10px;padding:1px 0 0;">
    """, unsafe_allow_html=True)

    st.dataframe(
        fwd_df,
        use_container_width=True,
        hide_index=True,
    )

    # ── Model details (collapsed) ─────────────────────────────────
    with st.expander("▾ Model details — MAPE / MAE / RMSE for all models"):
        focus_for_detail = forecasted[focus_sku]
        acc_df = pd.DataFrame(focus_for_detail["accuracy_results"])
        acc_df = acc_df[["Method", "MAE", "MAPE_%", "RMSE", "Folds"]].rename(columns={"MAPE_%": "WAPE_%"})
        acc_df["Recommended"] = acc_df["Method"].apply(
            lambda m: "★" if m == focus_for_detail["best_method"] else ""
        )
        acc_df = acc_df.sort_values("WAPE_%")
        st.dataframe(acc_df, use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:1.5rem 0 0.5rem;color:{TEXT4};font-size:0.75rem;">
    Built by Preet Patel · ForecastIQ v1.0
</div>
""", unsafe_allow_html=True)
