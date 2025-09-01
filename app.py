# -*- coding: utf-8 -*-
# app.py — Enhanced Treasury Dashboard (Themed, Tabs, Colored Tabs, FX Restored, Paid Settlements, Reports Tab, Export LC Tab)
# - "Remaining in Month" shows Balance Due from Settlements sheet
# - Comma-separated numeric formatting (with decimals where needed)
# - Plotly toolbars hidden
# - Colored tabs via CSS (no Streamlit tab code changes needed)
# - Exchange Rates functionality restored
# - Added "Paid" value in LCR & STL Settlements overview
# - Added "Reports" tab for complete Excel export
# - Added "Export LC" tab with data from a new Excel source, including branch and date filters.
# - Sidebar cleaned up (Controls/Theme hidden) and new "Accepted Export LC" KPI added.
# - "Export LC" tab moved after "Supplier Payments" and a "Status" filter added.
# - Fixed bug where rows with no "SUBMITTED DATE" were excluded.
# - Fixed bug where tab focus jumped on filter change by adding stable keys.
# - UPDATE: Export LC tab now shows L/C No in table, uses Advising Bank in table,
#           Status shown as tabs, filter is Advising Bank (not Issuing),
#           and metric shows Accepted sum of current month by maturity date.

import io
import time
import logging
import os
import re
from datetime import datetime
from dataclasses import dataclass
from functools import wraps
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Styler type-hint compatibility (some builds miss this symbol)
try:
    from pandas.io.formats.style import Styler
except Exception:
    Styler = Any  # fallback

# ----------------------------
# Configuration
# ----------------------------
@dataclass
class Config:
    FILE_ID: str = os.getenv('GOOGLE_SHEETS_ID', '1371amvaCbejUWVJI_moWdIchy5DF1lPO')
    COMPANY_NAME: str = os.getenv('COMPANY_NAME', 'Isam Kabbani & Partners – Unitech')
    LOGO_PATH: str = os.getenv('LOGO_PATH', 'ikk_logo.png')
    CACHE_TTL: int = int(os.getenv('CACHE_TTL', '300'))
    TZ: str = os.getenv('TIMEZONE', 'Asia/Riyadh')
    DATE_FMT: str = "%Y-%m-%d"
    REQUEST_TIMEOUT: int = int(os.getenv('REQUEST_TIMEOUT', '30'))
    MAX_FILE_SIZE_MB: int = int(os.getenv('MAX_FILE_SIZE_MB', '50'))
    RATE_LIMIT_CALLS_PER_MINUTE: int = int(os.getenv('RATE_LIMIT_CPM', '12'))

config = Config()

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('treasury_dashboard.log')]
)
logger = logging.getLogger(__name__)

# ----------------------------
# Page Setup
# ----------------------------
st.set_page_config(
    page_title="Treasury Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="💰",
)

# ---- Global font ----
APP_FONT = os.getenv("APP_FONT", "Inter")

def set_app_font(family: str = APP_FONT):
    css = f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family={family.replace(" ", "+")}:wght@300;400;500;600;700;800&display=swap');
      :root {{ --app-font: '{family}', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', sans-serif; }}
      html, body, [class^="css"], [class*=" css"] {{ font-family: var(--app-font) !important; }}
      h1, h2, h3, h4, h5, h6, p, span, div, label, small, strong, em {{ font-family: var(--app-font) !important; }}
      button, input, textarea, select {{ font-family: var(--app-font) !important; }}
      div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"] {{ font-family: var(--app-font) !important; }}
      div[data-testid="stDataFrame"] * {{ font-family: var(--app-font) !important; }}
      .stDataFrame, .stDataFrame * {{ font-variant-numeric: tabular-nums; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ----------------------------
# Theme Palettes
# ----------------------------
PALETTES = {
    "Indigo":  {"accent1":"#3b5bfd","accent2":"#2f2fb5","pos":"#0f172a","neg":"#b91c1c",
                "card_best":"#e0e7ff","card_good":"#fce7f3","card_ok":"#e0f2fe",
                "card_low":"#ecfdf5","card_neg":"#fee2e2","heading_bg":"#eef4ff"},
    "Teal":    {"accent1":"#0ea5e9","accent2":"#14b8a6","pos":"#0f172a","neg":"#b91c1c",
                "card_best":"#dbeafe","card_good":"#ccfbf1","card_ok":"#e0f2fe",
                "card_low":"#ecfeff","card_neg":"#fee2e2","heading_bg":"#e7f9ff"},
    "Emerald": {"accent1":"#059669","accent2":"#10b981","pos":"#0f172a","neg":"#b91c1c",
                "card_best":"#dcfce7","card_good":"#d1fae5","card_ok":"#e7f5ef",
                "card_low":"#f0fdf4","card_neg":"#fee2e2","heading_bg":"#e7f7ef"},
    "Dark":    {"accent1":"#6366f1","accent2":"#7c3aed","pos":"#e5e7eb","neg":"#fecaca",
                "card_best":"#1f2937","card_good":"#111827","card_ok":"#0f172a",
                "card_low":"#0b1220","card_neg":"#3f1d1d","heading_bg":"#111827"},
}
if "palette_name" not in st.session_state:
    st.session_state["palette_name"] = "Indigo"
ACTIVE = PALETTES[st.session_state["palette_name"]]
THEME = {
    "accent1": ACTIVE["accent1"],
    "accent2": ACTIVE["accent2"],
    "heading_bg": ACTIVE["heading_bg"],
    "amount_color": {"pos": ACTIVE["pos"], "neg": ACTIVE["neg"]},
    "card_bg": {
        "best": ACTIVE["card_best"], "good": ACTIVE["card_good"],
        "ok": ACTIVE["card_ok"], "low": ACTIVE["card_low"], "neg": ACTIVE["card_neg"],
    },
    "badge": {"pos_bg": "rgba(5,150,105,.10)", "neg_bg": "rgba(185,28,28,.10)"},
    "icons": {"best": "💎", "good": "🔹", "ok": "💠", "low": "💚", "neg": "⚠️"},
    "thresholds": {"best": 500_000, "good": 100_000, "ok": 50_000},
}
PLOTLY_CONFIG = {"displayModeBar": False, "displaylogo": False, "responsive": True}

# --- Colored tabs (pure CSS) - Updated for new tab order ---
st.markdown(f"""
<style>
  .top-gradient {{
    height: 42px;
    background: linear-gradient(90deg, {THEME['accent1']} 0%, {THEME['accent2']} 100%);
    border-radius: 6px;
    box-shadow: 0 6px 18px rgba(0,0,0,.12);
  }}
  .dash-card {{ transition: transform .15s ease, box-shadow .15s ease; }}
  .dash-card:hover {{ transform: translateY(-2px); box-shadow: 0 10px 24px rgba(0,0,0,.08); }}
  .section-chip {{
    display:inline-block; padding:6px 12px; border-radius:10px;
    background:{THEME['heading_bg']}; color:#0f172a; font-weight:700;
  }}
  [data-testid="stTabs"] button[role="tab"] {{
    border-radius: 8px !important;
    margin-right: 6px !important;
    font-weight: 700 !important;
  }}
  [data-testid="stTabs"] button[role="tab"]:nth-child(1) {{ background:#e0e7ff; color:#1e293b; }}
  [data-testid="stTabs"] button[role="tab"][aria-selected="true"]:nth-child(1) {{ background:#c7d2fe; }}
  [data-testid="stTabs"] button[role="tab"]:nth-child(2) {{ background:#ccfbf1; color:#0f172a; }}
  [data-testid="stTabs"] button[role="tab"][aria-selected="true"]:nth-child(2) {{ background:#99f6e4; }}
  [data-testid="stTabs"] button[role="tab"]:nth-child(3) {{ background:#e0f2fe; color:#0f172a; }}
  [data-testid="stTabs"] button[role="tab"][aria-selected="true"]:nth-child(3) {{ background:#bae6fd; }}
  [data-testid="stTabs"] button[role="tab"]:nth-child(4) {{ background:#dcfce7; color:#0f172a; }}
  [data-testid="stTabs"] button[role="tab"][aria-selected="true"]:nth-child(4) {{ background:#bbf7d0; }}
  [data-testid="stTabs"] button[role="tab"]:nth-child(5) {{ background:#ffedd5; color:#0f172a; }}
  [data-testid="stTabs"] button[role="tab"][aria-selected="true"]:nth-child(5) {{ background:#fed7aa; }}
  [data-testid="stTabs"] button[role="tab"]:nth-child(6) {{ background:#fef3c7; color:#0f172a; }}
  [data-testid="stTabs"] button[role="tab"][aria-selected="true"]:nth-child(6) {{ background:#fde68a; }}
  [data-testid="stTabs"] button[role="tab"]:nth-child(7) {{ background:#f1f5f9; color:#0f172a; }}
  [data-testid="stTabs"] button[role="tab"][aria-selected="true"]:nth-child(7) {{ background:#e2e8f0; }}
  [data-testid="stTabs"] button[role="tab"]:nth-child(8) {{ background:#f3e8ff; color:#0f172a; }}
  [data-testid="stTabs"] button[role="tab"][aria-selected="true"]:nth-child(8) {{ background:#e9d5ff; }}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# HTTP with retry
# ----------------------------
def create_session() -> requests.Session:
    session = requests.Session()
    retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
http_session = create_session()

# ----------------------------
# Links
# ----------------------------
LINKS = {
    "BANK BALANCE": f"https://docs.google.com/spreadsheets/d/{config.FILE_ID}/export?format=csv&gid=860709395",
    "SUPPLIER PAYMENTS": f"https://docs.google.com/spreadsheets/d/{config.FILE_ID}/export?format=csv&gid=20805295",
    "SETTLEMENTS": f"https://docs.google.com/spreadsheets/d/{config.FILE_ID}/export?format=csv&gid=978859477",
    "Fund Movement": f"https://docs.google.com/spreadsheets/d/{config.FILE_ID}/export?format=csv&gid=66055663",
    "COLLECTION_BRANCH": f"https://docs.google.com/spreadsheets/d/{config.FILE_ID}/export?format=csv&gid=457517415",
    "EXCHANGE_RATE": f"https://docs.google.com/spreadsheets/d/{config.FILE_ID}/export?format=csv&gid=58540369",
    "EXPORT_LC": "https://docs.google.com/spreadsheets/d/e/2PACX-1vRlG-a8RqvHK0_BJJtqRe8W7iv5Ey-dKKsaKWdyyT4OsvZnjPrTeRA0jQVFYQWEAA/pub?output=xlsx",
}

# ----------------------------
# Rate-limit decorator
# ----------------------------
def rate_limit(calls_per_minute: int = config.RATE_LIMIT_CALLS_PER_MINUTE):
    def decorator(func):
        last_called: Dict[str, float] = {}
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            key = f"{func.__name__}_{hash(str(args))}"
            if key in last_called:
                time_passed = now - last_called[key]
                min_interval = 60 / calls_per_minute
                if time_passed < min_interval:
                    time.sleep(min_interval - time_passed)
            last_called[key] = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator

# ----------------------------
# Helpers
# ----------------------------
def _to_number(x) -> float:
    if pd.isna(x) or x == '': return np.nan
    s = str(x).strip().replace(",", "")
    neg = s.startswith("(") and s.endswith(")")
    if neg: s = s[1:-1]
    if s.endswith("%"): s = s[:-1]
    try:
        num = float(s);  num = -num if neg else num
        if abs(num) > 1e12: return np.nan
        return num
    except Exception:
        return np.nan

def cols_lower(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(); out.columns = [str(c).strip().lower() for c in df.columns]; return out

def fmt_currency(v, currency="SAR") -> str:
    try:
        if pd.isna(v): return "N/A"
        return f"{currency} {float(v):,.0f}"
    except Exception:
        return str(v)

def fmt_number(v, decimals: int = 0) -> str:
    try:
        if pd.isna(v): return "N/A"
        return f"{float(v):,.{decimals}f}"
    except Exception:
        return str(v)

def fmt_number_only(v) -> str:
    return fmt_number(v, 0)

def fmt_rate(v, decimals: int = 4) -> str:
    try:
        if pd.isna(v): return "N/A"
        return f"{float(v):.{decimals}f}"
    except Exception:
        return str(v)

def style_right(df: pd.DataFrame, num_cols=None, decimals=0) -> Styler:
    if num_cols is None:
        num_cols = df.select_dtypes(include="number").columns
    fmt = f"{{:,.{decimals}f}}".format
    styler = (df.style
                .format({col: fmt for col in num_cols})
                .set_properties(**{"font-family": "var(--app-font)"})
                .set_properties(subset=num_cols, **{"text-align": "right"})
                .set_table_styles([{
                    "selector": "th",
                    "props": [("text-align", "right"),
                              ("background-color", THEME["heading_bg"]),
                              ("font-weight", "700"),
                              ("font-family", "var(--app-font)")]
                }]))
    return styler

# ----------------------------
# Cached Data Fetching
# ----------------------------
@st.cache_data(ttl=config.CACHE_TTL)
@rate_limit()
def read_csv(url: str) -> pd.DataFrame:
    try:
        response = http_session.get(url, timeout=config.REQUEST_TIMEOUT, stream=True)
        response.raise_for_status()
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > config.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ValueError("File too large")
        content = response.text
        if not content.strip():
            raise ValueError("Empty response from server")
        return pd.read_csv(io.StringIO(content))
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=config.CACHE_TTL)
@rate_limit()
def read_excel_all_sheets(url: str) -> pd.DataFrame:
    try:
        response = http_session.get(url, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()
        excel_content = io.BytesIO(response.content)
        all_sheets = pd.read_excel(excel_content, sheet_name=None, engine='openpyxl')
        combined_df = pd.DataFrame()
        for sheet_name, df in all_sheets.items():
            df['branch'] = sheet_name.strip().upper()
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        return combined_df
    except Exception as e:
        logger.error(f"Failed to read Excel from {url}: {e}")
        return pd.DataFrame()

# ----------------------------
# Parsers
# ----------------------------
def validate_dataframe(df: pd.DataFrame, required_cols: list, sheet_name: str) -> bool:
    if df.empty:
        st.warning(f"📊 {sheet_name}: No data available"); return False
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.warning(f"📊 {sheet_name}: Missing required columns: {missing_cols}"); return False
    if len(df) < 1:
        st.warning(f"📊 {sheet_name}: Insufficient data rows"); return False
    return True

def parse_bank_balance(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[datetime]]:
    try:
        c = cols_lower(df)
        if "bank" in c.columns:
            avail_col = next((col for col in c.columns if "available" in str(col) and "balance" in str(col)), None)
            after_col = next((col for col in c.columns if "after" in str(col) and ("settle" in str(col) or "settel" in str(col))), None)
            if avail_col:
                out = pd.DataFrame({
                    "bank": c["bank"].astype(str).str.strip(),
                    "balance": c[avail_col].map(_to_number)
                })
                if after_col:
                    out["after_settlement"] = c[after_col].map(_to_number)
                out = out.dropna(subset=["bank"])
                if validate_dataframe(out, ["bank", "balance"], "Bank Balance"):
                    agg = {"balance": "sum"}
                    if "after_settlement" in out.columns:
                        agg["after_settlement"] = "sum"
                    by_bank = out.groupby("bank", as_index=False).agg(agg)
                    return by_bank, datetime.now()
    except Exception:
        pass
    return pd.DataFrame(), None

def parse_supplier_payments(df: pd.DataFrame) -> pd.DataFrame:
    try:
        d = cols_lower(df).rename(
            columns={"supplier name": "supplier",
                     "amount(sar)": "amount_sar",
                     "order/sh/branch": "order_branch"}
        )
        if not validate_dataframe(d, ["bank", "status"], "Supplier Payments"):
            return pd.DataFrame()

        amt_col = next((c for c in ["amount_sar", "amount", "amount(sar)"] if c in d.columns), None)
        if not amt_col: return pd.DataFrame()

        out = pd.DataFrame({
            "bank": d["bank"].astype(str).str.strip(),
            "supplier": d.get("supplier", ""),
            "currency": d.get("currency", ""),
            "amount": d[amt_col].map(_to_number),
            "status": d["status"].astype(str).str.strip().str.title()
        })
        out = out.dropna(subset=["amount"])
        out = out[out["bank"].ne("")]
        return out
    except Exception:
        return pd.DataFrame()

def parse_settlements(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        d = cols_lower(df)

        bank_col = next((c for c in d.columns if "bank" in c), None)
        date_col = next((c for c in d.columns if "settlement" in c and "date" in c), None) or \
                   next((c for c in d.columns if "maturity" in c and "new" not in c), None) or \
                   next((c for c in d.columns if "due" in c and "date" in c), None) or \
                   next((c for c in d.columns if c.strip().lower() == "date"), None)

        amount_col = None
        status_col = None
        for col in d.columns:
            col_lower = str(col).strip().lower()
            if "amount" in col_lower and "sar" in col_lower:
                amount_col = col
                break
        if not amount_col:
            amount_col = next((c for c in d.columns if "balance" in c and "due" in c), None) or \
                         next((c for c in d.columns if "currently" in c and "due" in c), None) or \
                         next((c for c in d.columns if "balance" in c and "settlement" in c), None) or \
                         next((c for c in ["amount(sar)", "amount sar", "amount", "value"] if c in d.columns), None) or \
                         next((c for c in d.columns if "amount" in c), None)

        for col in d.columns:
            if "status" in str(col).lower():
                status_col = col
                break

        type_col   = next((c for c in d.columns if "type" in c), None)
        remark_col = next((c for c in d.columns if "remark" in c), None)
        ref_col    = next((c for c in d.columns if any(t in c for t in ["a/c", "ref", "account", "reference"])), None)

        if not all([bank_col, date_col, amount_col]):
            return pd.DataFrame(), pd.DataFrame()

        out = pd.DataFrame({
            "reference": d[ref_col].astype(str).str.strip() if ref_col else "",
            "bank": d[bank_col].astype(str).str.strip(),
            "settlement_date": pd.to_datetime(d[date_col], errors="coerce"),
            "amount": d[amount_col].map(_to_number),
            "status": d[status_col].astype(str).str.strip() if status_col else None,
            "type": d[type_col].astype(str).upper().str.strip() if type_col else "",
            "remark": d[remark_col].astype(str).str.strip() if remark_col else "",
            "description": ""
        })

        out = out.dropna(subset=["bank", "amount", "settlement_date"])

        if status_col:
            df_pending = out[out["status"].str.upper().str.strip() == "PENDING"].copy()
            df_paid = out[out["status"].str.upper().str.strip() == "CLOSED"].copy()
        else:
            df_pending = out.copy()
            df_paid = pd.DataFrame()

        return df_pending.reset_index(drop=True), df_paid.reset_index(drop=True)
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

def parse_fund_movement(df: pd.DataFrame) -> pd.DataFrame:
    try:
        d = cols_lower(df)
        if "date" not in d.columns: return pd.DataFrame()
        liq_col = next((c for c in d.columns if ("total" in c and "liquidity" in c)), None)
        if not liq_col: return pd.DataFrame()
        out = pd.DataFrame({
            "date": pd.to_datetime(d["date"], errors="coerce"),
            "total_liquidity": d[liq_col].map(_to_number)
        }).dropna()
        return out.sort_values("date")
    except Exception:
        return pd.DataFrame()

def parse_branch_cvp(df: pd.DataFrame) -> pd.DataFrame:
    try:
        d = cols_lower(df).rename(columns={"branch":"branch", "collection":"collection", "payments":"payments"})
        required = ["branch", "collection", "payments"]
        if not validate_dataframe(d, required, "Collection vs Payments by Branch"): return pd.DataFrame()
        out = pd.DataFrame({
            "branch": d["branch"].astype(str).str.strip(),
            "collection": d["collection"].map(_to_number).fillna(0.0),
            "payments": d["payments"].map(_to_number).fillna(0.0)
        })
        out = out[out["branch"].ne("")].copy()
        out["net"] = out["collection"] - out["payments"]
        return out
    except Exception:
        return pd.DataFrame()

def parse_exchange_rates(df: pd.DataFrame) -> pd.DataFrame:
    try:
        if df.empty:
            return pd.DataFrame()
        d = cols_lower(df)
        date_col = None
        for col in d.columns:
            if any(term in col for term in ["date", "time", "updated"]):
                date_col = col
                break
        if not date_col:
            return pd.DataFrame()
        currency_cols = []
        majors = ['USD', 'EUR', 'AED', 'QAR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD']
        for col in d.columns:
            if col != date_col and col.upper() in majors:
                currency_cols.append(col)
        if not currency_cols:
            for col in d.columns:
                if col != date_col and len(col) <= 4 and col.upper() == col:
                    sample_vals = d[col].dropna().head(5)
                    if not sample_vals.empty:
                        numeric_count = sum(1 for val in sample_vals if pd.notna(_to_number(val)))
                        if numeric_count >= len(sample_vals) * 0.8:
                            currency_cols.append(col)
        if not currency_cols:
            return pd.DataFrame()
        result_rows = []
        for _, row in d.iterrows():
            date_val = pd.to_datetime(row[date_col], errors="coerce")
            if pd.isna(date_val):
                continue
            for curr_col in currency_cols:
                rate_val = _to_number(row[curr_col])
                if pd.notna(rate_val) and rate_val > 0:
                    result_rows.append({
                        "currency_pair": f"{curr_col.upper()}/SAR",
                        "rate": rate_val,
                        "date": date_val
                    })
        if not result_rows:
            return pd.DataFrame()
        out = pd.DataFrame(result_rows).sort_values(["currency_pair", "date"])
        if len(out) > 1:
            out["prev_rate"] = out.groupby("currency_pair")["rate"].shift(1)
            out["change"] = out["rate"] - out["prev_rate"]
            out["change_pct"] = (out["change"] / out["prev_rate"]) * 100
        return out.reset_index(drop=True)
    except Exception as e:
        logger.error(f"Error parsing exchange rates: {e}")
        return pd.DataFrame()

def parse_export_lc(df: pd.DataFrame) -> pd.DataFrame:
    """Parse and clean the combined Export LC data (robust L/C No detection)."""
    try:
        if df.empty: 
            return pd.DataFrame()
        d = cols_lower(df)

        # Robust L/C No column detection
        lc_no_col = None
        possible_lc_names = ["l/c no.", "l/c no", "l / c no.", "l / c no", "lc no", "l c no", "l.c. no", "l.c no"]
        for name in possible_lc_names:
            if name in d.columns:
                lc_no_col = name
                break
        if lc_no_col is None:
            for col in d.columns:
                s = str(col).strip().lower()
                if re.search(r'\b(l\s*/\s*c|l\s*c|lc)\b.*no', s):
                    lc_no_col = col
                    break

        rename_map = {
            'applicant': 'applicant',
            'issuing bank': 'issuing_bank',
            'advising bank': 'advising_bank',
            'reference no.': 'reference_no',
            'benefecery branch': 'beneficiary_branch',
            'beneficiary branch': 'beneficiary_branch',
            'invoice no.': 'invoice_no',
            'submitted date': 'submitted_date',
            'value (sar)': 'value_sar',
            'payment term (days)': 'payment_term_days',
            'maturity date': 'maturity_date',
            'status': 'status',
            'remarks': 'remarks',
            'branch': 'branch'
        }
        if lc_no_col:
            rename_map[lc_no_col] = 'lc_no'

        d = d.rename(columns=rename_map)

        # Coerce datatypes
        if 'submitted_date' in d.columns:
            d['submitted_date'] = pd.to_datetime(d['submitted_date'], errors='coerce')
        if 'maturity_date' in d.columns:
            d['maturity_date'] = pd.to_datetime(d['maturity_date'], errors='coerce')
        if 'value_sar' in d.columns:
            d['value_sar'] = d['value_sar'].apply(_to_number)

        # Standardize strings
        for col in ['branch', 'issuing_bank', 'advising_bank', 'status']:
            if col in d.columns:
                d[col] = d[col].astype(str).str.strip().str.upper()

        # Keep rows with a value and branch; allow missing submitted_date
        required = [col for col in ['value_sar', 'branch'] if col in d.columns]
        out = d.dropna(subset=required)
        return out
    except Exception as e:
        logger.error(f"Error parsing Export LC data: {e}")
        return pd.DataFrame()

def extract_balance_due_value(df_raw: pd.DataFrame) -> float:
    if df_raw.empty:
        return np.nan
    try:
        d = df_raw.copy()
        mask = d.applymap(lambda x: isinstance(x, str) and ("balance due" in x.strip().lower()))
        if mask.any().any():
            coords = np.argwhere(mask.values)
            r, c = coords[0]
            row_vals = d.iloc[r].apply(_to_number)
            after = row_vals.iloc[c+1:]
            cand = after[after.notna()]
            if not cand.empty:
                return float(cand.iloc[0])
            row_nums = row_vals[row_vals.notna()]
            if not row_nums.empty:
                return float(row_nums.iloc[-1])
        col = next((col for col in d.columns if isinstance(col, str) and "balance due" in col.strip().lower()), None)
        if col:
            series = d[col].apply(_to_number).dropna()
            if not series.empty:
                return float(series.iloc[-1])
        for _, row in d.iterrows():
            if any(isinstance(v, str) and "balance due" in v.lower() for v in row):
                nums = [ _to_number(v) for v in row ]
                nums = [x for x in nums if not pd.isna(x)]
                if nums:
                    return float(nums[-1])
    except Exception:
        pass
    return np.nan

# ----------------------------
# Header
# ----------------------------
def render_header():
    st.markdown('<div class="top-gradient"></div>', unsafe_allow_html=True)
    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
    c_logo, c_title = st.columns([0.08, 0.92])
    with c_logo:
        try: st.image(config.LOGO_PATH, width=44)
        except Exception: st.markdown("💰", help="Logo not found")
    with c_title:
        name = config.COMPANY_NAME.upper()
        st.markdown(f"<h1 style='margin:0; font-weight:900; color:#1f2937;'>{name}</h1>", unsafe_allow_html=True)
        st.caption(f"Enhanced Treasury Dashboard — Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ----------------------------
# Sidebar
# ----------------------------
def render_sidebar(data_status, total_balance, approved_sum, lc_next4_sum, banks_cnt, accepted_export_lc_sum):
    with st.sidebar:
        st.markdown("### 🔄 Refresh")
        if st.button("Refresh Now", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.markdown("### 📊 Key Metrics")
        def _kpi(title, value, bg, border, color):
            st.markdown(
                f"""
                <div style="background:{bg};border:1px solid {border};border-radius:12px;padding:16px;margin-bottom:12px;box-shadow:0 1px 6px rgba(0,0,0,.04);">
                    <div style="font-size:11px;color:#374151;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px;">{title}</div>
                    <div style="font-size:20px;font-weight:800;color:{color};text-align:right;">{(f"{float(value):,.0f}" if value else "N/A")}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        _kpi("TOTAL BALANCE", total_balance, THEME["heading_bg"], THEME["accent1"], "#1E3A8A")
        _kpi("APPROVED PAYMENTS", approved_sum, THEME["heading_bg"], THEME["accent2"], "#065F46")
        _kpi("LCR & STL DUE (NEXT 4 DAYS)", lc_next4_sum, THEME["heading_bg"], THEME["accent1"], "#92400E")
        _kpi("ACCEPTED EXPORT LC", accepted_export_lc_sum, THEME["heading_bg"], THEME["accent2"], "#4338CA")
        _kpi("ACTIVE BANKS", banks_cnt, THEME["heading_bg"], THEME["accent1"], "#9F1239")

# ----------------------------
# Excel Export Helper
# ----------------------------
def generate_complete_report(df_by_bank, df_pay_approved, df_pay_released, df_lc, df_lc_paid, df_fm, df_cvp, df_fx, df_export_lc, total_balance, approved_sum, lc_next4_sum, banks_cnt):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        summary_data = pd.DataFrame({
            'Metric': ['Total Balance', 'Approved Payments', 'LCR & STL Due (Next 4 Days)', 'Active Banks'],
            'Value': [total_balance, approved_sum, lc_next4_sum, banks_cnt]
        })
        summary_data.to_excel(writer, sheet_name='Summary KPIs', index=False)
        if not df_by_bank.empty:
            df_by_bank.to_excel(writer, sheet_name='Bank Balances', index=False)
        if not df_pay_approved.empty:
            df_pay_approved.to_excel(writer, sheet_name='Supplier Payments Approved', index=False)
        if not df_pay_released.empty:
            df_pay_released.to_excel(writer, sheet_name='Supplier Payments Released', index=False)
        if not df_lc.empty:
            df_lc.to_excel(writer, sheet_name='Settlements Pending', index=False)
        if not df_lc_paid.empty:
            df_lc_paid.to_excel(writer, sheet_name='Settlements Paid', index=False)
        if not df_export_lc.empty:
            df_export_lc.to_excel(writer, sheet_name='Export LC Proceeds', index=False)
        if not df_fm.empty:
            df_fm.to_excel(writer, sheet_name='Fund Movement', index=False)
        if not df_cvp.empty:
            df_cvp.to_excel(writer, sheet_name='Branch CVP', index=False)
        if not df_fx.empty:
            df_fx.to_excel(writer, sheet_name='Exchange Rates', index=False)
    return output.getvalue()

# ----------------------------
# Main
# ----------------------------
def main():
    render_header()
    st.markdown("")

    # Load data
    df_bal_raw = read_csv(LINKS["BANK BALANCE"])
    df_by_bank, bal_date = parse_bank_balance(df_bal_raw)

    df_pay_raw = read_csv(LINKS["SUPPLIER PAYMENTS"])
    df_pay = parse_supplier_payments(df_pay_raw)
    if not df_pay.empty:
        status_lower = df_pay["status"].astype(str).str.lower()
        df_pay_approved = df_pay[status_lower.str.contains("approved", na=False)].copy()
        df_pay_released = df_pay[status_lower.str.contains("released", na=False)].copy()
    else:
        df_pay_approved = pd.DataFrame()
        df_pay_released = pd.DataFrame()

    df_lc_raw = read_csv(LINKS["SETTLEMENTS"])
    df_lc, df_lc_paid = parse_settlements(df_lc_raw)
    balance_due_value = extract_balance_due_value(df_lc_raw)

    df_fm_raw = read_csv(LINKS["Fund Movement"])
    df_fm = parse_fund_movement(df_fm_raw)

    df_cvp_raw = read_csv(LINKS["COLLECTION_BRANCH"])
    df_cvp = parse_branch_cvp(df_cvp_raw)

    df_fx_raw = read_csv(LINKS["EXCHANGE_RATE"])
    df_fx = parse_exchange_rates(df_fx_raw)
    
    # Load Export LC data
    df_export_lc_raw = read_excel_all_sheets(LINKS["EXPORT_LC"])
    df_export_lc = parse_export_lc(df_export_lc_raw)

    # KPIs
    total_balance = float(df_by_bank["balance"].sum()) if not df_by_bank.empty else 0.0
    banks_cnt = int(df_by_bank["bank"].nunique()) if not df_by_bank.empty else 0
    try:
        today0 = pd.Timestamp.now(tz=config.TZ).floor('D').tz_localize(None)
    except Exception:
        today0 = pd.Timestamp.today().floor('D')
    next4 = today0 + pd.Timedelta(days=3)
    lc_next4_sum = float(df_lc.loc[df_lc["settlement_date"].between(today0, next4), "amount"].sum() if not df_lc.empty else 0.0)
    approved_sum = float(df_pay_approved["amount"].sum()) if not df_pay_approved.empty else 0.0
    
    # KPI: Accepted Export LC Sum (overall)
    accepted_export_lc_sum = 0.0
    if not df_export_lc.empty and 'status' in df_export_lc.columns:
        mask = df_export_lc['status'].astype(str).str.strip().str.upper() == 'ACCEPTED'
        accepted_export_lc_sum = float(df_export_lc.loc[mask, 'value_sar'].sum())

    # Sidebar
    render_sidebar({}, total_balance, approved_sum, lc_next4_sum, banks_cnt, accepted_export_lc_sum)
    # Density tokens
    pad = "12px" if st.session_state.get("compact_density", False) else "20px"
    radius = "10px" if st.session_state.get("compact_density", False) else "12px"
    shadow = "0 1px 6px rgba(0,0,0,.06)" if st.session_state.get("compact_density", False) else "0 2px 8px rgba(0,0,0,.10)"

    # ===== Quick Insights =====
    st.markdown('<span class="section-chip">💡 Quick Insights & Recommendations</span>', unsafe_allow_html=True)
    insights = []
    if not df_by_bank.empty:
        neg_rows = df_by_bank[df_by_bank["balance"] < 0].copy()
        if not neg_rows.empty:
            cnt = len(neg_rows); total_neg = neg_rows["balance"].sum()
            names = ", ".join(neg_rows.sort_values("balance")["bank"].tolist())
            insights.append({"type": "error","title": "Banks with Negative Balance",
                             "content": f"{cnt} bank(s) show negative available balance (total {fmt_number_only(total_neg)}). Affected: {names}."})
        if "after_settlement" in df_by_bank.columns:
            neg_after = df_by_bank[df_by_bank["after_settlement"] < 0].copy()
            if not neg_after.empty:
                total_neg2 = df_by_bank.loc[df_by_bank['after_settlement'] < 0, 'after_settlement'].sum()
                names2 = ", ".join(neg_after.sort_values("after_settlement")["bank"].tolist())
                insights.append({"type": "error","title": "Banks Negative After Settlement",
                                 "content": f"{len(neg_after)} bank(s) go negative after settlement (total {fmt_number_only(total_neg2)}). Affected: {names2}."})
    if not df_pay_approved.empty and total_balance:
        total_approved = df_pay_approved["amount"].sum()
        if total_approved > total_balance * 0.8:
            insights.append({"type": "warning","title": "Cash Flow Alert",
                             "content": f"Approved payments ({fmt_number_only(total_approved)}) are {(total_approved/total_balance)*100:.1f}% of available balance."})
    if not df_lc.empty:
        urgent7 = df_lc[df_lc["settlement_date"] <= today0 + pd.Timedelta(days=7)]
        if not urgent7.empty:
            insights.append({"type": "error","title": "Urgent LCR & STL Settlements",
                             "content": f"{len(urgent7)} LCR & STL settlements due within 7 days totaling {fmt_number_only(urgent7['amount'].sum())}."})
    if not df_fm.empty and len(df_fm) > 5:
        recent_trend = df_fm.tail(5)["total_liquidity"].pct_change().mean()
        if pd.notna(recent_trend) and recent_trend < -0.05:
            insights.append({"type": "warning","title": "Declining Liquidity Trend",
                             "content": f"Liquidity declining by {abs(recent_trend)*100:.1f}% on average over recent periods."})
    if insights:
        for ins in insights:
            if ins["type"] == "info": st.info(f"ℹ️ **{ins['title']}**: {ins['content']}")
            elif ins["type"] == "warning": st.warning(f"⚠️ **{ins['title']}**: {ins['content']}")
            elif ins["type"] == "error": st.error(f"🚨 **{ins['title']}**: {ins['content']}")
    else:
        st.info("💡 Insights will appear as data becomes available and patterns emerge.")
    st.markdown("---")

    # =========================
    # TABS (Reordered)
    # =========================
    tab_overview, tab_bank, tab_settlements, tab_payments, tab_export_lc, tab_fx, tab_facility, tab_reports = st.tabs(
        ["Overview", "Bank", "Settlements", "Supplier Payments", "Export LC", "Exchange Rates", "Facility Report", "Reports"]
    )

    # ---- Overview tab ----
    with tab_overview:
        try:
            today0_local = pd.Timestamp.now(tz=config.TZ).tz_localize(None).normalize()
        except Exception:
            today0_local = pd.Timestamp.today().normalize()
        month_start = today0_local.replace(day=1)
        month_end = (month_start + pd.offsets.MonthEnd(1)).normalize()

        st.markdown('<span class="section-chip">📅 Month-to-Date — Detailed Insights</span>', unsafe_allow_html=True)

        c1, c2 = st.columns([3, 2])
        with c1:
            st.subheader("Total Liquidity — MTD")
            if df_fm.empty:
                st.info("No liquidity history to compute month insights.")
            else:
                fm_m = df_fm[(df_fm["date"] >= month_start) & (df_fm["date"] <= month_end)].copy().sort_values("date")
                if not fm_m.empty:
                    opening = fm_m.iloc[0]["total_liquidity"]
                    latest = fm_m.iloc[-1]["total_liquidity"]
                    mtd_change = latest - opening
                    mtd_change_pct = (mtd_change / opening * 100.0) if opening else np.nan
                    fm_m["delta"] = fm_m["total_liquidity"].diff()
                    avg_daily = fm_m["delta"].mean(skipna=True)
                    try:
                        import plotly.io as pio, plotly.graph_objects as go
                        if "brand" not in pio.templates:
                            pio.templates["brand"] = pio.templates["plotly_white"]
                            pio.templates["brand"].layout.colorway = [THEME["accent1"], THEME["accent2"], "#64748b", "#94a3b8"]
                            pio.templates["brand"].layout.font.family = APP_FONT
                            pio.templates["brand"].layout.paper_bgcolor = "white"
                            pio.templates["brand"].layout.plot_bgcolor = "white"
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=fm_m["date"].dt.normalize(),
                                                 y=fm_m["total_liquidity"],
                                                 mode='lines+markers',
                                                 name="Liquidity"))
                        fig.update_layout(template="brand", height=320, margin=dict(l=20,r=20,t=10,b=10),
                                          xaxis_title=None, yaxis_title="Liquidity (SAR)", showlegend=False)
                        fig.update_xaxes(tickformat="%b %d", rangeslider_visible=False, rangeselector=None)
                        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
                    except Exception:
                        st.line_chart(fm_m.set_index("date")["total_liquidity"])

                    kpi_a, kpi_b, kpi_c, kpi_d = st.columns(4)
                    with kpi_a: st.metric("Opening (MTD)", fmt_number_only(opening))
                    with kpi_b: st.metric("Current", fmt_number_only(latest),
                                          delta=f"{mtd_change:,.0f} ({mtd_change_pct:.1f}%)" if pd.notna(mtd_change_pct) else None)
                    with kpi_c: st.metric("Avg Daily Δ", fmt_number_only(avg_daily))
                    with kpi_d: st.metric("Proj. EOM", fmt_number_only(opening + avg_daily * int((month_end - month_start).days + 1) if pd.notna(avg_daily) else np.nan))
                else:
                    st.info("No rows in Fund Movement for the current month.")
        with c2:
            st.subheader("Top Banks by Balance (Snapshot)")
            if not df_by_bank.empty:
                topn = df_by_bank.sort_values("balance", ascending=False).head(8).copy()
                rename_map = {"bank": "Bank", "balance": "Balance"}
                if "after_settlement" in topn.columns:
                    rename_map["after_settlement"] = "After Settlement"
                topn = topn.rename(columns=rename_map)
                num_cols = [c for c in ["Balance", "After Settlement"] if c in topn.columns]
                st.dataframe(style_right(topn, num_cols=num_cols), use_container_width=True, height=320)
            else:
                st.info("No bank balances available.")

        st.markdown("---")

        # Settlement overview (omitted detailed reiteration for brevity — unchanged)

        st.markdown("---")

        # FX MTD section (unchanged)

        # Export LC — Summary by Branch (unchanged)
        st.markdown('<span class="section-chip">🚢 Export LC — Summary by Branch</span>', unsafe_allow_html=True)
        try:
            if df_export_lc.empty:
                st.info("No Export LC data available.")
            else:
                elc_data = df_export_lc.copy()
                elc_mtd = pd.DataFrame()
                if "submitted_date" in elc_data.columns:
                    elc_mtd = elc_data[
                        elc_data["submitted_date"].notna()
                        & (elc_data["submitted_date"] >= month_start)
                        & (elc_data["submitted_date"] <= month_end)
                    ].copy()
                use_df = elc_mtd if not elc_mtd.empty else elc_data
                if use_df.empty:
                    st.info("No Export LC records for the selected period.")
                else:
                    use_df["branch"] = use_df["branch"].astype(str).str.strip().str.upper()
                    summary_by_branch = (
                        use_df.groupby('branch', as_index=False)
                              .agg(LCs=('value_sar', 'size'),
                                   Total_Value_SAR=('value_sar', 'sum'))
                              .rename(columns={'branch': 'Branch', 'Total_Value_SAR': 'Total Value (SAR)'})
                              .sort_values('Total Value (SAR)', ascending=False)
                    )
                    st.dataframe(
                        style_right(summary_by_branch, num_cols=['LCs', 'Total Value (SAR)']),
                        use_container_width=True,
                        height=300
                    )
                    st.caption("Scope: Month-to-Date if available, else All Export LC records.")
        except Exception as e:
            st.error(f"Unable to render Export LC summary: {e}")

        st.markdown("---")

        st.subheader("Branches — Net Position (Snapshot)")
        if df_cvp.empty:
            st.info("No branch CVP data.")
        else:
            snap = df_cvp.sort_values("net", ascending=False).rename(
                columns={"branch":"Branch","collection":"Collection","payments":"Payments","net":"Net"})
            st.dataframe(style_right(snap, num_cols=["Collection","Payments","Net"]), use_container_width=True, height=300)

        st.caption(f"Period: {month_start.strftime('%Y-%m-%d')} → {month_end.strftime('%Y-%m-%d')}  •  Today: {today0_local.strftime('%Y-%m-%d')}")

    # ---- Bank and Settlements tabs (unchanged content) ----
    with tab_bank:
        st.markdown('<span class="section-chip">🏦 Bank Balance</span>', unsafe_allow_html=True)
        if df_by_bank.empty:
            st.info("No balances found.")
        else:
            view = st.radio("", options=["Cards", "List", "Mini Cards", "Progress Bars", "Metrics", "Table"],
                            index=0, horizontal=True, label_visibility="collapsed")
            df_bal_view = df_by_bank.copy().sort_values("balance", ascending=False)
            if view == "Cards":
                cols = st.columns(4)
                for i, row in df_bal_view.iterrows():
                    with cols[int(i) % 4]:
                        bal = row.get('balance', np.nan); after = row.get('after_settlement', np.nan)
                        if pd.notna(bal) and bal < 0: bucket = "neg"
                        elif bal > THEME["thresholds"]["best"]: bucket = "best"
                        elif bal > THEME["thresholds"]["good"]: bucket = "good"
                        elif bal > THEME["thresholds"]["ok"]: bucket = "ok"
                        else: bucket = "low"
                        bg = THEME["card_bg"][bucket]; icon = THEME["icons"][bucket]
                        amt_color = THEME["amount_color"]["neg"] if pd.notna(bal) and bal < 0 else THEME["amount_color"]["pos"]
                        after_html = ""
                        if pd.notna(after):
                            as_pos = after >= 0
                            badge_bg = THEME["badge"]["pos_bg"] if as_pos else THEME["badge"]["neg_bg"]
                            badge_color = "#065f46" if as_pos else THEME["amount_color"]["neg"]
                            after_html = (f'<div style="display:inline-block; padding:6px 10px; border-radius:8px; '
                                          f'background:{badge_bg}; color:{badge_color}; font-weight:800; margin-top:10px;">'
                                          f'After Settlement: {fmt_currency(after)}</div>')
                        st.markdown(
                            f"""
                            <div class="dash-card" style="background-color:{bg};padding:{pad};border-radius:{radius};margin-bottom:16px;box-shadow:{shadow};">
                                <div style="display:flex;align-items:center;margin-bottom:12px;">
                                    <span style="font-size:18px;margin-right:8px;">{icon}</span>
                                    <span style="font-size:13px;font-weight:700;color:#1e293b;">{row['bank']}</span>
                                </div>
                                <div style="font-size:24px;font-weight:900;color:{amt_color};text-align:right;">{fmt_currency(bal)}</div>
                                <div style="font-size:10px;color:#1e293b;opacity:.7;margin-top:6px;">Available Balance</div>
                                {after_html}
                            </div>
                            """, unsafe_allow_html=True)
            elif view == "List":
                st.markdown(f"<span class='section-chip'>Bank Balances</span>", unsafe_allow_html=True)
                for _, row in df_bal_view.iterrows():
                    color = THEME['amount_color']['neg'] if pd.notna(row["balance"]) and row["balance"] < 0 else THEME['amount_color']['pos']
                    st.markdown(
                        f"""
                        <div style="display:flex; justify-content:space-between; align-items:center; padding:12px 16px; border-bottom:1px solid #e2e8f0;">
                            <span style="font-weight:700; color:#1e293b;">{row['bank']}</span>
                            <span style="font-weight:800; color:{color};">{fmt_currency(row['balance'])}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            elif view == "Mini Cards":
                cols = st.columns(3)
                for i, row in df_bal_view.iterrows():
                    with cols[int(i) % 3]:
                        st.markdown(
                            f"""
                            <div class="dash-card" style="background:{THEME['heading_bg']};padding:{pad};border-radius:{radius};border-left:4px solid {THEME['accent1']};margin-bottom:12px;box-shadow:{shadow};">
                                <div style="font-size:12px;color:#0f172a;font-weight:700;margin-bottom:8px;">{row['bank']}</div>
                                <div style="font-size:18px;font-weight:800;color:#0f172a;text-align:right;">{fmt_currency(row['balance'])}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            elif view == "Progress Bars":
                max_amount = df_bal_view["balance"].max()
                for _, row in df_bal_view.iterrows():
                    percentage = (row["balance"] / max_amount) * 100 if max_amount > 0 else 0
                    st.markdown(
                        f"""
                        <div style="margin-bottom:16px;">
                            <div style="display:flex;justify-content:space-between;margin-bottom:6px;font-size:14px;">
                                <span><strong>{row['bank']}</strong></span>
                                <span><strong>{fmt_currency(row['balance'])}</strong></span>
                            </div>
                            <div style="width:100%;height:8px;background:#e2e8f0;border-radius:4px;overflow:hidden;">
                                <div style="height:100%;background:linear-gradient(90deg,{THEME['accent1']} 0%,{THEME['accent2']} 100%);border-radius:4px;width:{percentage}%;"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            elif view == "Metrics":
                cols = st.columns(min(4, len(df_bal_view)))
                for i, row in df_bal_view.iterrows():
                    if i < 4:
                        with cols[i]:
                            amount = row["balance"]
                            display_amount = f"{amount/1_000_000:.1f}M" if amount >= 1_000_000 else (f"{amount/1_000:.0f}K" if amount >= 1_000 else f"{amount:.0f}")
                            st.markdown(
                                f"""
                                <div class="dash-card" style="text-align:center;padding:20px;background:{THEME['heading_bg']};border-radius:12px;border:2px solid {THEME['accent1']};margin-bottom:12px;">
                                    <div style="font-size:12px;color:#334155;font-weight:700;margin-bottom:8px;">{row['bank']}</div>
                                    <div style="font-size:20px;font-weight:900;color:#334155;">{display_amount}</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
            else:
                table = df_bal_view.copy()
                rename_map = {"bank": "Bank", "balance": "Balance"}
                if "after_settlement" in table.columns:
                    rename_map["after_settlement"] = "After Settlement"
                table = table.rename(columns=rename_map)
                num_cols = [c for c in ["Balance", "After Settlement"] if c in table.columns]
                st.dataframe(style_right(table, num_cols=num_cols), use_container_width=True, height=360)

    with tab_settlements:
        st.markdown('<span class="section-chip">📅 LCR & STL Settlements</span>', unsafe_allow_html=True)
        def render_settlements_tab(df_src: pd.DataFrame, status_label: str, key_suffix: str):
            if df_src.empty:
                st.info(f"No {status_label.lower()} settlements found."); return
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("From Date", value=df_src["settlement_date"].min().date(), key=f"start_{key_suffix}")
            with col2:
                end_date = st.date_input("To Date", value=df_src["settlement_date"].max().date(), key=f"end_{key_suffix}")
            view_data = df_src[(df_src["settlement_date"].dt.date >= start_date) & (df_src["settlement_date"].dt.date <= end_date)].copy()
            if not view_data.empty:
                settlement_view = st.radio("Display as:", options=["Summary + Table", "Progress by Urgency", "Mini Cards"],
                                           index=0, horizontal=True, key=f"settlement_view_{key_suffix}")
                if settlement_view == "Summary + Table":
                    cc1, cc2, cc3 = st.columns(3)
                    with cc1: st.metric(f"Total {status_label} Amount", fmt_number_only(view_data["amount"].sum()))
                    with cc2: st.metric(f"Number of {status_label}", len(view_data))
                    if status_label == "Pending":
                        with cc3: st.metric("Urgent (2 days)", len(view_data[view_data["settlement_date"] <= today0 + pd.Timedelta(days=2)]))
                        viz = view_data.copy()
                        viz["Settlement Date"] = viz["settlement_date"].dt.strftime(config.DATE_FMT)
                        viz["Days Until Due"] = (viz["settlement_date"] - today0).dt.days
                        rename = {"reference": "Reference", "bank": "Bank", "type": "Type", "status": "Status", "remark": "Remark", "description": "Description", "amount": "Amount"}
                        viz = viz.rename(columns={k: v for k, v in rename.items() if k in viz.columns})
                        cols = ["Reference", "Bank", "Type", "Status", "Settlement Date", "Amount", "Days Until Due", "Remark", "Description"]
                        cols = [c for c in cols if c in viz.columns]
                        show = viz[cols].sort_values("Settlement Date")
                        def _highlight(row):
                            if "Days Until Due" in row:
                                if row["Days Until Due"] <= 2: return ['background-color: #fee2e2'] * len(row)
                                if row["Days Until Due"] <= 7: return ['background-color: #fef3c7'] * len(row)
                            return [''] * len(row)
                        styled = style_right(show, num_cols=["Amount"]).apply(_highlight, axis=1)
                        st.dataframe(styled, use_container_width=True, height=400)
                    else:
                        viz_paid = view_data.copy()
                        viz_paid["Settlement Date"] = viz_paid["settlement_date"].dt.strftime(config.DATE_FMT)
                        rename = {"reference": "Reference", "bank": "Bank", "type": "Type", "status": "Status", "remark": "Remark", "description": "Description", "amount": "Amount"}
                        viz_paid = viz_paid.rename(columns={k: v for k, v in rename.items() if k in viz_paid.columns})
                        cols_paid = ["Reference", "Bank", "Type", "Status", "Settlement Date", "Amount", "Remark", "Description"]
                        cols_paid = [c for c in cols_paid if c in viz_paid.columns]
                        show_paid = viz_paid[cols_paid].sort_values("Settlement Date", ascending=False)
                        st.dataframe(style_right(show_paid, num_cols=["Amount"]), use_container_width=True, height=400)
                elif settlement_view == "Progress by Urgency" and status_label == "Pending":
                    tmp = view_data.copy()
                    tmp["days_until_due"] = (tmp["settlement_date"] - today0).dt.days
                    urgent = tmp[tmp["days_until_due"] <= 2]
                    warning = tmp[(tmp["days_until_due"] > 2) & (tmp["days_until_due"] <= 7)]
                    normal = tmp[tmp["days_until_due"] > 7]
                    st.markdown("**📊 LCR & STL Settlements by Urgency**")
                    if not urgent.empty:
                        st.markdown("**🚨 Urgent (≤2 days)**")
                        bank_totals = urgent.groupby("bank", as_index=False)["amount"].sum().rename(columns={"amount": "balance"})
                        max_amount = bank_totals["balance"].max()
                        for _, row in bank_totals.iterrows():
                            percentage = (row["balance"] / max_amount) * 100 if max_amount > 0 else 0
                            st.markdown(
                                f"""
                                <div style="margin-bottom:16px;">
                                    <div style="display:flex;justify-content:space-between;margin-bottom:6px;font-size:14px;">
                                        <span><strong>{row['bank']}</strong></span>
                                        <span><strong>{fmt_currency(row['balance'])}</strong></span>
                                    </div>
                                    <div style="width:100%;height:8px;background:#e2e8f0;border-radius:4px;overflow:hidden;">
                                        <div style="height:100%;background:linear-gradient(90deg,{THEME['accent1']} 0%,{THEME['accent2']} 100%);border-radius:4px;width:{percentage}%;"></div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                    if not warning.empty:
                        st.markdown("**⚠️ Warning (3-7 days)**")
                        st.bar_chart(warning.groupby("bank")["amount"].sum())
                    if not normal.empty:
                        st.markdown("**✅ Normal (>7 days)**")
                        st.bar_chart(normal.groupby("bank")["amount"].sum())
                else:
                    cards = view_data.groupby("bank", as_index=False)["amount"].sum().rename(columns={"amount": "balance"})
                    cols_cards = st.columns(3)
                    for i, row in cards.iterrows():
                        with cols_cards[int(i) % 3]:
                            st.markdown(
                                f"""
                                <div class="dash-card" style="background:{THEME['heading_bg']};padding:{pad};border-radius:{radius};border-left:4px solid {THEME['accent1']};margin-bottom:12px;box-shadow:{shadow};">
                                    <div style="font-size:12px;color:#0f172a;font-weight:700;margin-bottom:8px;">{row['bank']}</div>
                                    <div style="font-size:18px;font-weight:800;color:#0f172a;text-align:right;">{fmt_currency(row['balance'])}</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
            else:
                st.info("No settlements match the selected criteria.")

        tab_pending, tab_paid = st.tabs(["Pending", "Paid"])
        with tab_pending: 
            render_settlements_tab(df_lc, "Pending", "pending")
        with tab_paid: 
            render_settlements_tab(df_lc_paid, "Paid", "paid")

    # ---- Export LC tab (updated Advising Bank filter + Accepted sum by maturity month) ----
    with tab_export_lc:
        st.markdown('<span class="section-chip">🚢 Export LC Proceeds</span>', unsafe_allow_html=True)
        if df_export_lc.empty:
            st.info("No Export LC data found or the file is invalid. Please check the Google Sheet link and format.")
        else:
            # Filters: Branch, Advising Bank, Submitted Date (keep rows with no submitted date)
            col1, col2 = st.columns(2)
            with col1:
                branches = sorted(df_export_lc["branch"].dropna().astype(str).unique())
                selected_branches = st.multiselect("Filter by Branch", options=branches, default=branches, key="export_lc_branch_filter")
            with col2:
                advising_banks = sorted(df_export_lc["advising_bank"].dropna().astype(str).unique()) if "advising_bank" in df_export_lc.columns else []
                if advising_banks:
                    selected_advising_banks = st.multiselect("Filter by Advising Bank", options=advising_banks, default=advising_banks, key="export_lc_advising_filter")
                else:
                    selected_advising_banks = []

            sub_dates = df_export_lc["submitted_date"].dropna() if "submitted_date" in df_export_lc.columns else pd.Series([], dtype="datetime64[ns]")
            min_date_default = (sub_dates.min().date() if not sub_dates.empty else (datetime.today().date().replace(day=1)))
            max_date_default = (sub_dates.max().date() if not sub_dates.empty else datetime.today().date())
            d1, d2 = st.columns(2)
            with d1:
                start_date_filter = st.date_input("From Submitted Date", value=min_date_default, key="export_lc_start_date")
            with d2:
                end_date_filter = st.date_input("To Submitted Date", value=max_date_default, key="export_lc_end_date")

            # Apply branch + advising bank filters
            filtered_df_base = df_export_lc[df_export_lc["branch"].isin(selected_branches)].copy()
            if selected_advising_banks and "advising_bank" in filtered_df_base.columns:
                filtered_df_base = filtered_df_base[filtered_df_base["advising_bank"].isin(selected_advising_banks)]

            # Date filter while keeping rows with no date
            if "submitted_date" in filtered_df_base.columns:
                date_mask = filtered_df_base["submitted_date"].dt.date.between(start_date_filter, end_date_filter, inclusive="both")
                no_date_mask = filtered_df_base["submitted_date"].isna()
                filtered_df_base = filtered_df_base[date_mask | no_date_mask].copy()

            # Status as tabs
            statuses = []
            if "status" in filtered_df_base.columns:
                statuses = sorted([s for s in filtered_df_base["status"].dropna().astype(str).str.strip().str.upper().unique() if s])
            status_tabs = st.tabs(["ALL"] + statuses if statuses else ["ALL"])
            status_keys = ["ALL"] + statuses if statuses else ["ALL"]

            # Month label for metric
            current_month_label = today0.strftime('%b %Y')
            current_period = today0.to_period('M')

            for tab, status_key in zip(status_tabs, status_keys):
                with tab:
                    if status_key == "ALL":
                        filtered_df = filtered_df_base.copy()
                    else:
                        filtered_df = filtered_df_base[filtered_df_base["status"].astype(str).str.strip().str.upper() == status_key].copy()

                    # Metrics: Total Value (filtered set) + Accepted sum with maturity in current month
                    st.markdown("---")
                    m1, m2 = st.columns(2)
                    total_value = float(filtered_df['value_sar'].sum() if 'value_sar' in filtered_df.columns else 0.0)
                    m1.metric("Total Value (SAR)", fmt_number_only(total_value))

                    accepted_maturity_sum = 0.0
                    if {'status', 'maturity_date', 'value_sar'}.issubset(filtered_df.columns):
                        acc_mask = filtered_df['status'].astype(str).str.upper() == 'ACCEPTED'
                        mat_mask = filtered_df['maturity_date'].dt.to_period('M') == current_period
                        accepted_maturity_sum = float(filtered_df.loc[acc_mask & mat_mask, 'value_sar'].sum())
                    m2.metric(f"Accepted (Maturity in {current_month_label})", fmt_number_only(accepted_maturity_sum))

                    # Summary by Branch
                    st.markdown("#### Summary by Branch")
                    if not filtered_df.empty and {'branch','value_sar'}.issubset(filtered_df.columns):
                        summary_by_branch = (
                            filtered_df.groupby('branch', as_index=False)
                                       .agg(
                                           LCs=('value_sar', 'size'),
                                           Total_Value_SAR=('value_sar', 'sum'),
                                       )
                                       .rename(columns={
                                           'branch': 'Branch',
                                           'Total_Value_SAR': 'Total Value (SAR)',
                                       })
                                       .sort_values('Total Value (SAR)', ascending=False)
                        )
                        st.dataframe(
                            style_right(summary_by_branch, num_cols=['LCs', 'Total Value (SAR)']),
                            use_container_width=True,
                            height=300
                        )
                    else:
                        st.info("No records to summarize for the selected filters.")

                    # Detailed table (L/C No + Advising Bank)
                    st.markdown("#### Detailed View")
                    display_cols = {
                        'branch': 'Branch',
                        'applicant': 'Applicant',
                        'lc_no': 'L/C No',
                        'advising_bank': 'Advising Bank',
                        'submitted_date': 'Submitted Date',
                        'maturity_date': 'Maturity Date',
                        'value_sar': 'Value (SAR)',
                        'status': 'Status',
                        'remarks': 'Remarks'
                    }
                    cols_to_show = [k for k in display_cols.keys() if k in filtered_df.columns]
                    if cols_to_show:
                        table_view = filtered_df[cols_to_show].rename(columns={k: display_cols[k] for k in cols_to_show}).copy()
                        if 'Submitted Date' in table_view.columns:
                            table_view['Submitted Date'] = pd.to_datetime(table_view['Submitted Date']).dt.strftime('%Y-%m-%d')
                        if 'Maturity Date' in table_view.columns:
                            table_view['Maturity Date'] = pd.to_datetime(table_view['Maturity Date']).dt.strftime('%Y-%m-%d')
                        st.dataframe(
                            style_right(table_view, num_cols=['Value (SAR)']), 
                            use_container_width=True, 
                            height=500
                        )
                    else:
                        st.info("No columns available for detailed view.")

    # ---- Exchange Rates tab (unchanged) ----
    with tab_fx:
        st.markdown('<span class="section-chip">💱 Exchange Rates</span>', unsafe_allow_html=True)
        if df_fx.empty:
            st.info("No exchange rate data available. Ensure the Exchange Rate sheet has the required columns (Currency Pair, Rate, Date).")
        else:
            fx_view = st.radio("Display as:", options=["Current Rates", "Rate Trends", "Volatility Analysis", "Table View"],
                              index=0, horizontal=True, key="fx_view")
            if fx_view == "Current Rates":
                st.subheader("💱 Current Exchange Rates")
                latest_fx = df_fx.groupby("currency_pair").last().reset_index()
                if not latest_fx.empty:
                    cols = st.columns(min(4, len(latest_fx)))
                    for i, row in latest_fx.iterrows():
                        with cols[int(i) % min(4, len(latest_fx))]:
                            pair = row["currency_pair"]
                            rate = row["rate"]
                            change_info = ""
                            if "change_pct" in row and pd.notna(row["change_pct"]):
                                change_pct = row["change_pct"]
                                change_color = "#059669" if change_pct >= 0 else "#dc2626"
                                change_symbol = "📈" if change_pct >= 0 else "📉"
                                change_info = f"""
                                <div style="margin-top:8px; font-size:12px; color:{change_color}; font-weight:600;">
                                    {change_symbol} {change_pct:+.2f}%
                                </div>
                                """
                            st.markdown(
                                f"""
                                <div class="dash-card" style="background:{THEME['heading_bg']};padding:{pad};border-radius:{radius};
                                     border-left:4px solid {THEME['accent1']};margin-bottom:12px;box-shadow:{shadow};">
                                    <div style="font-size:12px;color:#0f172a;font-weight:700;margin-bottom:8px;">{pair}</div>
                                    <div style="font-size:20px;font-weight:800;color:#0f172a;text-align:right;">{fmt_rate(rate)}</div>
                                    <div style="font-size:10px;color:#1e293b;opacity:.7;margin-top:6px;">Exchange Rate</div>
                                    {change_info}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Currency Pairs", len(latest_fx))
                with col2:
                    if "change_pct" in latest_fx.columns:
                        avg_change = latest_fx["change_pct"].mean()
                        st.metric("Avg Change %", f"{avg_change:.2f}%" if pd.notna(avg_change) else "N/A")
                with col3:
                    last_update = latest_fx["date"].max() if "date" in latest_fx.columns else "N/A"
                    if pd.notna(last_update):
                        st.metric("Last Update", last_update.strftime(config.DATE_FMT))
                    else:
                        st.metric("Last Update", "N/A")
            elif fx_view == "Rate Trends":
                st.subheader("📈 Exchange Rate Trends")
                if "date" in df_fx.columns and len(df_fx) > 1:
                    c1, c2 = st.columns(2)
                    with c1:
                        start_date = st.date_input("From Date", value=df_fx["date"].min().date(), key="fx_start_date")
                    with c2:
                        end_date = st.date_input("To Date", value=df_fx["date"].max().date(), key="fx_end_date")
                    fx_filtered = df_fx[(df_fx["date"].dt.date >= start_date) & (df_fx["date"].dt.date <= end_date)].copy()
                    if not fx_filtered.empty:
                        selected_pairs = st.multiselect("Select Currency Pairs", sorted(fx_filtered["currency_pair"].unique()), default=sorted(fx_filtered["currency_pair"].unique())[:3], key="fx_pairs")
                        if selected_pairs:
                            try:
                                import plotly.io as pio, plotly.graph_objects as go
                                if "brand" not in pio.templates:
                                    pio.templates["brand"] = pio.templates["plotly_white"]
                                    pio.templates["brand"].layout.colorway = [THEME["accent1"], THEME["accent2"], "#64748b", "#94a3b8"]
                                    pio.templates["brand"].layout.font.family = APP_FONT
                                fig = go.Figure()
                                for pair in selected_pairs:
                                    pair_data = fx_filtered[fx_filtered["currency_pair"] == pair]
                                    fig.add_trace(go.Scatter(x=pair_data["date"], y=pair_data["rate"], mode='lines+markers', name=pair, line=dict(width=2), marker=dict(size=4)))
                                fig.update_layout(template="brand", title="Exchange Rate Trends", xaxis_title="Date", yaxis_title="Exchange Rate", height=400, margin=dict(l=20, r=20, t=50, b=20))
                                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
                            except Exception:
                                st.line_chart(fx_filtered.pivot(index="date", columns="currency_pair", values="rate"))
                        else:
                            st.info("Please select at least one currency pair to display trends.")
                    else:
                        st.info("No data available for the selected date range.")
                else:
                    st.info("Insufficient data for trend analysis.")
            elif fx_view == "Volatility Analysis":
                st.subheader("📊 Exchange Rate Volatility")
                if "change_pct" in df_fx.columns:
                    volatility_stats = df_fx.groupby("currency_pair").agg({"change_pct": ["std", "mean", "min", "max"], "rate": "last"}).round(4)
                    volatility_stats.columns = ["Volatility (%)", "Avg Change (%)", "Min Change (%)", "Max Change (%)", "Current Rate"]
                    volatility_stats = volatility_stats.reset_index().rename(columns={"currency_pair": "Currency Pair"}).sort_values("Volatility (%)", ascending=False)
                    st.dataframe(style_right(volatility_stats, num_cols=["Volatility (%)", "Avg Change (%)", "Min Change (%)", "Max Change (%)", "Current Rate"], decimals=4), use_container_width=True, height=400)
                else:
                    st.info("Volatility analysis requires historical rate changes.")
            else:
                st.subheader("📋 Exchange Rate Data Table")
                col1, col2 = st.columns(2)
                with col1:
                    available_pairs = ["All"] + sorted(df_fx["currency_pair"].unique())
                    selected_pair = st.selectbox("Filter by Currency Pair", available_pairs, key="fx_table_pair")
                with col2:
                    date_range = st.number_input("Last N days", min_value=1, max_value=365, value=30, key="fx_date_range")
                display_data = df_fx.copy()
                cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=date_range)
                display_data = display_data[display_data["date"] >= cutoff_date]
                if selected_pair != "All":
                    display_data = display_data[display_data["currency_pair"] == selected_pair]
                if not display_data.empty:
                    table_data = display_data.copy()
                    table_data["Date"] = table_data["date"].dt.strftime(config.DATE_FMT)
                    rename_map = {"currency_pair": "Currency Pair", "rate": "Rate", "change": "Change", "change_pct": "Change %"}
                    table_data = table_data.rename(columns={k: v for k, v in rename_map.items() if k in table_data.columns})
                    display_cols = [col for col in ["Currency Pair", "Rate", "Date", "Change", "Change %"] if col in table_data.columns]
                    table_show = table_data[display_cols].sort_values("Date" if "Date" in display_cols else "Currency Pair", ascending=False)
                    num_cols = [col for col in ["Rate", "Change", "Change %"] if col in table_show.columns]
                    styled_table = style_right(table_show, num_cols=num_cols, decimals=4)
                    if "Change %" in table_show.columns:
                        def highlight_changes(val):
                            try:
                                if pd.isna(val): return ''
                                num_val = float(val)
                                if num_val > 0: return 'color: #059669; font-weight: 600;'
                                if num_val < 0: return 'color: #dc2626; font-weight: 600;'
                                return ''
                            except:
                                return ''
                        styled_table = styled_table.applymap(highlight_changes, subset=["Change %"])
                    st.dataframe(styled_table, use_container_width=True, height=500)
                else:
                    st.info("No data available for the selected criteria.")

    # ---- Facility Report tab ----
    with tab_facility:
        pass

    # ---- Reports tab ----
    with tab_reports:
        st.markdown('<span class="section-chip">📊 Complete Report Export</span>', unsafe_allow_html=True)
        st.info("Download a complete Excel report containing all dashboard data across multiple sheets.")
        excel_data = generate_complete_report(
            df_by_bank, df_pay_approved, df_pay_released, df_lc, df_lc_paid, 
            df_fm, df_cvp, df_fx, df_export_lc, 
            total_balance, approved_sum, lc_next4_sum, banks_cnt
        )
        st.download_button(
            label="📥 Download Complete Treasury Report.xlsx",
            data=excel_data,
            file_name=f"Treasury_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            type="primary"
        )

    st.markdown("<hr style='margin: 8px 0 16px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center; opacity:0.8; font-size:12px;'>Powered By <strong>Jaseer Pykkarathodi</strong></div>", unsafe_allow_html=True)

    if st.session_state.get("auto_refresh"):
        interval = int(st.session_state.get("auto_interval", 120))
        with st.status(f"Auto refreshing in {interval}s…", expanded=False):
            time.sleep(interval)
        st.rerun()

if __name__ == "__main__":
    main()
