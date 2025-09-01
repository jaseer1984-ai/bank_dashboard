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
# - Updates:
#   • All KPI figures in tabs now use the card model
#   • Export LC: Advising Bank filter, L/C No in table, Status as tabs, day-first parsing, DD-MM-YYYY display,
#                and “Accepted (Maturity in current month)” KPI, date filter is on MATURITY DATE
#   • Supplier Payments: robust parser, no NaN text in tables
#   • Settlements: robust date-col detection (incl. NEW MATURITY DATE), day-first parsing, normalized statuses,
#                  2-color highlight on remarks, and urgent settlement warnings restored
#   • Bank Balance: robust parser with fallback, red highlighting of negative numbers in cards
#   • Tables: remove “NAN/NaN/nan/None/null/NaT/NA/N/A” placeholders from display
#   • Weekly Settlement Schedule tooltip formatted with commas

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

# Global font
APP_FONT = os.getenv("APP_FONT", "Inter")
def set_app_font(family: str = APP_FONT):
    css = f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family={family.replace(" ", "+")}:wght@300;400;500;600;700;800&display=swap');
      :root {{ --app-font: '{family}', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', sans-serif; }}
      html, body, [class^="css"], [class*=" css"] {{ font-family: var(--app-font) !important; }}
      .stDataFrame, .stDataFrame * {{ font-variant-numeric: tabular-nums; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
set_app_font()

# ----------------------------
# Theme
# ----------------------------
PALETTES = {
    "Indigo":  {"accent1":"#3b5bfd","accent2":"#2f2fb5","pos":"#0f172a","neg":"#b91c1c",
                "card_best":"#e0e7ff","card_good":"#fce7f3","card_ok":"#e0f2fe",
                "card_low":"#ecfdf5","card_neg":"#fee2e2","heading_bg":"#eef4ff"},
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
  .kpi-card {{
    background: #f8fafc;
    border-radius: 12px;
    padding: 18px;
    text-align: center;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 8px rgba(0,0,0,.04);
  }}
  .kpi-label {{
    font-size: 13px;
    color: #475569;
    font-weight: 600;
    margin-bottom: 8px;
  }}
  .kpi-value {{
    font-size: 26px;
    font-weight: 800;
    color: #1e293b;
  }}
  [data-testid="stTabs"] button[role="tab"] {{ border-radius: 8px !important; margin-right: 6px !important; font-weight: 700 !important; }}
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

def tidy_display(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN-like text with blanks for nice display."""
    if df is None or df.empty: return df
    out = df.copy()
    out = out.replace({np.nan: ""})
    out = out.replace(to_replace=r'(?i)^\s*(nan|none|null|nat|na|n/a)\s*$', value="", regex=True)
    return out

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

def _find_after_settlement_col(columns: pd.Index, df: Optional[pd.DataFrame] = None) -> Optional[str]:
    for col in columns:
        c = str(col).strip().lower()
        if "after" in c and ("settle" in c or "settel" in c): return col
        if "balance after" in c and ("settle" in c or "settel" in c): return col
    if df is not None and not df.empty:
        try:
            head = df.head(5).applymap(lambda x: str(x).strip().lower())
            for col in df.columns:
                if head[col].str.contains(r"(balance\s*)?after\s*sett(el|le)ment", regex=True, na=False).any():
                    return col
        except Exception:
            pass
    return None

def _find_available_col(columns: pd.Index) -> Optional[str]:
    for col in columns:
        c = str(col).strip().lower()
        if "available" in c and "balance" in c:
            return col
    lc = [str(c).lower() for c in columns]
    if "amount" in lc: return "amount"
    if "amount(sar)" in lc: return "amount(sar)"
    if "balance" in lc: return "balance"
    return None

def parse_bank_balance(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[datetime]]:
    """Robust parser: direct 'available balance' columns or fallback to latest date column."""
    try:
        c = cols_lower(df)
        if "bank" in c.columns:
            avail_col = _find_available_col(c.columns)
            after_col = _find_after_settlement_col(c.columns, c)
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

        raw = df.copy().dropna(how="all").dropna(axis=1, how="all")
        bank_col = None
        for col in raw.columns:
            if raw[col].dtype == object:
                non_empty = (raw[col].dropna().astype(str).str.strip() != "").sum()
                if non_empty >= 3:
                    bank_col = col; break
        if bank_col is None:
            raise ValueError("Could not detect bank column")

        parsed = pd.to_datetime(pd.Index(raw.columns), errors="coerce", dayfirst=False)
        date_cols = [col for col, d in zip(raw.columns, parsed) if pd.notna(d)]
        if not date_cols:
            raise ValueError("No valid date columns found")
        date_map = {col: pd.to_datetime(col, errors="coerce", dayfirst=False) for col in date_cols}
        latest_col = max(date_cols, key=lambda c_: date_map[c_])

        after_col = _find_after_settlement_col(raw.columns, raw)
        s = raw[bank_col].astype(str).str.strip()
        mask = s.ne("") & ~s.str.contains("available|total", case=False, na=False)

        keep_cols = [bank_col, latest_col] + ([after_col] if after_col else [])
        sub = raw.loc[mask, keep_cols].copy()
        rename_map = {bank_col: "bank", latest_col: "balance"}
        if after_col: rename_map[after_col] = "after_settlement"
        sub = sub.rename(columns=rename_map)

        sub["balance"] = sub["balance"].astype(str).str.replace(",", "", regex=False).map(_to_number)
        sub["bank"] = sub["bank"].str.replace(r"\s*-\s*.*$", "", regex=True).str.strip()
        if "after_settlement" in sub.columns:
            sub["after_settlement"] = sub["after_settlement"].astype(str).str.replace(",", "", regex=False).map(_to_number)

        latest_date = date_map[latest_col]
        agg = {"balance": "sum"}
        if "after_settlement" in sub.columns: agg["after_settlement"] = "sum"
        by_bank = sub.dropna(subset=["bank"]).groupby("bank", as_index=False).agg(agg)
        if validate_dataframe(by_bank, ["bank", "balance"], "Bank Balance"):
            return by_bank, latest_date
    except Exception as e:
        logger.error(f"parse_bank_balance error: {e}")
    return pd.DataFrame(), None

def parse_supplier_payments(df: pd.DataFrame) -> pd.DataFrame:
    """Robust parser for Supplier Payments; avoids NaN text and column name drift."""
    try:
        d = cols_lower(df).rename(
            columns={
                "supplier name": "supplier",
                "order/sh/branch": "order_branch",
                "amount(sar)": "amount_sar",
                "amount (sar)": "amount_sar",
            }
        )
        if "bank" not in d.columns:
            bank_col = next((c for c in d.columns if c.strip().lower() == "bank"), None)
            if bank_col: d = d.rename(columns={bank_col: "bank"})
        if "status" not in d.columns:
            status_col = next((c for c in d.columns if "status" in c), None)
            if status_col: d = d.rename(columns={status_col: "status"})
        if not validate_dataframe(d, ["bank", "status"], "Supplier Payments"):
            return pd.DataFrame()

        amt_col = None
        for c in ["amount_sar", "amount (sar)", "amount", "value", "payment amount", "total"]:
            if c in d.columns:
                amt_col = c
                break
        if amt_col is None:
            best, hits = None, 0
            for c in d.columns:
                series = pd.to_numeric(d[c].astype(str).str.replace(",", ""), errors="coerce")
                score = series.notna().sum()
                if score > hits:
                    best, hits = c, score
            amt_col = best
        if amt_col is None: return pd.DataFrame()

        out = pd.DataFrame({
            "bank": d["bank"].astype(str).str.strip(),
            "supplier": d.get("supplier", pd.Series("", index=d.index)).astype(str).str.strip(),
            "currency": d.get("currency", pd.Series("", index=d.index)).astype(str).str.strip(),
            "amount": pd.to_numeric(d[amt_col].astype(str).str.replace(",", ""), errors="coerce"),
            "status": d["status"].astype(str).str.strip()
        })
        out = out.dropna(subset=["amount"])
        out = out[out["bank"].ne("")]
        return out
    except Exception as e:
        logger.error(f"parse_supplier_payments error: {e}")
        return pd.DataFrame()

def _normalize_settlement_status(s: Any) -> str:
    u = str(s).strip().upper()
    if u in ("CLOSED", "PAID", "COLLECTED", "SETTLED", "DONE", "REPAID", "REDEEMED"):
        return "CLOSED"
    if u in ("PENDING", "OPEN", "DUE", "UNPAID", "OUTSTANDING", "", "NAN", "NONE", "-"):
        return "PENDING"
    return u

def _pick_settlement_date_col(d: pd.DataFrame) -> Optional[str]:
    cols = list(d.columns)
    lc = [str(c).strip().lower() for c in cols]
    tests = [
        lambda c: ("settle" in c and "date" in c),
        lambda c: ("maturity" in c and "date" in c),
        lambda c: ("due" in c and "date" in c),
        lambda c: (c == "date")
    ]
    for t in tests:
        for col, c in zip(cols, lc):
            if t(c): return col
    best, score = None, 0
    for col in cols:
        series = pd.to_datetime(d[col], errors="coerce", dayfirst=True)
        s = series.notna().sum()
        if s > score:
            best, score = col, s
    return best

def parse_settlements(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        d = cols_lower(df)
        bank_col = next((c for c in d.columns if "bank" in c), None)
        date_col = _pick_settlement_date_col(d)

        amount_col = None
        for c in d.columns:
            cl = str(c).lower()
            if "amount" in cl and "sar" in cl:
                amount_col = c; break
        if not amount_col:
            for c in ["amount(sar)", "amount sar", "amount", "value", "balance due", "currently due", "balance settlement"]:
                if c in d.columns:
                    amount_col = c; break

        if not bank_col or not date_col or not amount_col:
            return pd.DataFrame(), pd.DataFrame()

        status_col = next((c for c in d.columns if "status" in c), None)
        type_col   = next((c for c in d.columns if "type" in c), None)
        remark_col = next((c for c in d.columns if "remark" in c), None)
        ref_col    = next((c for c in d.columns if any(t in c for t in ["a/c", "ref", "account", "reference"])), None)

        out = pd.DataFrame({
            "reference": d[ref_col].astype(str).str.strip() if ref_col else "",
            "bank": d[bank_col].astype(str).str.strip(),
            "settlement_date": pd.to_datetime(d[date_col], errors="coerce", dayfirst=True),
            "amount": pd.to_numeric(d[amount_col].astype(str).str.replace(",", ""), errors="coerce"),
            "status": d[status_col].apply(_normalize_settlement_status) if status_col else "PENDING",
            "type": d[type_col].astype(str).str.upper().str.strip() if type_col else "",
            "remark": d[remark_col].astype(str).str.strip() if remark_col else "",
            "description": ""
        })
        out = out.dropna(subset=["bank", "amount", "settlement_date"])

        df_pending = out[out["status"] == "PENDING"].copy()
        df_closed  = out[out["status"] == "CLOSED"].copy()
        return df_pending.reset_index(drop=True), df_closed.reset_index(drop=True)
    except Exception as e:
        logger.error(f"parse_settlements error: {e}")
        return pd.DataFrame(), pd.DataFrame()

def parse_fund_movement(df: pd.DataFrame) -> pd.DataFrame:
    try:
        d = cols_lower(df)
        if "date" not in d.columns: return pd.DataFrame()
        liq_col = next((c for c in d.columns if ("total" in c and "liquidity" in c)), None)
        if not liq_col: return pd.DataFrame()
        out = pd.DataFrame({
            "date": pd.to_datetime(d["date"], errors="coerce", dayfirst=True),
            "total_liquidity": pd.to_numeric(d[liq_col].astype(str).str.replace(",",""), errors="coerce")
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
            "collection": pd.to_numeric(d["collection"].astype(str).str.replace(",",""), errors="coerce").fillna(0.0),
            "payments": pd.to_numeric(d["payments"].astype(str).str.replace(",",""), errors="coerce").fillna(0.0)
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
        for col in d.columns:
            if col != date_col and col.upper() in ['USD', 'EUR', 'AED', 'QAR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD']:
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
            date_val = pd.to_datetime(row[date_col], errors="coerce", dayfirst=True)
            if pd.isna(date_val):
                continue
            for curr_col in currency_cols:
                rate_val = _to_number(row[curr_col])
                if pd.notna(rate_val) and rate_val > 0:
                    currency_pair = f"{curr_col.upper()}/SAR"
                    result_rows.append({"currency_pair": currency_pair, "rate": rate_val, "date": date_val})
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
    """Parse and clean the combined Export LC data (robust L/C No detection + day-first dates)."""
    try:
        if df.empty: 
            return pd.DataFrame()
        d = cols_lower(df)

        lc_no_col = None
        possible_lc_names = ["l/c no.", "l/c no", "l / c no.", "l / c no", "lc no", "l c no", "l.c. no", "l.c no"]
        for name in possible_lc_names:
            if name in d.columns: lc_no_col = name; break
        if lc_no_col is None:
            for col in d.columns:
                s = str(col).strip().lower()
                if re.search(r'\b(l\s*/\s*c|l\s*c|lc)\b.*no', s):
                    lc_no_col = col; break

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
        if lc_no_col: rename_map[lc_no_col] = 'lc_no'

        d = d.rename(columns=rename_map)

        if 'submitted_date' in d.columns:
            d['submitted_date'] = pd.to_datetime(d['submitted_date'], errors='coerce', dayfirst=True)
        if 'maturity_date' in d.columns:
            d['maturity_date'] = pd.to_datetime(d['maturity_date'], errors='coerce', dayfirst=True)
        if 'value_sar' in d.columns:
            d['value_sar'] = d['value_sar'].apply(_to_number)

        for col in ['branch','issuing_bank','advising_bank','status','lc_no','applicant','remarks']:
            if col in d.columns:
                d[col] = d[col].astype(str).str.strip().str.upper()

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
    
    accepted_export_lc_sum = 0.0
    if not df_export_lc.empty and 'status' in df_export_lc.columns:
        mask = df_export_lc['status'].astype(str).str.strip().str.upper() == 'ACCEPTED'
        accepted_export_lc_sum = float(df_export_lc.loc[mask, 'value_sar'].sum())

    # Sidebar
    render_sidebar({}, total_balance, approved_sum, lc_next4_sum, banks_cnt, accepted_export_lc_sum)
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
                cnt2 = len(neg_after); total_neg2 = df_by_bank.loc[df_by_bank['after_settlement'] < 0, 'after_settlement'].sum()
                names2 = ", ".join(neg_after.sort_values("after_settlement")["bank"].tolist())
                insights.append({"type": "error","title": "Banks Negative After Settlement",
                                 "content": f"{cnt2} bank(s) go negative after settlement (total {fmt_number_only(total_neg2)}). Affected: {names2}."})
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
    # TABS
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
                    total_days_in_month = int((month_end - month_start).days + 1)
                    proj_eom = (opening + avg_daily * total_days_in_month) if pd.notna(avg_daily) else np.nan

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
                    with kpi_d: st.metric("Proj. EOM", fmt_number_only(opening + avg_daily * total_days_in_month if pd.notna(avg_daily) else np.nan))
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
                st.dataframe(style_right(tidy_display(topn), num_cols=num_cols), use_container_width=True, height=320)
            else:
                st.info("No bank balances available.")

        st.markdown("---")

        st.markdown('<span class="section-chip">📅 LCR & STL Settlements — Overview</span>', unsafe_allow_html=True)
        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
        st.markdown('<div style="background:#f1f5f9;padding:12px;border-radius:8px;border-left:4px solid #3b82f6;margin-bottom:16px;"><small>📊 <strong>Metrics show ALL settlements</strong> | 📈 <strong>Chart & table show current month only</strong></small></div>', unsafe_allow_html=True)
        
        if df_lc.empty and df_lc_paid.empty:
            st.info("No LCR & STL data.")
        else:
            try:
                lc_m = df_lc[(df_lc["settlement_date"] >= month_start) & (df_lc["settlement_date"] <= month_end)].copy() if not df_lc.empty else pd.DataFrame()
                all_pending = df_lc.copy() if not df_lc.empty else pd.DataFrame()
                all_paid = df_lc_paid.copy() if not df_lc_paid.empty else pd.DataFrame()
                total_due = (all_pending["amount"].sum() if not all_pending.empty else 0.0) + (all_paid["amount"].sum() if not all_paid.empty else 0.0)

                current_due = 0.0
                if not all_pending.empty:
                    ok = all_pending["remark"].astype(str).str.strip().replace({"-":"","nan":""}).ne("")
                    current_due = all_pending.loc[(all_pending["status"]=="PENDING") & ok, "amount"].sum()
                paid_amount = all_paid["amount"].sum() if not all_paid.empty else 0.0
                balance_due = 0.0
                if not all_pending.empty:
                    empty = all_pending["remark"].astype(str).str.strip().replace({"-":"","nan":""}).eq("")
                    balance_due = all_pending.loc[(all_pending["status"]=="PENDING") & empty, "amount"].sum()

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(
                        f"""
                        <div class="dash-card" style="background:linear-gradient(135deg, #f3e8ff 0%, #faf5ff 100%);
                             padding:24px;border-radius:16px;border-left:6px solid #7c3aed;margin-bottom:20px;
                             box-shadow:0 4px 12px rgba(124,58,237,.15);position:relative;overflow:hidden;">
                            <div style="position:absolute;top:-20px;right:-20px;font-size:60px;opacity:0.1;">💰</div>
                            <div style="font-size:14px;color:#581c87;font-weight:600;margin-bottom:8px;text-transform:uppercase;letter-spacing:1px;">Total Due</div>
                            <div style="font-size:28px;font-weight:900;color:#581c87;margin-bottom:8px;">{fmt_number_only(total_due)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col2:
                    st.markdown(
                        f"""
                        <div class="dash-card" style="background:linear-gradient(135deg, #fee2e2 0%, #fef2f2 100%);
                             padding:24px;border-radius:16px;border-left:6px solid #dc2626;margin-bottom:20px;
                             box-shadow:0 4px 12px rgba(220,38,38,.15);position:relative;overflow:hidden;">
                            <div style="position:absolute;top:-20px;right:-20px;font-size:60px;opacity:0.1;">⚠️</div>
                            <div style="font-size:14px;color:#7f1d1d;font-weight:600;margin-bottom:8px;text-transform:uppercase;letter-spacing:1px;">Current Due</div>
                            <div style="font-size:28px;font-weight:900;color:#7f1d1d;margin-bottom:8px;">{fmt_number_only(current_due)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col3:
                    st.markdown(
                        f"""
                        <div class="dash-card" style="background:linear-gradient(135deg, #dcfce7 0%, #f0fdf4 100%);
                             padding:24px;border-radius:16px;border-left:6px solid #16a34a;margin-bottom:20px;
                             box-shadow:0 4px 12px rgba(22,163,74,.15);position:relative;overflow:hidden;">
                            <div style="position:absolute;top:-20px;right:-20px;font-size:60px;opacity:0.1;">✅</div>
                            <div style="font-size:14px;color:#14532d;font-weight:600;margin-bottom:8px;text-transform:uppercase;letter-spacing:1px;">Paid</div>
                            <div style="font-size:28px;font-weight:900;color:#14532d;margin-bottom:8px;">{fmt_number_only(paid_amount)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col4:
                    st.markdown(
                        f"""
                        <div class="dash-card" style="background:linear-gradient(135deg, #fef3c7 0%, #fffbeb 100%);
                             padding:24px;border-radius:16px;border-left:6px solid #d97706;margin-bottom:20px;
                             box-shadow:0 4px 12px rgba(217,119,6,.15);position:relative;overflow:hidden;">
                            <div style="position:absolute;top:-20px;right:-20px;font-size:60px;opacity:0.1;">📋</div>
                            <div style="font-size:14px;color:#92400e;font-weight:600;margin-bottom:8px;text-transform:uppercase;letter-spacing:1px;">Balance Due</div>
                            <div style="font-size:28px;font-weight:900;color:#92400e;margin-bottom:8px;">{fmt_number_only(balance_due)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                if not lc_m.empty:
                    weekly = (lc_m
                              .assign(week=lambda df_: df_["settlement_date"].dt.isocalendar().week.astype(int))
                              .groupby("week", as_index=False)["amount"].sum()
                              .sort_values("week"))
                    try:
                        import plotly.graph_objects as go
                        fig = go.Figure(go.Bar(
                            x=[f"Week {int(w)}" for w in weekly["week"]],
                            y=weekly["amount"],
                            marker_color=THEME["accent1"],
                            text=[f"SAR {v:,.0f}" for v in weekly["amount"]],
                            textposition="outside",
                            hovertemplate=" %{x}<br>Amount: SAR %{y:,.0f}<extra></extra>"
                        ))
                        fig.update_layout(template="plotly_white",
                                          height=350,
                                          margin=dict(l=20, r=20, t=20, b=40),
                                          xaxis_title="", yaxis_title="Amount (SAR)",
                                          showlegend=False)
                        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
                    except Exception:
                        st.bar_chart(weekly.set_index("week")["amount"])
            except Exception as e:
                st.error(f"Unable to render Settlements section: {e}")

        st.markdown("---")

        if st.session_state.get("show_fx", True) and not df_fx.empty:
            st.subheader("Exchange Rates — Month Overview")
            fx_m = df_fx[(df_fx["date"] >= month_start) & (df_fx["date"] <= month_end)].copy()
            if not fx_m.empty:
                f1, f2 = st.columns(2)
                with f1:
                    latest_fx = fx_m.groupby("currency_pair").last().reset_index()
                    fx_display = latest_fx[["currency_pair", "rate"]].rename(
                        columns={"currency_pair": "Pair", "rate": "Current Rate"})
                    st.dataframe(style_right(fx_display, num_cols=["Current Rate"], decimals=4), use_container_width=True, height=200)
                with f2:
                    if "change_pct" in fx_m.columns:
                        volatility = fx_m.groupby("currency_pair")["change_pct"].std().reset_index()
                        volatility = volatility.rename(columns={"currency_pair": "Pair", "change_pct": "Volatility %"})
                        st.dataframe(style_right(volatility, num_cols=["Volatility %"], decimals=2), use_container_width=True, height=200)
            else:
                st.info("No FX data for current month.")

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
                    st.caption("Scope: MTD if available, else all Export LC records.")
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

    # ---- Bank tab ----
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
                        bal = row.get('balance', np.nan)
                        after = row.get('after_settlement', np.nan)
                        bal_color = THEME["amount_color"]["neg"] if pd.notna(bal) and bal < 0 else THEME["amount_color"]["pos"]
                        after_color = THEME["amount_color"]["neg"] if pd.notna(after) and after < 0 else "#065f46"
                        after_line = ""
                        if pd.notna(after):
                            after_line = f"<div style='font-size:13px;font-weight:800;color:{after_color};margin-top:6px;text-align:right;'>After Settlement: {fmt_currency(after)}</div>"
                        st.markdown(
                            f"""
                            <div class="dash-card" style="background:{THEME['heading_bg']};padding:{pad};border-radius:{radius};margin-bottom:16px;box-shadow:{shadow};border-left:4px solid {THEME['accent1']};">
                                <div style="font-weight:700;margin-bottom:8px;">{row['bank']}</div>
                                <div style="font-size:22px;font-weight:900;color:{bal_color};text-align:right;">{fmt_currency(bal)}</div>
                                {after_line}
                            </div>
                            """, unsafe_allow_html=True)
            elif view == "List":
                display_as_list(df_bal_view, "bank", "balance", "Bank Balances")
            elif view == "Mini Cards":
                display_as_mini_cards(df_bal_view, "bank", "balance", pad=pad, radius=radius, shadow=shadow)
            elif view == "Progress Bars":
                display_as_progress_bars(df_bal_view, "bank", "balance")
            elif view == "Metrics":
                display_as_metrics(df_bal_view, "bank", "balance")
            else:
                table = df_bal_view.copy()
                rename_map = {"bank": "Bank", "balance": "Balance"}
                if "after_settlement" in table.columns:
                    rename_map["after_settlement"] = "After Settlement"
                table = table.rename(columns=rename_map)
                st.dataframe(style_right(table, num_cols=[c for c in ["Balance","After Settlement"] if c in table.columns]),
                             use_container_width=True, height=360)

    # ---- Settlements tab ----
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
                    with cc1:
                        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Total {status_label} Amount</div><div class="kpi-value">{fmt_number_only(view_data["amount"].sum())}</div></div>""", unsafe_allow_html=True)
                    with cc2:
                        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Number of {status_label}</div><div class="kpi-value">{len(view_data)}</div></div>""", unsafe_allow_html=True)
                    if status_label == "Pending":
                        with cc3:
                            st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Urgent (2 days)</div><div class="kpi-value">{len(view_data[view_data["settlement_date"] <= today0 + pd.Timedelta(days=2)])}</div></div>""", unsafe_allow_html=True)

                    viz = view_data.copy()
                    viz["Settlement Date"] = viz["settlement_date"].dt.strftime(config.DATE_FMT)
                    rename = {"reference": "Reference", "bank": "Bank", "type": "Type", "status": "Status",
                              "remark": "Remark", "description": "Description", "amount": "Amount"}
                    viz = viz.rename(columns={k: v for k, v in rename.items() if k in viz.columns})
                    cols = ["Reference", "Bank", "Type", "Status", "Settlement Date", "Amount", "Remark", "Description"]
                    cols = [c for c in cols if c in viz.columns]
                    table_df = tidy_display(viz[cols].sort_values("Settlement Date", ascending=(status_label=="Pending")))

                    def highlight_by_remark(row):
                        r = str(row.get("Remark", "")).strip().upper()
                        if r and r not in ("-", "NAN", "NONE", "NULL"):
                            return ['background-color: #fef3c7'] * len(row)
                        else:
                            return ['background-color: #fee2e2'] * len(row)
                    styled = style_right(table_df, num_cols=["Amount"]).apply(highlight_by_remark, axis=1)
                    st.dataframe(styled, use_container_width=True, height=420)
                
                elif settlement_view == "Progress by Urgency" and status_label == "Pending":
                    tmp = view_data.copy()
                    tmp["days_until_due"] = (tmp["settlement_date"] - today0).dt.days
                    urgent = tmp[tmp["days_until_due"] <= 2]
                    warning = tmp[(tmp["days_until_due"] > 2) & (tmp["days_until_due"] <= 7)]
                    normal = tmp[tmp["days_until_due"] > 7]
                    st.markdown("**📊 LCR & STL Settlements by Urgency**")
                    if not urgent.empty:
                        st.markdown("**🚨 Urgent (≤2 days)**")
                        display_as_progress_bars(urgent.groupby("bank", as_index=False)["amount"].sum().rename(columns={"amount": "balance"}))
                    if not warning.empty:
                        st.markdown("**⚠️ Warning (3-7 days)**")
                        display_as_progress_bars(warning.groupby("bank", as_index=False)["amount"].sum().rename(columns={"amount": "balance"}))
                    if not normal.empty:
                        st.markdown("**✅ Normal (>7 days)**")
                        display_as_progress_bars(normal.groupby("bank", as_index=False)["amount"].sum().rename(columns={"amount": "balance"}))
                
                elif settlement_view == "Progress by Urgency" and status_label == "Paid":
                    st.info("Progress by urgency view is only available for pending settlements.")
                    bank_totals = view_data.groupby("bank", as_index=False)["amount"].sum().rename(columns={"amount": "balance"})
                    display_as_progress_bars(bank_totals, "bank", "balance")
                
                else:
                    cards = view_data.groupby("bank", as_index=False)["amount"].sum().rename(columns={"amount": "balance"})
                    display_as_mini_cards(cards, "bank", "balance", pad=pad, radius=radius, shadow=shadow)
            else:
                st.info("No settlements match the selected criteria.")
        
        tab_pending, tab_paid = st.tabs(["Pending", "Paid"])
        with tab_pending: 
            render_settlements_tab(df_lc, "Pending", "pending")
        with tab_paid: 
            render_settlements_tab(df_lc_paid, "Paid", "paid")

    # ---- Supplier Payments tab ----
    with tab_payments:
        st.markdown('<span class="section-chip">💰 Supplier Payments</span>', unsafe_allow_html=True)
        def render_payments_tab(df_src: pd.DataFrame, status_label: str, key_suffix: str):
            if df_src.empty:
                st.info(f"No {status_label.lower()} payments found."); return
            col1, col2 = st.columns([2, 1])
            with col1:
                banks = sorted(df_src["bank"].dropna().unique())
                pick_banks = st.multiselect("Filter by Bank", banks, default=banks, key=f"banks_{key_suffix}")
            with col2:
                min_amount = st.number_input("Minimum Amount", min_value=0, value=0, key=f"min_{key_suffix}")
            view_data = df_src[(df_src["bank"].isin(pick_banks)) & (df_src["amount"] >= min_amount)].copy()
            if not view_data.empty:
                payment_view = st.radio("Display as:", options=["Summary + Table", "Mini Cards", "List", "Progress Bars"],
                                        index=0, horizontal=True, key=f"payment_view_{key_suffix}")
                if payment_view == "Summary + Table":
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(
                            f"""
                            <div class="kpi-card">
                                <div class="kpi-label">Total {status_label} Amount</div>
                                <div class="kpi-value">{fmt_number_only(view_data["amount"].sum())}</div>
                            </div>
                            """,
                            unsafe_allow_html=True)
                    with c2:
                        st.markdown(
                            f"""
                            <div class="kpi-card">
                                <div class="kpi-label">Number of Payments</div>
                                <div class="kpi-value">{len(view_data)}</div>
                            </div>
                            """,
                            unsafe_allow_html=True)
                    with c3:
                        st.markdown(
                            f"""
                            <div class="kpi-card">
                                <div class="kpi-label">Average Payment</div>
                                <div class="kpi-value">{fmt_number_only(view_data["amount"].mean())}</div>
                            </div>
                            """,
                            unsafe_allow_html=True)
                    st.markdown("**📊 Summary by Bank**")
                    grp = (view_data.groupby("bank", as_index=False)["amount"]
                           .sum().sort_values("amount", ascending=False)
                           .rename(columns={"bank": "Bank", "amount": "Amount"}))
                    st.dataframe(style_right(grp, num_cols=["Amount"]), use_container_width=True, height=220)
                    st.markdown("**📋 Detailed Payment List**")
                    show_cols = [c for c in ["bank", "supplier", "currency", "amount", "status"] if c in view_data.columns]
                    v = view_data[show_cols].rename(columns={"bank": "Bank", "supplier": "Supplier", "currency": "Currency",
                                                             "amount": "Amount", "status": "Status"})
                    v = tidy_display(v)
                    st.dataframe(style_right(v, num_cols=["Amount"]), use_container_width=True, height=360)
                elif payment_view == "Mini Cards":
                    bank_totals = view_data.groupby("bank", as_index=False)["amount"].sum().rename(columns={"amount": "balance"})
                    display_as_mini_cards(bank_totals, "bank", "balance", pad=pad, radius=radius, shadow=shadow)
                elif payment_view == "List":
                    bank_totals = view_data.groupby("bank", as_index=False)["amount"].sum().rename(columns={"amount": "balance"})
                    display_as_list(bank_totals, "bank", "balance", f"{status_label} Payments by Bank")
                else:
                    bank_totals = view_data.groupby("bank", as_index=False)["amount"].sum().rename(columns={"amount": "balance"})
                    display_as_progress_bars(bank_totals, "bank", "balance")
            else:
                st.info("No payments match the selected criteria.")
        tab_approved, tab_released = st.tabs(["Approved", "Released"])
        with tab_approved: render_payments_tab(df_pay_approved, "Approved", "approved")
        with tab_released: render_payments_tab(df_pay_released, "Released", "released")

    # ---- Export LC tab ----
    with tab_export_lc:
        st.markdown('<span class="section-chip">🚢 Export LC Proceeds</span>', unsafe_allow_html=True)
        if df_export_lc.empty:
            st.info("No Export LC data found or the file is invalid. Please check the Google Sheet link and format.")
        else:
            # Filters: Branch, Advising Bank, Maturity Date
            col1, col2 = st.columns(2)
            with col1:
                branches = sorted(df_export_lc["branch"].dropna().astype(str).unique())
                selected_branches = st.multiselect("Filter by Branch", options=branches, default=branches, key="export_lc_branch_filter")
            with col2:
                advising_banks = sorted(df_export_lc["advising_bank"].dropna().astype(str).unique()) if "advising_bank" in df_export_lc.columns else []
                selected_advising_banks = st.multiselect("Filter by Advising Bank", options=advising_banks, default=advising_banks, key="export_lc_advising_filter") if advising_banks else []

            mat_dates = df_export_lc["maturity_date"].dropna() if "maturity_date" in df_export_lc.columns else pd.Series([], dtype="datetime64[ns]")
            min_date_default = (mat_dates.min().date() if not mat_dates.empty else (datetime.today().date().replace(day=1)))
            max_date_default = (mat_dates.max().date() if not mat_dates.empty else datetime.today().date())
            d1, d2 = st.columns(2)
            with d1:
                start_date_filter = st.date_input("From Maturity Date", value=min_date_default, key="export_lc_start_date")
            with d2:
                end_date_filter = st.date_input("To Maturity Date", value=max_date_default, key="export_lc_end_date")

            # Apply filters
            filtered_df_base = df_export_lc[df_export_lc["branch"].isin(selected_branches)].copy()
            if selected_advising_banks and "advising_bank" in filtered_df_base.columns:
                filtered_df_base = filtered_df_base[filtered_df_base["advising_bank"].isin(selected_advising_banks)]
            if "maturity_date" in filtered_df_base.columns:
                date_mask = filtered_df_base["maturity_date"].dt.date.between(start_date_filter, end_date_filter, inclusive="both")
                no_date_mask = filtered_df_base["maturity_date"].isna()
                filtered_df_base = filtered_df_base[date_mask | no_date_mask].copy()

            # Status tabs
            statuses = sorted([s for s in filtered_df_base["status"].dropna().astype(str).str.strip().str.upper().unique()]) if "status" in filtered_df_base.columns else []
            status_tabs = st.tabs(["ALL"] + statuses if statuses else ["ALL"])
            status_keys = ["ALL"] + statuses if statuses else ["ALL"]

            try:
                now_local = pd.Timestamp.now(tz=config.TZ).tz_localize(None).normalize()
            except Exception:
                now_local = pd.Timestamp.today().normalize()
            current_period = now_local.to_period('M')
            current_month_label = now_local.strftime('%b %Y')

            for tab, status_key in zip(status_tabs, status_keys):
                with tab:
                    if status_key == "ALL":
                        filtered_df = filtered_df_base.copy()
                    else:
                        filtered_df = filtered_df_base[filtered_df_base["status"].astype(str).str.strip().str.upper() == status_key].copy()

                    st.markdown("---")
                    m1, m2 = st.columns(2)
                    with m1:
                        st.markdown(
                            f"""
                            <div class="kpi-card">
                                <div class="kpi-label">Total Value (SAR)</div>
                                <div class="kpi-value">{fmt_number_only(filtered_df['value_sar'].sum() if 'value_sar' in filtered_df.columns else 0.0)}</div>
                            </div>
                            """,
                            unsafe_allow_html=True)
                    with m2:
                        accepted_month_sum = 0.0
                        if not filtered_df.empty and {'status','maturity_date','value_sar'}.issubset(filtered_df.columns):
                            mask_acc = filtered_df['status'].astype(str).str.upper() == 'ACCEPTED'
                            mask_mat = filtered_df['maturity_date'].dt.to_period('M') == current_period
                            accepted_month_sum = float(filtered_df.loc[mask_acc & mask_mat, 'value_sar'].sum())
                        st.markdown(
                            f"""
                            <div class="kpi-card">
                                <div class="kpi-label">Accepted (Maturity in {current_month_label})</div>
                                <div class="kpi-value">{fmt_number_only(accepted_month_sum)}</div>
                            </div>
                            """,
                            unsafe_allow_html=True)

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
                        st.dataframe(style_right(summary_by_branch, num_cols=['LCs', 'Total Value (SAR)']),
                                     use_container_width=True, height=300)
                    else:
                        st.info("No records to summarize for the selected filters.")

                    # Detailed table
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
                            table_view['Submitted Date'] = pd.to_datetime(table_view['Submitted Date'], errors='coerce').dt.strftime('%d-%m-%Y')
                        if 'Maturity Date' in table_view.columns:
                            table_view['Maturity Date'] = pd.to_datetime(table_view['Maturity Date'], errors='coerce').dt.strftime('%d-%m-%Y')
                        table_view = tidy_display(table_view)
                        st.dataframe(style_right(table_view, num_cols=['Value (SAR)']), use_container_width=True, height=500)
                    else:
                        st.info("No columns available for detailed view.")

    # ---- Exchange Rates tab ----
    with tab_fx:
        st.markdown('<span class="section-chip">💱 Exchange Rates</span>', unsafe_allow_html=True)
        if df_fx.empty:
            st.info("No exchange rate data available.")
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
                            st.markdown(
                                f"""
                                <div class="kpi-card">
                                    <div class="kpi-label">{row["currency_pair"]}</div>
                                    <div class="kpi-value">{row['rate']:.4f}</div>
                                </div>
                                """,
                                unsafe_allow_html=True)
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Currency Pairs</div><div class="kpi-value">{len(latest_fx)}</div></div>""", unsafe_allow_html=True)
                with col2:
                    if "change_pct" in latest_fx.columns:
                        avg_change = latest_fx["change_pct"].mean()
                        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Avg Change %</div><div class="kpi-value">{avg_change:.2f}%</div></div>""", unsafe_allow_html=True)
                with col3:
                    last_update = latest_fx["date"].max() if "date" in latest_fx.columns else "N/A"
                    if pd.notna(last_update):
                        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Last Update</div><div class="kpi-value">{last_update.strftime(config.DATE_FMT)}</div></div>""", unsafe_allow_html=True)
            
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
                        available_pairs = sorted(fx_filtered["currency_pair"].unique())
                        selected_pairs = st.multiselect("Select Currency Pairs", available_pairs, default=available_pairs[:3], key="fx_pairs")
                        if selected_pairs:
                            pivot = fx_filtered[fx_filtered["currency_pair"].isin(selected_pairs)].pivot(index="date", columns="currency_pair", values="rate")
                            st.line_chart(pivot)
                        else:
                            st.info("Please select at least one currency pair to display trends.")
                    else:
                        st.info("No data available for the selected date range.")
                else:
                    st.info("Insufficient data for trend analysis.")
            
            elif fx_view == "Volatility Analysis":
                st.subheader("📊 Exchange Rate Volatility")
                if "change_pct" in df_fx.columns:
                    volatility_stats = df_fx.groupby("currency_pair").agg({
                        "change_pct": ["std", "mean", "min", "max"],
                        "rate": "last"
                    }).round(4)
                    volatility_stats.columns = ["Volatility (%)", "Avg Change (%)", "Min Change (%)", "Max Change (%)", "Current Rate"]
                    volatility_stats = volatility_stats.reset_index().rename(columns={"currency_pair": "Currency Pair"}).sort_values("Volatility (%)", ascending=False)
                    st.dataframe(style_right(volatility_stats, num_cols=["Volatility (%)", "Avg Change (%)", "Min Change (%)", "Max Change (%)", "Current Rate"], decimals=4),
                                 use_container_width=True, height=400)
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
                    display_cols = [c for c in ["Currency Pair", "Rate", "Date", "Change", "Change %"] if c in table_data.columns]
                    table_show = tidy_display(table_data[display_cols].sort_values("Date" if "Date" in display_cols else "Currency Pair", ascending=False))
                    num_cols = [col for col in ["Rate", "Change", "Change %"] if col in table_show.columns]
                    styled_table = style_right(table_show, num_cols=num_cols, decimals=4)
                    if "Change %" in table_show.columns:
                        def highlight_changes(val):
                            try:
                                if val == "" or pd.isna(val): return ''
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
