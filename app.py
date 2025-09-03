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
# - UPDATE: Export LC tab now shows L/C No in table, uses Advising Bank instead of Issuing Bank in table,
#           Status moved to tabs, and added Issuing Bank filter. Parsing of L/C No made robust.
# - CHANGE: Export LC tab top-level date filters now use Maturity Date (not Submitted Date).
# - CHANGE: Export LC Detailed View removes 'None'/NaT, adds table-only Maturity Date filters, and formats Maturity Date as DD-MM-YYYY.
# - UPDATE: Removed auto refresh.
# - UPDATE: "Accepted Due this Month (SAR)" metric in Export LC tab now sums "MATURING CURRENT MONTH" column for 'ACCEPTED' status.
# - UPDATE: Export LC Detailed View table now displays 'Maturity Date' (formatted DD-MM-YYYY) and excludes 'Submitted Date'.
# - FIX: Ensured 'Maturity Date' column is explicitly included and correctly formatted in Export LC Detailed View,
#        and added robust parsing for 'MATURITY DATE (DT FORMAT)' column name from the Excel sheet.

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
      /* Do NOT hide the toolbar—this holds the sidebar toggle */
      /* [data-testid="stToolbar"] {{ display: none !important; }} */
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

  /* Streamlit tabs colorization (index-based styling) */
  [data-testid="stTabs"] button[role="tab"] {{
    border-radius: 8px !important;
    margin-right: 6px !important;
    font-weight: 700 !important;
  }}
  /* Overview */
  [data-testid="stTabs"] button[role="tab"]:nth-child(1) {{ background:#e0e7ff; color:#1e293b; }}
  [data-testid="stTabs"] button[role="tab"][aria-selected="true"]:nth-child(1) {{ background:#c7d2fe; }}
  /* Bank */
  [data-testid="stTabs"] button[role="tab"]:nth-child(2) {{ background:#ccfbf1; color:#0f172a; }}
  [data-testid="stTabs"] button[role="tab"][aria-selected="true"]:nth-child(2) {{ background:#99f6e4; }}
  /* Settlements */
  [data-testid="stTabs"] button[role="tab"]:nth-child(3) {{ background:#e0f2fe; color:#0f172a; }}
  [data-testid="stTabs"] button[role="tab"][aria-selected="true"]:nth-child(3) {{ background:#bae6fd; }}
  /* Supplier Payments */
  [data-testid="stTabs"] button[role="tab"]:nth-child(4) {{ background:#dcfce7; color:#0f172a; }}
  [data-testid="stTabs"] button[role="tab"][aria-selected="true"]:nth-child(4) {{ background:#bbf7d0; }}
  /* Export LC (now 5th) */
  [data-testid="stTabs"] button[role="tab"]:nth-child(5) {{ background:#ffedd5; color:#0f172a; }}
  [data-testid="stTabs"] button[role="tab"][aria-selected="true"]:nth-child(5) {{ background:#fed7aa; }}
  /* Exchange Rates (now 6th) */
  [data-testid="stTabs"] button[role="tab"]:nth-child(6) {{ background:#fef3c7; color:#0f172a; }}
  [data-testid="stTabs"] button[role="tab"][aria-selected="true"]:nth-child(6) {{ background:#fde68a; }}
  /* Facility Report (now 7th) */
  [data-testid="stTabs"] button[role="tab"]:nth-child(7) {{ background:#f1f5f9; color:#0f172a; }}
  [data-testid="stTabs"] button[role="tab"][aria-selected="true"]:nth-child(7) {{ background:#e2e8f0; }}
  /* Reports (now 8th) */
  [data-testid="stTabs"] button[role="tab"]:nth-child(8) {{ background:#f3e8ff; color:#0f172a; }}
  [data-testid="stTabs"] button[aria-selected="true"]:nth-child(8) {{ background:#e9d5ff; }}
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
# Links - Added Export LC link
# ----------------------------
LINKS = {
    "BANK BALANCE": f"https://docs.google.com/spreadsheets/d/{config.FILE_ID}/export?format=csv&gid=860709395",
    "SUPPLIER PAYMENTS": f"https://docs.google.com/spreadsheets/d/{config.FILE_ID}/export?format=csv&gid=20805295",
    "SETTLEMENTS": f"https://docs.google.com/spreadsheets/d/{config.FILE_ID}/export?format=csv&gid=978859477",
    "Fund Movement": f"https://docs.google.com/spreadsheets/d/{config.FILE_ID}/export?format=csv&gid=66055663",
    "COLLECTION_BRANCH": f"https://docs.google.com/spreadsheets/d/{config.FILE_ID}/export?format=csv&gid=457517415",
    "EXCHANGE_RATE": f"https://docs.google.com/spreadsheets/d/{config.FILE_ID}/export?format=csv&gid=58540369",
    "EXPORT_LC_1": "https://docs.google.com/spreadsheets/d/e/2PACX-1vRPcr4Mo_ELNbhRer8xuonW9sF1rvBb3kG2W4hKSUI3d_ZRV5Rou_Y-G1HmcW7StQ/pub?output=xlsx",
    "EXPORT_LC_2": "https://docs.google.com/spreadsheets/d/e/2PACX-1vSfkIoyzXyzw5Q-xcc1lrdVJL41croZix9S8Q0lsXJEDjiCSTTFn980edt8jFXH6g/pub?output=xlsx",
    "EXPORT_LC_3": "https://docs.google.com/spreadsheets/d/e/2PACX-1vSrbVcu8T64yTgx9ofcAVvBXvRtHBKoQF7a088Gp0GMQAWyB_fGv9QjRdSUcdbCug/pub?output=xlsx",
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
    """Reads all sheets from an Excel file URL and combines them."""
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
# Display helpers
# ----------------------------
def display_as_list(df, bank_col="bank", amount_col="balance", title="Bank Balances"):
    st.markdown(f"<span class='section-chip'>{title}</span>", unsafe_allow_html=True)
    for _, row in df.iterrows():
        color = THEME['amount_color']['neg'] if pd.notna(row[amount_col]) and row[amount_col] < 0 else THEME['amount_color']['pos']
        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between; align-items:center; padding:12px 16px; border-bottom:1px solid #e2e8f0;">
                <span style="font-weight:700; color:#1e293b;">{row[bank_col]}</span>
                <span style="font-weight:800; color:{color};">{fmt_currency(row[amount_col])}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

def display_as_mini_cards(df, bank_col="bank", amount_col="balance", pad="20px", radius="12px", shadow="0 2px 8px rgba(0,0,0,0.1)"):
    cols = st.columns(3)
    for i, row in df.iterrows():
        with cols[int(i) % 3]:
            st.markdown(
                f"""
                <div class="dash-card" style="background:{THEME['heading_bg']};padding:{pad};border-radius:{radius};border-left:4px solid {THEME['accent1']};margin-bottom:12px;box-shadow:{shadow};">
                    <div style="font-size:12px;color:#0f172a;font-weight:700;margin-bottom:8px;">{row[bank_col]}</div>
                    <div style="font-size:18px;font-weight:800;color:#0f172a;text-align:right;">{fmt_currency(row[amount_col])}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

def display_as_progress_bars(df, bank_col="bank", amount_col="balance"):
    max_amount = df[amount_col].max()
    for _, row in df.iterrows():
        percentage = (row[amount_col] / max_amount) * 100 if max_amount > 0 else 0
        st.markdown(
            f"""
            <div style="margin-bottom:16px;">
                <div style="display:flex;justify-content:space-between;margin-bottom:6px;font-size:14px;">
                    <span><strong>{row[bank_col]}</strong></span>
                    <span><strong>{fmt_currency(row[amount_col])}</strong></span>
                </div>
                <div style="width:100%;height:8px;background:#e2e8f0;border-radius:4px;overflow:hidden;">
                    <div style="height:100%;background:linear-gradient(90deg,{THEME['accent1']} 0%,{THEME['accent2']} 100%);border-radius:4px;width:{percentage}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

def display_as_metrics(df, bank_col="bank", amount_col="balance"):
    cols = st.columns(min(4, len(df)))
    for i, row in df.iterrows():
        if i < 4:
            with cols[i]:
                amount = row[amount_col]
                display_amount = f"{amount/1_000_000:.1f}M" if amount >= 1_000_000 else (f"{amount/1_000:.0f}K" if amount >= 1_000 else f"{amount:.0f}")
                st.markdown(
                    f"""
                    <div class="dash-card" style="text-align:center;padding:20px;background:{THEME['heading_bg']};border-radius:12px;border:2px solid {THEME['accent1']};margin-bottom:12px;">
                        <div style="font-size:12px;color:#334155;font-weight:700;margin-bottom:8px;">{row[bank_col]}</div>
                        <div style="font-size:20px;font-weight:900;color:#334155;">{display_amount}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

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
        if bank_col is None: raise ValueError("Could not detect bank column")

        parsed = pd.to_datetime(pd.Index(raw.columns), errors="coerce", dayfirst=False)
        date_cols = [col for col, d in zip(raw.columns, parsed) if pd.notna(d)]
        if not date_cols: raise ValueError("No valid date columns found")
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
        if after_col:
            sub["after_settlement"] = sub["after_settlement"].astype(str).str.replace(",", "", regex=False).map(_to_number)

        latest_date = date_map[latest_col]
        agg = {"balance": "sum"}
        if after_col: agg["after_settlement"] = "sum"
        by_bank = sub.dropna(subset=["bank"]).groupby("bank", as_index=False).agg(agg)
        if validate_dataframe(by_bank, ["bank", "balance"], "Bank Balance"):
            return by_bank, latest_date
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
            col_lower = str(col).strip().lower()
            if "status" in col_lower:
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
            "type": d[type_col].astype(str).str.upper().str.strip() if type_col else "",
            "remark": d[remark_col].astype(str).str.strip() if remark_col else "",
            "description": ""
        })

        out = out.dropna(subset=["bank", "amount", "settlement_date"])

        df_pending = pd.DataFrame()
        df_paid = pd.DataFrame()
        
        if status_col:
            df_pending = out[out["status"].str.upper().str.strip() == "PENDING"].copy()
            df_paid = out[out["status"].str.upper().str.strip() == "CLOSED"].copy()
        else:
            df_pending = out.copy()

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
    """Parse exchange rates data from the spreadsheet"""
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
                    numeric_count = sum(1 for val in sample_vals if pd.notna(_to_number(val)))
                    if not sample_vals.empty and numeric_count >= len(sample_vals) * 0.8:
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
                    currency_pair = f"{curr_col.upper()}/SAR"
                    result_rows.append({
                        "currency_pair": currency_pair,
                        "rate": rate_val,
                        "date": date_val
                    })
        if not result_rows:
            return pd.DataFrame()
        out = pd.DataFrame(result_rows)
        if len(out) > 1:
            out = out.sort_values(["currency_pair", "date"])
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

        # Robustly detect L/C No column (handles: 'l/c no.', 'l / c no', 'lc no', 'l c no', etc.)
        lc_no_col = None
        possible_lc_names = [
            "l/c no.", "l/c no", "l / c no.", "l / c no", "lc no", "l c no", "l.c. no", "l.c no"
        ]
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

        # Build rename map
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
            # Robustly map 'maturity date' and its variations
            'maturity date (dt format)': 'maturity_date', # Specific to your new sheet column name
            'maturity date': 'maturity_date',
            'date of maturity': 'maturity_date',
            'due date': 'maturity_date',
            'lc maturity date': 'maturity_date',
            'status': 'status',
            'remarks': 'remarks',
            'branch': 'branch',
            'maturing current month': 'maturing_current_month' # New column added
        }
        if lc_no_col:
            rename_map[lc_no_col] = 'lc_no'
            
        # Apply rename map to DataFrame columns
        # First, ensure all keys in rename_map that are present in df.columns are lowercased
        # Then, create a new mapping based on the actual lowercased columns in d
        current_cols = [col for col in d.columns]
        actual_rename_map = {}
        for old_col, new_col in rename_map.items():
            if old_col in current_cols:
                actual_rename_map[old_col] = new_col
        
        d = d.rename(columns=actual_rename_map)

        # Coerce datatypes
        if 'submitted_date' in d.columns:
            d['submitted_date'] = pd.to_datetime(d['submitted_date'], errors='coerce')
        if 'maturity_date' in d.columns:
            d['maturity_date'] = pd.to_datetime(d['maturity_date'], errors='coerce')
        if 'value_sar' in d.columns:
            d['value_sar'] = d['value_sar'].apply(_to_number)
        # Convert new maturing_current_month column to numeric
        if 'maturing_current_month' in d.columns:
            d['maturing_current_month'] = d['maturing_current_month'].apply(_to_number)


        # Clean/standardize
        if 'branch' in d.columns:
            d['branch'] = d['branch'].astype(str).str.strip().str.upper()
        if 'issuing_bank' in d.columns:
            d['issuing_bank'] = d['issuing_bank'].astype(str).str.strip().str.upper()
        if 'advising_bank' in d.columns:
            d['advising_bank'] = d['advising_bank'].astype(str).str.strip().str.upper()
        if 'status' in d.columns:
            d['status'] = d['status'].astype(str).str.strip().str.upper()

        # Keep rows with a valid value and branch; allow missing submitted_date/maturity_date
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
    """Generate a complete Excel report with multiple sheets."""
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
    
    # Load Export LC data from 3 files and combine
    df_export_lc_raw_1 = read_excel_all_sheets(LINKS["EXPORT_LC_1"])
    df_export_lc_raw_2 = read_excel_all_sheets(LINKS["EXPORT_LC_2"])
    df_export_lc_raw_3 = read_excel_all_sheets(LINKS["EXPORT_LC_3"])
    
    df_export_lc_raw = pd.concat(
        [df_export_lc_raw_1, df_export_lc_raw_2, df_export_lc_raw_3],
        ignore_index=True
    )

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
    
    # KPI: Accepted Export LC Sum
    accepted_export_lc_sum = 0.0
    if not df_export_lc.empty and 'status' in df_export_lc.columns:
        # Sum 'value_sar' for all 'ACCEPTED' LCs
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

        # 1) Liquidity MTD
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
                    with kpi_d: st.metric("Proj. EOM", fmt_number_only(latest + (avg_daily or 0)))
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

        # 2) LCR & STL Settlements — Overview
        st.markdown("---")
        st.markdown('<span class="section-chip">📅 LCR & STL Settlements — Overview</span>', unsafe_allow_html=True)
        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
        st.markdown('<div style="background:#f1f5f9;padding:12px;border-radius:8px;border-left:4px solid #3b82f6;margin-bottom:16px;"><small>📊 <strong>Metrics show ALL settlements</strong> | 📈 <strong>Chart & table show current month only</strong></small></div>', unsafe_allow_html=True)
        
        if df_lc.empty and df_lc_paid.empty:
            st.info("No LCR & STL data.")
        else:
            try:
                lc_m = df_lc[(df_lc["settlement_date"] >= month_start) & (df_lc["settlement_date"] <= month_end)].copy() if not df_lc.empty else pd.DataFrame()
                lc_paid_m = df_lc_paid[(df_lc_paid["settlement_date"] >= month_start) & (df_lc_paid["settlement_date"] <= month_end)].copy() if not df_lc_paid.empty else pd.DataFrame()
                all_pending = df_lc.copy() if not df_lc.empty else pd.DataFrame()
                all_paid = df_lc_paid.copy() if not df_lc_paid.empty else pd.DataFrame()
                total_due = (all_pending["amount"].sum() if not all_pending.empty else 0.0) + (all_paid["amount"].sum() if not all_paid.empty else 0.0)
                if not all_pending.empty:
                    current_due_mask = (all_pending["status"].str.upper().str.strip() == "PENDING") & \
                                       (all_pending["remark"].notna()) & \
                                       (all_pending["remark"].astype(str).str.strip() != "") & \
                                       (all_pending["remark"].astype(str).str.strip() != "-") & \
                                       (all_pending["remark"].astype(str).str.strip().str.lower() != "nan")
                    current_due = all_pending.loc[current_due_mask, "amount"].sum()
                else:
                    current_due = 0.0
                paid_amount = all_paid["amount"].sum() if not all_paid.empty else 0.0
                if not all_pending.empty:
                    balance_due_mask = (all_pending["status"].str.upper().str.strip() == "PENDING") & \
                                       ((all_pending["remark"].isna()) | \
                                        (all_pending["remark"].astype(str).str.strip() == "") | \
                                        (all_pending["remark"].astype(str).str.strip() == "-") | \
                                        (all_pending["remark"].astype(str).str.strip().str.lower() == "nan"))
                    balance_due = all_pending.loc[balance_due_mask, "amount"].sum()
                else:
                    balance_due = 0.0
                completion_rate = (paid_amount / total_due * 100) if total_due > 0 else 0

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
            except Exception as e:
                st.error(f"Unable to render Settlements section: {e}")

        st.markdown("---")

        # 3) FX MTD section restored
        if st.session_state.get("show_fx", True) and not df_fx.empty:
            st.subheader("Exchange Rates — Month Overview")
            fx_m = df_fx[(df_fx["date"] >= month_start) & (df_fx["date"] <= month_end)].copy()
            if not fx_m.empty:
                f1, f2 = st.columns(2)
                with f1:
                    latest_fx = fx_m.groupby("currency_pair").last().reset_index()
                    fx_display = latest_fx[["currency_pair", "rate"]].rename(
                        columns={"currency_pair": "Pair", "rate": "Current Rate"})
                    st.dataframe(style_right(fx_display, num_cols=["Current Rate"], decimals=4), 
                               use_container_width=True, height=200)
                with f2:
                    if "change_pct" in fx_m.columns:
                        volatility = fx_m.groupby("currency_pair")["change_pct"].std().reset_index()
                        volatility = volatility.rename(columns={"currency_pair": "Pair", "change_pct": "Volatility %"})
                        st.dataframe(style_right(volatility, num_cols=["Volatility %"], decimals=2), 
                                   use_container_width=True, height=200)
            else:
                st.info("No FX data for current month.")

        # NEW: Export LC — Summary by Branch (MTD; fallback to ALL)
        st.markdown('<span class="section-chip">🚢 Export LC — Summary by Branch</span>', unsafe_allow_html=True)
        try:
            if df_export_lc.empty:
                st.info("No Export LC data available.")
            else:
                elc_data = df_export_lc.copy()
                # Prefer MTD; fallback to ALL
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
                num_cols = [c for c in ["Balance", "After Settlement"] if c in table.columns]
                st.dataframe(style_right(table, num_cols=num_cols), use_container_width=True, height=360)

        st.markdown("---")
        st.markdown('<span class="section-chip">📈 Liquidity Trend Analysis</span>', unsafe_allow_html=True)
        if df_fm.empty:
            st.info("No liquidity data available.")
        else:
            try:
                import plotly.io as pio, plotly.graph_objects as go
                if "brand" not in pio.templates:
                    pio.templates["brand"] = pio.templates["plotly_white"]
                    pio.templates["brand"].layout.colorway = [THEME["accent1"], THEME["accent2"], "#64748b", "#94a3b8"]
                    pio.templates["brand"].layout.font.family = APP_FONT
                    pio.templates["brand"].layout.paper_bgcolor = "white"
                    pio.templates["brand"].layout.plot_bgcolor = "white"
                latest_liquidity = df_fm.iloc[-1]["total_liquidity"]
                if len(df_fm) > 1:
                    prev = df_fm.iloc[-2]["total_liquidity"]
                    trend_change = latest_liquidity - prev
                    trend_pct = (trend_change / prev) * 100 if prev != 0 else 0
                    trend_text = f"{'📈' if trend_change > 0 else '📉'} {trend_pct:+.1f}%"
                else:
                    trend_text = "No trend data"
                c1, c2 = st.columns([3, 1])
                with c1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_fm["date"], y=df_fm["total_liquidity"], mode='lines+markers', line=dict(width=3), marker=dict(size=6)))
                    fig.update_layout(template="brand", title="Total Liquidity Trend",
                                      xaxis_title="Date", yaxis_title="Liquidity (SAR)", height=400,
                                      margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
                    fig.update_xaxes(rangeslider_visible=False, rangeselector=None)
                    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
                with c2:
                    st.markdown("### 📊 Liquidity Metrics")
                    st.metric("Current", fmt_number_only(latest_liquidity))
                    if len(df_fm) > 1: st.metric("Trend", trend_text)
                    st.markdown("**Statistics (30d)**")
                    last30 = df_fm.tail(30)
                    st.write(f"**Max:** {fmt_number_only(last30['total_liquidity'].max())}")
                    st.write(f"**Min:** {fmt_number_only(last30['total_liquidity'].min())}")
                    st.write(f"**Avg:** {fmt_number_only(last30['total_liquidity'].mean())}")
            except Exception:
                st.error("❌ Unable to display liquidity trend analysis")
                st.line_chart(df_fm.set_index("date")["total_liquidity"])

        st.markdown("---")
        st.markdown('<span class="section-chip">🏢 Collection vs Payments — by Branch</span>', unsafe_allow_html=True)
        if df_cvp.empty:
            st.info("No data in 'Collection vs Payments by Branch'. Make sure the sheet has 'Branch', 'Collection', 'Payments'.")
        else:
            cvp_view = st.radio("", options=["Bars", "Table", "Cards"], index=0, horizontal=True, label_visibility="collapsed")
            cvp_sorted = df_cvp.sort_values("net", ascending=False).reset_index(drop=True)
            if cvp_view == "Bars":
                try:
                    import plotly.io as pio, plotly.graph_objects as go
                    if "brand" not in pio.templates:
                        pio.templates["brand"] = pio.templates["plotly_white"]
                        pio.templates["brand"].layout.colorway = [THEME["accent1"], THEME["accent2"], "#64748b", "#94a3b8"]
                        pio.templates["brand"].layout.font.family = APP_FONT
                    fig = go.Figure()
                    fig.add_bar(name="Collection", x=cvp_sorted["branch"], y=cvp_sorted["collection"])
                    fig.add_bar(name="Payments", x=cvp_sorted["branch"], y=cvp_sorted["payments"])
                    fig.update_layout(template="brand", barmode="group",
                                      height=420, margin=dict(l=20, r=20, t=30, b=80),
                                      xaxis_title="Branch", yaxis_title="Amount (SAR)",
                                      legend_title_text="")
                    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
                except Exception:
                    st.bar_chart(cvp_sorted.set_index("branch")[["collection", "payments"]])
            elif cvp_view == "Table":
                tbl = cvp_sorted.rename(columns={"branch": "Branch", "collection": "Collection", "payments": "Payments", "net": "Net"})
                styled = style_right(tbl, num_cols=["Collection", "Payments", "Net"])
                def _net_red(val):
                    try: return 'color:#b91c1c;font-weight:700;' if float(val) < 0 else ''
                    except Exception: return ''
                styled = styled.applymap(_net_red, subset=["Net"])
                st.dataframe(styled, use_container_width=True, height=420)
            else:
                display_as_mini_cards(cvp_sorted.rename(columns={"net":"balance"}), "branch", "balance", pad=pad, radius=radius, shadow=shadow)

    # ---- Settlements tab ----
    with tab_settlements:
        st.markdown('<span class="section-chip">📅 LCR & STL Settlements</span>', unsafe_allow_html=True)
        
        def render_settlements_tab(df_src: pd.DataFrame, status_label: str, key_suffix: str):
            if df_src.empty:
                st.info(f"No {status_label.lower()} settlements found."); return
            
            # Date filtering
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("From Date", value=df_src["settlement_date"].min().date(), key=f"start_{key_suffix}")
            with col2:
                end_date = st.date_input("To Date", value=df_src["settlement_date"].max().date(), key=f"end_{key_suffix}")
            
            # Filter data by date range
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
                        
                        # Add urgency indicators for pending settlements
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
                        
                        # Urgency warnings for pending
                        urgent_settlements = view_data[view_data["settlement_date"] <= today0 + pd.Timedelta(days=3)]
                        if not urgent_settlements.empty:
                            st.warning(f"⚠️ {len(urgent_settlements)} settlement(s) due within 3 days!")
                            for _, settlement in urgent_settlements.iterrows():
                                days_left = (settlement["settlement_date"] - today0).days
                                st.write(f"• {settlement['bank']} - {fmt_number_only(settlement['amount'])} - {days_left} day(s) left")
                    
                    else:  # Paid settlements
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
                        display_as_progress_bars(urgent.groupby("bank", as_index=False)["amount"].sum().rename(columns={"amount": "balance"}))
                    if not warning.empty:
                        st.markdown("**⚠️ Warning (3-7 days)**")
                        display_as_progress_bars(warning.groupby("bank", as_index=False)["amount"].sum().rename(columns={"amount": "balance"}))
                    if not normal.empty:
                        st.markdown("**✅ Normal (>7 days)**")
                        display_as_progress_bars(normal.groupby("bank", as_index=False)["amount"].sum().rename(columns={"amount": "balance"}))
                
                elif settlement_view == "Progress by Urgency" and status_label == "Paid":
                    st.info("Progress by urgency view is only available for pending settlements.")
                    # Show bank summary for paid
                    bank_totals = view_data.groupby("bank", as_index=False)["amount"].sum().rename(columns={"amount": "balance"})
                    display_as_progress_bars(bank_totals, "bank", "balance")
                
                else:  # Mini Cards
                    cards = view_data.groupby("bank", as_index=False)["amount"].sum().rename(columns={"amount": "balance"})
                    display_as_mini_cards(cards, "bank", "balance", pad=pad, radius=radius, shadow=shadow)
            else:
                st.info("No settlements match the selected criteria.")
        
        # Create sub-tabs for Pending and Paid settlements
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
                    with c1: st.metric(f"Total {status_label} Amount", fmt_number_only(view_data["amount"].sum()))
                    with c2: st.metric("Number of Payments", len(view_data))
                    with c3: st.metric("Average Payment", fmt_number_only(view_data["amount"].mean()))
                    grp = (view_data.groupby("bank", as_index=False)["amount"]
                           .sum().sort_values("amount", ascending=False)
                           .rename(columns={"bank": "Bank", "amount": "Amount"}))
                    st.markdown("**📊 Summary by Bank**")
                    st.dataframe(style_right(grp, num_cols=["Amount"]), use_container_width=True, height=220)
                    st.markdown("**📋 Detailed Payment List**")
                    show_cols = [c for c in ["bank", "supplier", "currency", "amount", "status"] if c in view_data.columns]
                    v = view_data[show_cols].rename(columns={"bank": "Bank", "supplier": "Supplier", "currency": "Currency",
                                                             "amount": "Amount", "status": "Status"})
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
                # ===============================
                # Branch & Advising Bank (RADIO)
                # ===============================
                col1, col2 = st.columns(2)
        
                with col1:
                    if "branch" in df_export_lc.columns:
                        branches = sorted(df_export_lc["branch"].dropna().astype(str).unique().tolist())
                    else:
                        branches = []
                    branch_options = ["All"] + branches  # To hide "All", change to: branch_options = branches
                    branch_choice = st.radio(
                        "Filter by Branch",
                        options=branch_options,
                        index=0,
                        horizontal=True,
                        key="export_lc_branch_radio",
                    )
                    selected_branches = branches if branch_choice == "All" else [branch_choice]
        
                with col2:
                    if "advising_bank" in df_export_lc.columns:
                        advising_banks = sorted(
                            df_export_lc["advising_bank"].dropna().astype(str).unique().tolist()
                        )
                    else:
                        advising_banks = []
        
                    if advising_banks:
                        adv_options = ["All"] + advising_banks  # To hide "All", change to: adv_options = advising_banks
                        adv_choice = st.radio(
                            "Filter by Advising Bank",
                            options=adv_options,
                            index=0,
                            horizontal=True,
                            key="export_lc_advising_radio",
                        )
                        selected_advising_banks = advising_banks if adv_choice == "All" else [adv_choice]
                    else:
                        selected_advising_banks = []
        
                # ==============================================
                # Maturity Date (top-level; keep rows with NaT)
                # ==============================================
                if "maturity_date" in df_export_lc.columns:
                    mdates_all = pd.to_datetime(df_export_lc["maturity_date"], errors="coerce")
                    try:
                        mdates_all = mdates_all.dt.tz_localize(None)
                    except Exception:
                        try:
                            mdates_all = mdates_all.dt.tz_convert(None)
                        except Exception:
                            pass
                else:
                    mdates_all = pd.Series([], dtype="datetime64[ns]")
        
                if mdates_all.notna().any():
                    min_mat_default = mdates_all.dropna().min().normalize().date()
                    max_mat_default = mdates_all.dropna().max().normalize().date()
                else:
                    today_ = datetime.today().date()
                    min_mat_default = today_.replace(day=1)
                    max_mat_default = today_
        
                d1, d2 = st.columns(2)
                with d1:
                    start_maturity_filter = st.date_input(
                        "From Maturity Date",
                        value=min_mat_default,
                        key="export_lc_maturity_start",
                    )
                with d2:
                    end_maturity_filter = st.date_input(
                        "To Maturity Date",
                        value=max_mat_default,
                        key="export_lc_maturity_end",
                    )
        
                # ==========================
                # Apply top-level filtering
                # ==========================
                filtered_df_base = df_export_lc.copy()
        
                if "branch" in filtered_df_base.columns and selected_branches:
                    filtered_df_base = filtered_df_base[filtered_df_base["branch"].isin(selected_branches)]
        
                if "advising_bank" in filtered_df_base.columns and selected_advising_banks:
                    filtered_df_base = filtered_df_base[
                        filtered_df_base["advising_bank"].isin(selected_advising_banks)
                    ]
        
                if "maturity_date" in filtered_df_base.columns:
                    mnorm = pd.to_datetime(filtered_df_base["maturity_date"], errors="coerce")
                    try:
                        mnorm = mnorm.dt.tz_localize(None)
                    except Exception:
                        try:
                            mnorm = mnorm.dt.tz_convert(None)
                        except Exception:
                            pass
                    mnorm = mnorm.dt.normalize()
                    start_ts = pd.to_datetime(start_maturity_filter)
                    end_ts = pd.to_datetime(end_maturity_filter)
                    filtered_df_base = filtered_df_base[mnorm.isna() | mnorm.between(start_ts, end_ts)].copy()
        
                # ======================
                # Status as sub-tabs
                # ======================
                if "status" in filtered_df_base.columns:
                    statuses = sorted(
                        [
                            s
                            for s in filtered_df_base["status"]
                            .dropna()
                            .astype(str)
                            .str.strip()
                            .str.upper()
                            .unique()
                            if s
                        ]
                    )
                else:
                    statuses = []
        
                tab_labels = ["ALL"] + statuses if statuses else ["ALL"]
                status_tabs = st.tabs(tab_labels)
        
                for status_key, tab in zip(tab_labels, status_tabs):
                    with tab:
                        if status_key == "ALL":
                            filtered_df = filtered_df_base.copy()
                        else:
                            filtered_df = filtered_df_base[
                                filtered_df_base["status"].astype(str).str.strip().str.upper() == status_key
                            ].copy()
        
                        # =========
                        # KPIs
                        # =========
                        st.markdown("---")
                        total_value = float(filtered_df["value_sar"].sum()) if ("value_sar" in filtered_df.columns and not filtered_df.empty) else 0.0
                        accepted_mtd_value = 0.0
                        if not filtered_df.empty:
                            if {"status", "maturing_current_month"}.issubset(filtered_df.columns):
                        float(filtered_df["value_sar"].sum())
                            if (not filtered_df.empty and "value_sar" in filtered_df.columns)
                            else 0.0
                        )
        
                        # Accepted Due this Month (prefer 'maturing_current_month' column)
                        accepted_mtd_value = 0.0
                        if not filtered_df.empty:
                            if {"status", "maturing_current_month"}.issubset(filtered_df.columns):
                                mask_acc = filtered_df["status"].astype(str).str.strip().str.upper() == "ACCEPTED"
                                accepted_mtd_value = float(filtered_df.loc[mask_acc, "maturing_current_month"].sum())
                                if pd.isna(accepted_mtd_value): accepted_mtd_value = 0.0
                            elif {"status", "maturity_date", "value_sar"}.issubset(filtered_df.columns):
                                now = pd.Timestamp.now()
                                start_month = now.replace(day=1).normalize()
                                end_month = (start_month + pd.offsets.MonthEnd(1)).normalize()
                                mser = pd.to_datetime(filtered_df["maturity_date"], errors="coerce")
                                try:
                                    mser = mser.dt.tz_localize(None)
                                except Exception:
                                    try: mser = mser.dt.tz_convert(None)
                                    except Exception: pass
                                mask_acc = (
                                    filtered_df["status"].astype(str).str.strip().str.upper().eq("ACCEPTED")
                                    & mser.dt.normalize().between(start_month, end_month)
                                )
                                accepted_mtd_value = float(filtered_df.loc[mask_acc, "value_sar"].sum())
                                if pd.isna(accepted_mtd_value): accepted_mtd_value = 0.0

                        # Collected & Remaining
                        if not filtered_df.empty and {"status", "value_sar"}.issubset(filtered_df.columns):
                            mask_col = filtered_df["status"].astype(str).str.strip().str.upper() == "COLLECTED"  # change text if your status differs
                            collected_sum = float(filtered_df.loc[mask_col, "value_sar"].sum())
                            if pd.isna(collected_sum): collected_sum = 0.0
                        remaining_value = float(total_value - collected_sum)
                       
                        # Show 4 KPIs
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Total Value (SAR)", fmt_number_only(total_value))
                        m2.metric("Accepted Due this Month (SAR)", fmt_number_only(accepted_mtd_value))
                        m3.metric("Collected (SAR)", fmt_number_only(collected_sum))
                        m4.metric("Remaining (SAR)", fmt_number_only(remaining_value))
                        
                        # ============================
                        # Summary by Branch (table)
                        # ============================
                        st.markdown("#### Summary by Branch")
                        if not filtered_df.empty and {"branch", "value_sar"}.issubset(filtered_df.columns):
                            summary_by_branch = (
                                filtered_df.groupby("branch", as_index=False)
                                .agg(LCs=("value_sar", "size"), Total_Value_SAR=("value_sar", "sum"))
                                .rename(columns={"branch": "Branch", "Total_Value_SAR": "Total Value (SAR)"})
                                .sort_values("Total Value (SAR)", ascending=False)
                            )
                            st.dataframe(
                                style_right(summary_by_branch, num_cols=["LCs", "Total Value (SAR)"]),
                                use_container_width=True,
                                height=300,
                            )
                        else:
                            st.info("No records to summarize for the selected filters.")
        
                        # ==========================================
                        # Detailed View (own Maturity Date filters)
                        # ==========================================
                        st.markdown("#### Detailed View")
                        table_base = filtered_df.copy()
        
                        if "maturity_date" in table_base.columns:
                            mdates = pd.to_datetime(table_base["maturity_date"], errors="coerce")
                            try:
                                mdates = mdates.dt.tz_localize(None)
                            except Exception:
                                try:
                                    mdates = mdates.dt.tz_convert(None)
                                except Exception:
                                    pass
        
                            min_date = (
                                mdates.dropna().min().normalize().date()
                                if mdates.notna().any()
                                else datetime.today().date().replace(day=1)
                            )
                            max_date = (
                                mdates.dropna().max().normalize().date()
                                if mdates.notna().any()
                                else datetime.today().date()
                            )
        
                            dd1, dd2 = st.columns(2)
                            with dd1:
                                mat_start = st.date_input(
                                    "From Maturity Date (Table View)",
                                    value=min_date,
                                    key=f"export_lc_table_mstart_{status_key}",
                                )
                            with dd2:
                                mat_end = st.date_input(
                                    "To Maturity Date (Table View)",
                                    value=max_date,
                                    key=f"export_lc_table_mend_{status_key}",
                                )
        
                            mnorm = mdates.dt.normalize()
                            table_base = table_base[
                                mnorm.isna()
                                | mnorm.between(pd.to_datetime(mat_start), pd.to_datetime(mat_end))
                            ].copy()
        
                        # Columns order & labels
                        desired_columns_info = [
                            ("branch", "Branch"),
                            ("applicant", "Applicant"),
                            ("lc_no", "L/C No"),
                            ("advising_bank", "Advising Bank"),
                            ("maturity_date", "Maturity Date"),
                            ("value_sar", "Value (SAR)"),
                            ("maturing_current_month", "Maturing Current Month"),
                            ("status", "Status"),
                            ("remarks", "Remarks"),
                        ]
                        cols_to_show = [c for c, _ in desired_columns_info if c in table_base.columns]
                        display_name_map = {c: n for c, n in desired_columns_info if c in table_base.columns}
        
                        if not table_base.empty and cols_to_show:
                            table_view = table_base[cols_to_show].rename(columns=display_name_map)
        
                            # Format maturity date as DD-MM-YYYY
                            if "Maturity Date" in table_view.columns:
                                table_view["Maturity Date"] = (
                                    pd.to_datetime(table_view["Maturity Date"], errors="coerce")
                                    .dt.strftime("%d-%m-%Y")
                                    .fillna("")
                                )
        
                            # Clean "None"/"NaT"/"nan" texts in object columns
                            obj_cols = [
                                c for c in table_view.select_dtypes(include="object").columns
                                if c != "Maturity Date"
                            ]
                            for c in obj_cols:
                                table_view[c] = (
                                    table_view[c]
                                    .astype(str)
                                    .replace({"None": "", "none": "", "NaT": "", "nan": ""}, regex=True)
                                    .fillna("")
                                )
        
                            num_cols = [c for c in ["Value (SAR)", "Maturing Current Month"] if c in table_view.columns]
                            st.dataframe(
                                style_right(table_view, num_cols=num_cols),
                                use_container_width=True,
                                height=500,
                            )
                        else:
                            st.info("No records available for the detailed view after applying filters.")


    # ---- Exchange Rates tab ----
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
                        start_date = st.date_input("From Date", 
                                                 value=df_fx["date"].min().date(),
                                                 key="fx_start_date")
                    with c2:
                        end_date = st.date_input("To Date", 
                                               value=df_fx["date"].max().date(),
                                               key="fx_end_date")
                    fx_filtered = df_fx[
                        (df_fx["date"].dt.date >= start_date) & 
                        (df_fx["date"].dt.date <= end_date)
                    ].copy()
                    if not fx_filtered.empty:
                        available_pairs = sorted(fx_filtered["currency_pair"].unique())
                        selected_pairs = st.multiselect("Select Currency Pairs", 
                                                       available_pairs, 
                                                       default=available_pairs[:3],
                                                       key="fx_pairs")
                        if selected_pairs:
                            fx_chart_data = fx_filtered[fx_filtered["currency_pair"].isin(selected_pairs)]
                            try:
                                import plotly.io as pio, plotly.graph_objects as go
                                if "brand" not in pio.templates:
                                    pio.templates["brand"] = pio.templates["plotly_white"]
                                    pio.templates["brand"].layout.colorway = [THEME["accent1"], THEME["accent2"], "#64748b", "#94a3b8"]
                                    pio.templates["brand"].layout.font.family = APP_FONT
                                fig = go.Figure()
                                for pair in selected_pairs:
                                    pair_data = fx_chart_data[fx_chart_data["currency_pair"] == pair]
                                    fig.add_trace(go.Scatter(
                                        x=pair_data["date"],
                                        y=pair_data["rate"],
                                        mode='lines+markers',
                                        name=pair,
                                        line=dict(width=2),
                                        marker=dict(size=4)
                                    ))
                                fig.update_layout(
                                    template="brand",
                                    title="Exchange Rate Trends",
                                    xaxis_title="Date",
                                    yaxis_title="Exchange Rate",
                                    height=400,
                                    margin=dict(l=20, r=20, t=50, b=20)
                                )
                                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
                            except Exception:
                                pivot_data = fx_chart_data.pivot(index="date", columns="currency_pair", values="rate")
                                st.line_chart(pivot_data)
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
                    volatility_stats = volatility_stats.reset_index()
                    volatility_stats = volatility_stats.rename(columns={"currency_pair": "Currency Pair"})
                    volatility_stats = volatility_stats.sort_values("Volatility (%)", ascending=False)
                    st.dataframe(
                        style_right(volatility_stats, 
                                  num_cols=["Volatility (%)", "Avg Change (%)", "Min Change (%)", "Max Change (%)", "Current Rate"],
                                  decimals=4),
                        use_container_width=True,
                        height=400
                    )
                    if len(volatility_stats) > 1:
                        try:
                            import plotly.io as pio, plotly.graph_objects as go
                            fig = go.Figure(go.Bar(
                                x=volatility_stats["Currency Pair"],
                                y=volatility_stats["Volatility (%)"],
                                marker_color=THEME["accent1"]
                            ))
                            fig.update_layout(
                                template="brand",
                                title="Exchange Rate Volatility by Currency Pair",
                                xaxis_title="Currency Pair",
                                yaxis_title="Volatility (%)",
                                height=300,
                                margin=dict(l=20, r=20, t=50, b=80)
                            )
                            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
                        except Exception:
                            st.bar_chart(volatility_stats.set_index("Currency Pair")["Volatility (%)"])
                else:
                    st.info("Volatility analysis requires historical rate changes.")
            
            else:  # Table View
                st.subheader("📋 Exchange Rate Data Table")
                col1, col2 = st.columns(2)
                with col1:
                    if "currency_pair" in df_fx.columns:
                        available_pairs = ["All"] + sorted(df_fx["currency_pair"].unique())
                        selected_pair = st.selectbox("Filter by Currency Pair", available_pairs, key="fx_table_pair")
                with col2:
                    if "date" in df_fx.columns:
                        date_range = st.number_input("Last N days", min_value=1, max_value=365, value=30, key="fx_date_range")
                display_data = df_fx.copy()
                if "date" in display_data.columns:
                    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=date_range)
                    display_data = display_data[display_data["date"] >= cutoff_date]
                if selected_pair != "All":
                    display_data = display_data[display_data["currency_pair"] == selected_pair]
                if not display_data.empty:
                    table_data = display_data.copy()
                    if "date" in table_data.columns:
                        table_data["Date"] = table_data["date"].dt.strftime(config.DATE_FMT)
                    rename_map = {
                        "currency_pair": "Currency Pair",
                        "rate": "Rate",
                        "change": "Change",
                        "change_pct": "Change %"
                    }
                    table_data = table_data.rename(columns={k: v for k, v in rename_map.items() if k in table_data.columns})
                    display_cols = ["Currency Pair", "Rate"]
                    if "Date" in table_data.columns:
                        display_cols.append("Date")
                    if "Change" in table_data.columns:
                        display_cols.append("Change")
                    if "Change %" in table_data.columns:
                        display_cols.append("Change %")
                    display_cols = [col for col in display_cols if col in table_data.columns]
                    table_show = table_data[display_cols].sort_values("Date" if "Date" in display_cols else "Currency Pair", ascending=False)
                    num_cols = [col for col in ["Rate", "Change", "Change %"] if col in table_show.columns]
                    styled_table = style_right(table_show, num_cols=num_cols, decimals=4)
                    if "Change %" in table_show.columns:
                        def highlight_changes(val):
                            try:
                                if pd.isna(val):
                                    return ''
                                num_val = float(val)
                                if num_val > 0:
                                    return 'color: #059669; font-weight: 600;'
                                elif num_val < 0:
                                    return 'color: #dc2626; font-weight: 600;'
                                else:
                                    return ''
                            except:
                                return ''
                        styled_table = styled_table.applymap(highlight_changes, subset=["Change %"])
                    st.dataframe(styled_table, use_container_width=True, height=500)
                else:
                    st.info("No data available for the selected criteria.")

    # ---- Facility Report tab ----
    with tab_facility:
        # Intentionally no placeholder text (per preference)
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

    # Removed the auto refresh block as requested.

if __name__ == "__main__":
    set_app_font() # Ensure font is set at the start
    main()




