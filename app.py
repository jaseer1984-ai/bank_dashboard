# app.py — Complete Enhanced Treasury Dashboard
# Enhanced with interactive visualizations, animations, modern UI, and improved After Settlement logic
# Includes: gauge charts, waterfall charts, donut charts, enhanced cards, interactive timelines

import io
import re
import time
import logging
import os
from datetime import datetime, timedelta
from dataclasses import dataclass
from functools import wraps
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --- Fix for pandas Styler type-hint on some builds ---
try:
    from pandas.io.formats.style import Styler
except Exception:
    Styler = Any  # fallback so annotations don't crash on import

# ----------------------------
# Configuration Management
# ----------------------------
@dataclass
class Config:
    FILE_ID: str = os.getenv('GOOGLE_SHEETS_ID', '1371amvaCbejUWVJI_moWdIchy5DF1lPO')
    COMPANY_NAME: str = os.getenv('COMPANY_NAME', 'Issam Kabbani & Partners – Unitech')
    LOGO_PATH: str = os.getenv('LOGO_PATH', 'ikk_logo.png')
    CACHE_TTL: int = int(os.getenv('CACHE_TTL', '300'))
    TZ: str = os.getenv('TIMEZONE', 'Asia/Riyadh')
    DATE_FMT: str = "%Y-%m-%d"
    REQUEST_TIMEOUT: int = int(os.getenv('REQUEST_TIMEOUT', '30'))
    MAX_FILE_SIZE_MB: int = int(os.getenv('MAX_FILE_SIZE_MB', '50'))
    RATE_LIMIT_CALLS_PER_MINUTE: int = int(os.getenv('RATE_LIMIT_CPM', '12'))

config = Config()

# ---- Enhanced Theme (colors/icons all in one place) ----
THEME = {
    "icons": {
        "neg": "🚫",
        "best": "💎",
        "good": "🔹",
        "ok": "💠",
        "low": "💚",
    },
    "card_bg": {
        "neg": "#fee2e2",  # light red
        "best": "#e0e7ff",  # indigo-100
        "good": "#fce7f3",  # pink-100
        "ok": "#e0f2fe",    # sky-100
        "low": "#ecfdf5",   # green-100
    },
    "amount_color": {
        "pos": "#1e293b",  # slate-800
        "neg": "#b91c1c",  # red-700
        "subtle": "#334155"
    },
    "kpi": {
        "total_bg": "#EEF2FF", "total_bd": "#C7D2FE", "total_fg": "#1E3A8A",
        "appr_bg": "#E9FFF2", "appr_bd": "#C7F7DD", "appr_fg": "#065F46",
        "lc_bg": "#FFF7E6", "lc_bd": "#FDE9C8", "lc_fg": "#92400E",
        "bank_bg": "#FFF1F2", "bank_bd": "#FBD5D8", "bank_fg": "#9F1239",
    },
    "card_thresholds": [500_000, 100_000, 50_000],
    "colors": {
        "primary": "#3b82f6",
        "success": "#10b981", 
        "warning": "#f59e0b",
        "danger": "#ef4444",
        "purple": "#8b5cf6",
        "cyan": "#06b6d4"
    }
}

def pick_card_style(balance: float) -> Tuple[str, str, str]:
    """Return (bg_color, icon, amount_color) based on balance with negative case first."""
    try:
        b = float(balance)
    except Exception:
        b = np.nan
    if pd.notna(b) and b < 0:
        return THEME["card_bg"]["neg"], THEME["icons"]["neg"], THEME["amount_color"]["neg"]
    t1, t2, t3 = THEME["card_thresholds"]
    if pd.notna(b) and b > t1:
        return THEME["card_bg"]["best"], THEME["icons"]["best"], THEME["amount_color"]["pos"]
    if pd.notna(b) and b > t2:
        return THEME["card_bg"]["good"], THEME["icons"]["good"], THEME["amount_color"]["pos"]
    if pd.notna(b) and b > t3:
        return THEME["card_bg"]["ok"], THEME["icons"]["ok"], THEME["amount_color"]["pos"]
    return THEME["card_bg"]["low"], THEME["icons"]["low"], THEME["amount_color"]["pos"]

# ----------------------------
# Logging Setup
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
    page_title="Enhanced Treasury Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="💰"
)

# ---- Global font (one place to change) ----
APP_FONT = os.getenv("APP_FONT", "Inter")

def set_app_font(family: str = APP_FONT):
    css = """
    <style>
      @import url('https://fonts.googleapis.com/css2?family={font_q}:wght@300;400;500;600;700;800&display=swap');
      :root {{ --app-font: '{font}', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', sans-serif; }}
      html, body, [class^="css"], [class*=" css"] {{ font-family: var(--app-font) !important; }}
      h1, h2, h3, h4, h5, h6, p, span, div, label, small, strong, em {{ font-family: var(--app-font) !important; }}
      button, input, textarea, select {{ font-family: var(--app-font) !important; }}
      div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"] {{ font-family: var(--app-font) !important; }}
      div[data-testid="stDataFrame"] * {{ font-family: var(--app-font) !important; }}
    </style>
    """.format(font_q=family.replace(" ", "+"), font=family)
    st.markdown(css, unsafe_allow_html=True)

set_app_font()

# Enhanced CSS with animations and modern styling
st.markdown("""
<style>
    .main-header { 
        position: sticky; 
        top: 0; 
        background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%); 
        z-index: 999; 
        padding: 20px 0; 
        border-bottom: 2px solid #e6eaf0; 
        margin-bottom: 20px; 
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .main-header h1 { 
        font-size: 32px !important; 
        font-weight: 900 !important; 
        color: white !important; 
        text-transform: uppercase !important; 
        letter-spacing: 1px !important; 
        margin: 0 !important; 
        text-align: center;
    }
    .main-header p {
        text-align: center;
        margin-top: 8px;
        opacity: 0.9;
        font-size: 16px;
    }
    
    /* Enhanced Cards */
    .enhanced-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 16px;
        padding: 25px;
        margin-bottom: 20px;
        border-left: 5px solid;
        transition: all 0.3s ease;
        cursor: pointer;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        position: relative;
        overflow: hidden;
    }
    .enhanced-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    }
    .enhanced-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #3b82f6, #06b6d4, #10b981);
    }
    .card-excellent { border-left-color: #10b981; }
    .card-good { border-left-color: #3b82f6; }
    .card-warning { border-left-color: #f59e0b; }
    .card-critical { border-left-color: #ef4444; }
    
    .bank-name {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 15px;
    }
    .bank-balance {
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 10px;
    }
    .progress-container {
        width: 100%;
        height: 8px;
        background: #e2e8f0;
        border-radius: 4px;
        overflow: hidden;
        margin: 15px 0;
    }
    .progress-bar {
        height: 100%;
        border-radius: 4px;
        transition: width 1.5s ease;
    }
    .status-badge {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* After Settlement Enhanced Styling */
    .after-settlement-box {
        margin-top: 15px;
        padding: 12px;
        background: rgba(248, 250, 252, 0.7);
        border-radius: 8px;
        border-left: 3px solid;
        transition: all 0.2s ease;
    }
    .after-settlement-box:hover {
        background: rgba(248, 250, 252, 0.9);
        transform: translateX(2px);
    }
    .settlement-header {
        display: flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 6px;
    }
    .settlement-status {
        font-size: 0.75rem;
        padding: 2px 6px;
        border-radius: 4px;
        margin-left: auto;
        font-weight: 600;
    }
    
    /* Control Buttons */
    .control-button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        margin: 5px;
        font-family: var(--app-font);
    }
    .control-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(59, 130, 246, 0.3);
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
        margin: 30px 0 20px 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Animation Classes */
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-in {
        animation: slideInUp 0.6s ease forwards;
    }
    
    /* Loading Spinner */
    .loading {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #3b82f6;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
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
# Google Sheets Links
# ----------------------------
LINKS = {
    "BANK BALANCE": f"https://docs.google.com/spreadsheets/d/{config.FILE_ID}/export?format=csv&gid=860709395",
    "SUPPLIER PAYMENTS": f"https://docs.google.com/spreadsheets/d/{config.FILE_ID}/export?format=csv&gid=20805295",
    "SETTLEMENTS": f"https://docs.google.com/spreadsheets/d/{config.FILE_ID}/export?format=csv&gid=978859477",
    "Fund Movement": f"https://docs.google.com/spreadsheets/d/{config.FILE_ID}/export?format=csv&gid=66055663",
    "COLLECTION_BRANCH": f"https://docs.google.com/spreadsheets/d/{config.FILE_ID}/export?format=csv&gid=457517415",
}

# ----------------------------
# Rate limit decorator
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
# Helpers: parsing + formatting
# ----------------------------
def _to_number(x) -> float:
    """Robust numeric parser"""
    if pd.isna(x) or x == '':
        return np.nan
    s = str(x).strip()
    if s == '':
        return np.nan

    s = s.replace('\u2212', '-')
    neg_paren = False
    if '(' in s and ')' in s:
        neg_paren = True
        s = s.replace('(', '').replace(')', '')

    s = s.replace(',', '')
    is_pct = s.endswith('%')
    if is_pct:
        s = s[:-1]

    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
    if not m:
        return np.nan
    try:
        num = float(m.group())
        if neg_paren:
            num = -num
        if is_pct:
            num = num / 100.0
        if abs(num) > 1e12:
            logger.warning(f"Unusually large number detected: {num}")
            return np.nan
        return num
    except Exception as e:
        logger.debug(f"Number conversion failed for '{x}': {e}")
        return np.nan

def cols_lower(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in df.columns]
    return out

def fmt_currency(v, currency="SAR") -> str:
    try:
        if pd.isna(v): return "N/A"
        return f"{currency} {float(v):,.0f}"
    except (ValueError, TypeError):
        return str(v)

def fmt_number_only(v) -> str:
    try:
        if pd.isna(v): return "N/A"
        return f"{float(v):,.0f}"
    except (ValueError, TypeError):
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
                              ("background-color", "#f8f9fa"),
                              ("font-weight", "600"),
                              ("font-family", "var(--app-font)")]
                }]))
    return styler

def days_until(d, ref):
    if pd.isna(d):
        return np.nan
    return int((pd.to_datetime(d) - pd.to_datetime(ref)).days)

# ----------------------------
# Cached CSV fetch
# ----------------------------
@st.cache_data(ttl=config.CACHE_TTL)
@rate_limit()
def read_csv(url: str) -> pd.DataFrame:
    try:
        logger.info(f"Fetching data from: {url}")
        response = http_session.get(url, timeout=config.REQUEST_TIMEOUT, stream=True)
        response.raise_for_status()
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > config.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ValueError(f"File too large: {content_length} bytes")
        content = response.text
        if not content.strip():
            raise ValueError("Empty response from server")
        df = pd.read_csv(io.StringIO(content))
        logger.info(f"Successfully loaded {len(df)} rows")
        return df
    except requests.Timeout:
        logger.error(f"Timeout while fetching {url}")
        st.error("⏱️ Data source timed out. Please try refreshing.")
        return pd.DataFrame()
    except requests.ConnectionError:
        logger.error(f"Connection error for {url}")
        st.error("🔌 Unable to connect to data source. Check your internet connection.")
        return pd.DataFrame()
    except requests.HTTPError as e:
        logger.error(f"HTTP error {e.response.status_code} for {url}")
        st.error(f"🌐 Server error: {e.response.status_code}")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        logger.error(f"Empty CSV data from {url}")
        st.error("📋 Data source returned empty file")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error loading {url}: {e}")
        st.error(f"❌ Unexpected error: {str(e)}")
        return pd.DataFrame()

# ----------------------------
# Parsers (keeping original logic)
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
        if bank_col is None:
            raise ValueError("Could not detect bank column")

        parsed = pd.to_datetime(pd.Index(raw.columns), errors="coerce", dayfirst=False)
        date_cols = [col for col, d in zip(raw.columns, parsed) if pd.notna(d)]
        if not date_cols: raise ValueError("No valid date columns found")
        date_map = {col: pd.to_datetime(col, errors="coerce", dayfirst=False) for col in date_cols}
        latest_col = max(date_cols, key=lambda c: date_map[c])

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
    except Exception as e:
        logger.error(f"Bank balance parsing error: {e}")
        st.error(f"❌ Bank balance parsing failed: {str(e)}")
    return pd.DataFrame(), None

def parse_supplier_payments(df: pd.DataFrame) -> pd.DataFrame:
    try:
        d = cols_lower(df).rename(columns={"supplier name": "supplier", "amount(sar)": "amount_sar", "order/sh/branch": "order_branch"})
        if not validate_dataframe(d, ["bank", "status"], "Supplier Payments"):
            return pd.DataFrame()
        status_norm = d["status"].astype("string").str.strip().str.lower()
        mask = status_norm.str.contains(r"\bapproved\b", na=False)
        if not mask.any():
            logger.info("No approved payments found"); return pd.DataFrame()
        d = d.loc[mask].copy()
        amt_col = next((c for c in ["amount_sar", "amount", "amount(sar)"] if c in d.columns), None)
        if not amt_col:
            logger.error("No amount column found in supplier payments"); return pd.DataFrame()
        d["amount"] = d[amt_col].map(_to_number)
        d["bank"] = d["bank"].astype(str).str.strip()
        keep = [c for c in ["bank", "supplier", "currency", "amount", "status"] if c in d.columns]
        out = d[keep].dropna(subset=["amount"]).copy()
        if "status" in out.columns:
            out["status"] = out["status"].astype(str).str.title()
        return out
    except Exception as e:
        logger.error(f"Supplier payments parsing error: {e}")
        st.error(f"❌ Supplier payments parsing failed: {str(e)}")
        return pd.DataFrame()

def parse_settlements(df: pd.DataFrame) -> pd.DataFrame:
    try:
        d = cols_lower(df)
        ref_col = next((c for c in d.columns if any(t in c for t in ["a/c", "ref", "account", "reference"])), None)
        bank_col = next((c for c in d.columns if (c.startswith("bank") or "bank" in c)), None)
        date_col = next((c for c in d.columns if ("maturity" in c and "new" not in c)), None) or \
                   next((c for c in d.columns if ("new" in c and "maturity" in c)), None)
        amount_col = next((c for c in d.columns if ("balance" in c and "settlement" in c)), None) or \
                     next((c for c in d.columns if ("currently" in c and "due" in c)), None) or \
                     next((c for c in ["amount(sar)", "amount"] if c in d.columns), None)
        if not all([bank_col, amount_col, date_col]):
            return pd.DataFrame()
        status_col = next((c for c in d.columns if "status" in c), None)
        type_col = next((c for c in d.columns if "type" in c), None)
        remark_col = next((c for c in d.columns if "remark" in c), None)

        out = pd.DataFrame({
            "reference": d[ref_col].astype(str).str.strip() if ref_col else "",
            "bank": d[bank_col].astype(str).str.strip(),
            "settlement_date": pd.to_datetime(d[date_col], errors="coerce"),
            "amount": d[amount_col].map(_to_number),
            "status": d[status_col].astype(str).str.title().str.strip() if status_col else "",
            "type": d[type_col].astype(str).str.upper().str.strip() if type_col else "",
            "remark": d[remark_col].astype(str).str.strip() if remark_col else "",
            "description": ""
        })
        out = out.dropna(subset=["bank", "amount", "settlement_date"])
        out = out[out["status"].str.lower() == "pending"].copy()
        return out
    except Exception as e:
        logger.error(f"Settlements parsing error: {e}")
        st.error(f"❌ Settlements parsing failed: {str(e)}")
        return pd.DataFrame()

def parse_fund_movement(df: pd.DataFrame) -> pd.DataFrame:
    try:
        d = cols_lower(df)
        date_col = "date" if "date" in d.columns else None
        liq_col = next((c for c in d.columns if ("total" in c and "liquidity" in c)), None)
        if not date_col or not liq_col:
            return pd.DataFrame()
        out = pd.DataFrame({
            "date": pd.to_datetime(d[date_col], errors="coerce"),
            "total_liquidity": d[liq_col].map(_to_number)
        }).dropna()
        return out.sort_values("date")
    except Exception as e:
        logger.error(f"Fund movement parsing error: {e}")
        st.error(f"❌ Fund movement parsing failed: {str(e)}")
        return pd.DataFrame()

def parse_branch_cvp(df: pd.DataFrame) -> pd.DataFrame:
    try:
        d = cols_lower(df).rename(columns={
            "branch": "branch",
            "collection": "collection",
            "payments": "payments"
        })
        required = ["branch", "collection", "payments"]
        if not validate_dataframe(d, required, "Collection vs Payments by Branch"):
            return pd.DataFrame()

        out = pd.DataFrame({
            "branch": d["branch"].astype(str).str.strip(),
            "collection": d["collection"].map(_to_number).fillna(0.0),
            "payments": d["payments"].map(_to_number).fillna(0.0)
        })
        out = out[out["branch"].ne("")].copy()
        out["net"] = out["collection"] - out["payments"]
        return out
    except Exception as e:
        logger.error(f"Branch CVP parsing error: {e}")
        st.error(f"❌ Branch CVP parsing failed: {str(e)}")
        return pd.DataFrame()

# ----------------------------
# Enhanced Visualization Functions
# ----------------------------

def create_enhanced_bank_balance_section(df_by_bank):
    """Enhanced bank balance visualization with multiple view options"""
    
    st.markdown('<h2 class="section-header">🏦 Enhanced Bank Balance</h2>', unsafe_allow_html=True)
    
    # View selector buttons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        show_cards = st.button("💳 Interactive Cards", use_container_width=True, key="cards_btn")
    with col2:
        show_gauge = st.button("⚡ Gauge Charts", use_container_width=True, key="gauge_btn")
    with col3:
        show_donut = st.button("🍩 Donut Chart", use_container_width=True, key="donut_btn")
    with col4:
        show_waterfall = st.button("📊 Waterfall", use_container_width=True, key="waterfall_btn")
    
    # Initialize session state for view
    if 'bank_view' not in st.session_state:
        st.session_state.bank_view = 'cards'
    
    # Update view based on button clicks
    if show_cards:
        st.session_state.bank_view = 'cards'
    elif show_gauge:
        st.session_state.bank_view = 'gauge'
    elif show_donut:
        st.session_state.bank_view = 'donut'
    elif show_waterfall:
        st.session_state.bank_view = 'waterfall'
    
    df_sorted = df_by_bank.sort_values("balance", ascending=False)
    
    if st.session_state.bank_view == 'cards':
        render_enhanced_cards(df_sorted)
    elif st.session_state.bank_view == 'gauge':
        render_gauge_charts(df_sorted)
    elif st.session_state.bank_view == 'donut':
        render_donut_chart(df_sorted)
    elif st.session_state.bank_view == 'waterfall':
        render_waterfall_chart(df_sorted)

def render_enhanced_cards(df):
    """Render interactive animated cards with improved after settlement logic"""
    
    max_balance = df["balance"].max() if not df.empty else 1
    
    cols = st.columns(3)
    for i, (_, row) in enumerate(df.iterrows()):
        with cols[i % 3]:
            balance = row["balance"]
            percentage = (balance / max_balance) * 100 if max_balance > 0 else 0
            
            # Determine status and color
            if balance > 1000000:
                status, color = "excellent", THEME["colors"]["success"]
            elif balance > 500000:
                status, color = "good", THEME["colors"]["primary"]
            elif balance > 100000:
                status, color = "warning", THEME["colors"]["warning"]
            else:
                status, color = "critical", THEME["colors"]["danger"]
            
            # IMPROVED: Enhanced after settlement logic
            after_settlement_html = ""
            if "after_settlement" in df.columns and pd.notna(row.get("after_settlement")):
                after_val = row["after_settlement"]
                
                # Only show if value is not zero or if it's significantly different from current balance
                show_after_settlement = (
                    after_val != 0 or  # Not zero
                    abs(after_val - balance) > 1000  # Significant difference (more than 1K SAR)
                )
                
                if show_after_settlement:
                    # Determine styling based on value and comparison to current balance
                    if after_val < 0:
                        after_color = THEME["colors"]["danger"]
                        after_icon = "⚠️"
                        after_status = "Deficit"
                    elif after_val > balance:
                        after_color = THEME["colors"]["success"]
                        after_icon = "📈"
                        after_status = "Improvement"
                    elif after_val < balance:
                        after_color = THEME["colors"]["warning"]
                        after_icon = "📉"
                        after_status = "Reduction"
                    else:
                        after_color = THEME["colors"]["primary"]
                        after_icon = "➡️"
                        after_status = "No Change"
                    
                    # Calculate difference
                    difference = after_val - balance
                    diff_text = ""
                    if abs(difference) > 100:  # Only show difference if > 100 SAR
                        diff_sign = "+" if difference > 0 else ""
                        diff_text = f"""
                        <div style="font-size: 0.8rem; color: {after_color}; margin-top: 3px;">
                            {diff_sign}{fmt_number_only(difference)} vs current
                        </div>
                        """
                    
                    after_settlement_html = f"""
                    <div class="after-settlement-box" style="border-left-color: {after_color};">
                        <div class="settlement-header">
                            <span>{after_icon}</span>
                            <span style="font-size: 0.85rem; color: #64748b; font-weight: 600;">After Settlement</span>
                            <span class="settlement-status" style="color: {after_color}; background: {after_color}20;">{after_status}</span>
                        </div>
                        <div style="font-size: 1.2rem; font-weight: 700; color: {after_color};">
                            {fmt_currency(after_val)}
                        </div>
                        {diff_text}
                    </div>
                    """
            
            st.markdown(f"""
            <div class="enhanced-card card-{status}">
                <div class="bank-name">{row['bank']}</div>
                <div class="bank-balance" style="color: {color};">
                    {fmt_currency(balance)}
                </div>
                <div class="progress-container">
                    <div class="progress-bar" style="width: {percentage}%; background: {color};"></div>
                </div>
                <div class="status-badge" style="color: {color};">
                    {status.upper()}
                </div>
                {after_settlement_html}
            </div>
            """, unsafe_allow_html=True)

def render_gauge_charts(df):
    """Create gauge charts for top banks"""
    
    # Limit to top 4 banks for better display
    top_banks = df.head(4)
    
    if top_banks.empty:
        st.info("No bank data available for gauge charts")
        return
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=top_banks['bank'].tolist(),
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}]]
    )
    
    max_value = df['balance'].max() / 1000000  # Convert to millions
    
    for i, (_, bank) in enumerate(top_banks.iterrows()):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        value = bank['balance'] / 1000000  # Convert to millions
        
        # Determine gauge color based on value
        if value > max_value * 0.7:
            bar_color = THEME["colors"]["success"]
        elif value > max_value * 0.4:
            bar_color = THEME["colors"]["primary"]
        elif value > max_value * 0.2:
            bar_color = THEME["colors"]["warning"]
        else:
            bar_color = THEME["colors"]["danger"]
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"{bank['bank']}<br><span style='font-size:12px'>Millions SAR</span>"},
                gauge={
                    'axis': {'range': [None, max_value]},
                    'bar': {'color': bar_color},
                    'steps': [
                        {'range': [0, max_value * 0.2], 'color': "#fee2e2"},
                        {'range': [max_value * 0.2, max_value * 0.4], 'color': "#fef3c7"},
                        {'range': [max_value * 0.4, max_value * 0.7], 'color': "#e0f2fe"},
                        {'range': [max_value * 0.7, max_value], 'color': "#ecfdf5"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': max_value * 0.9
                    }
                }
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        height=500,
        font=dict(family=APP_FONT, size=12),
        title_text="Bank Balance Gauges",
        title_x=0.5
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_donut_chart(df):
    """Create interactive donut chart"""
    
    if df.empty:
        st.info("No bank data available for donut chart")
        return
    
    total_balance = df['balance'].sum()
    
    fig = go.Figure(data=[go.Pie(
        labels=df['bank'],
        values=df['balance'],
        hole=0.4,
        hovertemplate='<b>%{label}</b><br>' +
                      'Balance: SAR %{value:,.0f}<br>' +
                      'Percentage: %{percent}<br>' +
                      '<extra></extra>',
        textinfo='label+percent',
        textposition='outside',
        marker=dict(
            colors=[THEME["colors"]["primary"], THEME["colors"]["success"], 
                   THEME["colors"]["warning"], THEME["colors"]["danger"], 
                   THEME["colors"]["purple"], THEME["colors"]["cyan"]],
            line=dict(color='#FFFFFF', width=3)
        )
    )])
    
    fig.update_layout(
        title="Bank Balance Distribution",
        font=dict(family=APP_FONT, size=14),
        height=500,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        ),
        annotations=[dict(
            text=f'Total<br>SAR {total_balance/1000000:.1f}M',
            x=0.5, y=0.5,
            font_size=20,
            showarrow=False,
            font_color="#1e293b"
        )]
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_waterfall_chart(df):
    """Create waterfall chart showing bank contributions"""
    
    if df.empty:
        st.info("No bank data available for waterfall chart")
        return
    
    fig = go.Figure(go.Waterfall(
        name="Bank Balances",
        orientation="v",
        measure=["relative"] * len(df) + ["total"],
        x=df['bank'].tolist() + ["Total"],
        textposition="outside",
        text=[f"SAR {int(val/1000)}K" for val in df['balance']] + [f"SAR {int(df['balance'].sum()/1000)}K"],
        y=df['balance'].tolist() + [df['balance'].sum()],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": THEME["colors"]["success"]}},
        decreasing={"marker": {"color": THEME["colors"]["danger"]}},
        totals={"marker": {"color": THEME["colors"]["primary"]}},
        hovertemplate='<b>%{x}</b><br>Balance: SAR %{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Bank Balance Waterfall Analysis",
        xaxis_title="Banks",
        yaxis_title="Balance (SAR)",
        font=dict(family=APP_FONT, size=14),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_enhanced_liquidity_trend(df_fm):
    """Enhanced liquidity trend with multiple periods"""
    
    st.markdown('<h2 class="section-header">📈 Interactive Liquidity Trend</h2>', unsafe_allow_html=True)
    
    if df_fm.empty:
        st.info("No liquidity data available")
        return
    
    # Period selector
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        show_7d = st.button("7 Days", use_container_width=True, key="liq_7d")
    with col2:
        show_30d = st.button("30 Days", use_container_width=True, key="liq_30d")
    with col3:
        show_90d = st.button("90 Days", use_container_width=True, key="liq_90d")
    with col4:
        show_1y = st.button("1 Year", use_container_width=True, key="liq_1y")
    
    # Initialize session state for period
    if 'liquidity_period' not in st.session_state:
        st.session_state.liquidity_period = 'all'
    
    # Update period based on button clicks
    if show_7d:
        st.session_state.liquidity_period = '7d'
    elif show_30d:
        st.session_state.liquidity_period = '30d'
    elif show_90d:
        st.session_state.liquidity_period = '90d'
    elif show_1y:
        st.session_state.liquidity_period = '1y'
    
    # Filter data based on selection
    if st.session_state.liquidity_period == '7d':
        df_filtered = df_fm.tail(7)
        period_title = "7 Days"
    elif st.session_state.liquidity_period == '30d':
        df_filtered = df_fm.tail(30)
        period_title = "30 Days"
    elif st.session_state.liquidity_period == '90d':
        df_filtered = df_fm.tail(90)
        period_title = "90 Days"
    elif st.session_state.liquidity_period == '1y':
        df_filtered = df_fm.tail(365)
        period_title = "1 Year"
    else:
        df_filtered = df_fm
        period_title = "All Time"
    
    if not df_filtered.empty:
        fig = go.Figure()
        
        # Add main trend line
        fig.add_trace(go.Scatter(
            x=df_filtered['date'],
            y=df_filtered['total_liquidity'],
            mode='lines+markers',
            name='Liquidity',
            line=dict(
                color=THEME["colors"]["primary"],
                width=3,
                shape='spline'
            ),
            marker=dict(
                size=8,
                color=THEME["colors"]["primary"],
                line=dict(color='white', width=2)
            ),
            fill='tonexty',
            fillcolor=f'rgba(59, 130, 246, 0.1)',
            hovertemplate='<b>%{x}</b><br>Liquidity: SAR %{y:,.0f}<extra></extra>'
        ))
        
        # Add trend line
        if len(df_filtered) > 2:
            z = np.polyfit(range(len(df_filtered)), df_filtered['total_liquidity'], 1)
            p = np.poly1d(z)
            trend_line = p(range(len(df_filtered)))
            
            fig.add_trace(go.Scatter(
                x=df_filtered['date'],
                y=trend_line,
                mode='lines',
                name='Trend',
                line=dict(
                    color=THEME["colors"]["danger"],
                    width=2,
                    dash='dash'
                ),
                hovertemplate='<b>Trend Line</b><br>Value: SAR %{y:,.0f}<extra></extra>'
            ))
        
        fig.update_layout(
            title=f'Liquidity Trend Analysis - {period_title}',
            xaxis_title='Date',
            yaxis_title='Liquidity (SAR)',
            font=dict(family=APP_FONT, size=14),
            height=450,
            hovermode='x unified',
            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current", fmt_number_only(df_filtered['total_liquidity'].iloc[-1]))
        with col2:
            if len(df_filtered) > 1:
                change = df_filtered['total_liquidity'].iloc[-1] - df_filtered['total_liquidity'].iloc[-2]
                st.metric("Change", fmt_number_only(change), delta=fmt_number_only(change))
            else:
                st.metric("Change", "N/A")
        with col3:
            st.metric("Max", fmt_number_only(df_filtered['total_liquidity'].max()))
        with col4:
            st.metric("Average", fmt_number_only(df_filtered['total_liquidity'].mean()))

def create_enhanced_cvp_section(df_cvp):
    """Enhanced Collection vs Payments visualization"""
    
    st.markdown('<h2 class="section-header">🏢 Collection vs Payments - Enhanced</h2>', unsafe_allow_html=True)
    
    if df_cvp.empty:
        st.info("No collection vs payments data available")
        return
    
    # View selector
    col1, col2, col3 = st.columns(3)
    with col1:
        show_waterfall = st.button("🌊 Waterfall View", use_container_width=True, key="cvp_waterfall")
    with col2:
        show_comparison = st.button("⚖️ Comparison View", use_container_width=True, key="cvp_comparison")
    with col3:
        show_heatmap = st.button("🔥 Heatmap View", use_container_width=True, key="cvp_heatmap")
    
    # Initialize session state for CVP view
    if 'cvp_view' not in st.session_state:
        st.session_state.cvp_view = 'waterfall'
    
    # Update view based on button clicks
    if show_waterfall:
        st.session_state.cvp_view = 'waterfall'
    elif show_comparison:
        st.session_state.cvp_view = 'comparison'
    elif show_heatmap:
        st.session_state.cvp_view = 'heatmap'
    
    if st.session_state.cvp_view == 'waterfall':
        render_cvp_waterfall(df_cvp)
    elif st.session_state.cvp_view == 'comparison':
        render_cvp_comparison(df_cvp)
    elif st.session_state.cvp_view == 'heatmap':
        render_cvp_heatmap(df_cvp)

def render_cvp_waterfall(df_cvp):
    """Waterfall chart for net cash flow"""
    
    net_values = df_cvp['collection'] - df_cvp['payments']
    
    fig = go.Figure(go.Waterfall(
        name="Net Cash Flow",
        orientation="v",
        measure=["relative"] * len(net_values),
        x=df_cvp['branch'],
        textposition="outside",
        text=[f"{'+'if v >= 0 else ''}{int(v/1000)}K" for v in net_values],
        y=net_values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": THEME["colors"]["success"]}},
        decreasing={"marker": {"color": THEME["colors"]["danger"]}},
        hovertemplate='<b>%{x}</b><br>Net Flow: SAR %{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Net Cash Flow by Branch",
        xaxis_title="Branch",
        yaxis_title="Net Amount (SAR)",
        font=dict(family=APP_FONT, size=14),
        height=450
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_cvp_comparison(df_cvp):
    """Side-by-side comparison chart"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Collection',
        x=df_cvp['branch'],
        y=df_cvp['collection'],
        marker_color=THEME["colors"]["success"],
        hovertemplate='<b>%{x}</b><br>Collection: SAR %{y:,.0f}<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='Payments',
        x=df_cvp['branch'],
        y=df_cvp['payments'],
        marker_color=THEME["colors"]["danger"],
        hovertemplate='<b>%{x}</b><br>Payments: SAR %{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Collection vs Payments by Branch',
        barmode='group',
        xaxis_title='Branch',
        yaxis_title='Amount (SAR)',
        font=dict(family=APP_FONT, size=14),
        height=450
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_cvp_heatmap(df_cvp):
    """Heatmap visualization"""
    
    # Prepare data for heatmap
    heatmap_data = []
    heatmap_data.append(df_cvp['collection'].tolist())
    heatmap_data.append(df_cvp['payments'].tolist())
    heatmap_data.append((df_cvp['collection'] - df_cvp['payments']).tolist())
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=df_cvp['branch'],
        y=['Collection', 'Payments', 'Net'],
        colorscale='RdYlGn',
        hovertemplate='<b>%{y}</b> - %{x}<br>Amount: SAR %{z:,.0f}<extra></extra>',
        colorbar=dict(title="Amount (SAR)")
    ))
    
    fig.update_layout(
        title='Collection vs Payments Heatmap',
        xaxis_title='Branch',
        yaxis_title='Metric',
        font=dict(family=APP_FONT, size=14),
        height=450
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_lc_timeline_enhanced(df_lc):
    """Enhanced LC timeline with bubble chart"""
    
    st.markdown('<h2 class="section-header">📅 LC Settlements Timeline</h2>', unsafe_allow_html=True)
    
    if df_lc.empty:
        st.info("No LC settlements data available")
        return
    
    # Calculate days until due
    try:
        today = pd.Timestamp.now(tz=config.TZ).floor('D').tz_localize(None)
    except Exception:
        today = pd.Timestamp.today().floor('D')
    
    df_lc = df_lc.copy()
    df_lc['days_until'] = (df_lc['settlement_date'] - today).dt.days
    
    # Create bubble chart
    fig = go.Figure()
    
    # Determine colors based on urgency
    colors = []
    for days in df_lc['days_until']:
        if days <= 7:
            colors.append(THEME["colors"]["danger"])  # Red for urgent
        elif days <= 14:
            colors.append(THEME["colors"]["warning"])  # Orange for warning
        else:
            colors.append(THEME["colors"]["success"])  # Green for normal
    
    fig.add_trace(go.Scatter(
        x=df_lc['settlement_date'],
        y=df_lc['amount'],
        mode='markers+lines',
        marker=dict(
            size=[max(10, amount/25000) for amount in df_lc['amount']],
            color=colors,
            line=dict(color='white', width=2),
            opacity=0.8
        ),
        line=dict(color=THEME["colors"]["primary"], width=2, dash='dot'),
        text=[f"{bank}<br>SAR {amount:,.0f}<br>{days} days" 
              for bank, amount, days in zip(df_lc['bank'], df_lc['amount'], df_lc['days_until'])],
        textposition='top center',
        hovertemplate='<b>%{text}</b><br>Due Date: %{x}<extra></extra>',
        name='LC Settlements'
    ))
    
    fig.update_layout(
        title='LC Settlements Timeline (Bubble Size = Amount, Color = Urgency)',
        xaxis_title='Settlement Date',
        yaxis_title='Amount (SAR)',
        font=dict(family=APP_FONT, size=14),
        height=450,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add urgency summary
    col1, col2, col3 = st.columns(3)
    urgent = df_lc[df_lc['days_until'] <= 7]
    warning = df_lc[(df_lc['days_until'] > 7) & (df_lc['days_until'] <= 14)]
    normal = df_lc[df_lc['days_until'] > 14]
    
    with col1:
        st.metric(
            "🚨 Urgent (≤7 days)", 
            len(urgent),
            delta=f"SAR {urgent['amount'].sum():,.0f}" if not urgent.empty else "SAR 0"
        )
    with col2:
        st.metric(
            "⚠️ Warning (8-14 days)", 
            len(warning),
            delta=f"SAR {warning['amount'].sum():,.0f}" if not warning.empty else "SAR 0"
        )
    with col3:
        st.metric(
            "✅ Normal (>14 days)", 
            len(normal),
            delta=f"SAR {normal['amount'].sum():,.0f}" if not normal.empty else "SAR 0"
        )

# ----------------------------
# Header
# ----------------------------
def render_header():
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    c_logo, c_title = st.columns([0.08, 0.92])
    with c_logo:
        try: 
            st.image(config.LOGO_PATH, width=44)
        except Exception: 
            st.markdown("💰", help="Logo not found")
    with c_title:
        company_name_upper = config.COMPANY_NAME.upper()
        st.markdown(f'<h1>{company_name_upper}</h1>', unsafe_allow_html=True)
        st.markdown(f'<p>Enhanced Treasury Dashboard - Last refresh: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Sidebar (includes Auto Refresh)
# ----------------------------
def render_enhanced_sidebar(data_status, total_balance, approved_sum, lc_next4_sum, banks_cnt, bal_date):
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        do_auto = st.toggle("Auto refresh", value=st.session_state.get("auto_refresh", False),
                            help="Automatically refresh the dashboard at the chosen interval.")
        every_sec = st.number_input("Every (seconds)", min_value=15, max_value=900,
                                    value=int(st.session_state.get("auto_interval", 120)), step=15,
                                    help="How often to refresh when Auto refresh is ON.")
        st.session_state["auto_refresh"] = bool(do_auto)
        st.session_state["auto_interval"] = int(every_sec)

        if st.button("🔄 Refresh Now", type="primary", use_container_width=True):
            st.cache_data.clear()
            logger.info("Manual refresh triggered from sidebar")
            st.rerun()

        st.markdown("---")
        st.markdown("### 💰 Treasury Dashboard")
        st.markdown("---")
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
        _kpi("TOTAL BALANCE", total_balance, THEME["kpi"]["total_bg"], THEME["kpi"]["total_bd"], THEME["kpi"]["total_fg"])
        _kpi("APPROVED PAYMENTS", approved_sum, THEME["kpi"]["appr_bg"], THEME["kpi"]["appr_bd"], THEME["kpi"]["appr_fg"])
        _kpi("LC DUE (NEXT 4 DAYS)", lc_next4_sum, THEME["kpi"]["lc_bg"], THEME["kpi"]["lc_bd"], THEME["kpi"]["lc_fg"])
        _kpi("ACTIVE BANKS", banks_cnt, THEME["kpi"]["bank_bg"], THEME["kpi"]["bank_bd"], THEME["kpi"]["bank_fg"])

# ----------------------------
# Main Function
# ----------------------------
def main():
    render_header()
    st.markdown("")

    data_status = {}

    # Load data
    try:
        df_bal_raw = read_csv(LINKS["BANK BALANCE"])
        df_by_bank, bal_date = parse_bank_balance(df_bal_raw)
        data_status['bank_balance'] = 'success' if not df_by_bank.empty else 'warning'
    except Exception as e:
        logger.error(f"Bank balance processing failed: {e}")
        df_by_bank, bal_date = pd.DataFrame(), None
        data_status['bank_balance'] = 'error'

    try:
        df_pay_raw = read_csv(LINKS["SUPPLIER PAYMENTS"])
        df_pay = parse_supplier_payments(df_pay_raw)
        data_status['supplier_payments'] = 'success' if not df_pay.empty else 'warning'
    except Exception as e:
        logger.error(f"Supplier payments processing failed: {e}")
        df_pay = pd.DataFrame()
        data_status['supplier_payments'] = 'error'

    try:
        df_lc_raw = read_csv(LINKS["SETTLEMENTS"])
        df_lc = parse_settlements(df_lc_raw)
        data_status['settlements'] = 'success' if not df_lc.empty else 'warning'
    except Exception as e:
        logger.error(f"Settlements processing failed: {e}")
        df_lc = pd.DataFrame()
        data_status['settlements'] = 'error'

    try:
        df_fm_raw = read_csv(LINKS["Fund Movement"])
        df_fm = parse_fund_movement(df_fm_raw)
        data_status['fund_movement'] = 'success' if not df_fm.empty else 'warning'
    except Exception as e:
        logger.error(f"Fund movement processing failed: {e}")
        df_fm = pd.DataFrame()
        data_status['fund_movement'] = 'error'

    try:
        df_cvp_raw = read_csv(LINKS["COLLECTION_BRANCH"])
        df_cvp = parse_branch_cvp(df_cvp_raw)
        data_status['collection_branch'] = 'success' if not df_cvp.empty else 'warning'
    except Exception as e:
        logger.error(f"CVP processing failed: {e}")
        df_cvp = pd.DataFrame()
        data_status['collection_branch'] = 'error'

    # KPIs
    total_balance = float(df_by_bank["balance"].sum()) if not df_by_bank.empty else 0.0
    banks_cnt = int(df_by_bank["bank"].nunique()) if not df_by_bank.empty else 0

    # Timezone-safe "today"
    try:
        today0 = pd.Timestamp.now(tz=config.TZ).floor('D').tz_localize(None)
    except Exception:
        today0 = pd.Timestamp.today().floor('D')

    next4 = today0 + pd.Timedelta(days=3)
    lc_next4_sum = float(df_lc.loc[df_lc["settlement_date"].between(today0, next4), "amount"].sum() if not df_lc.empty else 0.0)
    approved_sum = float(df_pay["amount"].sum()) if not df_pay.empty else 0.0

    # Sidebar
    render_enhanced_sidebar(data_status, total_balance, approved_sum, lc_next4_sum, banks_cnt, bal_date)

    # ===== Enhanced Bank Balance Section =====
    if not df_by_bank.empty:
        create_enhanced_bank_balance_section(df_by_bank)
    else:
        st.info("No bank balance data available.")

    st.markdown("---")

    # ===== Enhanced Supplier Payments =====
    st.markdown('<h2 class="section-header">💰 Approved Payments</h2>', unsafe_allow_html=True)
    if df_pay.empty:
        st.info("No approved payments found.")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            banks = sorted(df_pay["bank"].dropna().unique())
            pick_banks = st.multiselect("Filter by Bank", banks, default=banks)
        with col2:
            min_amount = st.number_input("Minimum Amount", min_value=0, value=0)

        view_data = df_pay[(df_pay["bank"].isin(pick_banks)) & (df_pay["amount"] >= min_amount)].copy()
        if not view_data.empty:
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Total Amount", fmt_number_only(view_data["amount"].sum()))
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
        else:
            st.info("No payments match the selected criteria.")

    st.markdown("---")

    # ===== Enhanced LC Settlements =====
    if not df_lc.empty:
        create_lc_timeline_enhanced(df_lc)
    else:
        st.info("No LC settlements data available.")

    st.markdown("---")

    # ===== Enhanced Liquidity Trend =====
    if not df_fm.empty:
        create_enhanced_liquidity_trend(df_fm)
    else:
        st.info("No liquidity trend data available.")

    st.markdown("---")

    # ===== Enhanced Collection vs Payments =====
    if not df_cvp.empty:
        create_enhanced_cvp_section(df_cvp)
    else:
        st.info("No collection vs payments data available.")

    st.markdown("---")

    # ===== Quick Insights =====
    st.markdown('<h2 class="section-header">💡 Quick Insights & Recommendations</h2>', unsafe_allow_html=True)
    insights = []
    if not df_by_bank.empty:
        top_bank = df_by_bank.sort_values("balance", ascending=False).iloc[0]
        insights.append({"type": "info", "title": "Top Bank Balance", "content": f"**{top_bank['bank']}** holds the highest balance: {fmt_number_only(top_bank['balance'])}"})
        total_bal = df_by_bank["balance"].sum()
        if total_bal:
            top_3_pct = df_by_bank.nlargest(3, "balance")["balance"].sum() / total_bal * 100
            if top_3_pct > 80:
                insights.append({"type": "warning", "title": "Concentration Risk", "content": f"Top 3 banks hold {top_3_pct:.1f}% of total balance. Consider diversification."})
        neg_df = df_by_bank[df_by_bank["balance"] < 0]
        if not neg_df.empty:
            insights.append({"type": "error", "title": "Negative Bank Balances", "content": f"{len(neg_df)} bank(s) show negative balances totaling {fmt_number_only(neg_df['balance'].sum())}."})
    if not df_pay.empty and total_balance:
        total_approved = df_pay["amount"].sum()
        if total_approved > total_balance * 0.8:
            insights.append({"type": "warning", "title": "Cash Flow Alert", "content": f"Approved payments ({fmt_number_only(total_approved)}) represent {(total_approved/total_balance)*100:.1f}% of available balance."})
    if not df_lc.empty:
        urgent7 = df_lc[df_lc["settlement_date"] <= today0 + pd.Timedelta(days=7)]
        if not urgent7.empty:
            insights.append({"type": "error", "title": "Urgent LC Settlements", "content": f"{len(urgent7)} LC settlements due within 7 days totaling {fmt_number_only(urgent7['amount'].sum())}"})
    if not df_fm.empty and len(df_fm) > 5:
        recent_trend = df_fm.tail(5)["total_liquidity"].pct_change().mean()
        if pd.notna(recent_trend) and recent_trend < -0.05:
            insights.append({"type": "warning", "title": "Declining Liquidity Trend", "content": f"Liquidity has been declining by an average of {abs(recent_trend)*100:.1f}% over recent periods."})
    
    if insights:
        for ins in insights:
            if ins["type"] == "info":
                st.info(f"ℹ️ **{ins['title']}**: {ins['content']}")
            elif ins["type"] == "warning":
                st.warning(f"⚠️ **{ins['title']}**: {ins['content']}")
            elif ins["type"] == "error":
                st.error(f"🚨 **{ins['title']}**: {ins['content']}")
    else:
        st.info("💡 Insights will appear as data becomes available and patterns emerge.")

    # --- Footer ---
    st.markdown("<hr style='margin: 8px 0 16px 0;'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center; opacity:0.8; font-size:12px;'>Enhanced Treasury Dashboard - Created By <strong>Jaseer Pykkarathodi</strong></div>",
        unsafe_allow_html=True
    )

    # === Auto-refresh (end of app) ===
    if st.session_state.get("auto_refresh"):
        interval = int(st.session_state.get("auto_interval", 120))
        with st.status(f"Auto refreshing in {interval}s…", expanded=False):
            time.sleep(interval)
        st.rerun()

if __name__ == "__main__":
    main()
