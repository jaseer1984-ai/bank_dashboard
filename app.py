# app.py — Treasury Dashboard
# Always show: Available Balance + Balance After Settlement (no "Projected").
# Negative-safe parsing, red font for negatives, light-red card bg if any value is negative.
# Global font, right-aligned tables, auto-refresh, logo fallback hidden, footer.

import io
import time
import logging
import os
import re
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

# ---- pandas Styler type hint guard ----
try:
    from pandas.io.formats.style import Styler
except Exception:
    Styler = Any

# ----------------------------
# Config
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

# ---------- THEME ----------
THEME = {
    "palette": {
        "text_dark": "#1e293b",
        "text_negative": "#B91C1C",
        "after_label": "#334155",
        "light_red": "#FFE4E6",  # background for any negative
        "lavender": "#e0e7ff",
        "pink": "#fce7f3",
        "sky": "#e0f2fe",
        "mint": "#ecfdf5",
    },
    "icons": {
        "very_high": "💎",
        "high": "🔹",
        "medium": "💠",
        "low": "💚",
        "negative": "🔻",
    },
    # Choose by available first; if after-settlement exists and is negative, use negative style.
    "card_rules": [
        {"when": lambda bal, aft: (aft is not None) and (pd.notna(aft)) and (float(aft) < 0), "bg": "light_red", "icon": "negative"},
        {"when": lambda bal, aft: (bal is not None) and (pd.notna(bal)) and (float(bal) < 0), "bg": "light_red", "icon": "negative"},
        {"when": lambda bal, aft: (bal is not None) and (pd.notna(bal)) and (float(bal) > 500_000), "bg": "lavender", "icon": "very_high"},
        {"when": lambda bal, aft: (bal is not None) and (pd.notna(bal)) and (float(bal) > 100_000), "bg": "pink", "icon": "high"},
        {"when": lambda bal, aft: (bal is not None) and (pd.notna(bal)) and (float(bal) > 50_000), "bg": "sky", "icon": "medium"},
        {"when": lambda bal, aft: True, "bg": "mint", "icon": "low"},
    ],
    "kpi": {
        "total_balance":     {"bg": "#EEF2FF", "border": "#C7D2FE", "color": "#1E3A8A"},
        "approved_payments": {"bg": "#E9FFF2", "border": "#C7F7DD", "color": "#065F46"},
        "lc_due":            {"bg": "#FFF7E6", "border": "#FDE9C8", "color": "#92400E"},
        "active_banks":      {"bg": "#FFF1F2", "border": "#FBD5D8", "color": "#9F1239"},
    },
}

def get_card_style(bal: float, aft: Optional[float]) -> dict:
    pal, icons = THEME["palette"], THEME["icons"]
    for rule in THEME["card_rules"]:
        try:
            if rule["when"](bal, aft):
                return {
                    "bg": pal[rule["bg"]],
                    "icon": icons[rule["icon"]],
                    "text": pal["text_dark"],
                    "text_negative": pal["text_negative"],
                    "after_label": pal["after_label"],
                }
        except Exception:
            continue
    return {"bg": pal["mint"], "icon": icons["low"], "text": pal["text_dark"], "text_negative": pal["text_negative"], "after_label": pal["after_label"]}

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
# Page
# ----------------------------
st.set_page_config(
    page_title="Treasury Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="💰"
)

APP_FONT = os.getenv("APP_FONT", "Inter")

def set_app_font(family: str = APP_FONT):
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=%s:wght@300;400;500;600;700;800&display=swap');
          :root { --app-font: '%s', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', sans-serif; }
          html, body, [class^="css"], [class*=" css"] { font-family: var(--app-font) !important; }
          h1, h2, h3, h4, h5, h6, p, span, div, label, small, strong, em { font-family: var(--app-font) !important; }
          button, input, textarea, select { font-family: var(--app-font) !important; }
          div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"] { font-family: var(--app-font) !important; }
          div[data-testid="stDataFrame"] * { font-family: var(--app-font) !important; }
        </style>
        """ % (family.replace(" ", "+"), family),
        unsafe_allow_html=True
    )
set_app_font()

st.markdown("""
<style>
    .main-header { position: sticky; top: 0; background: white; z-index: 999; padding: 15px 0; border-bottom: 2px solid #e6eaf0; margin-bottom: 20px; }
    .main-header h1 { font-size: 28px !important; font-weight: 900 !important; color: #1a202c !important; text-transform: uppercase !important; letter-spacing: .5px !important; margin: 0 !important; }
    .list-item { display:flex; justify-content:space-between; align-items:center; padding:12px 16px; border-bottom:1px solid #e2e8f0; }
    .list-item:last-child { border-bottom:none; }
    .list-bank { font-weight:600; color:#1e293b; }
    .list-amount { font-weight:700; color:#059669; }
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
}

# ----------------------------
# Rate limit
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
    """
    Robust numeric parser:
    - commas, currency text, unicode minus, accounting '(5,000)'
    """
    if pd.isna(x) or str(x).strip() == "":
        return np.nan
    s = str(x).strip()
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")  # normalize dash/minus
    negative = False
    if re.match(r"^\(.*\)$", s):
        negative = True
        s = s[1:-1]
    s = re.sub(r"[A-Za-z\s]", "", s)  # remove currency/space
    s = s.replace(",", "")            # remove thousands
    if s in {"", "-", "."}:
        return np.nan
    try:
        num = float(s)
        if negative: num = -num
        if abs(num) > 1e12:
            logger.warning(f"Unusually large number detected: {num}")
            return np.nan
        return num
    except Exception:
        logger.debug(f"Number conversion failed for '{x}'")
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
    def red_if_negative(v):
        try:
            return "color: #B91C1C" if float(v) < 0 else ""
        except Exception:
            return ""
    for col in num_cols:
        styler = styler.applymap(red_if_negative, subset=[col])
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
# UI helpers
# ----------------------------
def display_as_list(df, bank_col="bank", amount_col="balance", title="Bank Balances"):
    st.markdown(f"**{title}**")
    for _, row in df.iterrows():
        st.markdown(
            f"""
            <div class="list-item">
                <span class="list-bank">{row[bank_col]}</span>
                <span class="list-amount">{fmt_currency(row[amount_col])}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

def display_as_mini_cards(df, bank_col="bank", amount_col="balance"):
    cols = st.columns(3)
    for i, row in df.iterrows():
        with cols[int(i) % 3]:
            st.markdown(
                f"""
                <div style="background:linear-gradient(135deg,#e0f2fe 0%,#bae6fd 100%);padding:16px;border-radius:12px;border-left:4px solid #0284c7;margin-bottom:12px;">
                    <div style="font-size:12px;color:#0f172a;font-weight:600;margin-bottom:8px;">{row[bank_col]}</div>
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
                    <div style="height:100%;background:linear-gradient(90deg,#3b82f6 0%,#06b6d4 100%);border-radius:4px;width:{percentage}%;"></div>
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
                if amount >= 1_000_000:
                    display_amount = f"{amount/1_000_000:.1f}M"
                elif amount >= 1_000:
                    display_amount = f"{amount/1_000:.0f}K"
                else:
                    display_amount = f"{amount:.0f}"
                st.markdown(
                    f"""
                    <div style="text-align:center;padding:20px;background:linear-gradient(135deg,#fef3c7 0%,#fde68a 100%);border-radius:12px;border:2px solid #f59e0b;margin-bottom:12px;">
                        <div style="font-size:12px;color:#92400e;font-weight:600;margin-bottom:8px;">{row[bank_col]}</div>
                        <div style="font-size:20px;font-weight:800;color:#92400e;">{display_amount}</div>
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
        if "after" in c and ("settle" in c or "settel" in c):
            return col
        if "balance after" in c and ("settle" in c or "settel" in c):
            return col
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
    """
    Output columns:
      - bank
      - balance  (Available Balance)
      - after_settlement (Balance After Settlement) — if present
    """
    try:
        c = cols_lower(df)

        # Structured
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
                    if "after_settlement" in out.columns: agg["after_settlement"] = "sum"
                    by_bank = out.groupby("bank", as_index=False).agg(agg)
                    return by_bank, datetime.now()

        # Legacy matrix
        raw = df.copy().dropna(how="all").dropna(axis=1, how="all")

        # bank col
        bank_col = None
        for col in raw.columns:
            if raw[col].dtype == object:
                non_empty = (raw[col].dropna().astype(str).str.strip() != "").sum()
                if non_empty >= 3:
                    bank_col = col; break
        if bank_col is None:
            raise ValueError("Could not detect bank column")

        # latest date col
        parsed = pd.to_datetime(pd.Index(raw.columns), errors="coerce", dayfirst=False)
        date_cols = [col for col, d in zip(raw.columns, parsed) if pd.notna(d)]
        if not date_cols:
            raise ValueError("No valid date columns found")
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
        d = cols_lower(df).rename(columns={
            "supplier name": "supplier",
            "amount(sar)": "amount_sar",
            "order/sh/branch": "order_branch"
        })
        if not validate_dataframe(d, ["bank", "status"], "Supplier Payments"):
            return pd.DataFrame()

        # normalize and filter APPROVED rows
        status_norm = d["status"].astype("string").str.strip().str.lower()
        mask = status_norm.str.contains(r"\bapproved\b", na=False)
        if not mask.any():
            logger.info("No approved payments found")
            return pd.DataFrame()
        d = d.loc[mask].copy()

        # pick amount column and clean numbers
        amt_col = next((c for c in ["amount_sar", "amount", "amount(sar)"] if c in d.columns), None)
        if not amt_col:
            logger.error("No amount column found in supplier payments")
            return pd.DataFrame()

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

# ----------------------------
# Header
# ----------------------------
def render_header():
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    c_logo, c_title = st.columns([0.08, 0.92])
    with c_logo:
        # Hide emoji/tooltip fallback: only show if file exists
        try:
            if os.path.exists(config.LOGO_PATH):
                st.image(config.LOGO_PATH, width=44)
            else:
                st.write("")
        except Exception:
            st.write("")
    with c_title:
        company_name_upper = config.COMPANY_NAME.upper()
        st.markdown(f'<h1 class="main-header h1">{company_name_upper}</h1>', unsafe_allow_html=True)
        st.caption(datetime.now().strftime("Last refresh: %Y-%m-%d %H:%M:%S"))
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Sidebar
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

        def _kpi(title, value, key):
            s = THEME["kpi"][key]
            st.markdown(
                f"""
                <div style="background:{s['bg']};border:1px solid {s['border']};border-radius:12px;padding:16px;margin-bottom:12px;box-shadow:0 1px 6px rgba(0,0,0,.04);">
                    <div style="font-size:11px;color:#374151;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px;">{title}</div>
                    <div style="font-size:20px;font-weight:800;color:{s['color']};text-align:right;">{(f"{float(value):,.0f}" if value or value == 0 else "N/A")}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        _kpi("TOTAL BALANCE", total_balance, "total_balance")
        _kpi("APPROVED PAYMENTS", approved_sum, "approved_payments")
        _kpi("LC DUE (NEXT 4 DAYS)", lc_next4_sum, "lc_due")
        _kpi("ACTIVE BANKS", banks_cnt, "active_banks")

# ----------------------------
# Main
# ----------------------------
def main():
    render_header()
    st.markdown("")

    data_status = {}

    # Load
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

    # KPIs
    total_balance = float(df_by_bank["balance"].sum()) if not df_by_bank.empty else 0.0
    banks_cnt = int(df_by_bank["bank"].nunique()) if not df_by_bank.empty else 0

    # Today
    try:
        today0 = pd.Timestamp.now(tz=config.TZ).floor('D').tz_localize(None)
    except Exception:
        today0 = pd.Timestamp.today().floor('D')

    next4 = today0 + pd.Timedelta(days=3)
    lc_next4_sum = float(df_lc.loc[df_lc["settlement_date"].between(today0, next4), "amount"].sum() if not df_lc.empty else 0.0)
    approved_sum = float(df_pay["amount"].sum()) if not df_pay.empty else 0.0

    # Sidebar
    render_enhanced_sidebar(data_status, total_balance, approved_sum, lc_next4_sum, banks_cnt, bal_date)

    # ===== Bank Balance =====
    st.header("🏦 Bank Balance")
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
                    aft = row.get('after_settlement', np.nan)

                    # Style choice
                    style = get_card_style(bal, aft)
                    bg = style["bg"]; icon = style["icon"]
                    text = style["text"]; neg = style["text_negative"]; after_label = style["after_label"]

                    bal_color = neg if (pd.notna(bal) and float(bal) < 0) else text
                    aft_color = neg if (pd.notna(aft) and float(aft) < 0) else text

                    aft_block = (
                        f"""
                        <div style="display:flex;justify-content:space-between;align-items:baseline;margin-top:10px;">
                            <div style="font-size:11px;color:{after_label};opacity:.85;">Balance After Settlement</div>
                            <div style="font-size:20px;font-weight:800;color:{aft_color};">{fmt_currency(aft)}</div>
                        </div>
                        """
                        if "after_settlement" in row.index else
                        f"""
                        <div style="display:flex;justify-content:space-between;align-items:baseline;margin-top:10px;">
                            <div style="font-size:11px;color:{after_label};opacity:.85;">Balance After Settlement</div>
                            <div style="font-size:20px;font-weight:800;color:{after_label};">N/A</div>
                        </div>
                        """
                    )

                    st.markdown(
                        f"""
                        <div style="background-color:{bg};padding:20px;border-radius:12px;margin-bottom:16px;box-shadow:0 2px 8px rgba(0,0,0,0.1);">
                            <div style="display:flex;align-items:center;margin-bottom:12px;">
                                <span style="font-size:18px;margin-right:8px;">{icon}</span>
                                <span style="font-size:13px;font-weight:600;color:{text};">{row['bank']}</span>
                            </div>
                            <div style="font-size:24px;font-weight:800;color:{bal_color};text-align:right;">{fmt_currency(bal)}</div>
                            <div style="font-size:11px;color:{text};opacity:.7;margin-top:6px;">Available Balance</div>
                            {aft_block}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        elif view == "List":
            display_as_list(df_bal_view, "bank", "balance", "Bank Balances")
        elif view == "Mini Cards":
            display_as_mini_cards(df_bal_view, "bank", "balance")
        elif view == "Progress Bars":
            display_as_progress_bars(df_bal_view, "bank", "balance")
        elif view == "Metrics":
            display_as_metrics(df_bal_view, "bank", "balance")
        else:
            # Show both columns if present
            table = df_bal_view.rename(columns={
                "bank": "Bank",
                "balance": "Available",
                "after_settlement": "Balance After Settlement"
            })
            num_cols = [c for c in ["Available", "Balance After Settlement"] if c in table.columns]
            st.dataframe(style_right(table, num_cols=num_cols), use_container_width=True, height=360)

    st.markdown("---")

    # ===== Supplier Payments =====
    st.header("💰 Approved Payments")
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
            payment_view = st.radio("Display as:", options=["Summary + Table", "Mini Cards", "List", "Progress Bars"],
                                    index=0, horizontal=True, key="payment_view")
            if payment_view == "Summary + Table":
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

            elif payment_view == "Mini Cards":
                bank_totals = view_data.groupby("bank", as_index=False)["amount"].sum().rename(columns={"amount": "balance"})
                display_as_mini_cards(bank_totals, "bank", "balance")
            elif payment_view == "List":
                bank_totals = view_data.groupby("bank", as_index=False)["amount"].sum().rename(columns={"amount": "balance"})
                display_as_list(bank_totals, "bank", "balance", "Approved Payments by Bank")
            elif payment_view == "Progress Bars":
                bank_totals = view_data.groupby("bank", as_index=False)["amount"].sum().rename(columns={"amount": "balance"})
                display_as_progress_bars(bank_totals, "bank", "balance")
        else:
            st.info("No payments match the selected criteria.")

    st.markdown("---")

    # ===== LC Settlements =====
    st.header("📅 LC Settlements — Pending")
    if df_lc.empty:
        st.info("No LC (Pending) data. Ensure sheet has Bank, Maturity Date/New Maturity Date, and any of: Balance for Settlement / Currently Due / Amount(SAR).")
    else:
        c1, c2 = st.columns(2)
        with c1:
            start_date = st.date_input("From Date", value=df_lc["settlement_date"].min().date())
        with c2:
            end_date = st.date_input("To Date", value=df_lc["settlement_date"].max().date())

        lc_view = df_lc[(df_lc["settlement_date"].dt.date >= start_date) & (df_lc["settlement_date"].dt.date <= end_date)].copy()
        if not lc_view.empty:
            lc_display = st.radio("Display as:", options=["Summary + Table", "Progress by Urgency", "Mini Cards"],
                                  index=0, horizontal=True, key="lc_view")
            if lc_display == "Summary + Table":
                cc1, cc2, cc3 = st.columns(3)
                with cc1: st.metric("Total LC Amount", fmt_number_only(lc_view["amount"].sum()))
                with cc2: st.metric("Number of LCs", len(lc_view))
                with cc3: st.metric("Urgent (2 days)", len(lc_view[lc_view["settlement_date"] <= today0 + pd.Timedelta(days=2)]))

                viz = lc_view.copy()
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

                styled = style_right(show, num_cols=["Amount"])
                styled = styled.apply(_highlight, axis=1)
                st.dataframe(styled, use_container_width=True, height=400)

            elif lc_display == "Progress by Urgency":
                tmp = lc_view.copy()
                tmp["days_until_due"] = (tmp["settlement_date"] - today0).dt.days
                urgent = tmp[tmp["days_until_due"] <= 2]
                warning = tmp[(tmp["days_until_due"] > 2) & (tmp["days_until_due"] <= 7)]
                normal = tmp[tmp["days_until_due"] > 7]

                st.markdown("**📊 LC Settlements by Urgency**")
                if not urgent.empty:
                    st.markdown("**🚨 Urgent (≤2 days)**")
                    display_as_progress_bars(urgent.groupby("bank", as_index=False)["amount"].sum().rename(columns={"amount": "balance"}))
                if not warning.empty:
                    st.markdown("**⚠️ Warning (3-7 days)**")
                    display_as_progress_bars(warning.groupby("bank", as_index=False)["amount"].sum().rename(columns={"amount": "balance"}))
                if not normal.empty:
                    st.markdown("**✅ Normal (>7 days)**")
                    display_as_progress_bars(normal.groupby("bank", as_index=False)["amount"].sum().rename(columns={"amount": "balance"}))

            urgent_lcs = lc_view[lc_view["settlement_date"] <= today0 + pd.Timedelta(days=3)]
            if not urgent_lcs.empty:
                st.warning(f"⚠️ {len(urgent_lcs)} LC(s) due within 3 days!")
                for _, lc in urgent_lcs.iterrows():
                    days_left = (lc["settlement_date"] - today0).days
                    st.write(f"• {lc['bank']} - {fmt_number_only(lc['amount'])} - {days_left} day(s) left")

    st.markdown("---")

    # ===== Quick Insights =====
    st.header("💡 Quick Insights & Recommendations")
    insights = []
    if not df_by_bank.empty:
        top_bank = df_by_bank.sort_values("balance", ascending=False).iloc[0]
        insights.append({"type": "info", "title": "Top Bank Balance", "content": f"**{top_bank['bank']}** has the highest available: {fmt_number_only(top_bank['balance'])}"})
        total_bal = df_by_bank["balance"].sum()
        if "after_settlement" in df_by_bank.columns:
            total_after = df_by_bank["after_settlement"].sum()
            insights.append({"type": "info", "title": "Total After Settlement", "content": f"Sum of 'Balance After Settlement' across banks: **{fmt_number_only(total_after)}**"})
            neg_after = df_by_bank[df_by_bank["after_settlement"] < 0]
            if not neg_after.empty:
                deficit = -neg_after["after_settlement"].sum()
                worst = neg_after.nsmallest(1, "after_settlement").iloc[0]
                insights.append({"type": "error", "title": "Negative After Settlement", "content": f"{len(neg_after)} bank(s) negative after settlement. Total deficit **{fmt_number_only(deficit)}**. Worst: **{worst['bank']} ({fmt_number_only(worst['after_settlement'])})**"})
        if total_bal:
            top_3_pct = df_by_bank.nlargest(3, "balance")["balance"].sum() / total_bal * 100
            if top_3_pct > 80:
                insights.append({"type": "warning", "title": "Concentration Risk", "content": f"Top 3 banks hold {top_3_pct:.1f}% of total available. Consider diversification."})
        neg_avail_cnt = int((df_by_bank["balance"] < 0).sum())
        if neg_avail_cnt > 0:
            insights.append({"type": "warning", "title": "Negative Available Balances", "content": f"{neg_avail_cnt} bank(s) showing negative available balances."})

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
        "<div style='text-align:center; opacity:0.8; font-size:12px;'>Created By <strong>Jaseer Pykkarathodi</strong></div>",
        unsafe_allow_html=True
    )

    # === Auto-refresh ===
    if st.session_state.get("auto_refresh"):
        interval = int(st.session_state.get("auto_interval", 120))
        with st.status(f"Auto refreshing in {interval}s…", expanded=False):
            time.sleep(interval)
        st.rerun()

if __name__ == "__main__":
    main()


