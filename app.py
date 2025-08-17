# app.py — Enhanced Treasury Dashboard with Sidebar KPIs, Clean Header, and Fixed Right-Aligned Tables
# Major improvements in v2.1:
# - FIX: Timezone-safe "today" handling (no tz errors)
# - NEW: Consistent right-aligned numeric tables using Pandas Styler (keeps numbers sortable; no SAR prefix)
# - Tidy: Safer requests retry config; minor robustness and comments

import io
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

# ----------------------------
# Configuration Management
# ----------------------------
@dataclass
class Config:
    """Centralized configuration with environment variable support"""
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

# ----------------------------
# Logging Setup
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('treasury_dashboard.log')
    ]
)
logger = logging.getLogger(__name__)

# ----------------------------
# Page Setup (keep this early)
# ----------------------------
st.set_page_config(
    page_title="Treasury Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="💰"
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .status-indicator { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; }
    .status-healthy { background-color: #10b981; }
    .status-warning { background-color: #f59e0b; }
    .status-error { background-color: #ef4444; }

    /* Sticky Header */
    .main-header { position: sticky; top: 0; background: white; z-index: 999; padding: 15px 0; border-bottom: 2px solid #e6eaf0; margin-bottom: 20px; }
    .main-header h1 { font-size: 28px !important; font-weight: 900 !important; color: #1a202c !important; text-transform: uppercase !important; letter-spacing: 0.5px !important; margin: 0 !important; padding: 0 !important; line-height: 1.2 !important; }

    /* Sidebar customization */
    .sidebar .sidebar-content { padding-top: 1rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# Enhanced HTTP Session with Retry Logic
# ----------------------------
def create_session() -> requests.Session:
    """Create requests session with retry strategy."""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],  # be explicit
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

# Global session
http_session = create_session()

# ----------------------------
# Google Sheets Links
# ----------------------------
LINKS = {
    "BANK BALANCE": f"https://docs.google.com/spreadsheets/d/{config.FILE_ID}/export?format=csv&gid=860709395",
    "SUPPLIER PAYMENTS": f"https://docs.google.com/spreadsheets/d/{config.FILE_ID}/export?format=csv&gid=20805295",
    "SETTLEMENTS": f"https://docs.google.com/spreadsheets/d/{config.FILE_ID}/export?format=csv&gid=978859477",
    "Fund Movement": f"https://docs.google.com/spreadsheets/d/{config.FILE_ID}/export?format=csv&gid=66055663",
}

# ----------------------------
# Rate Limiting Decorator
# ----------------------------
def rate_limit(calls_per_minute: int = config.RATE_LIMIT_CALLS_PER_MINUTE):
    """Rate limiting decorator to prevent API abuse"""
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
                    time.sleep(max(0, min_interval - time_passed))
            last_called[key] = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator

# ----------------------------
# Helpers: number formatting & styling
# ----------------------------
def _to_number(x) -> float:
    """Enhanced number conversion with validation and bounds checking"""
    if pd.isna(x) or x == '':
        return np.nan
    s = str(x).strip().replace(",", "")
    if s.endswith("%"):
        s = s[:-1]
    try:
        num = float(s)
        if abs(num) > 1e12:  # sanity bound
            logger.warning(f"Unusually large number detected: {num}")
            return np.nan
        return num
    except (ValueError, OverflowError) as e:
        logger.debug(f"Number conversion failed for '{x}': {e}")
        return np.nan

def cols_lower(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in df.columns]
    return out

def fmt_currency(v, currency="SAR") -> str:
    try:
        if pd.isna(v):
            return "N/A"
        return f"{currency} {float(v):,.0f}"
    except (ValueError, TypeError):
        return str(v)

# Keep numbers numeric for sorting; we only format in display via Styler or metric text

def style_right(df: pd.DataFrame, thousands=True) -> pd.io.formats.style.Styler:
    """Return a right-aligned Styler with thousands separator but NO currency prefix."""
    styler = df.style.format(
        {col: ("{0:,.0f}" if thousands else "{0}") for col in df.select_dtypes(include=["number"]).columns}
    )
    styler = styler.set_properties(**{"text-align": "right"})
    styler = styler.set_table_styles([
        {"selector": "th", "props": "text-align: right; background-color: #f8f9fa; font-weight: 600;"}
    ])
    return styler

# ----------------------------
# Cached CSV reader
# ----------------------------
@st.cache_data(ttl=config.CACHE_TTL)
@rate_limit()
def read_csv(url: str) -> pd.DataFrame:
    try:
        logger.info(f"Fetching data from: {url}")
        response = http_session.get(url, timeout=config.REQUEST_TIMEOUT, stream=True)
        response.raise_for_status()
        content_length = response.headers.get('content-length')
        if content_length is not None:
            try:
                if int(content_length) > config.MAX_FILE_SIZE_MB * 1024 * 1024:
                    raise ValueError(f"File too large: {content_length} bytes")
            except ValueError:
                pass  # non-integer header; ignore
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
# Table Alternative Display Functions (unchanged visuals)
# ----------------------------
def display_as_list(df, bank_col="bank", amount_col="balance", title="Bank Balances"):
    st.markdown(f"**{title}**")
    for _, row in df.iterrows():
        st.markdown(
            f"""
            <div class="list-item" style="display:flex;justify-content:space-between;align-items:center;padding:12px 16px;border-bottom:1px solid #e2e8f0;">
                <span class="list-bank" style="font-weight:600;color:#1e293b;">{row[bank_col]}</span>
                <span class="list-amount" style="font-weight:700;color:#059669;">{fmt_currency(row[amount_col])}</span>
            </div>
            """,
            unsafe_allow_html=True,
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
                unsafe_allow_html=True,
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
            unsafe_allow_html=True,
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
                    unsafe_allow_html=True,
                )

# ----------------------------
# Parsers
# ----------------------------
def validate_dataframe(df: pd.DataFrame, required_cols: list, sheet_name: str) -> bool:
    if df.empty:
        st.warning(f"📊 {sheet_name}: No data available")
        return False
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.warning(f"📊 {sheet_name}: Missing required columns: {missing_cols}")
        return False
    if len(df) < 1:
        st.warning(f"📊 {sheet_name}: Insufficient data rows")
        return False
    return True

def parse_bank_balance(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[datetime]]:
    try:
        c = cols_lower(df)
        if "bank" in c.columns and ("amount" in c.columns or "amount(sar)" in c.columns):
            amt_col = "amount" if "amount" in c.columns else "amount(sar)"
            out = pd.DataFrame(
                {"bank": c["bank"].astype(str).str.strip(), "balance": c[amt_col].map(_to_number)}
            ).dropna()
            if validate_dataframe(out, ["bank", "balance"], "Bank Balance"):
                by_bank = out.groupby("bank", as_index=False)["balance"].sum()
                return by_bank, datetime.now()
        # Fallback: detect bank col + latest date col from headers
        raw = df.copy().dropna(how="all").dropna(axis=1, how="all")
        bank_col = None
        for col in raw.columns:
            if raw[col].dtype == object:
                non_empty_count = (raw[col].dropna().astype(str).str.strip() != "").sum()
                if non_empty_count >= 3:
                    bank_col = col
                    break
        if bank_col is None:
            raise ValueError("Could not detect bank column")
        parsed = pd.to_datetime(pd.Index(raw.columns), errors="coerce", dayfirst=False)
        date_cols = [col for col, d in zip(raw.columns, parsed) if pd.notna(d)]
        if not date_cols:
            raise ValueError("No valid date columns found")
        date_map = {col: pd.to_datetime(col, errors="coerce", dayfirst=False) for col in date_cols}
        latest_col = max(date_cols, key=lambda c: date_map[c])
        s = raw[bank_col].astype(str).str.strip()
        mask = s.ne("") & ~s.str.contains("available|total", case=False, na=False)
        sub = raw.loc[mask, [bank_col, latest_col]].copy()
        sub.columns = ["bank", "balance"]
        sub["balance"] = sub["balance"].astype(str).str.replace(",", "", regex=False).map(_to_number)
        sub["bank"] = sub["bank"].str.replace(r"\s*-\s*.*$", "", regex=True).str.strip()
        latest_date = date_map[latest_col]
        by_bank = sub.dropna().groupby("bank", as_index=False)["balance"].sum()
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
            "order/sh/branch": "order_branch",
        })
        if not validate_dataframe(d, ["bank", "status"], "Supplier Payments"):
            return pd.DataFrame()
        status_norm = d["status"].astype(str).str.strip().str.lower()
        approved_mask = status_norm.str.contains("approved", na=False)
        d = d.loc[approved_mask].copy()
        if d.empty:
            logger.info("No approved payments found")
            return pd.DataFrame()
        amt_col = next((c for c in ["amount_sar", "amount", "amount(sar)"] if c in d.columns), None)
        if amt_col is None:
            logger.error("No amount column found in supplier payments")
            return pd.DataFrame()
        d["amount"] = d[amt_col].map(_to_number)
        d["bank"] = d["bank"].astype(str).str.strip()
        keep = [c for c in ["bank", "supplier", "currency", "amount", "status"] if c in d.columns]
        out = d[keep].dropna(subset=["amount"]).copy()
        if "status" in out.columns:
            out["status"] = out["status"].astype(str).str.title()
        logger.info(f"Parsed {len(out)} approved supplier payments")
        return out
    except Exception as e:
        logger.error(f"Supplier payments parsing error: {e}")
        st.error(f"❌ Supplier payments parsing failed: {str(e)}")
        return pd.DataFrame()

def parse_settlements(df: pd.DataFrame) -> pd.DataFrame:
    try:
        d = cols_lower(df)
        ref_col = next((c for c in d.columns if any(term in c for term in ["a/c", "ref", "account", "reference"])), None)
        bank_col = next((c for c in d.columns if (c.startswith("bank") or "bank" in c)), None)
        date_col = next((c for c in d.columns if ("maturity" in c and "new" not in c)), None) or \
                   next((c for c in d.columns if ("new" in c and "maturity" in c)), None)
        amount_col = next((c for c in d.columns if ("balance" in c and "settlement" in c)), None) or \
                     next((c for c in d.columns if ("currently" in c and "due" in c)), None) or \
                     next((c for c in ["amount(sar)", "amount"] if c in d.columns), None)
        if not all([bank_col, amount_col, date_col]):
            missing = [name for name, col in [("bank", bank_col), ("amount", amount_col), ("date", date_col)] if not col]
            logger.error(f"Missing required columns for settlements: {missing}")
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
            "description": "",
        })
        out = out.dropna(subset=["bank", "amount", "settlement_date"])  # keep only valid
        out = out[out["status"].str.lower() == "pending"].copy()
        logger.info(f"Parsed {len(out)} pending settlements")
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
            logger.error(f"Fund movement missing required columns. Date: {date_col}, Liquidity: {liq_col}")
            return pd.DataFrame()
        out = pd.DataFrame({"date": pd.to_datetime(d[date_col], errors="coerce"), "total_liquidity": d[liq_col].map(_to_number)}).dropna()
        logger.info(f"Parsed {len(out)} fund movement records")
        return out.sort_values("date")
    except Exception as e:
        logger.error(f"Fund movement parsing error: {e}")
        st.error(f"❌ Fund movement parsing failed: {str(e)}")
        return pd.DataFrame()

# ----------------------------
# Header Section
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
        st.markdown(f'<h1 class="main-header h1">{company_name_upper}</h1>', unsafe_allow_html=True)
        current_time = datetime.now().strftime("Last refresh: %Y-%m-%d %H:%M:%S")
        st.caption(current_time)
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Sidebar
# ----------------------------
def render_enhanced_sidebar(data_status, total_balance, approved_sum, lc_next4_sum, banks_cnt, bal_date):
    with st.sidebar:
        if st.button("🔄 Refresh All Data", type="primary", use_container_width=True):
            st.cache_data.clear()
            logger.info("Manual refresh triggered from sidebar")
            st.rerun()
        st.markdown("---")
        st.markdown("### 💰 Treasury Dashboard")
        st.markdown("---")
        st.markdown("### 📊 Key Metrics")
        # KPI cards (numbers only, no SAR prefix)
        def _kpi(title, value, bg, border, color):
            st.markdown(
                f"""
                <div style="background:{bg};border:1px solid {border};border-radius:12px;padding:16px;margin-bottom:12px;box-shadow:0 1px 6px rgba(0,0,0,.04);">
                    <div style="font-size:11px;color:#374151;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px;">{title}</div>
                    <div style="font-size:20px;font-weight:800;color:{color};text-align:right;">{(f"{float(value):,.0f}" if value else "N/A")}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        _kpi("TOTAL BALANCE", total_balance, "#EEF2FF", "#C7D2FE", "#1E3A8A")
        _kpi("APPROVED PAYMENTS", approved_sum, "#E9FFF2", "#C7F7DD", "#065F46")
        _kpi("LC DUE (NEXT 4 DAYS)", lc_next4_sum, "#FFF7E6", "#FDE9C8", "#92400E")
        _kpi("ACTIVE BANKS", banks_cnt, "#FFF1F2", "#FBD5D8", "#9F1239")
        if bal_date:
            st.markdown(
                f"""
                <div style="font-size:10px;color:#6b7280;text-align:center;margin-bottom:20px;padding:8px;background:#f9fafb;border-radius:6px;">
                    💡 Balance updated: {bal_date.strftime('%Y-%m-%d at %H:%M')}
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("---")
        st.markdown(
            f"""
            <div style="font-size:10px;color:#6b7280;line-height:1.4;">
                <div><strong>📊 Dashboard</strong></div>
                <div>Version: Enhanced v2.1</div>
                <div>Cache: {config.CACHE_TTL}s</div>
                <div>TZ: {config.TZ}</div>
                <br>
                <div><strong>📈 Sources</strong></div>
                <div>Active: {sum(1 for status in data_status.values() if status == 'success')}/4</div>
                <div>Refresh: {datetime.now().strftime('%H:%M:%S')}</div>
                <br>
                <div><strong>🏗️ By</strong></div>
                <div>Jaseer Pykarathodi</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ----------------------------
# Main Application
# ----------------------------
def main():
    render_header()
    st.markdown("")

    data_status: Dict[str, str] = {}
    # Load all sources
    try:
        df_bal_raw = read_csv(LINKS["BANK BALANCE"]) ; df_by_bank, bal_date = parse_bank_balance(df_bal_raw)
        data_status['bank_balance'] = 'success' if not df_by_bank.empty else 'warning'
    except Exception as e:
        logger.error(f"Bank balance processing failed: {e}")
        df_by_bank, bal_date = pd.DataFrame(), None
        data_status['bank_balance'] = 'error'

    try:
        df_pay_raw = read_csv(LINKS["SUPPLIER PAYMENTS"]) ; df_pay = parse_supplier_payments(df_pay_raw)
        data_status['supplier_payments'] = 'success' if not df_pay.empty else 'warning'
    except Exception as e:
        logger.error(f"Supplier payments processing failed: {e}")
        df_pay = pd.DataFrame()
        data_status['supplier_payments'] = 'error'

    try:
        df_lc_raw = read_csv(LINKS["SETTLEMENTS"]) ; df_lc = parse_settlements(df_lc_raw)
        data_status['settlements'] = 'success' if not df_lc.empty else 'warning'
    except Exception as e:
        logger.error(f"Settlements processing failed: {e}")
        df_lc = pd.DataFrame()
        data_status['settlements'] = 'error'

    try:
        df_fm_raw = read_csv(LINKS["Fund Movement"]) ; df_fm = parse_fund_movement(df_fm_raw)
        data_status['fund_movement'] = 'success' if not df_fm.empty else 'warning'
    except Exception as e:
        logger.error(f"Fund movement processing failed: {e}")
        df_fm = pd.DataFrame()
        data_status['fund_movement'] = 'error'

    # KPIs
    try:
        total_balance = float(df_by_bank["balance"].sum()) if not df_by_bank.empty else 0.0
    except Exception:
        total_balance = 0.0
    try:
        banks_cnt = int(df_by_bank["bank"].nunique()) if not df_by_bank.empty else 0
    except Exception:
        banks_cnt = 0

    # Timezone-safe "today"
    local_now = pd.Timestamp.now(tz=config.TZ)
    today0 = pd.Timestamp(local_now.date())  # naive midnight of local date
    try:
        next4 = today0 + pd.Timedelta(days=3)
        lc_next4_sum = float(df_lc.loc[df_lc["settlement_date"].between(today0, next4), "amount"].sum() if not df_lc.empty else 0.0)
    except Exception:
        lc_next4_sum = 0.0
    try:
        approved_sum = float(df_pay["amount"].sum()) if not df_pay.empty else 0.0
    except Exception:
        approved_sum = 0.0

    # Sidebar
    render_enhanced_sidebar(data_status, total_balance, approved_sum, lc_next4_sum, banks_cnt, bal_date)

    # ===== Bank Balance =====
    st.header("🏦 Bank Balance")
    if df_by_bank.empty:
        st.info("No balances found.")
    else:
        view = st.radio("", options=["Cards", "List", "Mini Cards", "Progress Bars", "Metrics", "Table"], index=0, horizontal=True, label_visibility="collapsed")
        df_bal_view = df_by_bank.copy().sort_values("balance", ascending=False)
        if view == "Cards":
            cols = st.columns(4)
            for i, row in df_bal_view.iterrows():
                with cols[int(i) % 4]:
                    balance = row['balance']
                    if balance > 500_000:
                        bg_color, icon = "#e0e7ff", "💎"
                    elif balance > 100_000:
                        bg_color, icon = "#fce7f3", "🔹"
                    elif balance > 50_000:
                        bg_color, icon = "#e0f2fe", "💠"
                    else:
                        bg_color, icon = "#ecfdf5", "💚"
                    st.markdown(
                        f"""
                        <div style="background-color:{bg_color};padding:20px;border-radius:12px;margin-bottom:16px;box-shadow:0 2px 8px rgba(0,0,0,0.1);">
                            <div style="display:flex;align-items:center;margin-bottom:12px;"><span style="font-size:18px;margin-right:8px;">{icon}</span><span style="font-size:13px;font-weight:600;color:#1e293b;">{row['bank']}</span></div>
                            <div style="font-size:24px;font-weight:800;color:#1e293b;text-align:right;">{fmt_currency(row['balance'])}</div>
                            <div style="font-size:9px;color:#1e293b;opacity:0.7;margin-top:8px;">Available Balance</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
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
            df_bal_table = df_bal_view[["bank", "balance"]].rename(columns={"bank": "Bank", "balance": "Balance"})
            st.dataframe(style_right(df_bal_table), use_container_width=True, height=360)

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
            payment_view = st.radio("Display as:", options=["Summary + Table", "Mini Cards", "List", "Progress Bars"], index=0, horizontal=True, key="payment_view")
            if payment_view == "Summary + Table":
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("Total Amount", f"{view_data['amount'].sum():,.0f}")
                with c2: st.metric("Number of Payments", len(view_data))
                with c3: st.metric("Average Payment", f"{view_data['amount'].mean():,.0f}")
                grp = view_data.groupby("bank", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
                st.markdown("**📊 Summary by Bank**")
                grp2 = grp.rename(columns={"bank": "Bank", "amount": "Amount"})
                st.dataframe(style_right(grp2), use_container_width=True, height=220)
                st.markdown("**📋 Detailed Payment List**")
                show_cols = [c for c in ["bank", "supplier", "currency", "amount", "status"] if c in view_data.columns]
                v = view_data[show_cols].rename(columns={"bank": "Bank", "supplier": "Supplier", "currency": "Currency", "amount": "Amount", "status": "Status"})
                st.dataframe(style_right(v), use_container_width=True, height=360)
            elif payment_view == "Mini Cards":
                bank_totals = view_data.groupby("bank")["amount"].sum().reset_index().rename(columns={"amount": "balance"})
                display_as_mini_cards(bank_totals, "bank", "balance")
            elif payment_view == "List":
                bank_totals = view_data.groupby("bank")["amount"].sum().reset_index().rename(columns={"amount": "balance"})
                display_as_list(bank_totals, "bank", "balance", "Approved Payments by Bank")
            elif payment_view == "Progress Bars":
                bank_totals = view_data.groupby("bank")["amount"].sum().reset_index().rename(columns={"amount": "balance"})
                display_as_progress_bars(bank_totals, "bank", "balance")
        else:
            st.info("No payments match the selected criteria.")

    st.markdown("---")

    # ===== LC Settlements =====
    st.header("📅 LC Settlements — Pending")
    if df_lc.empty:
        st.info("No LC (Pending) data. Ensure sheet has Bank, Maturity Date/New Maturity Date, and any of: Balance for Settlement / Currently Due / Amount(SAR).")
    else:
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("From Date", value=df_lc["settlement_date"].min().date())
        with col2:
            end_date = st.date_input("To Date", value=df_lc["settlement_date"].max().date())
        lc_view = df_lc[(df_lc["settlement_date"].dt.date >= start_date) & (df_lc["settlement_date"].dt.date <= end_date)].copy()
        if not lc_view.empty:
            lc_display = st.radio("Display as:", options=["Summary + Table", "Progress by Urgency", "Mini Cards"], index=0, horizontal=True, key="lc_view")
            if lc_display == "Summary + Table":
                cc1, cc2, cc3 = st.columns(3)
                with cc1: st.metric("Total LC Amount", f"{lc_view['amount'].sum():,.0f}")
                with cc2: st.metric("Number of LCs", len(lc_view))
                with cc3: st.metric("Urgent (2 days)", len(lc_view[lc_view["settlement_date"] <= today0 + pd.Timedelta(days=2)]))
                viz = lc_view.copy()
                viz["Settlement Date"] = viz["settlement_date"].dt.strftime(config.DATE_FMT)
                viz["Amount"] = viz["amount"]  # keep numeric for sorting; format via styler
                viz["Days Until Due"] = (viz["settlement_date"] - today0).dt.days
                rename_map = {"reference": "Reference", "bank": "Bank", "type": "Type", "status": "Status", "remark": "Remark", "description": "Description"}
                viz = viz.rename(columns={k: v for k, v in rename_map.items() if k in viz.columns})
                potential = ["Reference", "Bank", "Type", "Status", "Settlement Date", "Amount", "Days Until Due", "Remark", "Description"]
                display_cols = [c for c in potential if c in viz.columns]
                df_show = viz[display_cols]
                if "Settlement Date" in df_show.columns:
                    df_show = df_show.sort_values("Settlement Date")
                def _highlight(row):
                    if "Days Until Due" in row:
                        if row["Days Until Due"] <= 2:
                            return ['background-color: #fee2e2'] * len(row)
                        elif row["Days Until Due"] <= 7:
                            return ['background-color: #fef3c7'] * len(row)
                    return [''] * len(row)
                st.dataframe(style_right(df_show).apply(_highlight, axis=1), use_container_width=True, height=400)
            elif lc_display == "Progress by Urgency":
                lc_temp = lc_view.copy()
                lc_temp["days_until_due"] = (lc_temp["settlement_date"] - today0).dt.days
                urgent = lc_temp[lc_temp["days_until_due"] <= 2]
                warning = lc_temp[(lc_temp["days_until_due"] > 2) & (lc_temp["days_until_due"] <= 7)]
                normal = lc_temp[lc_temp["days_until_due"] > 7]
                st.markdown("**📊 LC Settlements by Urgency**")
                if not urgent.empty:
                    st.markdown("**🚨 Urgent (≤2 days)**")
                    display_as_progress_bars(urgent.groupby("bank")["amount"].sum().reset_index().rename(columns={"amount": "balance"}))
                if not warning.empty:
                    st.markdown("**⚠️ Warning (3-7 days)**")
                    display_as_progress_bars(warning.groupby("bank")["amount"].sum().reset_index().rename(columns={"amount": "balance"}))
                if not normal.empty:
                    st.markdown("**✅ Normal (>7 days)**")
                    display_as_progress_bars(normal.groupby("bank")["amount"].sum().reset_index().rename(columns={"amount": "balance"}))
            elif lc_display == "Mini Cards":
                lc_grouped = lc_view.groupby("bank")["amount"].sum().reset_index().rename(columns={"amount": "balance"})
                display_as_mini_cards(lc_grouped, "bank", "balance")
            urgent_lcs = lc_view[lc_view["settlement_date"] <= today0 + pd.Timedelta(days=3)]
            if not urgent_lcs.empty:
                st.warning(f"⚠️ {len(urgent_lcs)} LC(s) due within 3 days!")
                for _, lc in urgent_lcs.iterrows():
                    days_left = (lc["settlement_date"] - today0).days
                    st.write(f"• {lc['bank']} - {lc['amount']:,.0f} - {days_left} day(s) left")

    st.markdown("---")

    # ===== Liquidity Trend =====
    st.header("📈 Liquidity Trend Analysis")
    if df_fm.empty:
        st.info("No liquidity data available.")
    else:
        try:
            latest_liquidity = df_fm.iloc[-1]["total_liquidity"]
            trend_text = "No trend data"
            if len(df_fm) > 1:
                previous_liquidity = df_fm.iloc[-2]["total_liquidity"]
                trend_change = latest_liquidity - previous_liquidity
                trend_pct = (trend_change / previous_liquidity) * 100 if previous_liquidity != 0 else 0
                trend_text = f"{'📈' if trend_change > 0 else '📉'} {trend_pct:+.1f}%"
            col1, col2 = st.columns([3, 1])
            with col1:
                try:
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_fm["date"], y=df_fm["total_liquidity"], mode='lines+markers', name='Total Liquidity', line=dict(width=3), marker=dict(size=6)))
                    fig.update_layout(title="Total Liquidity Trend", xaxis_title="Date", yaxis_title="Liquidity (SAR)", margin=dict(l=20, r=20, t=50, b=20), height=400, showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.line_chart(df_fm.set_index("date")["total_liquidity"])
            with col2:
                st.markdown("### 📊 Liquidity Metrics")
                st.metric("Current", f"{latest_liquidity:,.0f}")
                if len(df_fm) > 1:
                    st.metric("Trend", trend_text)
                st.markdown("**Statistics (30d)**")
                recent = df_fm.tail(30)
                st.write(f"**Max:** {recent['total_liquidity'].max():,.0f}")
                st.write(f"**Min:** {recent['total_liquidity'].min():,.0f}")
                st.write(f"**Avg:** {recent['total_liquidity'].mean():,.0f}")
        except Exception as e:
            logger.error(f"Liquidity trend analysis error: {e}")
            st.error("❌ Unable to display liquidity trend analysis")
            st.line_chart(df_fm.set_index("date")["total_liquidity"])

    st.markdown("---")

    # ===== Quick Insights =====
    st.header("💡 Quick Insights & Recommendations")
    insights = []
    if 'df_by_bank' in locals() and not df_by_bank.empty:
        top_bank = df_by_bank.sort_values("balance", ascending=False).iloc[0]
        insights.append({"type": "info", "title": "Top Bank Balance", "content": f"**{top_bank['bank']}** holds the highest balance: {top_bank['balance']:,.0f}"})
        total_bal = df_by_bank["balance"].sum()
        if total_bal > 0:
            top_3_pct = df_by_bank.nlargest(3, "balance")["balance"].sum() / total_bal * 100
            if top_3_pct > 80:
                insights.append({"type": "warning", "title": "Concentration Risk", "content": f"Top 3 banks hold {top_3_pct:.1f}% of total balance. Consider diversification."})
    if 'df_pay' in locals() and not df_pay.empty and total_balance > 0:
        total_approved = df_pay["amount"].sum()
        if total_approved > total_balance * 0.8:
            insights.append({"type": "warning", "title": "Cash Flow Alert", "content": f"Approved payments ({total_approved:,.0f}) represent {(total_approved/total_balance)*100:.1f}% of available balance."})
    if 'df_lc' in locals() and not df_lc.empty:
        urgent_lcs = df_lc[df_lc["settlement_date"] <= today0 + pd.Timedelta(days=7)]
        if not urgent_lcs.empty:
            insights.append({"type": "error", "title": "Urgent LC Settlements", "content": f"{len(urgent_lcs)} LC settlements due within 7 days totaling {urgent_lcs['amount'].sum():,.0f}"})
    if 'df_fm' in locals() and not df_fm.empty and len(df_fm) > 5:
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

if __name__ == "__main__":
    main()
