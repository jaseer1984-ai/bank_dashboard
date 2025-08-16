# app.py — Enhanced Treasury Dashboard
# Major improvements: Better error handling, configuration management, performance optimization,
# data validation, security enhancements, and new features

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
# Page Setup
# ----------------------------
st.set_page_config(
    page_title="Treasury Dashboard", 
    layout="wide", 
    initial_sidebar_state="collapsed",
    page_icon="💰"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-healthy { background-color: #10b981; }
    .status-warning { background-color: #f59e0b; }
    .status-error { background-color: #ef4444; }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Enhanced HTTP Session with Retry Logic
# ----------------------------
def create_session() -> requests.Session:
    """Create requests session with retry strategy"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
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
        last_called = {}
        
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
# Enhanced Helper Functions
# ----------------------------
def simple_kpi_card(title, value, bg="#EEF2FF", border="#C7D2FE", text="#111827"):
    """Ultra-simple KPI card to avoid any naming conflicts"""
    try:
        # Simple number formatting
        if pd.isna(value) or value is None:
            display_value = "N/A"
        elif isinstance(value, (int, float)):
            display_value = f"{float(value):,.0f}"
        else:
            display_value = str(value)
        
        card_html = f"""
        <div style="
            background:{bg};
            border:1px solid {border};
            border-radius:12px;
            padding:14px 16px;
            box-shadow:0 1px 6px rgba(0,0,0,.04);">
            <div style="font-size:12px;color:#374151;text-transform:uppercase;letter-spacing:.08em">
                {title}
            </div>
            <div style="font-size:28px;font-weight:800;color:{text};margin-top:4px;text-align:right;direction:ltr;">
                {display_value}
            </div>
        </div>
        """
        
        st.markdown(card_html, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Card error for {title}: {str(e)}")

def _to_number(x) -> float:
    """Enhanced number conversion with validation and bounds checking"""
    if pd.isna(x) or x == '':
        return np.nan
    
    s = str(x).strip().replace(",", "")
    if s.endswith("%"):
        s = s[:-1]
    
    try:
        num = float(s)
        # Add reasonable bounds checking for financial data
        if abs(num) > 1e12:  # Trillion+ seems unreasonable
            logger.warning(f"Unusually large number detected: {num}")
            return np.nan
        return num
    except (ValueError, OverflowError) as e:
        logger.debug(f"Number conversion failed for '{x}': {e}")
        return np.nan

def validate_dataframe(df: pd.DataFrame, required_cols: list, sheet_name: str) -> bool:
    """Validate DataFrame has required columns and data"""
    if df.empty:
        st.warning(f"📊 {sheet_name}: No data available")
        return False
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.warning(f"📊 {sheet_name}: Missing required columns: {missing_cols}")
        return False
    
    # Check for minimum data rows
    if len(df) < 1:
        st.warning(f"📊 {sheet_name}: Insufficient data rows")
        return False
    
    return True

@st.cache_data(ttl=config.CACHE_TTL)
@rate_limit()
def read_csv(url: str) -> pd.DataFrame:
    """Enhanced CSV reader with comprehensive error handling"""
    try:
        logger.info(f"Fetching data from: {url}")
        
        # Make request with timeout and size checking
        response = http_session.get(url, timeout=config.REQUEST_TIMEOUT, stream=True)
        response.raise_for_status()
        
        # Check content size to prevent memory issues
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > config.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ValueError(f"File too large: {content_length} bytes")
        
        # Read and validate CSV
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

def cols_lower(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase"""
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in df.columns]
    return out

def fmt_full_int(v) -> str:
    """Format number as integer with thousands separators"""
    try:
        if pd.isna(v):
            return "N/A"
        return f"{float(v):,.0f}"
    except (ValueError, TypeError):
        return str(v)

def fmt_currency(v, currency="SAR") -> str:
    """Format number as currency"""
    try:
        if pd.isna(v):
            return "N/A"
        return f"{currency} {float(v):,.0f}"
    except (ValueError, TypeError):
        return str(v)

def get_simple_freshness_text(last_update: Optional[datetime]) -> str:
    """Return simple text for data freshness (no HTML)"""
    if not last_update:
        return "No update info"
    
    now = datetime.now()
    hours_old = (now - last_update).total_seconds() / 3600
    
    if hours_old < 1:
        return f"Updated {int(hours_old * 60)} min ago"
    elif hours_old < 24:
        return f"Updated {int(hours_old)} hours ago"
    else:
        return f"Updated {int(hours_old / 24)} days ago"

def get_data_freshness_indicator(last_update: Optional[datetime]) -> str:
    """Return HTML for data freshness indicator"""
    if not last_update:
        return '<span class="status-indicator status-error"></span>No update info'
    
    now = datetime.now()
    hours_old = (now - last_update).total_seconds() / 3600
    
    if hours_old < 1:
        status = "status-healthy"
        text = f"Updated {int(hours_old * 60)} min ago"
    elif hours_old < 24:
        status = "status-warning" if hours_old > 6 else "status-healthy"
        text = f"Updated {int(hours_old)} hours ago"
    else:
        status = "status-error"
        text = f"Updated {int(hours_old / 24)} days ago"
    
    return f'<span class="status-indicator {status}"></span>{text}'

def enhanced_kpi_card(title: str, value: Any, subtitle: str = "", 
                     bg: str = "#EEF2FF", border: str = "#C7D2FE", 
                     text: str = "#111827", trend: Optional[str] = None):
    """Enhanced KPI card with trend indicators and data freshness"""
    formatted_value = fmt_full_int(value) if isinstance(value, (int, float, np.integer, np.floating)) else str(value)
    
    trend_html = ""
    if trend:
        trend_color = "#10b981" if trend.startswith("+") else "#ef4444" if trend.startswith("-") else "#6b7280"
        trend_html = f'<div style="font-size:11px;color:{trend_color};margin-top:2px;">{trend}</div>'
    
    subtitle_html = f'<div style="font-size:10px;color:#9ca3af;margin-top:2px;">{subtitle}</div>' if subtitle else ""
    
    st.markdown(
        f"""
        <div style="
            background:{bg};border:1px solid {border};
            border-radius:12px;padding:14px 16px;
            box-shadow:0 2px 8px rgba(0,0,0,.06);
            transition:transform 0.2s ease;
        " onmouseover="this.style.transform='translateY(-2px)'" 
           onmouseout="this.style.transform='translateY(0)'">
            <div style="font-size:12px;color:#374151;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px;">
                {title}
            </div>
            <div style="font-size:28px;font-weight:800;color:{text};text-align:right;">
                {formatted_value}
            </div>
            {trend_html}
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True
    )

# ----------------------------
# Enhanced Parser Functions
# ----------------------------
def parse_bank_balance(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[datetime]]:
    """Enhanced bank balance parser with better error handling"""
    try:
        c = cols_lower(df)
        
        # Try structured format first
        if "bank" in c.columns and ("amount" in c.columns or "amount(sar)" in c.columns):
            amt_col = "amount" if "amount" in c.columns else "amount(sar)"
            out = pd.DataFrame({
                "bank": c["bank"].astype(str).str.strip(),
                "balance": c[amt_col].map(_to_number)
            }).dropna()
            
            if validate_dataframe(out, ["bank", "balance"], "Bank Balance"):
                by_bank = out.groupby("bank", as_index=False)["balance"].sum()
                return by_bank, datetime.now()
        
        # Fallback to original parsing logic
        raw = df.copy().dropna(how="all").dropna(axis=1, how="all")
        
        # Find bank column
        bank_col = None
        for col in raw.columns:
            if raw[col].dtype == object:
                non_empty_count = (raw[col].dropna().astype(str).str.strip() != "").sum()
                if non_empty_count >= 3:
                    bank_col = col
                    break
        
        if bank_col is None:
            raise ValueError("Could not detect bank column")
        
        # Find date columns
        parsed = pd.to_datetime(pd.Index(raw.columns), errors="coerce", dayfirst=False)
        date_cols = [col for col, d in zip(raw.columns, parsed) if pd.notna(d)]
        
        if not date_cols:
            raise ValueError("No valid date columns found")
        
        date_map = {col: pd.to_datetime(col, errors="coerce", dayfirst=False) for col in date_cols}
        latest_col = max(date_cols, key=lambda c: date_map[c])
        
        # Process data
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
    """Enhanced supplier payments parser"""
    try:
        d = cols_lower(df).rename(columns={
            "supplier name": "supplier",
            "amount(sar)": "amount_sar",
            "order/sh/branch": "order_branch"
        })
        
        required_cols = ["bank", "status"]
        if not validate_dataframe(d, required_cols, "Supplier Payments"):
            return pd.DataFrame()
        
        # Filter for approved payments
        status_norm = d["status"].astype(str).str.strip().str.lower()
        approved_mask = status_norm.str.contains("approved", na=False)
        d = d.loc[approved_mask].copy()
        
        if d.empty:
            logger.info("No approved payments found")
            return pd.DataFrame()
        
        # Find amount column
        amt_col = None
        for col_name in ["amount_sar", "amount", "amount(sar)"]:
            if col_name in d.columns:
                amt_col = col_name
                break
        
        if amt_col is None:
            logger.error("No amount column found in supplier payments")
            return pd.DataFrame()
        
        d["amount"] = d[amt_col].map(_to_number)
        d["bank"] = d["bank"].astype(str).str.strip()
        
        # Select relevant columns
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
    """Enhanced settlements parser with robust column detection"""
    try:
        d = cols_lower(df)
        
        # Enhanced column detection
        ref_col = None
        for col in d.columns:
            if any(term in col for term in ["a/c", "ref", "account", "reference"]):
                ref_col = col
                break
        
        bank_col = None
        for col in d.columns:
            if col.startswith("bank") or "bank" in col:
                bank_col = col
                break
        
        date_col = None
        for col in d.columns:
            if "maturity" in col and "new" not in col:
                date_col = col
                break
        if not date_col:
            for col in d.columns:
                if "new" in col and "maturity" in col:
                    date_col = col
                    break
        
        amount_col = None
        for col in d.columns:
            if "balance" in col and "settlement" in col:
                amount_col = col
                break
        if not amount_col:
            for col in d.columns:
                if "currently" in col and "due" in col:
                    amount_col = col
                    break
        if not amount_col:
            for col_name in ["amount(sar)", "amount"]:
                if col_name in d.columns:
                    amount_col = col_name
                    break
        
        # Validate required columns
        if not all([bank_col, amount_col, date_col]):
            missing = [name for name, col in [("bank", bank_col), ("amount", amount_col), ("date", date_col)] if not col]
            logger.error(f"Missing required columns for settlements: {missing}")
            return pd.DataFrame()
        
        # Additional columns
        status_col = next((c for c in d.columns if "status" in c), None)
        type_col = next((c for c in d.columns if "type" in c), None)
        remark_col = next((c for c in d.columns if "remark" in c), None)
        
        # Build output DataFrame
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
        
        # Filter and validate
        out = out.dropna(subset=["bank", "amount", "settlement_date"])
        out = out[out["status"].str.lower() == "pending"].copy()
        
        logger.info(f"Parsed {len(out)} pending settlements")
        return out
        
    except Exception as e:
        logger.error(f"Settlements parsing error: {e}")
        st.error(f"❌ Settlements parsing failed: {str(e)}")
        return pd.DataFrame()

def parse_fund_movement(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced fund movement parser"""
    try:
        d = cols_lower(df)
        
        date_col = "date" if "date" in d.columns else None
        liq_col = None
        for col in d.columns:
            if "total" in col and "liquidity" in col:
                liq_col = col
                break
        
        if not date_col or not liq_col:
            logger.error(f"Fund movement missing required columns. Date: {date_col}, Liquidity: {liq_col}")
            return pd.DataFrame()
        
        out = pd.DataFrame({
            "date": pd.to_datetime(d[date_col], errors="coerce"),
            "total_liquidity": d[liq_col].map(_to_number)
        }).dropna()
        
        logger.info(f"Parsed {len(out)} fund movement records")
        return out.sort_values("date")
        
    except Exception as e:
        logger.error(f"Fund movement parsing error: {e}")
        st.error(f"❌ Fund movement parsing failed: {str(e)}")
        return pd.DataFrame()

# ----------------------------
# Data Export Function
# ----------------------------
def export_data_to_excel(df_bank, df_payments, df_settlements, df_liquidity):
    """Export all data to Excel file"""
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            if not df_bank.empty:
                df_bank.to_excel(writer, sheet_name='Bank_Balances', index=False)
            if not df_payments.empty:
                df_payments.to_excel(writer, sheet_name='Supplier_Payments', index=False)
            if not df_settlements.empty:
                df_settlements.to_excel(writer, sheet_name='LC_Settlements', index=False)
            if not df_liquidity.empty:
                df_liquidity.to_excel(writer, sheet_name='Liquidity_Trend', index=False)
        
        output.seek(0)
        return output.getvalue()
    except Exception as e:
        logger.error(f"Excel export error: {e}")
        st.error(f"❌ Export failed: {str(e)}")
        return None

# ----------------------------
# Header Section
# ----------------------------
def render_header():
    """Render application header with logo and refresh functionality"""
    c_logo, c_title, c_refresh = st.columns([0.08, 0.74, 0.18])
    
    with c_logo:
        try:
            st.image(config.LOGO_PATH, width=44)
        except Exception:
            st.markdown("💰", help="Logo not found")
    
    with c_title:
        st.markdown(
            f"<h2 style='margin:0;padding-top:6px;display:flex;align-items:center;'>{config.COMPANY_NAME} — Treasury Dashboard</h2>",
            unsafe_allow_html=True
        )
    
    with c_refresh:
        current_time = datetime.now().strftime("Last refresh: %Y-%m-%d %H:%M:%S")
        st.caption(current_time)
        
        if st.button("🔄 Refresh", type="primary", use_container_width=True):
            st.cache_data.clear()
            logger.info("Manual refresh triggered")
            st.rerun()

# ----------------------------
# Main Application
# ----------------------------
def main():
    """Main application function"""
    render_header()
    st.markdown("")
    
    # Initialize session state
    if 'show_export' not in st.session_state:
        st.session_state['show_export'] = False
    
    # Load and parse data with enhanced error handling
    data_status = {}
    
    # Bank Balance
    try:
        df_bal_raw = read_csv(LINKS["BANK BALANCE"])
        df_by_bank, bal_date = parse_bank_balance(df_bal_raw)
        data_status['bank_balance'] = 'success' if not df_by_bank.empty else 'warning'
    except Exception as e:
        logger.error(f"Bank balance processing failed: {e}")
        df_by_bank, bal_date = pd.DataFrame(), None
        data_status['bank_balance'] = 'error'
    
    # Supplier Payments
    try:
        df_pay_raw = read_csv(LINKS["SUPPLIER PAYMENTS"])
        df_pay = parse_supplier_payments(df_pay_raw)
        data_status['supplier_payments'] = 'success' if not df_pay.empty else 'warning'
    except Exception as e:
        logger.error(f"Supplier payments processing failed: {e}")
        df_pay = pd.DataFrame()
        data_status['supplier_payments'] = 'error'
    
    # LC Settlements
    try:
        df_lc_raw = read_csv(LINKS["SETTLEMENTS"])
        df_lc = parse_settlements(df_lc_raw)
        data_status['settlements'] = 'success' if not df_lc.empty else 'warning'
    except Exception as e:
        logger.error(f"Settlements processing failed: {e}")
        df_lc = pd.DataFrame()
        data_status['settlements'] = 'error'
    
    # Fund Movement
    try:
        df_fm_raw = read_csv(LINKS["Fund Movement"])
        df_fm = parse_fund_movement(df_fm_raw)
        data_status['fund_movement'] = 'success' if not df_fm.empty else 'warning'
    except Exception as e:
        logger.error(f"Fund movement processing failed: {e}")
        df_fm = pd.DataFrame()
        data_status['fund_movement'] = 'error'
    
    # Calculate KPIs with safe defaults and error handling
    try:
        total_balance = float(df_by_bank["balance"].sum()) if not df_by_bank.empty else 0.0
    except Exception:
        total_balance = 0.0
        
    try:
        banks_cnt = int(df_by_bank["bank"].nunique()) if not df_by_bank.empty else 0
    except Exception:
        banks_cnt = 0
    
    try:
        today0 = pd.Timestamp.now(tz=config.TZ).normalize().tz_localize(None)
        next4 = today0 + pd.Timedelta(days=3)
        
        lc_next4_sum = float(
            df_lc.loc[df_lc["settlement_date"].between(today0, next4), "amount"].sum()
            if not df_lc.empty else 0.0
        )
    except Exception:
        lc_next4_sum = 0.0
        
    try:
        approved_sum = float(df_pay["amount"].sum()) if not df_pay.empty else 0.0
    except Exception:
        approved_sum = 0.0
    
    # Display KPIs with simple cards - using try/catch for each
    k1, k2, k3, k4 = st.columns(4)
    
    with k1:
        try:
            simple_kpi_card("Total Balance", total_balance, "#E6F0FF", "#C7D8FE", "#1E3A8A")
        except Exception as e:
            st.error(f"KPI 1 Error: {e}")
            
    with k2:
        try:
            simple_kpi_card("Approved Payments", approved_sum, "#E9FFF2", "#C7F7DD", "#065F46")
        except Exception as e:
            st.error(f"KPI 2 Error: {e}")
            
    with k3:
        try:
            simple_kpi_card("LC due (next 4 days)", lc_next4_sum, "#FFF7E6", "#FDE9C8", "#92400E")
        except Exception as e:
            st.error(f"KPI 3 Error: {e}")
            
    with k4:
        try:
            simple_kpi_card("Active Banks", banks_cnt, "#FFF1F2", "#FBD5D8", "#9F1239")
        except Exception as e:
            st.error(f"KPI 4 Error: {e}")
    
    # Add data freshness info below KPIs
    if bal_date:
        st.caption(f"💡 Bank balance data last updated: {bal_date.strftime('%Y-%m-%d at %H:%M')}")
    
    # Data Health Dashboard
    st.markdown("---")
    st.subheader("📊 Data Health Status")
    
    status_cols = st.columns(4)
    status_items = [
        ("Bank Balance", data_status.get('bank_balance', 'error')),
        ("Supplier Payments", data_status.get('supplier_payments', 'error')),
        ("LC Settlements", data_status.get('settlements', 'error')),
        ("Fund Movement", data_status.get('fund_movement', 'error'))
    ]
    
    for i, (name, status) in enumerate(status_items):
        with status_cols[i]:
            if status == 'success':
                st.success(f"✅ {name}")
            elif status == 'warning':
                st.warning(f"⚠️ {name}")
            else:
                st.error(f"❌ {name}")
    
    # Rest of the dashboard sections remain the same but with enhanced error handling
    # [Previous sections for Bank Balances, Supplier Payments, LC Settlements, etc.]
    # [I'll include the key sections here but abbreviated for space]
    
    st.markdown("---")
    
    # Bank Balances Section
    st.header("🏦 Bank Balances")
    if df_by_bank.empty:
        st.info("No bank balance data available.")
    else:
        view = st.radio("View Mode", options=["Cards", "Table"], index=0, horizontal=True)
        df_bal_view = df_by_bank.copy().sort_values("balance", ascending=False)
        
        if view == "Cards":
            cols = st.columns(4)
            for i, row in df_bal_view.iterrows():
                with cols[int(i) % 4]:
                    st.markdown(
                        f"""
                        <div style="
                            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
                            padding: 20px;
                            border-radius: 12px;
                            box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
                            margin-bottom: 16px;
                            border-left: 4px solid #3b82f6;
                            transition: transform 0.2s ease;
                        " onmouseover="this.style.transform='translateY(-2px)'" 
                           onmouseout="this.style.transform='translateY(0)'">
                            <div style="font-size:13px;color:#6b7280;margin-bottom:8px;font-weight:500;">{row['bank']}</div>
                            <div style="font-size:28px;font-weight:800;color:#111827;text-align:right;direction:ltr;">
                                {fmt_currency(row['balance'])}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            df_bal_table = df_bal_view[["bank", "balance"]].rename(columns={"bank": "Bank", "balance": "Balance"})
            df_bal_table["Balance"] = df_bal_table["Balance"].map(lambda x: fmt_currency(x))
            
            # Apply custom styling for right alignment of numbers
            def style_table(df):
                return df.style.set_properties(**{
                    'text-align': 'right'
                }, subset=['Balance']).set_properties(**{
                    'text-align': 'left'
                }, subset=['Bank'])
            
            st.dataframe(style_table(df_bal_table), use_container_width=True, height=360)

    st.markdown("---")

    # Supplier Payments Section
    st.header("💰 Approved Payments")
    if df_pay.empty:
        st.info("No approved payments found.")
    else:
        # Enhanced filtering
        col1, col2 = st.columns([2, 1])
        with col1:
            banks = sorted(df_pay["bank"].dropna().unique())
            pick_banks = st.multiselect("Filter by Bank", banks, default=banks)
        with col2:
            min_amount = st.number_input("Minimum Amount", min_value=0, value=0)

        # Apply filters
        view = df_pay[
            (df_pay["bank"].isin(pick_banks)) & 
            (df_pay["amount"] >= min_amount)
        ].copy()

        if not view.empty:
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Amount", fmt_currency(view["amount"].sum()))
            with col2:
                st.metric("Number of Payments", len(view))
            with col3:
                st.metric("Average Payment", fmt_currency(view["amount"].mean()))

            # Bank-wise breakdown
            grp = view.groupby("bank", as_index=False)["amount"].agg(['sum', 'count']).round(0)
            grp.columns = ["Bank", "Total Amount", "Count"]
            grp["Total Amount"] = grp["Total Amount"].map(fmt_currency)
            
            st.markdown("**📊 Summary by Bank**")
            st.dataframe(grp, use_container_width=True, height=220)

            # Detailed list
            st.markdown("**📋 Detailed Payment List**")
            show_cols = [c for c in ["bank", "supplier", "currency", "amount", "status"] if c in view.columns]
            v = view[show_cols].rename(columns={
                "bank": "Bank", "supplier": "Supplier", "currency": "Currency", 
                "amount": "Amount", "status": "Status"
            }).copy()
            v["Amount"] = v["Amount"].map(fmt_currency)
            
            # Style the detailed table
            def style_detailed_table(df):
                return df.style.set_properties(**{
                    'text-align': 'right'
                }, subset=['Amount']).set_properties(**{
                    'text-align': 'left'
                }, subset=[col for col in df.columns if col != 'Amount'])
            
            st.dataframe(style_detailed_table(v), use_container_width=True, height=360)
        else:
            st.info("No payments match the selected criteria.")

    st.markdown("---")

    # LC Settlements Section
    st.header("📅 LC Settlements — Pending")
    if df_lc.empty:
        st.info("No pending LC settlements found.")
    else:
        # Date range filter
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("From Date", value=df_lc["settlement_date"].min().date())
        with col2:
            end_date = st.date_input("To Date", value=df_lc["settlement_date"].max().date())

        # Filter data
        lc_view = df_lc[
            (df_lc["settlement_date"].dt.date >= start_date) & 
            (df_lc["settlement_date"].dt.date <= end_date)
        ].copy()

        if not lc_view.empty:
            # Summary metrics
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                enhanced_kpi_card("Total LC Amount", lc_view["amount"].sum(), 
                                bg="#FFF7E6", border="#FDE9C8", text="#92400E")
            with cc2:
                enhanced_kpi_card("Number of LCs", len(lc_view), 
                                bg="#E6F0FF", border="#C7D8FE", text="#1E3A8A")
            with cc3:
                urgent_count = len(lc_view[lc_view["settlement_date"] <= today0 + pd.Timedelta(days=2)])
                enhanced_kpi_card("Urgent (2 days)", urgent_count, 
                                bg="#FEE2E2", border="#FECACA", text="#991B1B")

            # LC Table with enhanced formatting
            viz = lc_view.copy()
            viz["Settlement Date"] = viz["settlement_date"].dt.strftime(config.DATE_FMT)
            viz["Amount"] = viz["amount"].map(fmt_currency)
            viz["Days Until Due"] = (viz["settlement_date"] - today0).dt.days

            # Rename columns for display, but only if they exist
            rename_dict = {
                "reference": "Reference",
                "bank": "Bank", 
                "type": "Type",
                "status": "Status",
                "remark": "Remark",
                "description": "Description"
            }
            
            # Only rename columns that actually exist
            existing_renames = {k: v for k, v in rename_dict.items() if k in viz.columns}
            viz = viz.rename(columns=existing_renames)

            # Color coding for urgency
            def highlight_urgent(row):
                if "Days Until Due" in row and row["Days Until Due"] <= 2:
                    return ['background-color: #fee2e2'] * len(row)
                elif "Days Until Due" in row and row["Days Until Due"] <= 7:
                    return ['background-color: #fef3c7'] * len(row)
                else:
                    return [''] * len(row)

            # Build display columns list more carefully
            potential_cols = ["Reference", "Bank", "Type", "Status", "Settlement Date", 
                            "Amount", "Days Until Due", "Remark", "Description"]
            display_cols = [c for c in potential_cols if c in viz.columns]
            
            df_show = viz[display_cols]
            
            # Safe sorting - only sort if the column exists
            if "Settlement Date" in df_show.columns:
                df_show = df_show.sort_values("Settlement Date")
            elif "settlement_date" in viz.columns:
                # If we still have the original column, sort by that and then select display columns
                viz_sorted = viz.sort_values("settlement_date")
                df_show = viz_sorted[display_cols]
            
            st.dataframe(
                df_show.style.apply(highlight_urgent, axis=1),
                use_container_width=True, 
                height=400
            )

            # Upcoming deadlines alert
            urgent_lcs = lc_view[lc_view["settlement_date"] <= today0 + pd.Timedelta(days=3)]
            if not urgent_lcs.empty:
                st.warning(f"⚠️ {len(urgent_lcs)} LC(s) due within 3 days!")
                for _, lc in urgent_lcs.iterrows():
                    days_left = (lc["settlement_date"] - today0).days
                    st.write(f"• {lc['bank']} - {fmt_currency(lc['amount'])} - {days_left} day(s) left")

    st.markdown("---")

    # Enhanced Liquidity Trend
    st.header("📈 Liquidity Trend Analysis")
    if df_fm.empty:
        st.info("No liquidity data available.")
    else:
        try:
            # Calculate trend metrics
            latest_liquidity = df_fm.iloc[-1]["total_liquidity"]
            if len(df_fm) > 1:
                previous_liquidity = df_fm.iloc[-2]["total_liquidity"]
                trend_change = latest_liquidity - previous_liquidity
                trend_pct = (trend_change / previous_liquidity) * 100 if previous_liquidity != 0 else 0
                trend_text = f"{'📈' if trend_change > 0 else '📉'} {trend_pct:+.1f}%"
            else:
                trend_text = "No trend data"

            # Display trend chart
            col1, col2 = st.columns([3, 1])
            
            with col1:
                try:
                    import plotly.express as px
                    import plotly.graph_objects as go
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_fm["date"],
                        y=df_fm["total_liquidity"],
                        mode='lines+markers',
                        name='Total Liquidity',
                        line=dict(color='#3b82f6', width=3),
                        marker=dict(size=6, color='#1e40af')
                    ))
                    
                    fig.update_layout(
                        title="Total Liquidity Trend",
                        xaxis_title="Date",
                        yaxis_title="Liquidity (SAR)",
                        margin=dict(l=20, r=20, t=50, b=20),
                        height=400,
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
                    
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.line_chart(df_fm.set_index("date")["total_liquidity"])
            
            with col2:
                st.markdown("### 📊 Liquidity Metrics")
                enhanced_kpi_card("Current", latest_liquidity, bg="#E6F0FF", border="#C7D8FE", text="#1E3A8A")
                if len(df_fm) > 1:
                    enhanced_kpi_card("Trend", trend_text, bg="#F0FDF4", border="#BBF7D0", text="#065F46")
                
                # Liquidity statistics
                st.markdown("**Statistics (30d)**")
                recent_data = df_fm.tail(30)
                st.write(f"**Max:** {fmt_currency(recent_data['total_liquidity'].max())}")
                st.write(f"**Min:** {fmt_currency(recent_data['total_liquidity'].min())}")
                st.write(f"**Avg:** {fmt_currency(recent_data['total_liquidity'].mean())}")
        
        except Exception as e:
            logger.error(f"Liquidity trend analysis error: {e}")
            st.error("❌ Unable to display liquidity trend analysis")
            # Fallback to simple chart
            st.line_chart(df_fm.set_index("date")["total_liquidity"])

    st.markdown("---")

    # Enhanced Quick Insights
    st.header("💡 Quick Insights & Recommendations")
    insights = []
    
    if not df_by_bank.empty:
        top_bank = df_by_bank.sort_values("balance", ascending=False).iloc[0]
        insights.append({
            "type": "info",
            "title": "Top Bank Balance",
            "content": f"**{top_bank['bank']}** holds the highest balance: {fmt_currency(top_bank['balance'])}"
        })
        
        # Concentration risk
        total_balance = df_by_bank["balance"].sum()
        top_3_pct = df_by_bank.nlargest(3, "balance")["balance"].sum() / total_balance * 100
        if top_3_pct > 80:
            insights.append({
                "type": "warning",
                "title": "Concentration Risk",
                "content": f"Top 3 banks hold {top_3_pct:.1f}% of total balance. Consider diversification."
            })

    if not df_pay.empty:
        total_approved = df_pay["amount"].sum()
        if total_approved > total_balance * 0.8:
            insights.append({
                "type": "warning",
                "title": "Cash Flow Alert",
                "content": f"Approved payments ({fmt_currency(total_approved)}) represent {(total_approved/total_balance)*100:.1f}% of available balance."
            })

    if not df_lc.empty:
        urgent_lcs = df_lc[df_lc["settlement_date"] <= today0 + pd.Timedelta(days=7)]
        if not urgent_lcs.empty:
            insights.append({
                "type": "error",
                "title": "Urgent LC Settlements",
                "content": f"{len(urgent_lcs)} LC settlements due within 7 days totaling {fmt_currency(urgent_lcs['amount'].sum())}"
            })

    if not df_fm.empty and len(df_fm) > 5:
        recent_trend = df_fm.tail(5)["total_liquidity"].pct_change().mean()
        if recent_trend < -0.05:  # 5% decline trend
            insights.append({
                "type": "warning",
                "title": "Declining Liquidity Trend",
                "content": f"Liquidity has been declining by an average of {abs(recent_trend)*100:.1f}% over recent periods."
            })

    # Display insights
    if insights:
        for insight in insights:
            if insight["type"] == "info":
                st.info(f"ℹ️ **{insight['title']}**: {insight['content']}")
            elif insight["type"] == "warning":
                st.warning(f"⚠️ **{insight['title']}**: {insight['content']}")
            elif insight["type"] == "error":
                st.error(f"🚨 **{insight['title']}**: {insight['content']}")
    else:
        st.info("💡 Insights will appear as data becomes available and patterns emerge.")

    # Export functionality
    if st.session_state.get('show_export', False):
        st.markdown("---")
        st.subheader("📁 Export Data")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("📊 Export to Excel", type="primary"):
                excel_data = export_data_to_excel(df_by_bank, df_pay, df_lc, df_fm)
                if excel_data:
                    st.download_button(
                        label="⬇️ Download Excel File",
                        data=excel_data,
                        file_name=f"treasury_dashboard_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        with col2:
            if st.button("❌ Cancel"):
                st.session_state['show_export'] = False
                st.rerun()

    # Footer with enhanced information
    st.markdown("---")
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    
    with footer_col1:
        st.markdown("**📊 Dashboard Info**")
        st.caption(f"Timezone: {config.TZ}")
        st.caption(f"Cache TTL: {config.CACHE_TTL}s")
        st.caption(f"Version: Enhanced v2.0")
    
    with footer_col2:
        st.markdown("**📈 Data Sources**")
        successful_sources = sum(1 for status in data_status.values() if status == 'success')
        st.caption(f"Active Sources: {successful_sources}/4")
        st.caption(f"Last Refresh: {datetime.now().strftime('%H:%M:%S')}")
    
    with footer_col3:
        st.markdown("**🏗️ Created By**")
        st.caption("**Jaseer Pykarathodi**")

if __name__ == "__main__":
    main()
