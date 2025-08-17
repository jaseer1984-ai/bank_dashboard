# app.py — Enhanced Treasury Dashboard with Sidebar KPIs and Health Status
# Major improvements: KPIs and Data Health moved to left sidebar for better space utilization

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
    initial_sidebar_state="expanded",  # Changed to show sidebar by default
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
    
    /* Sticky Header */
    .main-header {
        position: sticky;
        top: 0;
        background: white;
        z-index: 999;
        padding: 15px 0;
        border-bottom: 2px solid #e6eaf0;
        margin-bottom: 20px;
    }
    
    .main-header h1 {
        font-size: 28px !important;
        font-weight: 900 !important;
        color: #1a202c !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.2 !important;
    }
    
    /* Table Alternatives Styling */
    .list-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 16px;
        border-bottom: 1px solid #e2e8f0;
        transition: background 0.2s;
        margin: 0;
    }
    .list-item:hover {
        background-color: #f8fafc;
    }
    .list-item:last-child {
        border-bottom: none;
    }
    .list-bank {
        font-weight: 600;
        color: #1e293b;
    }
    .list-amount {
        font-weight: 700;
        color: #059669;
    }
    
    .progress-item {
        margin-bottom: 16px;
    }
    .progress-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 6px;
        font-size: 14px;
    }
    .progress-bar {
        width: 100%;
        height: 8px;
        background: #e2e8f0;
        border-radius: 4px;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #3b82f6 0%, #06b6d4 100%);
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    
    /* Sidebar customization */
    .sidebar .sidebar-content {
        padding-top: 1rem;
    }
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
# Enhanced Sidebar Function
# ----------------------------
def render_enhanced_sidebar(data_status, total_balance, approved_sum, lc_next4_sum, banks_cnt, bal_date):
    """Render both KPI cards and data health status in the left sidebar"""
    with st.sidebar:
        # Refresh button at the top
        if st.button("🔄 Refresh All Data", type="primary", use_container_width=True):
            st.cache_data.clear()
            logger.info("Manual refresh triggered from sidebar")
            st.rerun()
        
        st.markdown("---")
        
        # Company Logo/Title
        st.markdown("### 💰 Treasury Dashboard")
        st.markdown("---")
        
        # KPI Cards Section
        st.markdown("### 📊 Key Metrics")
        
        # Total Balance KPI
        st.markdown(
            f"""
            <div style="
                background: #EEF2FF;
                border: 1px solid #C7D2FE;
                border-radius: 12px;
                padding: 16px;
                margin-bottom: 12px;
                box-shadow: 0 1px 6px rgba(0,0,0,.04);
            ">
                <div style="
                    font-size: 11px;
                    color: #374151;
                    text-transform: uppercase;
                    letter-spacing: .08em;
                    margin-bottom: 6px;
                ">
                    TOTAL BALANCE
                </div>
                <div style="
                    font-size: 20px;
                    font-weight: 800;
                    color: #1E3A8A;
                    text-align: right;
                ">
                    {f"{float(total_balance):,.0f}" if total_balance else "N/A"}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Approved Payments KPI
        st.markdown(
            f"""
            <div style="
                background: #E9FFF2;
                border: 1px solid #C7F7DD;
                border-radius: 12px;
                padding: 16px;
                margin-bottom: 12px;
                box-shadow: 0 1px 6px rgba(0,0,0,.04);
            ">
                <div style="
                    font-size: 11px;
                    color: #374151;
                    text-transform: uppercase;
                    letter-spacing: .08em;
                    margin-bottom: 6px;
                ">
                    APPROVED PAYMENTS
                </div>
                <div style="
                    font-size: 20px;
                    font-weight: 800;
                    color: #065F46;
                    text-align: right;
                ">
                    {f"{float(approved_sum):,.0f}" if approved_sum else "N/A"}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # LC Due KPI
        st.markdown(
            f"""
            <div style="
                background: #FFF7E6;
                border: 1px solid #FDE9C8;
                border-radius: 12px;
                padding: 16px;
                margin-bottom: 12px;
                box-shadow: 0 1px 6px rgba(0,0,0,.04);
            ">
                <div style="
                    font-size: 11px;
                    color: #374151;
                    text-transform: uppercase;
                    letter-spacing: .08em;
                    margin-bottom: 6px;
                ">
                    LC DUE (NEXT 4 DAYS)
                </div>
                <div style="
                    font-size: 20px;
                    font-weight: 800;
                    color: #92400E;
                    text-align: right;
                ">
                    {f"{float(lc_next4_sum):,.0f}" if lc_next4_sum else "N/A"}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Active Banks KPI
        st.markdown(
            f"""
            <div style="
                background: #FFF1F2;
                border: 1px solid #FBD5D8;
                border-radius: 12px;
                padding: 16px;
                margin-bottom: 16px;
                box-shadow: 0 1px 6px rgba(0,0,0,.04);
            ">
                <div style="
                    font-size: 11px;
                    color: #374151;
                    text-transform: uppercase;
                    letter-spacing: .08em;
                    margin-bottom: 6px;
                ">
                    ACTIVE BANKS
                </div>
                <div style="
                    font-size: 20px;
                    font-weight: 800;
                    color: #9F1239;
                    text-align: right;
                ">
                    {int(banks_cnt) if banks_cnt else "N/A"}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Data freshness info
        if bal_date:
            st.markdown(
                f"""
                <div style="
                    font-size: 10px;
                    color: #6b7280;
                    text-align: center;
                    margin-bottom: 20px;
                    padding: 8px;
                    background: #f9fafb;
                    border-radius: 6px;
                ">
                    💡 Balance updated: {bal_date.strftime('%Y-%m-%d at %H:%M')}
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Data Health Status Section - HIDDEN
        # Uncomment the section below if you want to show data health status
        
        # st.markdown("---")
        # st.markdown("### 🔍 Data Health")
        # 
        # status_items = [
        #     ("Bank Balance", data_status.get('bank_balance', 'error')),
        #     ("Supplier Payments", data_status.get('supplier_payments', 'error')),
        #     ("LC Settlements", data_status.get('settlements', 'error')),
        #     ("Fund Movement", data_status.get('fund_movement', 'error'))
        # ]
        # 
        # for name, status in status_items:
        #     if status == 'success':
        #         st.success(f"✅ {name}")
        #     elif status == 'warning':
        #         st.warning(f"⚠️ {name}")
        #     else:
        #         st.error(f"❌ {name}")
        
        st.markdown("---")
        
        # Footer info (compact)
        st.markdown(
            f"""
            <div style="font-size: 10px; color: #6b7280; line-height: 1.4;">
                <div><strong>📊 Dashboard</strong></div>
                <div>Version: Enhanced v2.0</div>
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
            unsafe_allow_html=True
        )

# ----------------------------
# Enhanced Helper Functions
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

def fmt_currency_aligned(v, currency="SAR") -> str:
    """Format number as currency with right alignment using HTML"""
    try:
        if pd.isna(v):
            return "N/A"
        # Add extra spaces to push the text to the right
        formatted = f"{currency} {float(v):,.0f}"
        return f"{'':>20}{formatted}"  # Right-pad with spaces
    except (ValueError, TypeError):
        return str(v)

# ----------------------------
# Table Alternative Display Functions
# ----------------------------
def display_as_list(df, bank_col="bank", amount_col="balance", title="Bank Balances"):
    """Display data as a clean list"""
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
    """Display data as mini card grid"""
    cols = st.columns(3)
    for i, row in df.iterrows():
        with cols[int(i) % 3]:
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
                    padding: 16px;
                    border-radius: 12px;
                    border-left: 4px solid #0284c7;
                    margin-bottom: 12px;
                ">
                    <div style="font-size: 12px; color: #0f172a; font-weight: 600; margin-bottom: 8px;">
                        {row[bank_col]}
                    </div>
                    <div style="font-size: 18px; font-weight: 800; color: #0f172a; text-align: right;">
                        {fmt_currency(row[amount_col])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

def display_as_progress_bars(df, bank_col="bank", amount_col="balance"):
    """Display data as progress bars showing relative amounts"""
    max_amount = df[amount_col].max()
    
    for _, row in df.iterrows():
        percentage = (row[amount_col] / max_amount) * 100 if max_amount > 0 else 0
        
        st.markdown(
            f"""
            <div class="progress-item">
                <div class="progress-header">
                    <span><strong>{row[bank_col]}</strong></span>
                    <span><strong>{fmt_currency(row[amount_col])}</strong></span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {percentage}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

def display_as_metrics(df, bank_col="bank", amount_col="balance"):
    """Display data as metric boxes"""
    cols = st.columns(min(4, len(df)))
    for i, row in df.iterrows():
        if i < 4:  # Limit to 4 metrics per row
            with cols[i]:
                # Format amount in K/M for metric display
                amount = row[amount_col]
                if amount >= 1000000:
                    display_amount = f"{amount/1000000:.1f}M"
                elif amount >= 1000:
                    display_amount = f"{amount/1000:.0f}K"
                else:
                    display_amount = f"{amount:.0f}"
                
                st.markdown(
                    f"""
                    <div style="
                        text-align: center;
                        padding: 20px;
                        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
                        border-radius: 12px;
                        border: 2px solid #f59e0b;
                        margin-bottom: 12px;
                    ">
                        <div style="font-size: 12px; color: #92400e; font-weight: 600; margin-bottom: 8px;">
                            {row[bank_col]}
                        </div>
                        <div style="font-size: 20px; font-weight: 800; color: #92400e;">
                            {display_amount}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# ----------------------------
# Parser Functions
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
# Header Section
# ----------------------------
def render_header():
    """Render application header with logo and title - sticky/frozen"""
    # Create sticky header container
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    
    c_logo, c_title = st.columns([0.08, 0.92])
    
    with c_logo:
        try:
            st.image(config.LOGO_PATH, width=44)
        except Exception:
            st.markdown("💰", help="Logo not found")
    
    with c_title:
        # Convert company name to uppercase and make it bold with custom styling
        company_name_upper = config.COMPANY_NAME.upper()
        st.markdown(
            f'<h1 class="main-header h1">{company_name_upper}</h1>',
            unsafe_allow_html=True
        )
        current_time = datetime.now().strftime("Last refresh: %Y-%m-%d %H:%M:%S")
        st.caption(current_time)
    
    # Close sticky header container
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Main Application
# ----------------------------
def main():
    """Main application function with enhanced sidebar"""
    render_header()
    st.markdown("")
    
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
    
    # Render enhanced sidebar with KPIs and health status
    render_enhanced_sidebar(data_status, total_balance, approved_sum, lc_next4_sum, banks_cnt, bal_date)
    
    # Bank Balance Section with Multiple View Options
    st.header("🏦 Bank Balance")

    if df_by_bank.empty:
        st.info("No balances found.")
    else:
        view = st.radio("",
                        options=["Cards", "List", "Mini Cards", "Progress Bars", "Metrics", "Table"],
                        index=0,
                        horizontal=True,
                        label_visibility="collapsed")

        df_bal_view = df_by_bank.copy().sort_values("balance", ascending=False)

        if view == "Cards":
            cols = st.columns(4)
            for i, row in df_bal_view.iterrows():
                with cols[int(i) % 4]:
                    # Determine card color based on balance amount - VERY LIGHT COLORS
                    balance = row['balance']
                    if balance > 500000:
                        bg_color = "#e0e7ff"  # Very light lavender
                        icon = "💎"
                    elif balance > 100000:
                        bg_color = "#fce7f3"  # Very light pink
                        icon = "🔹"
                    elif balance > 50000:
                        bg_color = "#e0f2fe"  # Very light sky blue
                        icon = "💠"
                    else:
                        bg_color = "#ecfdf5"  # Very light mint green
                        icon = "💚"
                    
                    # Use Streamlit's built-in container instead of raw HTML
                    with st.container():
                        st.markdown(
                            f"""
                            <div style="
                                background-color: {bg_color};
                                padding: 20px;
                                border-radius: 12px;
                                margin-bottom: 16px;
                                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                            ">
                                <div style="display: flex; align-items: center; margin-bottom: 12px;">
                                    <span style="font-size: 18px; margin-right: 8px;">{icon}</span>
                                    <span style="font-size: 13px; font-weight: 600; color: #1e293b;">{row['bank']}</span>
                                </div>
                                <div style="
                                    font-size: 24px; 
                                    font-weight: 800; 
                                    color: #1e293b; 
                                    text-align: right;
                                ">
                                    {fmt_currency(row['balance'])}
                                </div>
                                <div style="font-size: 9px; color: #1e293b; opacity: 0.7; margin-top: 8px;">
                                    Available Balance
                                </div>
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
        
        else:  # Table view
            df_bal_table = df_bal_view[["bank", "balance"]].rename(columns={"bank": "Bank", "balance": "Balance"})
            # Use right-aligned currency formatting
            df_bal_table["Balance"] = df_bal_table["Balance"].map(lambda x: f"{fmt_currency(x):>20}")
            
            st.dataframe(
                df_bal_table, 
                use_container_width=True, 
                height=360,
                hide_index=True
            )

    st.markdown("---")

    # Supplier Payments Section with Multiple View Options
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
        view_data = df_pay[
            (df_pay["bank"].isin(pick_banks)) & 
            (df_pay["amount"] >= min_amount)
        ].copy()

        if not view_data.empty:
            # View selector for supplier payments
            payment_view = st.radio("Display as:",
                                   options=["Summary + Table", "Mini Cards", "List", "Progress Bars"],
                                   index=0,
                                   horizontal=True,
                                   key="payment_view")
            
            if payment_view == "Summary + Table":
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Amount", fmt_currency(view_data["amount"].sum()))
                with col2:
                    st.metric("Number of Payments", len(view_data))
                with col3:
                    st.metric("Average Payment", fmt_currency(view_data["amount"].mean()))

                # Bank-wise breakdown
                grp = view_data.groupby("bank", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
                st.markdown("**📊 Summary by Bank**")
                grp2 = grp.rename(columns={"bank": "Bank", "amount": "Amount"}).copy()
                grp2["Amount"] = grp2["Amount"].map(lambda x: f"{fmt_currency(x):>20}")
                
                st.dataframe(grp2, use_container_width=True, height=220, hide_index=True)

                st.markdown("**📋 Detailed Payment List**")
                show_cols = [c for c in ["bank", "supplier", "currency", "amount", "status"] if c in view_data.columns]
                v = view_data[show_cols].rename(columns={
                    "bank": "Bank", "supplier": "Supplier", "currency": "Currency", 
                    "amount": "Amount", "status": "Status"
                }).copy()
                v["Amount"] = v["Amount"].map(lambda x: f"{fmt_currency(x):>20}")
                
                st.dataframe(v, use_container_width=True, height=360, hide_index=True)
            
            elif payment_view == "Mini Cards":
                # Group by bank for mini cards
                bank_totals = view_data.groupby("bank")["amount"].sum().reset_index()
                bank_totals.columns = ["bank", "balance"]  # Rename for consistency with display function
                display_as_mini_cards(bank_totals, "bank", "balance")
            
            elif payment_view == "List":
                bank_totals = view_data.groupby("bank")["amount"].sum().reset_index()
                bank_totals.columns = ["bank", "balance"]
                display_as_list(bank_totals, "bank", "balance", "Approved Payments by Bank")
            
            elif payment_view == "Progress Bars":
                bank_totals = view_data.groupby("bank")["amount"].sum().reset_index()
                bank_totals.columns = ["bank", "balance"]
                display_as_progress_bars(bank_totals, "bank", "balance")
                
        else:
            st.info("No payments match the selected criteria.")

    st.markdown("---")

    # LC Settlements Section
    st.header("📅 LC Settlements — Pending")
    if df_lc.empty:
        st.info(
            "No LC (Pending) data. Ensure sheet has Bank, Maturity Date/New Maturity Date, "
            "and any of: Balance for Settlement / Currently Due / Amount(SAR)."
        )
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
            # LC View selector
            lc_display = st.radio("Display as:",
                                options=["Summary + Table", "Progress by Urgency", "Mini Cards"],
                                index=0,
                                horizontal=True,
                                key="lc_view")
            
            if lc_display == "Summary + Table":
                # Summary metrics
                cc1, cc2, cc3 = st.columns(3)
                with cc1:
                    st.metric("Total LC Amount", fmt_currency(lc_view["amount"].sum()))
                with cc2:
                    st.metric("Number of LCs", len(lc_view))
                with cc3:
                    urgent_count = len(lc_view[lc_view["settlement_date"] <= today0 + pd.Timedelta(days=2)])
                    st.metric("Urgent (2 days)", urgent_count)

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
            
            elif lc_display == "Progress by Urgency":
                # Group by urgency levels
                lc_temp = lc_view.copy()
                lc_temp["days_until_due"] = (lc_temp["settlement_date"] - today0).dt.days
                
                # Create urgency categories
                urgent = lc_temp[lc_temp["days_until_due"] <= 2]
                warning = lc_temp[(lc_temp["days_until_due"] > 2) & (lc_temp["days_until_due"] <= 7)]
                normal = lc_temp[lc_temp["days_until_due"] > 7]
                
                st.markdown("**📊 LC Settlements by Urgency**")
                
                # Display as progress bars grouped by urgency
                if not urgent.empty:
                    st.markdown("**🚨 Urgent (≤2 days)**")
                    urgent_grouped = urgent.groupby("bank")["amount"].sum().reset_index()
                    urgent_grouped.columns = ["bank", "balance"]
                    display_as_progress_bars(urgent_grouped, "bank", "balance")
                
                if not warning.empty:
                    st.markdown("**⚠️ Warning (3-7 days)**")
                    warning_grouped = warning.groupby("bank")["amount"].sum().reset_index()
                    warning_grouped.columns = ["bank", "balance"]
                    display_as_progress_bars(warning_grouped, "bank", "balance")
                
                if not normal.empty:
                    st.markdown("**✅ Normal (>7 days)**")
                    normal_grouped = normal.groupby("bank")["amount"].sum().reset_index()
                    normal_grouped.columns = ["bank", "balance"]
                    display_as_progress_bars(normal_grouped, "bank", "balance")
            
            elif lc_display == "Mini Cards":
                # Group by bank for mini cards
                lc_grouped = lc_view.groupby("bank")["amount"].sum().reset_index()
                lc_grouped.columns = ["bank", "balance"]
                display_as_mini_cards(lc_grouped, "bank", "balance")

            # Upcoming deadlines alert (always show)
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
                st.metric("Current", fmt_currency(latest_liquidity))
                if len(df_fm) > 1:
                    st.metric("Trend", trend_text)
                
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

if __name__ == "__main__":
    main()
