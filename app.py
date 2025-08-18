# app.py — Enhanced Treasury Dashboard with Advanced Analytics
# - All original features PLUS:
# - Risk Scoring Dashboard
# - Smart Alerts System  
# - Liquidity Forecasting
# - Enhanced KPIs
# - Cash Flow Projections
# - Performance Analytics

import io
import time
import logging
import os
from datetime import datetime, timedelta
from dataclasses import dataclass
from functools import wraps
from typing import Optional, Tuple, Dict, Any, List

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

# ---- Global font (one place to change) ----
APP_FONT = os.getenv("APP_FONT", "Inter")  # e.g., "Inter", "Poppins", "Rubik"

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
      /* Make numbers align nicely */
      .stDataFrame, .stDataFrame * {{ font-variant-numeric: tabular-nums; }}
      /* Hide viewer/help badges */
      [data-testid="stDecoration"], [data-testid="stStatusWidget"], [data-testid="stToolbar"] {{ display: none !important; }}
    </style>
    """.format(font_q=family.replace(" ", "+"), font=family)
    st.markdown(css, unsafe_allow_html=True)

set_app_font()

# ----------------------------
# Theme Palettes (one-line swap)
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

# Default active palette (can be changed from sidebar)
if "palette_name" not in st.session_state:
    st.session_state["palette_name"] = "Indigo"
ACTIVE = PALETTES[st.session_state["palette_name"]]

# Theme tokens used across components
THEME = {
    "accent1": ACTIVE["accent1"],
    "accent2": ACTIVE["accent2"],
    "heading_bg": ACTIVE["heading_bg"],
    "amount_color": {"pos": ACTIVE["pos"], "neg": ACTIVE["neg"]},
    "card_bg": {
        "best": ACTIVE["card_best"], "good": ACTIVE["card_good"],
        "ok": ACTIVE["card_ok"], "low": ACTIVE["card_low"], "neg": ACTIVE["card_neg"],
    },
    "badge": {
        "pos_bg": "rgba(5,150,105,.10)",   # greenish
        "neg_bg": "rgba(185,28,28,.10)",   # reddish
    },
    "icons": {"best": "💎", "good": "🔹", "ok": "💠", "low": "💚", "neg": "⚠️"},
    "thresholds": {"best": 500_000, "good": 100_000, "ok": 50_000},  # else low; negative -> neg
}

# Subtle hover for cards + section chips
st.markdown(f"""
<style>
  .top-gradient {{
    height: 42px;
    background: linear-gradient(90deg, {THEME['accent1']} 0%, {THEME['accent2']} 100%);
    border-radius: 6px;
    box-shadow: 0 6px 18px rgba(0,0,0,.12);
  }}
  .dash-card {{
    transition: transform .15s ease, box-shadow .15s ease;
  }}
  .dash-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 10px 24px rgba(0,0,0,.08);
  }}
  .section-chip {{
    display:inline-block; padding:6px 12px; border-radius:10px;
    background:{THEME['heading_bg']}; color:#0f172a; font-weight:700;
  }}
  .risk-critical {{ background: #fee2e2; border-left: 4px solid #dc2626; }}
  .risk-high {{ background: #fef3c7; border-left: 4px solid #f59e0b; }}
  .risk-medium {{ background: #dbeafe; border-left: 4px solid #3b82f6; }}
  .risk-low {{ background: #dcfce7; border-left: 4px solid #059669; }}
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
# Helpers: parsing + formatting
# ----------------------------
def _to_number(x) -> float:
    """Convert text with commas/percent/() negatives to float; NaN on failure."""
    if pd.isna(x) or x == '':
        return np.nan
    s = str(x).strip().replace(",", "")
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    if s.endswith("%"):
        s = s[:-1]
    try:
        num = float(s)
        if neg:
            num = -num
        if abs(num) > 1e12:
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
    """Right-align numeric columns, keep numbers sortable, format with separators."""
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
# NEW: Advanced Analytics Functions
# ----------------------------

def calculate_risk_score(df_balances: pd.DataFrame, df_payments: pd.DataFrame, df_lc: pd.DataFrame) -> Dict[str, float]:
    """Calculate comprehensive risk score using existing data"""
    if df_balances.empty:
        return {'risk_score': 0, 'liquidity_ratio': 0, 'concentration_risk': 0, 'negative_banks': 0, 'settlement_pressure': 0}
    
    total_balance = df_balances['balance'].sum()
    
    # Get approved payments
    approved_payments = 0
    if not df_payments.empty:
        status_lower = df_payments["status"].astype(str).str.lower()
        approved_mask = status_lower.str.contains("approved", na=False)
        approved_payments = df_payments[approved_mask]['amount'].sum()
    
    # Liquidity ratio
    liquidity_ratio = total_balance / approved_payments if approved_payments > 0 else float('inf')
    
    # Concentration risk (single bank dominance)
    max_bank_balance = df_balances['balance'].max()
    concentration_risk = (max_bank_balance / total_balance) * 100 if total_balance > 0 else 0
    
    # Negative balance count
    negative_banks = len(df_balances[df_balances['balance'] < 0])
    
    # Settlement pressure (next 7 days)
    settlement_pressure = 0
    if not df_lc.empty:
        today = pd.Timestamp.now().floor('D')
        urgent_settlements = df_lc[df_lc['settlement_date'] <= today + pd.Timedelta(days=7)]['amount'].sum()
        settlement_pressure = (urgent_settlements / total_balance) * 100 if total_balance > 0 else 0
    
    # Calculate composite risk score (0-100, higher is worse)
    risk_components = [
        min(30, 30 if liquidity_ratio < 1.2 else max(0, 30 - (liquidity_ratio - 1.2) * 10)),  # Liquidity risk
        min(25, concentration_risk * 0.25),  # Concentration risk  
        min(25, negative_banks * 12.5),  # Negative balance risk
        min(20, settlement_pressure * 0.4)  # Settlement pressure risk
    ]
    
    risk_score = sum(risk_components)
    
    return {
        'risk_score': round(risk_score, 1),
        'liquidity_ratio': round(liquidity_ratio, 2),
        'concentration_risk': round(concentration_risk, 1),
        'negative_banks': negative_banks,
        'settlement_pressure': round(settlement_pressure, 1)
    }

def generate_smart_alerts(df_balances: pd.DataFrame, df_payments: pd.DataFrame, df_lc: pd.DataFrame, df_cvp: pd.DataFrame) -> List[Dict]:
    """Generate intelligent alerts based on current data"""
    alerts = []
    today = pd.Timestamp.now().floor('D')
    
    if df_balances.empty:
        return alerts
    
    # Critical balance alerts
    critical_threshold = 50000
    critical_banks = df_balances[df_balances['balance'] < critical_threshold]
    if not critical_banks.empty:
        alerts.append({
            'level': 'critical',
            'type': 'Low Balance',
            'message': f"{len(critical_banks)} banks below {critical_threshold:,} SAR critical threshold",
            'details': critical_banks[['bank', 'balance']].to_dict('records'),
            'action': 'Immediate fund transfer required'
        })
    
    # Negative balance alerts
    negative_banks = df_balances[df_balances['balance'] < 0]
    if not negative_banks.empty:
        total_negative = negative_banks['balance'].sum()
        alerts.append({
            'level': 'critical',
            'type': 'Negative Balance',
            'message': f"Negative balances totaling {total_negative:,.0f} SAR across {len(negative_banks)} banks",
            'details': negative_banks[['bank', 'balance']].to_dict('records'),
            'action': 'Overdraft attention required immediately'
        })
    
    # After-settlement negative alerts
    if 'after_settlement' in df_balances.columns:
        negative_after = df_balances[df_balances['after_settlement'] < 0]
        if not negative_after.empty:
            total_neg_after = negative_after['after_settlement'].sum()
            alerts.append({
                'level': 'urgent',
                'type': 'Post-Settlement Risk',
                'message': f"{len(negative_after)} banks will be negative after settlements ({total_neg_after:,.0f} SAR)",
                'details': negative_after[['bank', 'after_settlement']].to_dict('records'),
                'action': 'Prepare additional funding before settlements'
            })
    
    # LC settlement alerts
    if not df_lc.empty:
        urgent_lc = df_lc[df_lc['settlement_date'] <= today + pd.Timedelta(days=3)]
        if not urgent_lc.empty:
            urgent_amount = urgent_lc['amount'].sum()
            alerts.append({
                'level': 'urgent',
                'type': 'Urgent Settlements',
                'message': f"{len(urgent_lc)} LC settlements due in next 3 days totaling {urgent_amount:,.0f} SAR",
                'details': urgent_lc[['bank', 'amount', 'settlement_date']].to_dict('records'),
                'action': 'Ensure settlement funds are available'
            })
    
    # Cash flow pressure alert
    if not df_payments.empty:
        status_lower = df_payments["status"].astype(str).str.lower()
        approved_payments = df_payments[status_lower.str.contains("approved", na=False)]
        if not approved_payments.empty:
            total_balance = df_balances['balance'].sum()
            total_approved = approved_payments['amount'].sum()
            
            if total_approved > total_balance * 0.8:
                alerts.append({
                    'level': 'warning',
                    'type': 'Cash Flow Pressure',
                    'message': f"Approved payments ({total_approved:,.0f}) consume {(total_approved/total_balance)*100:.1f}% of available balance",
                    'details': {'approved_amount': total_approved, 'available_balance': total_balance},
                    'action': 'Monitor daily cash position closely'
                })
    
    # Concentration risk alert
    if len(df_balances) > 1:
        total_balance = df_balances['balance'].sum()
        max_bank = df_balances.loc[df_balances['balance'].idxmax()]
        concentration = (max_bank['balance'] / total_balance) * 100
        
        if concentration > 70:
            alerts.append({
                'level': 'warning',
                'type': 'Concentration Risk',
                'message': f"{max_bank['bank']} holds {concentration:.1f}% of total liquidity",
                'details': {'dominant_bank': max_bank['bank'], 'concentration_pct': concentration},
                'action': 'Consider diversifying bank balances'
            })
    
    # Branch performance alerts
    if not df_cvp.empty:
        negative_branches = df_cvp[df_cvp['net'] < -50000]  # Significant deficits only
        if not negative_branches.empty:
            total_deficit = negative_branches['net'].sum()
            alerts.append({
                'level': 'info',
                'type': 'Branch Performance',
                'message': f"{len(negative_branches)} branches showing significant deficits ({total_deficit:,.0f} SAR)",
                'details': negative_branches[['branch', 'net']].to_dict('records'),
                'action': 'Review branch cash management'
            })
    
    return alerts

def simple_liquidity_forecast(df_fm: pd.DataFrame, days_ahead: int = 30) -> pd.DataFrame:
    """Generate simple trend-based liquidity forecast"""
    if df_fm.empty or len(df_fm) < 3:
        return pd.DataFrame()
    
    df_fm_sorted = df_fm.sort_values('date').copy()
    
    # Calculate trend using linear regression on recent data
    recent_data = df_fm_sorted.tail(min(10, len(df_fm_sorted)))  # Use last 10 points or all if less
    x_vals = np.arange(len(recent_data))
    y_vals = recent_data['total_liquidity'].values
    
    # Fit linear trend
    if len(x_vals) >= 2:
        z = np.polyfit(x_vals, y_vals, 1)
        trend_slope = z[0]
        last_value = y_vals[-1]
        
        # Generate forecast dates
        last_date = df_fm_sorted['date'].max()
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=days_ahead,
            freq='D'
        )
        
        # Generate forecast values with some reasonable bounds
        forecast_values = []
        for i in range(1, days_ahead + 1):
            projected = last_value + (trend_slope * i)
            # Add some constraints to prevent unrealistic projections
            projected = max(projected, last_value * 0.5)  # Don't go below 50% of current
            projected = min(projected, last_value * 1.5)  # Don't go above 150% of current
            forecast_values.append(projected)
        
        return pd.DataFrame({
            'date': forecast_dates,
            'forecast_liquidity': forecast_values,
            'trend_direction': 'increasing' if trend_slope > 0 else 'decreasing',
            'daily_change': trend_slope
        })
    
    return pd.DataFrame()

def calculate_treasury_kpis(df_balances: pd.DataFrame, df_payments: pd.DataFrame, df_lc: pd.DataFrame, df_fm: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive treasury KPIs"""
    kpis = {}
    
    if df_balances.empty:
        return kpis
    
    # Basic liquidity metrics
    total_balance = df_balances['balance'].sum()
    kpis['total_liquidity'] = total_balance
    kpis['active_banks'] = len(df_balances[df_balances['balance'] > 0])
    kpis['negative_banks'] = len(df_balances[df_balances['balance'] < 0])
    
    # Concentration metrics
    if total_balance > 0:
        kpis['largest_bank_pct'] = (df_balances['balance'].max() / total_balance) * 100
        # Herfindahl index for concentration
        shares = (df_balances['balance'] / total_balance) ** 2
        kpis['concentration_index'] = shares.sum() * 100
    else:
        kpis['largest_bank_pct'] = 0
        kpis['concentration_index'] = 0
    
    # Payment metrics
    if not df_payments.empty:
        status_lower = df_payments["status"].astype(str).str.lower()
        approved_payments = df_payments[status_lower.str.contains("approved", na=False)]
        released_payments = df_payments[status_lower.str.contains("released", na=False)]
        
        kpis['approved_payments'] = approved_payments['amount'].sum() if not approved_payments.empty else 0
        kpis['released_payments'] = released_payments['amount'].sum() if not released_payments.empty else 0
        
        if kpis['approved_payments'] > 0:
            kpis['payment_coverage_days'] = total_balance / (kpis['approved_payments'] / 30) if kpis['approved_payments'] > 0 else float('inf')
            kpis['payment_coverage_ratio'] = total_balance / kpis['approved_payments']
        else:
            kpis['payment_coverage_days'] = float('inf')
            kpis['payment_coverage_ratio'] = float('inf')
    
    # Settlement metrics
    if not df_lc.empty:
        today = pd.Timestamp.now().floor('D')
        next_7_days = df_lc[df_lc['settlement_date'] <= today + pd.Timedelta(days=7)]
        next_30_days = df_lc[df_lc['settlement_date'] <= today + pd.Timedelta(days=30)]
        
        kpis['settlements_7d'] = next_7_days['amount'].sum()
        kpis['settlements_30d'] = next_30_days['amount'].sum()
        kpis['settlement_coverage_7d'] = total_balance / kpis['settlements_7d'] if kpis['settlements_7d'] > 0 else float('inf')
    
    # Trend metrics
    if not df_fm.empty and len(df_fm) >= 2:
        df_fm_sorted = df_fm.sort_values('date')
        latest = df_fm_sorted.iloc[-1]['total_liquidity']
        
        if len(df_fm_sorted) >= 2:
            previous = df_fm_sorted.iloc[-2]['total_liquidity']
            kpis['liquidity_trend_1d'] = ((latest - previous) / previous) * 100 if previous != 0 else 0
        
        if len(df_fm_sorted) >= 7:
            week_ago = df_fm_sorted.iloc[-7]['total_liquidity']
            kpis['liquidity_trend_7d'] = ((latest - week_ago) / week_ago) * 100 if week_ago != 0 else 0
    
    return kpis

def project_cash_flow(df_balances: pd.DataFrame, df_payments: pd.DataFrame, df_lc: pd.DataFrame, days_ahead: int = 30) -> pd.DataFrame:
    """Project cash flow using known commitments"""
    if df_balances.empty:
        return pd.DataFrame()
    
    total_balance = df_balances['balance'].sum()
    today = pd.Timestamp.now().floor('D')
    projection_dates = pd.date_range(start=today, periods=days_ahead, freq='D')
    
    # Get approved payments (assume spread evenly over next 30 days)
    approved_payments = 0
    if not df_payments.empty:
        status_lower = df_payments["status"].astype(str).str.lower()
        approved_mask = status_lower.str.contains("approved", na=False)
        approved_payments = df_payments[approved_mask]['amount'].sum()
    
    daily_payments = approved_payments / 30 if approved_payments > 0 else 0
    
    # Project day by day
    cash_flow_projection = []
    running_balance = total_balance
    
    for date in projection_dates:
        # Known settlements for this date
        daily_settlements = 0
        if not df_lc.empty:
            daily_settlements = df_lc[df_lc['settlement_date'].dt.date == date.date()]['amount'].sum()
        
        # Total daily outflow
        daily_outflow = daily_settlements + daily_payments
        running_balance -= daily_outflow
        
        # Determine risk level
        if running_balance < 0:
            risk_level = 'Critical'
        elif running_balance < total_balance * 0.1:
            risk_level = 'High'
        elif running_balance < total_balance * 0.3:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        cash_flow_projection.append({
            'date': date,
            'settlements': daily_settlements,
            'payments': daily_payments,
            'total_outflow': daily_outflow,
            'projected_balance': running_balance,
            'risk_level': risk_level
        })
    
    return pd.DataFrame(cash_flow_projection)

# ----------------------------
# Display helpers (keeping original functions)
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
                if amount >= 1_000_000:
                    display_amount = f"{amount/1_000_000:.1f}M"
                elif amount >= 1_000:
                    display_amount = f"{amount/1_000:.0f}K"
                else:
                    display_amount = f"{amount:.0f}"
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
# Parsers (keeping all original functions)
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

        # Fallback: legacy layout with date columns
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
    except Exception as e:
        logger.error(f"Bank balance parsing error: {e}")
        st.error(f"❌ Bank balance parsing failed: {str(e)}")
    return pd.DataFrame(), None

def parse_supplier_payments(df: pd.DataFrame) -> pd.DataFrame:
    """Return ALL supplier payments with normalized columns."""
    try:
        d = cols_lower(df).rename(
            columns={"supplier name": "supplier",
                     "amount(sar)": "amount_sar",
                     "order/sh/branch": "order_branch"}
        )
        if not validate_dataframe(d, ["bank", "status"], "Supplier Payments"):
            return pd.DataFrame()

        amt_col = next((c for c in ["amount_sar", "amount", "amount(sar)"] if c in d.columns), None)
        if not amt_col:
            logger.error("No amount column found in supplier payments"); return pd.DataFrame()

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
        d = cols_lower(df).rename(columns={"branch":"branch", "collection":"collection", "payments":"payments"})
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

def parse_exchange_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Supports both layouts: DATE | USD | AED... OR Currency | 18-Aug-2025 | 19-Aug-2025..."""
    try:
        d = df.copy()
        d.columns = [str(c).strip().strip('"').strip("'") for c in d.columns]

        # Detect orientation
        has_date_col = any("date" in c.lower() for c in d.columns)
        if has_date_col:
            # Layout A: DATE column + currency columns
            date_col = next(c for c in d.columns if "date" in c.lower())
            value_cols = [c for c in d.columns if c != date_col]
            long_df = d.melt(id_vars=[date_col], value_vars=value_cols,
                             var_name="currency", value_name="rate")
            long_df = long_df.rename(columns={date_col: "date"})
            long_df["date"] = pd.to_datetime(long_df["date"], errors="coerce", dayfirst=True)
            long_df["currency"] = long_df["currency"].astype(str).str.strip()
            long_df["rate"] = long_df["rate"].map(_to_number)
        else:
            # Layout B: Currency first column + date columns
            cur_col = next((c for c in d.columns if "currency" in c.lower()), d.columns[0])
            candidate_cols = [c for c in d.columns if c != cur_col]
            parsed = pd.to_datetime(pd.Index(candidate_cols), errors="coerce", dayfirst=True)
            date_cols = [col for col, dt_ in zip(candidate_cols, parsed) if pd.notna(dt_)]
            if not date_cols:
                return pd.DataFrame()
            long_df = d.melt(id_vars=[cur_col], value_vars=date_cols,
                             var_name="date", value_name="rate").rename(columns={cur_col: "currency"})
            long_df["date"] = pd.to_datetime(long_df["date"], errors="coerce", dayfirst=True)
            long_df["currency"] = long_df["currency"].astype(str).str.strip()
            long_df["rate"] = long_df["rate"].map(_to_number)

        long_df = long_df.dropna(subset=["currency", "date", "rate"])
        long_df = long_df[long_df["currency"] != ""]
        long_df = long_df.sort_values(["currency", "date"]).reset_index(drop=True)
        return long_df

    except Exception as e:
        logger.error(f"Exchange rate parsing error: {e}")
        st.error(f"❌ Exchange rate parsing failed: {str(e)}")
        return pd.DataFrame()

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
        st.caption(f"Enhanced Treasury Dashboard with Risk Analytics — Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ----------------------------
# Sidebar (Enhanced with Risk Metrics)
# ----------------------------
def render_sidebar(data_status, treasury_kpis):
    with st.sidebar:
        # --- REFRESH ---
        st.markdown("### 🔄 Refresh")
        if st.button("Refresh Now", type="primary", use_container_width=True):
            st.cache_data.clear()
            logger.info("Manual refresh triggered (sidebar)")
            st.rerun()

        # --- FX TOGGLE BUTTON ---
        if "show_fx" not in st.session_state:
            st.session_state["show_fx"] = False
        if st.button("📈 Exchange Rates", use_container_width=True, help="Show/Hide exchange rate chart"):
            st.session_state["show_fx"] = not st.session_state["show_fx"]
            st.rerun()

        # --- ENHANCED KEY METRICS ---
        st.markdown("### 📊 Treasury KPIs")

        def _kpi(title, value, bg, border, color, help_text=""):
            formatted_value = f"{float(value):,.0f}" if value and value != float('inf') else "N/A"
            st.markdown(
                f"""
                <div style="background:{bg};border:1px solid {border};border-radius:12px;padding:16px;margin-bottom:12px;box-shadow:0 1px 6px rgba(0,0,0,.04);" title="{help_text}">
                    <div style="font-size:11px;color:#374151;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px;">{title}</div>
                    <div style="font-size:20px;font-weight:800;color:{color};text-align:right;">{formatted_value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Core metrics
        _kpi("TOTAL LIQUIDITY", treasury_kpis.get('total_liquidity', 0), THEME["heading_bg"], THEME["accent1"], "#1E3A8A", "Total available balance across all banks")
        _kpi("ACTIVE BANKS", treasury_kpis.get('active_banks', 0), THEME["heading_bg"], THEME["accent2"], "#065F46", "Banks with positive balances")
        
        # Risk metrics
        if treasury_kpis.get('payment_coverage_ratio', 0) != float('inf'):
            coverage = treasury_kpis.get('payment_coverage_ratio', 0)
            coverage_color = "#059669" if coverage > 1.5 else "#f59e0b" if coverage > 1.0 else "#dc2626"
            _kpi("PAYMENT COVERAGE", f"{coverage:.1f}x", THEME["heading_bg"], THEME["accent1"], coverage_color, "Available balance / Approved payments")
        
        # Settlement metrics
        if treasury_kpis.get('settlements_7d', 0) > 0:
            _kpi("SETTLEMENTS (7D)", treasury_kpis.get('settlements_7d', 0), THEME["heading_bg"], THEME["accent2"], "#92400E", "LC settlements due in next 7 days")

        st.markdown("---")

        # --- CONTROLS ---
        st.markdown("### ⚙️ Controls")
        do_auto = st.toggle("Auto refresh",
                            value=st.session_state.get("auto_refresh", False),
                            help="Automatically refresh the dashboard.")
        every_sec = st.number_input("Every (seconds)", min_value=15, max_value=900,
                                    value=int(st.session_state.get("auto_interval", 120)), step=15)
        st.session_state["auto_refresh"] = bool(do_auto)
        st.session_state["auto_interval"] = int(every_sec)

        st.markdown("### 🎨 Theme")
        sel = st.selectbox("Palette", list(PALETTES.keys()),
                           index=list(PALETTES.keys()).index(st.session_state["palette_name"]))
        if sel != st.session_state["palette_name"]:
            st.session_state["palette_name"] = sel
            st.rerun()

        density = st.toggle("Compact density", value=st.session_state.get("compact_density", False))
        st.session_state["compact_density"] = density

# ----------------------------
# NEW: Risk Dashboard Section
# ----------------------------
def render_risk_dashboard(df_balances, df_payments, df_lc, df_cvp):
    """Render comprehensive risk dashboard"""
    st.markdown('<span class="section-chip">⚠️ Risk Analytics Dashboard</span>', unsafe_allow_html=True)
    
    # Calculate risk metrics
    risk_metrics = calculate_risk_score(df_balances, df_payments, df_lc)
    
    # Risk score with color coding
    risk_score = risk_metrics['risk_score']
    if risk_score >= 70:
        risk_color = "#dc2626"
        risk_level = "Critical"
        risk_class = "risk-critical"
    elif risk_score >= 50:
        risk_color = "#f59e0b"
        risk_level = "High"
        risk_class = "risk-high"
    elif risk_score >= 30:
        risk_color = "#3b82f6"
        risk_level = "Medium"
        risk_class = "risk-medium"
    else:
        risk_color = "#059669"
        risk_level = "Low"
        risk_class = "risk-low"
    
    # Risk metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"""
            <div class="dash-card {risk_class}" style="padding:20px;border-radius:12px;margin-bottom:16px;text-align:center;">
                <div style="font-size:12px;color:#374151;text-transform:uppercase;margin-bottom:8px;">Overall Risk Score</div>
                <div style="font-size:32px;font-weight:900;color:{risk_color};">{risk_score}/100</div>
                <div style="font-size:14px;font-weight:700;color:{risk_color};">{risk_level} Risk</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        liquidity_ratio = risk_metrics['liquidity_ratio']
        liq_color = "#059669" if liquidity_ratio > 1.5 else "#f59e0b" if liquidity_ratio > 1.0 else "#dc2626"
        st.metric("Liquidity Ratio", f"{liquidity_ratio:.2f}x", 
                 help="Available balance ÷ Approved payments. >1.5 is healthy")
    
    with col3:
        concentration = risk_metrics['concentration_risk']
        conc_color = "#059669" if concentration < 50 else "#f59e0b" if concentration < 70 else "#dc2626"
        st.markdown(
            f"""
            <div style="text-align:center;padding:16px;">
                <div style="font-size:12px;color:#374151;">Concentration Risk</div>
                <div style="font-size:24px;font-weight:800;color:{conc_color};">{concentration:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col4:
        settlement_pressure = risk_metrics['settlement_pressure']
        settle_color = "#059669" if settlement_pressure < 20 else "#f59e0b" if settlement_pressure < 50 else "#dc2626"
        st.markdown(
            f"""
            <div style="text-align:center;padding:16px;">
                <div style="font-size:12px;color:#374151;">Settlement Pressure</div>
                <div style="font-size:24px;font-weight:800;color:{settle_color};">{settlement_pressure:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Generate and display alerts
    st.markdown("### 🚨 Smart Alerts")
    alerts = generate_smart_alerts(df_balances, df_payments, df_lc, df_cvp)
    
    if alerts:
        for alert in alerts:
            if alert['level'] == 'critical':
                st.error(f"🚨 **{alert['type']}**: {alert['message']}")
                with st.expander(f"Details - {alert['type']}", expanded=False):
                    st.write(f"**Action Required**: {alert['action']}")
                    if 'details' in alert:
                        st.json(alert['details'])
            elif alert['level'] == 'urgent':
                st.warning(f"⚠️ **{alert['type']}**: {alert['message']}")
                with st.expander(f"Details - {alert['type']}", expanded=False):
                    st.write(f"**Recommended Action**: {alert['action']}")
                    if 'details' in alert:
                        st.json(alert['details'])
            else:
                st.info(f"ℹ️ **{alert['type']}**: {alert['message']}")
                with st.expander(f"Details - {alert['type']}", expanded=False):
                    st.write(f"**Suggestion**: {alert['action']}")
                    if 'details' in alert:
                        st.json(alert['details'])
    else:
        st.success("✅ No active alerts. All risk metrics are within acceptable ranges.")

# ----------------------------
# NEW: Cash Flow Projection Section
# ----------------------------
def render_cash_flow_projection(df_balances, df_payments, df_lc):
    """Render cash flow projection analysis"""
    st.markdown('<span class="section-chip">📊 Cash Flow Projection (30 Days)</span>', unsafe_allow_html=True)
    
    if df_balances.empty:
        st.info("No balance data available for cash flow projection.")
        return
    
    # Generate projection
    projection = project_cash_flow(df_balances, df_payments, df_lc, days_ahead=30)
    
    if projection.empty:
        st.info("Unable to generate cash flow projection.")
        return
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    current_balance = df_balances['balance'].sum()
    min_projected = projection['projected_balance'].min()
    critical_days = len(projection[projection['risk_level'] == 'Critical'])
    avg_daily_outflow = projection['total_outflow'].mean()
    
    with col1:
        st.metric("Current Balance", fmt_number_only(current_balance))
    with col2:
        color = "#dc2626" if min_projected < 0 else "#f59e0b" if min_projected < current_balance * 0.2 else "#059669"
        st.markdown(
            f"""
            <div style="text-align:center;padding:16px;">
                <div style="font-size:12px;color:#374151;">Minimum Projected</div>
                <div style="font-size:18px;font-weight:800;color:{color};">{fmt_number_only(min_projected)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.metric("Critical Risk Days", critical_days, help="Days with projected negative balance")
    with col4:
        st.metric("Avg Daily Outflow", fmt_number_only(avg_daily_outflow))
    
    # Create projection chart
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        
        # Set up theme
        if "brand" not in pio.templates:
            pio.templates["brand"] = pio.templates["plotly_white"]
            pio.templates["brand"].layout.colorway = [THEME["accent1"], THEME["accent2"], "#64748b", "#94a3b8"]
            pio.templates["brand"].layout.font.family = APP_FONT
        
        fig = go.Figure()
        
        # Add projected balance line
        fig.add_trace(go.Scatter(
            x=projection['date'],
            y=projection['projected_balance'],
            mode='lines+markers',
            name='Projected Balance',
            line=dict(width=3, color=THEME["accent1"]),
            marker=dict(size=6)
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Zero Balance")
        
        # Color code by risk level
        risk_colors = {'Low': '#059669', 'Medium': '#3b82f6', 'High': '#f59e0b', 'Critical': '#dc2626'}
        for risk_level, color in risk_colors.items():
            risk_data = projection[projection['risk_level'] == risk_level]
            if not risk_data.empty:
                fig.add_trace(go.Scatter(
                    x=risk_data['date'],
                    y=risk_data['projected_balance'],
                    mode='markers',
                    name=f'{risk_level} Risk',
                    marker=dict(size=8, color=color),
                    showlegend=True
                ))
        
        fig.update_layout(
            template="brand",
            title="30-Day Cash Flow Projection",
            xaxis_title="Date",
            yaxis_title="Projected Balance (SAR)",
            height=400,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Cash flow chart error: {e}")
        st.line_chart(projection.set_index('date')['projected_balance'])
    
    # Show critical periods
    critical_periods = projection[projection['risk_level'].isin(['Critical', 'High'])]
    if not critical_periods.empty:
        st.warning("⚠️ **High Risk Periods Identified**")
        st.dataframe(
            critical_periods[['date', 'projected_balance', 'total_outflow', 'risk_level']].head(10),
            use_container_width=True
        )

# ----------------------------
# NEW: Liquidity Forecast Section  
# ----------------------------
def render_liquidity_forecast(df_fm):
    """Render liquidity forecasting based on historical trends"""
    st.markdown('<span class="section-chip">📈 Liquidity Trend Forecast</span>', unsafe_allow_html=True)
    
    if df_fm.empty or len(df_fm) < 3:
        st.info("Insufficient historical data for liquidity forecasting. Need at least 3 data points.")
        return
    
    # Generate forecast
    forecast = simple_liquidity_forecast(df_fm, days_ahead=30)
    
    if forecast.empty:
        st.info("Unable to generate liquidity forecast.")
        return
    
    # Display forecast summary
    col1, col2, col3 = st.columns(3)
    
    current_liquidity = df_fm['total_liquidity'].iloc[-1]
    forecast_end = forecast['forecast_liquidity'].iloc[-1]
    trend_direction = forecast['trend_direction'].iloc[0]
    daily_change = forecast['daily_change'].iloc[0]
    
    with col1:
        st.metric("Current Liquidity", fmt_number_only(current_liquidity))
    with col2:
        change_pct = ((forecast_end - current_liquidity) / current_liquidity) * 100
        st.metric("30-Day Forecast", fmt_number_only(forecast_end), 
                 delta=f"{change_pct:+.1f}%")
    with col3:
        trend_icon = "📈" if trend_direction == "increasing" else "📉"
        st.metric("Trend", f"{trend_icon} {trend_direction.title()}", 
                 delta=fmt_number_only(daily_change))
    
    # Create forecast chart
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        
        if "brand" not in pio.templates:
            pio.templates["brand"] = pio.templates["plotly_white"]
            pio.templates["brand"].layout.colorway = [THEME["accent1"], THEME["accent2"], "#64748b", "#94a3b8"]
            pio.templates["brand"].layout.font.family = APP_FONT
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=df_fm['date'],
            y=df_fm['total_liquidity'],
            mode='lines+markers',
            name='Historical',
            line=dict(width=3, color=THEME["accent1"]),
            marker=dict(size=6)
        ))
        
        # Forecast data
        fig.add_trace(go.Scatter(
            x=forecast['date'],
            y=forecast['forecast_liquidity'],
            mode='lines+markers',
            name='Forecast',
            line=dict(width=2, dash='dash', color=THEME["accent2"]),
            marker=dict(size=4)
        ))
        
        fig.update_layout(
            template="brand",
            title="Liquidity Historical Trend & 30-Day Forecast",
            xaxis_title="Date",
            yaxis_title="Total Liquidity (SAR)",
            height=400,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Forecast chart error: {e}")
        # Fallback to simple chart
        combined_data = pd.concat([
            df_fm[['date', 'total_liquidity']].rename(columns={'total_liquidity': 'value'}),
            forecast[['date', 'forecast_liquidity']].rename(columns={'forecast_liquidity': 'value'})
        ])
        st.line_chart(combined_data.set_index('date')['value'])

# ----------------------------
# Main Function (Enhanced)
# ----------------------------
def main():
    render_header()
    st.markdown("")

    data_status = {}

    # Load all data (keeping original logic)
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

    # Build Approved/Released subsets
    if not df_pay.empty:
        status_lower = df_pay["status"].astype(str).str.lower()
        df_pay_approved = df_pay[status_lower.str.contains("approved", na=False)].copy()
        df_pay_released = df_pay[status_lower.str.contains("released", na=False)].copy()
    else:
        df_pay_approved = pd.DataFrame()
        df_pay_released = pd.DataFrame()

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

    try:
        df_fx_raw = read_csv(LINKS["EXCHANGE_RATE"])
        df_fx = parse_exchange_rate(df_fx_raw)
        data_status['exchange_rate'] = 'success' if not df_fx.empty else 'warning'
    except Exception as e:
        logger.error(f"Exchange rate processing failed: {e}")
        df_fx = pd.DataFrame()
        data_status['exchange_rate'] = 'error'

    # Calculate enhanced KPIs
    treasury_kpis = calculate_treasury_kpis(df_by_bank, df_pay_approved, df_lc, df_fm)

    # Sidebar (enhanced with new metrics)
    render_sidebar(data_status, treasury_kpis)

    # Density tokens
    pad = "12px" if st.session_state.get("compact_density", False) else "20px"
    radius = "10px" if st.session_state.get("compact_density", False) else "12px"
    shadow = "0 1px 6px rgba(0,0,0,.06)" if st.session_state.get("compact_density", False) else "0 2px 8px rgba(0,0,0,.10)"

    # NEW: Risk Dashboard (top priority)
    render_risk_dashboard(df_by_bank, df_pay_approved, df_lc, df_cvp)
    st.markdown("---")

    # NEW: Cash Flow Projection
    render_cash_flow_projection(df_by_bank, df_pay_approved, df_lc)
    st.markdown("---")

    # NEW: Liquidity Forecast
    render_liquidity_forecast(df_fm)
    st.markdown("---")

    # Exchange Rate (if toggled)
    if st.session_state.get("show_fx", False):
        st.markdown('<span class="section-chip">💱 Exchange Rate — Variation</span>', unsafe_allow_html=True)
        if df_fx.empty:
            st.info("No exchange rate data.")
        else:
            all_curr = sorted(df_fx["currency"].unique().tolist())
            default_pick = [c for c in ["USD", "AED", "EUR", "QAR"] if c in all_curr] or all_curr[:3]
            col1, col2 = st.columns([2, 1])
            with col1:
                pick_curr = st.multiselect("Currencies", all_curr, default=default_pick, key="fx_curr")
            with col2:
                dmin = df_fx["date"].min().date()
                dmax = df_fx["date"].max().date()
                start_d = st.date_input("From", value=dmin, min_value=dmin, max_value=dmax, key="fx_from")
                end_d = st.date_input("To", value=dmax, min_value=dmin, max_value=dmax, key="fx_to")

            view_fx = df_fx[
                (df_fx["currency"].isin(pick_curr)) &
                (df_fx["date"].dt.date >= start_d) &
                (df_fx["date"].dt.date <= end_d)
            ].copy()

            if view_fx.empty:
                st.info("No data for the selected filters.")
            else:
                try:
                    import plotly.io as pio, plotly.graph_objects as go
                    if "brand" not in pio.templates:
                        pio.templates["brand"] = pio.templates["plotly_white"]
                        pio.templates["brand"].layout.colorway = [THEME["accent1"], THEME["accent2"], "#64748b", "#94a3b8"]
                        pio.templates["brand"].layout.font.family = APP_FONT

                    fig = go.Figure()
                    for cur in pick_curr:
                        sub = view_fx[view_fx["currency"] == cur]
                        if not sub.empty:
                            fig.add_trace(go.Scatter(x=sub["date"], y=sub["rate"], mode="lines+markers", name=cur))
                    fig.update_layout(template="brand", height=420,
                                      margin=dict(l=20, r=20, t=40, b=20),
                                      title="Daily Exchange Rate Variation",
                                      xaxis_title="Date", yaxis_title="Rate (SAR)")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    logger.error(f"FX chart error: {e}")
                    st.line_chart(view_fx.pivot_table(index="date", columns="currency", values="rate"))

        st.markdown("---")

    # Rest of the original dashboard sections...
    # Bank Balance
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

                    # choose bucket
                    if pd.notna(bal) and bal < 0:
                        bucket = "neg"
                    elif bal > THEME["thresholds"]["best"]:
                        bucket = "best"
                    elif bal > THEME["thresholds"]["good"]:
                        bucket = "good"
                    elif bal > THEME["thresholds"]["ok"]:
                        bucket = "ok"
                    else:
                        bucket = "low"

                    bg = THEME["card_bg"][bucket]
                    icon = THEME["icons"][bucket]
                    amt_color = THEME["amount_color"]["neg"] if pd.notna(bal) and bal < 0 else THEME["amount_color"]["pos"]

                    # After Settlement badge
                    after_html = ""
                    if pd.notna(after):
                        as_pos = after >= 0
                        badge_bg = THEME["badge"]["pos_bg"] if as_pos else THEME["badge"]["neg_bg"]
                        badge_color = "#065f46" if as_pos else THEME["amount_color"]["neg"]
                        after_html = (
                            f'<div style="display:inline-block; padding:6px 10px; border-radius:8px; '
                            f'background:{badge_bg}; color:{badge_color}; font-weight:800; '
                            f'margin-top:10px;">After Settlement: {fmt_currency(after)}</div>'
                        )

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
                        """,
                        unsafe_allow_html=True
                    )

        elif view == "List":
            display_as_list(df_bal_view, "bank", "balance", "Bank Balances")
        elif view == "Mini Cards":
            display_as_mini_cards(df_bal_view, "bank", "balance", pad=pad, radius=radius, shadow=shadow)
        elif view == "Progress Bars":
            display_as_progress_bars(df_bal_view, "bank", "balance")
        elif view == "Metrics":
            display_as_metrics(df_bal_view, "bank", "balance")
        else:
            table = df_bal_view.rename(columns={"bank": "Bank", "balance": "Balance"})
            st.dataframe(style_right(table, num_cols=["Balance"]), use_container_width=True, height=360)

    st.markdown("---")

    # Supplier Payments (keeping original implementation)
    st.markdown('<span class="section-chip">💰 Supplier Payments</span>', unsafe_allow_html=True)

    def render_payments_tab(df_src: pd.DataFrame, status_label: str, key_suffix: str):
        if df_src.empty:
            st.info(f"No {status_label.lower()} payments found.")
            return
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
            elif payment_view == "Progress Bars":
                bank_totals = view_data.groupby("bank", as_index=False)["amount"].sum().rename(columns={"amount": "balance"})
                display_as_progress_bars(bank_totals, "bank", "balance")
        else:
            st.info("No payments match the selected criteria.")

    if df_pay.empty and df_pay_approved.empty and df_pay_released.empty:
        st.info("No supplier payments found.")
    else:
        tab_approved, tab_released = st.tabs(["Approved", "Released"])
        with tab_approved:
            render_payments_tab(df_pay_approved, "Approved", "approved")
        with tab_released:
            render_payments_tab(df_pay_released, "Released", "released")

    st.markdown("---")

    # LC Settlements (keeping original)
    st.markdown('<span class="section-chip">📅 LC Settlements — Pending</span>', unsafe_allow_html=True)
    if df_lc.empty:
        st.info("No LC (Pending) data. Ensure the sheet has the required columns.")
    else:
        c1, c2 = st.columns(2)
        today0 = pd.Timestamp.now(tz=config.TZ).floor('D').tz_localize(None) if config.TZ else pd.Timestamp.today().floor('D')
        
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

            elif lc_display == "Mini Cards":
                cards = lc_view.groupby("bank", as_index=False)["amount"].sum().rename(columns={"amount": "balance"})
                display_as_mini_cards(cards, "bank", "balance", pad=pad, radius=radius, shadow=shadow)

            urgent_lcs = lc_view[lc_view["settlement_date"] <= today0 + pd.Timedelta(days=3)]
            if not urgent_lcs.empty:
                st.warning(f"⚠️ {len(urgent_lcs)} LC(s) due within 3 days!")
                for _, lc in urgent_lcs.iterrows():
                    days_left = (lc["settlement_date"] - today0).days
                    st.write(f"• {lc['bank']} - {fmt_number_only(lc['amount'])} - {days_left} day(s) left")

    st.markdown("---")

    # Liquidity Trend (keeping original but enhanced)
    st.markdown('<span class="section-chip">📈 Historical Liquidity Trend</span>', unsafe_allow_html=True)
    if df_fm.empty:
        st.info("No liquidity data available.")
    else:
        try:
            import plotly.io as pio, plotly.graph_objects as go
            if "brand" not in pio.templates:
                pio.templates["brand"] = pio.templates["plotly_white"]
                pio.templates["brand"].layout.colorway = [THEME["accent1"], THEME["accent2"], "#64748b", "#94a3b8"]
                pio.templates["brand"].layout.font.family = APP_FONT

            latest_liquidity = df_fm.iloc[-1]["total_liquidity"]
            trend_text = "No trend data"
            if len(df_fm) > 1:
                prev = df_fm.iloc[-2]["total_liquidity"]
                trend_change = latest_liquidity - prev
                trend_pct = (trend_change / prev) * 100 if prev != 0 else 0
                trend_text = f"{'📈' if trend_change > 0 else '📉'} {trend_pct:+.1f}%"

            c1, c2 = st.columns([3, 1])
            with c1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_fm["date"], y=df_fm["total_liquidity"], mode='lines+markers', line=dict(width=3), marker=dict(size=6)))
                fig.update_layout(template="brand", title="Total Liquidity Trend",
                                  xaxis_title="Date", yaxis_title="Liquidity (SAR)", height=400,
                                  margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.markdown("### 📊 Liquidity Metrics")
                st.metric("Current", fmt_number_only(latest_liquidity))
                if len(df_fm) > 1:
                    st.metric("Trend", trend_text)
                st.markdown("**Statistics (30d)**")
                last30 = df_fm.tail(30)
                st.write(f"**Max:** {fmt_number_only(last30['total_liquidity'].max())}")
                st.write(f"**Min:** {fmt_number_only(last30['total_liquidity'].min())}")
                st.write(f"**Avg:** {fmt_number_only(last30['total_liquidity'].mean())}")
        except Exception as e:
            logger.error(f"Liquidity trend analysis error: {e}")
            st.error("❌ Unable to display liquidity trend analysis")
            st.line_chart(df_fm.set_index("date")["total_liquidity"])

    st.markdown("---")

    # Collection vs Payments by Branch (keeping original)
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
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.bar_chart(cvp_sorted.set_index("branch")[["collection", "payments"]])

        elif cvp_view == "Table":
            tbl = cvp_sorted.rename(columns={"branch": "Branch", "collection": "Collection", "payments": "Payments", "net": "Net"})
            styled = style_right(tbl, num_cols=["Collection", "Payments", "Net"])
            def _net_red(val):
                try:
                    return 'color:#b91c1c;font-weight:700;' if float(val) < 0 else ''
                except Exception:
                    return ''
            styled = styled.applymap(_net_red, subset=["Net"])
            st.dataframe(styled, use_container_width=True, height=420)

        else:  # Cards
            cols = st.columns(3)
            for i, row in cvp_sorted.iterrows():
                with cols[i % 3]:
                    net = row["net"]
                    pos = net >= 0
                    bg = THEME["card_bg"]["low"] if pos else THEME["card_bg"]["neg"]
                    title = "Net Surplus" if pos else "Net Deficit"
                    net_color = "#065f46" if pos else THEME["amount_color"]["neg"]
                    st.markdown(
                        f"""
                        <div class="dash-card" style="background:{bg};padding:{pad};border-radius:{radius};border:1px solid rgba(0,0,0,0.05);margin-bottom:14px;box-shadow:{shadow};">
                            <div style="display:flex;justify-content:space-between;margin-bottom:10px;">
                                <div style="font-weight:800;color:#0f172a;">{row['branch']}</div>
                                <div style="opacity:.7">{title}</div>
                            </div>
                            <div style="display:flex;justify-content:space-between;">
                                <div>
                                    <div style="font-size:12px;opacity:.7">Collection</div>
                                    <div style="font-weight:900">{fmt_currency(row['collection'])}</div>
                                </div>
                                <div>
                                    <div style="font-size:12px;opacity:.7">Payments</div>
                                    <div style="font-weight:900">{fmt_currency(row['payments'])}</div>
                                </div>
                            </div>
                            <div style="text-align:right;margin-top:10px;font-weight:900;color:{net_color};">Net: {fmt_currency(net)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    st.markdown("---")

    # NEW: Executive Summary Section
    st.markdown('<span class="section-chip">📋 Executive Summary</span>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Key Financial Position")
        
        # Current position summary
        total_liquidity = treasury_kpis.get('total_liquidity', 0)
        approved_payments = treasury_kpis.get('approved_payments', 0)
        settlements_7d = treasury_kpis.get('settlements_7d', 0)
        
        # Calculate key ratios
        if approved_payments > 0:
            payment_coverage = total_liquidity / approved_payments
            coverage_status = "Strong" if payment_coverage > 2.0 else "Adequate" if payment_coverage > 1.5 else "Tight" if payment_coverage > 1.0 else "Critical"
            coverage_color = "#059669" if payment_coverage > 1.5 else "#f59e0b" if payment_coverage > 1.0 else "#dc2626"
        else:
            coverage_status = "No Pending Payments"
            coverage_color = "#059669"
            payment_coverage = float('inf')
        
        # Risk assessment
        risk_metrics = calculate_risk_score(df_by_bank, df_pay_approved, df_lc)
        risk_score = risk_metrics['risk_score']
        risk_status = "Low" if risk_score < 30 else "Medium" if risk_score < 50 else "High" if risk_score < 70 else "Critical"
        
        st.markdown(f"""
        **💰 Liquidity Position**: {fmt_currency(total_liquidity)}  
        **📊 Payment Coverage**: {coverage_status} <span style="color:{coverage_color}">({payment_coverage:.1f}x)</span>  
        **⚠️ Overall Risk**: {risk_status} ({risk_score:.0f}/100)  
        **🏦 Active Banks**: {treasury_kpis.get('active_banks', 0)} banks with positive balances  
        **📅 Near-term Settlements**: {fmt_currency(settlements_7d)} due in next 7 days
        """, unsafe_allow_html=True)
        
        # Key recommendations
        recommendations = []
        if risk_score > 50:
            recommendations.append("🚨 **High Risk Detected** - Review liquidity position and reduce concentration")
        if payment_coverage < 1.5 and approved_payments > 0:
            recommendations.append("💰 **Cash Flow Alert** - Consider securing additional funding or delaying non-critical payments")
        if treasury_kpis.get('negative_banks', 0) > 0:
            recommendations.append("🏦 **Negative Balances** - Address overdrafts immediately")
        if treasury_kpis.get('concentration_index', 0) > 70:
            recommendations.append("📊 **Diversify Holdings** - Reduce dependence on single bank relationship")
        
        if recommendations:
            st.markdown("### 🎯 Priority Actions")
            for rec in recommendations:
                st.markdown(f"• {rec}")
        else:
            st.success("✅ **All Key Metrics Healthy** - Treasury position is well-managed")
    
    with col2:
        st.markdown("### 📈 Performance Indicators")
        
        # Quick performance metrics
        if 'liquidity_trend_7d' in treasury_kpis:
            trend_7d = treasury_kpis['liquidity_trend_7d']
            trend_icon = "📈" if trend_7d > 0 else "📉" if trend_7d < -2 else "➡️"
            st.metric("7-Day Liquidity Trend", f"{trend_icon} {trend_7d:+.1f}%")
        
        # Bank utilization
        active_banks = treasury_kpis.get('active_banks', 0)
        total_banks = len(df_by_bank) if not df_by_bank.empty else 0
        if total_banks > 0:
            utilization = (active_banks / total_banks) * 100
            st.metric("Bank Utilization", f"{utilization:.0f}%", help="Percentage of banks with positive balances")
        
        # Settlement efficiency
        if settlements_7d > 0 and total_liquidity > 0:
            settlement_efficiency = (total_liquidity / settlements_7d) * 100
            efficiency_status = "Excellent" if settlement_efficiency > 200 else "Good" if settlement_efficiency > 150 else "Adequate" if settlement_efficiency > 100 else "Poor"
            st.metric("Settlement Efficiency", efficiency_status, help="Liquidity as % of upcoming settlements")

    st.markdown("---")

    # Footer
    st.markdown("<hr style='margin: 8px 0 16px 0;'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center; opacity:0.8; font-size:12px;'>Enhanced Treasury Dashboard with Advanced Analytics — Powered By <strong>Jaseer Pykkarathodi</strong></div>",
        unsafe_allow_html=True
    )

    # Auto-refresh (keeping original logic)
    if st.session_state.get("auto_refresh"):
        interval = int(st.session_state.get("auto_interval", 120))
        with st.status(f"Auto refreshing in {interval}s…", expanded=False):
            time.sleep(interval)
        st.rerun()

if __name__ == "__main__":
    main()
