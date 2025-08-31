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

# --- Styler type-hint compatibility
try:
    from pandas.io.formats.style import Styler
except Exception:
    Styler = Any

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

# ----------------------------
# Theme Palettes
# ----------------------------
PALETTES = {
    "Indigo": {
        "accent1": "#3b5bfd",
        "accent2": "#2f2fb5",
        "pos": "#0f172a",
        "neg": "#b91c1c",
        "card_best": "#e0e7ff",
        "card_good": "#fce7f3",
        "card_ok": "#e0f2fe",
        "card_low": "#ecfdf5",
        "card_neg": "#fee2e2",
        "heading_bg": "#eef4ff"
    },
    "Modern": {
        "accent1": "#3b82f6",  # Blue-500
        "accent2": "#1d4ed8",  # Blue-700
        "pos": "#059669",       # Emerald-600
        "neg": "#dc2626",      # Red-600
        "card_best": "#dbeafe", # Blue-50
        "card_good": "#d1fae5", # Emerald-50
        "card_ok": "#e0f2fe",   # Blue-50
        "card_low": "#f0fdf4",  # Emerald-50
        "card_neg": "#fee2e2",  # Red-50
        "heading_bg": "#f8fafc" # Slate-50
    }
}

if "palette_name" not in st.session_state:
    st.session_state["palette_name"] = "Modern"

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

# ----------------------------
# Global CSS Styling
# ----------------------------
st.markdown("""
<style>
    :root {
        --app-font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        --primary-color: """ + THEME['accent1'] + """;
        --secondary-color: """ + THEME['accent2'] + """;
        --success-color: #10b981;
        --danger-color: #ef4444;
        --warning-color: #f59e0b;
        --info-color: #3b82f6;
        --light-color: #f8fafc;
        --dark-color: #1e293b;
        --border-radius: 12px;
        --box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        --transition: all 0.3s ease;
    }

    html, body, [class*="css"] {
        font-family: var(--app-font) !important;
    }

    .stButton>button {
        border-radius: var(--border-radius);
        border: none;
        box-shadow: var(--box-shadow);
        transition: var(--transition);
        padding: 0.5rem 1rem;
        font-weight: 600;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }

    .stSelectbox, .stMultiSelect, .stDateInput {
        border-radius: var(--border-radius) !important;
        border: 1px solid #e2e8f0 !important;
    }

    .dash-card {
        background: white;
        border-radius: var(--border-radius);
        box-shadow: var(--box-shadow);
        transition: var(--transition);
        border: 1px solid #e2e8f0;
    }

    .dash-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }

    [data-testid="stTabs"] {
        gap: 8px;
        margin-bottom: 1rem;
    }

    [data-testid="stTabs"] button {
        border-radius: 8px !important;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border: none;
        transition: var(--transition);
        background: #f1f5f9;
        color: #475569;
    }

    [data-testid="stTabs"] button[aria-selected="true"] {
        background: white;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
    }

    .stDataFrame {
        border-radius: var(--border-radius);
        overflow: hidden;
        box-shadow: var(--box-shadow);
        border: 1px solid #e2e8f0;
    }

    [data-testid="stMetric"] {
        background: white;
        border-radius: var(--border-radius);
        padding: 1rem;
        box-shadow: var(--box-shadow);
        border: 1px solid #e2e8f0;
    }

    .section-chip {
        display: inline-block;
        padding: 0.375rem 0.75rem;
        border-radius: 9999px;
        background: """ + THEME['heading_bg'] + """;
        color: """ + THEME['pos'] + """;
        font-weight: 600;
        font-size: 0.875rem;
        margin-bottom: 1rem;
    }

    /* Streamlit tabs colorization */
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
    /* Export LC */
    [data-testid="stTabs"] button[role="tab"]:nth-child(5) {{ background:#fef3c7; color:#0f172a; }}
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"]:nth-child(5) {{ background:#fde68a; }}
    /* Exchange Rates */
    [data-testid="stTabs"] button[role="tab"]:nth-child(6) {{ background:#f1f5f9; color:#0f172a; }}
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"]:nth-child(6) {{ background:#e2e8f0; }}
    /* Facility Report */
    [data-testid="stTabs"] button[role="tab"]:nth-child(7) {{ background:#f3e8ff; color:#0f172a; }}
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"]:nth-child(7) {{ background:#e9d5ff; }}
    /* Reports */
    [data-testid="stTabs"] button[role="tab"]:nth-child(8) {{ background:#f8fafc; color:#0f172a; }}
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"]:nth-child(8) {{ background:#f1f5f9; }}
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
    "EXPORT_LC": "https://docs.google.com/spreadsheets/d/e/2PACX-1vRlG-a8RqvHK0_BJJtqRe8W7iv5Ey-dKKsaKWdyyT4OsvZnjPrTeRA0jQVFYQWEAA/export?format=csv&gid=0"
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
        num = float(s); num = -num if neg else num
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
# Styled Dataframe Function
# ----------------------------
def styled_dataframe(df, num_cols=None, decimals=0):
    if num_cols is None:
        num_cols = df.select_dtypes(include="number").columns.tolist()

    # Format numbers with commas
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:,.{decimals}f}" if pd.notna(x) else "N/A")

    # Apply custom styling
    return df.style \
        .set_properties(**{
            "font-family": "var(--app-font)",
            "font-size": "0.9rem"
        }) \
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", THEME["heading_bg"]),
                    ("color", THEME["pos"]),
                    ("font-weight", "600"),
                    ("text-align", "left"),
                    ("padding", "0.75rem"),
                    ("border-bottom", f"1px solid {THEME['accent1']}")
                ]
            },
            {
                "selector": "td",
                "props": [
                    ("padding", "0.5rem 0.75rem"),
                    ("border-bottom", "1px solid #e2e8f0")
                ]
            },
            {
                "selector": ".row_heading, .blank",
                "props": [
                    ("display", "none")
                ]
            }
        ]) \
        .applymap(lambda x: 'color: ' + THEME['amount_color']['neg'] + '; font-weight: 600;' if
                 isinstance(x, str) and x.startswith('-') and any(col in x for col in num_cols) else '') \
        .applymap(lambda x: 'color: ' + THEME['amount_color']['pos'] + '; font-weight: 600;' if
                 isinstance(x, str) and not x.startswith('-') and any(col in x for col in num_cols) else '')

# ----------------------------
# Cached CSV fetch
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
    except Exception as e:
        logger.error(f"Error reading CSV from {url}: {e}")
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
            bal = row.get(amount_col, np.nan)
            color = THEME['amount_color']['neg'] if pd.notna(bal) and bal < 0 else THEME['amount_color']['pos']

            # Modern card design
            st.markdown(
                f"""
                <div class="dash-card" style="
                    background: linear-gradient(135deg, {THEME['heading_bg']} 0%, white 100%);
                    padding: {pad};
                    border-radius: {radius};
                    border-left: 4px solid {THEME['accent1']};
                    margin-bottom: 16px;
                    box-shadow: {shadow};
                    transition: all 0.3s ease;
                    position: relative;
                    overflow: hidden;
                ">
                    <div style="
                        position: absolute;
                        top: 0;
                        right: 0;
                        width: 100%;
                        height: 4px;
                        background: linear-gradient(90deg, {THEME['accent1']}, {THEME['accent2']});
                    "></div>
                    <div style="
                        font-size: 14px;
                        color: #374151;
                        font-weight: 600;
                        margin-bottom: 8px;
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                    ">
                        <span>{row[bank_col]}</span>
                        <span style="
                            font-size: 12px;
                            color: {THEME['accent1']};
                            font-weight: 500;
                        ">Balance</span>
                    </div>
                    <div style="
                        font-size: 20px;
                        font-weight: 700;
                        color: {color};
                        text-align: right;
                    ">{fmt_currency(bal)}</div>
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
# Modern Header Design
# ----------------------------
def render_header():
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {THEME['accent1']} 0%, {THEME['accent2']} 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        color: white;
        position: relative;
        overflow: hidden;
    ">
        <div style="
            display: flex;
            align-items: center;
            justify-content: space-between;
        ">
            <div style="
                display: flex;
                align-items: center;
                gap: 1rem;
            ">
                <div style="
                    background: rgba(255, 255, 255, 0.2);
                    padding: 0.5rem;
                    border-radius: 8px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                ">
                    <img src='https://via.placeholder.com/40' style='width: 2.5rem; height: 2.5rem; border-radius: 4px;'>
                </div>
                <div>
                    <h1 style='margin: 0; font-size: 1.5rem; font-weight: 700;'>{config.COMPANY_NAME.upper()}</h1>
                    <p style='margin: 0; font-size: 0.875rem; opacity: 0.9;'>Treasury Dashboard</p>
                </div>
            </div>
            <div style="
                background: rgba(255, 255, 255, 0.2);
                padding: 0.5rem 1rem;
                border-radius: 8px;
                font-size: 0.875rem;
            ">
                {datetime.now().strftime('%A, %B %d, %Y')}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# Sidebar
# ----------------------------
def render_sidebar(total_balance, approved_sum, lc_next4_sum, banks_cnt, total_accepted_export_lc=0):
    with st.sidebar:
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
        _kpi("ACTIVE BANKS", banks_cnt, THEME["heading_bg"], THEME["accent2"], "#9F1239")
        _kpi("ACCEPTED EXPORT LC", total_accepted_export_lc, THEME["heading_bg"], THEME["accent1"], "#059669")

        st.markdown("---")
        st.markdown("### 🎨 Theme")
        sel = st.selectbox("Palette", list(PALETTES.keys()),
                           index=list(PALETTES.keys()).index(st.session_state["palette_name"]))
        if sel != st.session_state["palette_name"]:
            st.session_state["palette_name"] = sel
            st.rerun()

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
    except Exception as e:
        logger.error(f"Error parsing bank balance: {e}")
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
    except Exception as e:
        logger.error(f"Error parsing supplier payments: {e}")
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

        type_col = next((c for c in d.columns if "type" in c), None)
        remark_col = next((c for c in d.columns if "remark" in c), None)
        ref_col = next((c for c in d.columns if any(t in c for t in ["a/c", "ref", "account", "reference"])), None)

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
    except Exception as e:
        logger.error(f"Error parsing settlements: {e}")
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
    except Exception as e:
        logger.error(f"Error parsing fund movement: {e}")
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
    except Exception as e:
        logger.error(f"Error parsing branch CVP: {e}")
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
    try:
        if df.empty:
            return pd.DataFrame()

        d = cols_lower(df)

        required_cols = {
            "applicant": "applicant",
            "l/c no.": "lc_no",
            "issuing bank": "issuing_bank",
            "advising bank": "advising_bank",
            "reference no.": "reference_no",
            "beneficiary branch": "beneficiary_branch",
            "invoice no.": "invoice_no",
            "submitted date": "submitted_date",
            "value (sar)": "value_sar",
            "payment term (days)": "payment_term_days",
            "maturity date": "maturity_date",
            "status": "status",
            "remarks": "remarks"
        }

        col_mapping = {}
        for req_col, expected_name in required_cols.items():
            for col in d.columns:
                if req_col in str(col).lower():
                    col_mapping[expected_name] = col
                    break

        if len(col_mapping) < len(required_cols):
            missing = [req for req in required_cols.values() if req not in col_mapping]
            logger.warning(f"Missing columns in Export LC data: {missing}")
            return pd.DataFrame()

        out = pd.DataFrame({
            "applicant": d[col_mapping["applicant"]].astype(str).str.strip(),
            "lc_no": d[col_mapping["lc_no"]].astype(str).str.strip(),
            "issuing_bank": d[col_mapping["issuing_bank"]].astype(str).str.strip(),
            "advising_bank": d[col_mapping["advising_bank"]].astype(str).str.strip(),
            "reference_no": d[col_mapping["reference_no"]].astype(str).str.strip(),
            "branch": d[col_mapping["beneficiary_branch"]].astype(str).str.strip(),
            "invoice_no": d[col_mapping["invoice_no"]].astype(str).str.strip(),
            "submitted_date": pd.to_datetime(d[col_mapping["submitted_date"]], errors="coerce"),
            "value_sar": d[col_mapping["value_sar"]].map(_to_number),
            "payment_term_days": d[col_mapping["payment_term_days"]].map(_to_number),
            "maturity_date": pd.to_datetime(d[col_mapping["maturity_date"]], errors="coerce"),
            "status": d[col_mapping["status"]].astype(str).str.strip(),
            "remarks": d[col_mapping["remarks"]].astype(str).str.strip()
        })

        out = out.dropna(subset=["applicant", "value_sar", "maturity_date"])
        out["branch"] = out["branch"].str.extract(r'([A-Z]{3})')[0]

        return out.reset_index(drop=True)
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
# Main
# ----------------------------
def main():
    # Initialize session state for active tab
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0

    render_header()

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

    df_export_lc_raw = read_csv(LINKS["EXPORT_LC"])
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
    total_accepted_export_lc = float(df_export_lc[df_export_lc["status"].str.upper() == "ACCEPTED"]["value_sar"].sum()) if not df_export_lc.empty else 0.0

    # Sidebar
    render_sidebar(total_balance, approved_sum, lc_next4_sum, banks_cnt, total_accepted_export_lc)

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
                cnt2 = len(neg_after); total_neg2 = neg_after["after_settlement"].sum()
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

    # Tab state preservation
    st.markdown(
        """
        <script>
        function preserveTabState() {
            const tabs = document.querySelectorAll('[data-testid="stTabs"] [role="tab"]');
            tabs.forEach((tab, index) => {
                tab.addEventListener('click', () => {
                    window.localStorage.setItem('activeTab', index);
                });
            });

            const activeTab = window.localStorage.getItem('activeTab');
            if (activeTab) {
                const event = new CustomEvent('streamlit:setComponentValue', {
                    detail: { value: parseInt(activeTab) }
                });
                const tabContainer = document.querySelector('[data-testid="stTabs"]');
                if (tabContainer) {
                    tabContainer.dispatchEvent(event);
                }
            }
        }
        setTimeout(preserveTabState, 100);
        </script>
        """,
        unsafe_allow_html=True
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
                    best_row = fm_m.loc[fm_m["delta"].idxmax()] if fm_m["delta"].notna().any() else None
                    worst_row = fm_m.loc[fm_m["delta"].idxmin()] if fm_m["delta"].notna().any() else None
                    total_days_in_month = int((month_end - month_start).days + 1)
                    proj_eom = (opening + avg_daily * total_days_in_month) if pd.notna(avg_daily) else np.nan
                    cummax = fm_m["total_liquidity"].cummax()
                    drawdowns = fm_m["total_liquidity"] - cummax
                    max_dd = drawdowns.min() if not drawdowns.empty else np.nan

                    try:
                        import plotly.io as pio, plotly.graph_objects as go
                        if "brand" not in pio.templates:
                            pio.templates["brand"] = pio.templates["plotly_white"]
                            pio.templates["brand"].layout.colorway = [THEME["accent1"], THEME["accent2"], "#64748b", "#94a3b8"]
                            pio.templates["brand"].layout.font.family = "Inter"
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
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    except Exception:
                        st.line_chart(fm_m.set_index("date")["total_liquidity"])

                    kpi_a, kpi_b, kpi_c, kpi_d = st.columns(4)
                    with kpi_a: st.metric("Opening (MTD)", fmt_number_only(opening))
                    with kpi_b: st.metric("Current", fmt_number_only(latest),
                                        delta=f"{mtd_change:,.0f} ({mtd_change_pct:.1f}%)" if pd.notna(mtd_change_pct) else None)
                    with kpi_c: st.metric("Avg Daily Δ", fmt_number_only(avg_daily))
                    with kpi_d: st.metric("Proj. EOM", fmt_number_only(proj_eom))

                    st.markdown("**Daily Dynamics (MTD)**")
                    d1, d2, d3 = st.columns(3)
                    with d1:
                        st.write(f"**Best Day:** {best_row['date'].strftime('%b %d') if best_row is not None else 'N/A'} — {fmt_number_only(best_row['delta']) if best_row is not None else 'N/A'}")
                    with d2:
                        st.write(f"**Worst Day:** {worst_row['date'].strftime('%b %d') if worst_row is not None else 'N/A'} — {fmt_number_only(worst_row['delta']) if worst_row is not None else 'N/A'}")
                    with d3:
                        vol = fm_m["delta"].std(skipna=True)
                        st.write(f"**Volatility (σ Δ):** {fmt_number_only(vol) if pd.notna(vol) else 'N/A'}")
                        st.write(f"**Max Drawdown:** {fmt_number_only(max_dd) if pd.notna(max_dd) else 'N/A'}")
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
                st.dataframe(styled_dataframe(topn, num_cols=num_cols), use_container_width=True, height=320)
            else:
                st.info("No bank balances available.")

        st.markdown("---")

        # 2) LCR & STL Settlements this month
        st.markdown("---")
        st.markdown('<span class="section-chip">📅 LCR & STL Settlements — Overview</span>', unsafe_allow_html=True)
        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
        st.markdown('<div style="background:#f1f5f9;padding:12px;border-radius:8px;border-left:4px solid #3b82f6;margin-bottom:16px;"><small>📊 <strong>Metrics show ALL settlements</strong> | 📈 <strong>Chart & table show current month only</strong></small></div>', unsafe_allow_html=True)

        if df_lc.empty and df_lc_paid.empty:
            st.info("No LCR & STL data.")
        else:
            lc_m = df_lc[(df_lc["settlement_date"] >= month_start) & (df_lc["settlement_date"] <= month_end)].copy() if not df_lc.empty else pd.DataFrame()
            lc_paid_m = df_lc_paid[(df_lc_paid["settlement_date"] >= month_start) & (df_lc_paid["settlement_date"] <= month_end)].copy() if not df_lc_paid.empty else pd.DataFrame()

            if lc_m.empty and lc_paid_m.empty:
                st.write("No LCR & STL for this month.")
            else:
                all_pending = df_lc.copy() if not df_lc.empty else pd.DataFrame()
                all_paid = df_lc_paid.copy() if not df_lc_paid.empty else pd.DataFrame()

                total_due = (all_pending["amount"].sum() if not all_pending.empty else 0.0) + \
                           (all_paid["amount"].sum() if not all_paid.empty else 0.0)

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
                    count_balance_due = len(all_pending.loc[balance_due_mask])
                else:
                    balance_due = 0.0
                    count_balance_due = 0

                count_pending = len(all_pending) if not all_pending.empty else 0
                count_paid = len(all_paid) if not all_paid.empty else 0
                count_current_due = len(all_pending.loc[current_due_mask]) if not all_pending.empty and 'current_due_mask' in locals() else 0

                completion_rate = (paid_amount / total_due * 100) if total_due > 0 else 0

                lc_m_chart = lc_m.copy() if not lc_m.empty else pd.DataFrame()

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

                st.markdown(
                    f"""
                    <div class="dash-card" style="background:linear-gradient(135deg, {THEME['heading_bg']} 0%, #ffffff 100%);
                         padding:24px;border-radius:16px;border:2px solid {THEME['accent1']};margin-bottom:24px;
                         box-shadow:0 8px 24px rgba(0,0,0,.08);">
                        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;">
                            <div style="font-size:18px;font-weight:800;color:#1f2937;">📈 Settlement Progress</div>
                            <div style="font-size:24px;font-weight:900;color:{THEME['accent1']};">{completion_rate:.1f}%</div>
                        </div>
                        <div style="width:100%;height:12px;background:#e5e7eb;border-radius:6px;overflow:hidden;margin-bottom:16px;">
                            <div style="height:100%;background:linear-gradient(90deg,{THEME['accent1']} 0%,{THEME['accent2']} 100%);
                                 border-radius:6px;width:{completion_rate}%;transition:width 0.3s ease;"></div>
                        </div>
                        <div style="display:flex;justify-content:space-between;font-size:14px;flex-wrap:wrap;gap:8px;">
                            <span style="color:#7c3aed;font-weight:600;">💰 Total: {fmt_number_only(total_due)}</span>
                            <span style="color:#dc2626;font-weight:600;">⚠️ Current: {fmt_number_only(current_due)}</span>
                            <span style="color:#16a34a;font-weight:600;">✅ Paid: {fmt_number_only(paid_amount)}</span>
                            <span style="color:#d97706;font-weight:600;">📋 Balance: {fmt_number_only(balance_due)}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if not lc_m_chart.empty:
                    lc_m_chart["week"] = lc_m_chart["settlement_date"].dt.isocalendar().week.astype(int)
                    weekly = lc_m_chart.groupby("week", as_index=False)["amount"].sum().sort_values("week")

                    chart_col1, chart_col2 = st.columns([2, 1])

                    with chart_col1:
                        st.markdown("### 📊 Weekly Settlement Schedule")
                        try:
                            import plotly.io as pio, plotly.graph_objects as go
                            if "brand" not in pio.templates:
                                pio.templates["brand"] = pio.templates["plotly_white"]
                                pio.templates["brand"].layout.colorway = [THEME["accent1"], THEME["accent2"], "#64748b", "#94a3b8"]
                                pio.templates["brand"].layout.font.family = "Inter"

                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                x=[f"Week {w}" for w in weekly["week"]],
                                y=weekly["amount"],
                                marker=dict(
                                    color=weekly["amount"],
                                    colorscale=[[0, '#fee2e2'], [0.5, '#fecaca'], [1, '#dc2626']],
                                    line=dict(width=2, color='white')
                                ),
                                text=[f"SAR {v:,.0f}" for v in weekly["amount"]],
                                textposition="outside",
                                hovertemplate="<b>%{x}</b><br>Amount: SAR %{y:,.0f}<extra></extra>"
                            ))

                            fig.update_layout(
                                template="brand",
                                height=350,
                                margin=dict(l=20,r=20,t=20,b=40),
                                xaxis_title="",
                                yaxis_title="Amount (SAR)",
                                showlegend=False,
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                xaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
                                yaxis=dict(gridcolor='rgba(0,0,0,0.1)')
                            )
                            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                        except Exception:
                            st.bar_chart(weekly.set_index("week")["amount"])

                    with chart_col2:
                        st.markdown("### 🎯 Key Insights")
                        if not weekly.empty:
                            peak_week = weekly.loc[weekly["amount"].idxmax()]
                            avg_weekly = weekly["amount"].mean()

                            st.markdown(
                                f"""
                                <div style="background:#f8fafc;padding:20px;border-radius:12px;border-left:4px solid {THEME['accent1']};margin-bottom:16px;">
                                    <div style="font-size:12px;color:#64748b;margin-bottom:8px;font-weight:600;">PEAK WEEK</div>
                                    <div style="font-size:16px;font-weight:800;color:#1e293b;">Week {int(peak_week['week'])}</div>
                                    <div style="font-size:14px;color:{THEME['accent1']};font-weight:600;">{fmt_number_only(peak_week['amount'])}</div>
                                </div>

                                <div style="background:#f8fafc;padding:20px;border-radius:12px;border-left:4px solid {THEME['accent2']};margin-bottom:16px;">
                                    <div style="font-size:12px;color:#64748b;margin-bottom:8px;font-weight:600;">AVG WEEKLY</div>
                                    <div style="font-size:16px;font-weight:800;color:#1e293b;">{fmt_number_only(avg_weekly)}</div>
                                    <div style="font-size:12px;color:#64748b;">per week</div>
                                </div>

                                <div style="background:#f8fafc;padding:20px;border-radius:12px;border-left:4px solid #64748b;">
                                    <div style="font-size:12px;color:#64748b;margin-bottom:8px;font-weight:600;">WEEKS ACTIVE</div>
                                    <div style="font-size:16px;font-weight:800;color:#1e293b;">{len(weekly)}</div>
                                    <div style="font-size:12px;color:#64748b;">this month</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

        st.markdown("---")

        # 3) FX MTD section
        if not df_fx.empty:
            st.subheader("Exchange Rates — Month Overview")
            fx_m = df_fx[(df_fx["date"] >= month_start) & (df_fx["date"] <= month_end)].copy()
            if not fx_m.empty:
                f1, f2 = st.columns(2)
                with f1:
                    latest_fx = fx_m.groupby("currency_pair").last().reset_index()
                    fx_display = latest_fx[["currency_pair", "rate"]].rename(
                        columns={"currency_pair": "Pair", "rate": "Current Rate"})
                    st.dataframe(styled_dataframe(fx_display, num_cols=["Current Rate"], decimals=4),
                               use_container_width=True, height=200)
                with f2:
                    if "change_pct" in fx_m.columns:
                        volatility = fx_m.groupby("currency_pair")["change_pct"].std().reset_index()
                        volatility = volatility.rename(columns={"currency_pair": "Pair", "change_pct": "Volatility %"})
                        st.dataframe(styled_dataframe(volatility, num_cols=["Volatility %"], decimals=2),
                                   use_container_width=True, height=200)
            else:
                st.info("No FX data for current month.")

        st.subheader("Branches — Net Position (Snapshot)")
        if df_cvp.empty:
            st.info("No branch CVP data.")
        else:
            snap = df_cvp.sort_values("net", ascending=False).rename(
                columns={"branch":"Branch","collection":"Collection","payments":"Payments","net":"Net"})
            st.dataframe(styled_dataframe(snap, num_cols=["Collection","Payments","Net"]), use_container_width=True, height=300)

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
                            <div class="dash-card" style="background-color:{bg};padding:20px;border-radius:12px;margin-bottom:16px;box-shadow:0 2px 8px rgba(0,0,0,.1);">
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
                display_as_mini_cards(df_bal_view, "bank", "balance")
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
                st.dataframe(styled_dataframe(table, num_cols=num_cols), use_container_width=True, height=360)

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
                    pio.templates["brand"].layout.font.family = "Inter"
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
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
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
            st.info("No data in \'Collection vs Payments by Branch\'. Make sure the sheet has \'Branch\', \'Collection\', \'Payments\'.")
        else:
            cvp_view = st.radio("", options=["Bars", "Table", "Cards"], index=0, horizontal=True, label_visibility="collapsed")
            cvp_sorted = df_cvp.sort_values("net", ascending=False).reset_index(drop=True)
            if cvp_view == "Bars":
                try:
                    import plotly.io as pio, plotly.graph_objects as go
                    if "brand" not in pio.templates:
                        pio.templates["brand"] = pio.templates["plotly_white"]
                        pio.templates["brand"].layout.colorway = [THEME["accent1"], THEME["accent2"], "#64748b", "#94a3b8"]
                        pio.templates["brand"].layout.font.family = "Inter"
                    fig = go.Figure()
                    fig.add_bar(name="Collection", x=cvp_sorted["branch"], y=cvp_sorted["collection"])
                    fig.add_bar(name="Payments", x=cvp_sorted["branch"], y=cvp_sorted["payments"])
                    fig.update_layout(template="brand", barmode="group",
                                    height=420, margin=dict(l=20, r=20, t=30, b=80),
                                    xaxis_title="Branch", yaxis_title="Amount (SAR)",
                                    legend_title_text="")
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                except Exception:
                    st.bar_chart(cvp_sorted.set_index("branch")[["collection", "payments"]])
            elif cvp_view == "Table":
                tbl = cvp_sorted.rename(columns={"branch": "Branch", "collection": "Collection", "payments": "Payments", "net": "Net"})
                st.dataframe(styled_dataframe(tbl, num_cols=["Collection", "Payments", "Net"]), use_container_width=True, height=420)
            else:
                display_as_mini_cards(cvp_sorted.rename(columns={"net":"balance"}), "branch", "balance")

    # ---- Settlements tab ----
    with tab_settlements:
        st.markdown('<span class="section-chip">📅 LCR & STL Settlements</span>', unsafe_allow_html=True)

        def render_settlements_tab(df_src: pd.DataFrame, status_label: str, key_suffix: str):
            if df_src.empty:
                st.info(f"No {status_label.lower()} settlements found.")
                return

            # Date filters
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

                        # Create a styled dataframe with proper highlighting
                        viz = view_data.copy()
                        viz["Settlement Date"] = viz["settlement_date"].dt.strftime(config.DATE_FMT)
                        viz["Days Until Due"] = (viz["settlement_date"] - today0).dt.days

                        # Define the highlight function
                        def highlight_row(row):
                            if "Days Until Due" in row:
                                if row["Days Until Due"] <= 2:
                                    return ['background-color: #fee2e2'] * len(row)
                                elif row["Days Until Due"] <= 7:
                                    return ['background-color: #fef3c7'] * len(row)
                            return [''] * len(row)

                        rename = {
                            "reference": "Reference",
                            "bank": "Bank",
                            "type": "Type",
                            "status": "Status",
                            "settlement_date": "Settlement Date",
                            "amount": "Amount",
                            "remark": "Remark",
                            "description": "Description"
                        }

                        viz = viz.rename(columns={k: v for k, v in rename.items() if k in viz.columns})
                        cols = ["Reference", "Bank", "Type", "Status", "Settlement Date", "Amount", "Days Until Due", "Remark", "Description"]
                        cols = [c for c in cols if c in viz.columns]
                        show = viz[cols].sort_values("Settlement Date")

                        # Use the styled_dataframe function
                        styled = styled_dataframe(show, num_cols=["Amount"])

                        # Apply the highlight function
                        styled = styled.apply(highlight_row, axis=1)

                        st.dataframe(styled, use_container_width=True, height=400)

                        # Urgency warnings for pending
                        urgent_settlements = view_data[view_data["settlement_date"] <= today0 + pd.Timedelta(days=3)]
                        if not urgent_settlements.empty:
                            st.warning(f"⚠️ {len(urgent_settlements)} settlement(s) due within 3 days!")
                            for _, settlement in urgent_settlements.iterrows():
                                days_left = (settlement["settlement_date"] - today0).days
                                st.write(f"• {settlement['bank']} - {fmt_number_only(settlement['amount'])} - {days_left} day(s) left")

                    else:  # Paid settlements
                        with cc3: st.metric("Avg. Payment Term", f"{view_data['payment_term_days'].mean():.1f} days")

                        viz_paid = view_data.copy()
                        viz_paid["Settlement Date"] = viz_paid["settlement_date"].dt.strftime(config.DATE_FMT)
                        rename = {
                            "reference": "Reference",
                            "bank": "Bank",
                            "type": "Type",
                            "status": "Status",
                            "settlement_date": "Settlement Date",
                            "amount": "Amount",
                            "remark": "Remark",
                            "description": "Description"
                        }
                        viz_paid = viz_paid.rename(columns={k: v for k, v in rename.items() if k in viz_paid.columns})
                        cols_paid = ["Reference", "Bank", "Type", "Status", "Settlement Date", "Amount", "Remark", "Description"]
                        cols_paid = [c for c in cols_paid if c in viz_paid.columns]
                        show_paid = viz_paid[cols_paid].sort_values("Settlement Date", ascending=False)

                        # Use styled_dataframe instead of style_right
                        st.dataframe(styled_dataframe(show_paid, num_cols=["Amount"]), use_container_width=True, height=400)

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
                    display_as_mini_cards(cards, "bank", "balance")
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
                    st.dataframe(styled_dataframe(grp, num_cols=["Amount"]), use_container_width=True, height=220)

                    st.markdown("**📋 Detailed Payment List**")
                    show_cols = [c for c in ["bank", "supplier", "currency", "amount", "status"] if c in view_data.columns]
                    v = view_data[show_cols].rename(columns={"bank": "Bank", "supplier": "Supplier", "currency": "Currency",
                                                             "amount": "Amount", "status": "Status"})
                    st.dataframe(styled_dataframe(v, num_cols=["Amount"]), use_container_width=True, height=360)

                elif payment_view == "Mini Cards":
                    bank_totals = view_data.groupby("bank", as_index=False)["amount"].sum().rename(columns={"amount": "balance"})
                    display_as_mini_cards(bank_totals, "bank", "balance")

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
        st.markdown('<span class="section-chip">📄 Export LC Proceedings</span>', unsafe_allow_html=True)

        if df_export_lc.empty:
            st.info("No Export LC data available.")
        else:
            # Create filter controls with modern styling
            with st.container():
                st.markdown("""
                <div style="
                    background: white;
                    padding: 1.5rem;
                    border-radius: 12px;
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                    margin-bottom: 1.5rem;
                    border: 1px solid #e2e8f0;
                ">
                    <h3 style="margin-top: 0; color: #1e293b; margin-bottom: 1.25rem;">Filters</h3>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;">
                        <div>
                            <label style="display: block; margin-bottom: 0.25rem; font-weight: 600; color: #475569;">Start Date</label>
                            {start_date_input}
                        </div>
                        <div>
                            <label style="display: block; margin-bottom: 0.25rem; font-weight: 600; color: #475569;">End Date</label>
                            {end_date_input}
                        </div>
                        <div>
                            <label style="display: block; margin-bottom: 0.25rem; font-weight: 600; color: #475569;">Status</label>
                            {status_select}
                        </div>
                    </div>
                    <div style="margin-top: 1rem;">
                        <label style="display: block; margin-bottom: 0.25rem; font-weight: 600; color: #475569;">Branch</label>
                        {branch_select}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Create the actual filter inputs
                col1, col2, col3 = st.columns(3)
                with col1:
                    # Get min and max dates, handling NaT values
                    min_date = df_export_lc["submitted_date"].min()
                    max_date = df_export_lc["submitted_date"].max()

                    # If all dates are NaT, use today's date as default
                    if pd.isna(min_date) or pd.isna(max_date):
                        default_start = datetime.today().date() - timedelta(days=30)
                        default_end = datetime.today().date()
                    else:
                        default_start = min_date.date()
                        default_end = max_date.date()

                    start_date = st.date_input("start_date", value=default_start, label_visibility="collapsed")
                with col2:
                    end_date = st.date_input("end_date", value=default_end, label_visibility="collapsed")
                with col3:
                    all_statuses = ["All"] + sorted(df_export_lc["status"].str.upper().dropna().unique().tolist())
                    selected_status = st.selectbox("status_select", options=all_statuses, label_visibility="collapsed")

                branches = ["All"] + sorted(df_export_lc["branch"].dropna().unique().tolist())
                selected_branch = st.selectbox("branch_select", options=branches, label_visibility="collapsed")

            # Filter data - handle missing dates
            filtered_data = df_export_lc.copy()

            # Only apply date filter if dates are not NaT
            if 'submitted_date' in filtered_data.columns:
                date_mask = (
                    filtered_data["submitted_date"].isna() |  # Include records with missing dates
                    ((filtered_data["submitted_date"].dt.date >= start_date) &
                     (filtered_data["submitted_date"].dt.date <= end_date))
                )
                filtered_data = filtered_data[date_mask]

            if selected_status != "All":
                filtered_data = filtered_data[filtered_data["status"].str.upper() == selected_status]

            if selected_branch != "All":
                filtered_data = filtered_data[filtered_data["branch"] == selected_branch]

            # Create summary cards
            st.markdown("""
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1.5rem;">
            """, unsafe_allow_html=True)

            # Summary cards for each status
            for status in ["ACCEPTED", "UNDER PROCESS", "COLLECTED"]:
                status_data = filtered_data[filtered_data["status"].str.upper() == status]
                count = len(status_data)
                total_value = status_data["value_sar"].sum() if not status_data.empty else 0

                color_map = {
                    "ACCEPTED": "#10b981",  # Green
                    "UNDER PROCESS": "#f59e0b",  # Amber
                    "COLLECTED": "#3b82f6"   # Blue
                }

                st.markdown(f"""
                <div style="
                    background: white;
                    border-radius: 12px;
                    padding: 1.25rem;
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                    border-top: 4px solid {color_map[status]};
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                        <span style="font-weight: 600; color: #475569;">{status}</span>
                        <span style="background: {color_map[status]}; color: white; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">
                            {count} records
                        </span>
                    </div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #1e293b; text-align: right;">
                        {fmt_number_only(total_value)}
                    </div>
                    <div style="font-size: 0.875rem; color: #64748b; text-align: right;">
                        Total Value
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # Create sub-tabs for different statuses with modern styling
            export_lc_accepted, export_lc_under_process, export_lc_collected = st.tabs(
                ["📝 ACCEPTED", "⏳ UNDER PROCESS", "✅ COLLECTED"]
            )

            # Function to display data in a status tab
            def display_status_data(data, status_name, status_emoji):
                if data.empty:
                    st.info(f"No {status_name} records found for the selected filters.")
                else:
                    st.markdown(f"""
                    <div style="
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        margin-bottom: 1rem;
                    ">
                        <h3 style="margin: 0; color: #1e293b;">{status_emoji} {status_name} Export LCs</h3>
                        <span style="
                            background: #f1f5f9;
                            padding: 0.25rem 0.75rem;
                            border-radius: 9999px;
                            font-size: 0.875rem;
                            color: #475569;
                            font-weight: 600;
                        ">{len(data)} records</span>
                    </div>
                    """, unsafe_allow_html=True)

                    # Format and display data
                    display_data = data.copy()
                    display_data["submitted_date"] = display_data["submitted_date"].apply(
                        lambda x: x.strftime(config.DATE_FMT) if pd.notna(x) else "N/A"
                    )
                    display_data["maturity_date"] = display_data["maturity_date"].apply(
                        lambda x: x.strftime(config.DATE_FMT) if pd.notna(x) else "N/A"
                    )
                    display_data["value_sar"] = display_data["value_sar"].apply(fmt_number_only)

                    # Select and rename columns for display
                    display_cols = {
                        "applicant": "Applicant",
                        "lc_no": "L/C No.",
                        "issuing_bank": "Issuing Bank",
                        "branch": "Branch",
                        "submitted_date": "Submitted Date",
                        "value_sar": "Value (SAR)",
                        "maturity_date": "Maturity Date",
                        "status": "Status",
                        "remarks": "Remarks"
                    }

                    display_data = display_data.rename(columns=display_cols)
                    display_data = display_data[[col for col in display_cols.values() if col in display_data.columns]]

                    # Display the styled dataframe
                    st.dataframe(
                        styled_dataframe(display_data, num_cols=["Value (SAR)"]),
                        use_container_width=True,
                        height=400
                    )

                    # Summary metrics with modern cards
                    st.markdown("""
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1.5rem;">
                    """, unsafe_allow_html=True)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"""
                        <div style="
                            background: white;
                            border-radius: 12px;
                            padding: 1rem;
                            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                            border: 1px solid #e2e8f0;
                            text-align: center;
                        ">
                            <div style="font-size: 0.875rem; color: #64748b; margin-bottom: 0.25rem;">Total Value</div>
                            <div style="font-size: 1.25rem; font-weight: 700; color: #1e293b;">
                                {fmt_number_only(data["value_sar"].sum())}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"""
                        <div style="
                            background: white;
                            border-radius: 12px;
                            padding: 1rem;
                            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                            border: 1px solid #e2e8f0;
                            text-align: center;
                        ">
                            <div style="font-size: 0.875rem; color: #64748b; margin-bottom: 0.25rem;">Average Value</div>
                            <div style="font-size: 1.25rem; font-weight: 700; color: #1e293b;">
                                {fmt_number_only(data["value_sar"].mean())}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        st.markdown(f"""
                        <div style="
                            background: white;
                            border-radius: 12px;
                            padding: 1rem;
                            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                            border: 1px solid #e2e8f0;
                            text-align: center;
                        ">
                            <div style="font-size: 0.875rem; color: #64748b; margin-bottom: 0.25rem;">Records</div>
                            <div style="font-size: 1.25rem; font-weight: 700; color: #1e293b;">
                                {len(data)}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)

            # Display each status tab
            with export_lc_accepted:
                display_status_data(
                    filtered_data[filtered_data["status"].str.upper() == "ACCEPTED"],
                    "Accepted",
                    "📝"
                )

            with export_lc_under_process:
                display_status_data(
                    filtered_data[filtered_data["status"].str.upper() == "UNDER PROCESS"],
                    "Under Process",
                    "⏳"
                )

            with export_lc_collected:
                display_status_data(
                    filtered_data[filtered_data["status"].str.upper() == "COLLECTED"],
                    "Collected",
                    "✅"
                )

                # Collection date analysis for COLLECTED tab
                st.markdown("""
                <div style="
                    background: white;
                    border-radius: 12px;
                    padding: 1.5rem;
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                    margin-top: 1.5rem;
                    border: 1px solid #e2e8f0;
                ">
                    <h3 style="margin-top: 0; color: #1e293b; margin-bottom: 1rem;">📊 Collection Analysis</h3>
                """, unsafe_allow_html=True)

                # Extract collection dates from remarks if available
                collection_dates = []
                collected_data = filtered_data[filtered_data["status"].str.upper() == "COLLECTED"]
                for remark in collected_data["remarks"]:
                    if pd.notna(remark) and "collected on" in remark.lower():
                        try:
                            date_str = remark.split("collected on")[-1].strip().split()[0]
                            collection_dates.append(pd.to_datetime(date_str, errors="coerce"))
                        except:
                            continue

                if collection_dates:
                    avg_collection_time = (pd.Series(collection_dates) - collected_data["submitted_date"].iloc[0]).mean()
                    st.markdown(f"""
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
                        <div style="
                            background: #f8fafc;
                            border-radius: 8px;
                            padding: 1rem;
                            border: 1px solid #e2e8f0;
                        ">
                            <div style="font-size: 0.875rem; color: #64748b; margin-bottom: 0.25rem;">Avg Collection Time</div>
                            <div style="font-size: 1.25rem; font-weight: 700; color: #1e293b;">
                                {avg_collection_time.days} days
                            </div>
                        </div>
                        <div style="
                            background: #f8fafc;
                            border-radius: 8px;
                            padding: 1rem;
                            border: 1px solid #e2e8f0;
                        ">
                            <div style="font-size: 0.875rem; color: #64748b; margin-bottom: 0.25rem;">Total Collections</div>
                            <div style="font-size: 1.25rem; font-weight: 700; color: #1e293b;">
                                {len(collection_dates)} records
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="
                        background: #fef3c7;
                        border-radius: 8px;
                        padding: 1rem;
                        border-left: 4px solid #f59e0b;
                    ">
                        <p style="margin: 0; color: #92400e; font-weight: 600;">⚠️ No collection date information found</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

    # ---- Exchange Rates tab ----
    with tab_fx:
        st.markdown('<span class="section-chip">💱 Exchange Rates</span>', unsafe_allow_html=True)

        if df_fx.empty:
            st.info("No exchange rate data available. Ensure the Exchange Rate sheet has the required columns (Currency Pair, Rate, Date).")
        else:
            # FX Display Options
            fx_view = st.radio("Display as:", options=["Current Rates", "Rate Trends", "Volatility Analysis", "Table View"],
                              index=0, horizontal=True, key="fx_view")

            if fx_view == "Current Rates":
                st.subheader("💱 Current Exchange Rates")

                # Get latest rates for each currency pair
                latest_fx = df_fx.groupby("currency_pair").last().reset_index()

                # Display as cards
                if not latest_fx.empty:
                    cols = st.columns(min(4, len(latest_fx)))
                    for i, row in latest_fx.iterrows():
                        with cols[int(i) % min(4, len(latest_fx))]:
                            pair = row["currency_pair"]
                            rate = row["rate"]

                            # Calculate change if available
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
                                <div class="dash-card" style="background:{THEME['heading_bg']};padding:20px;border-radius:12px;
                                     border-left:4px solid {THEME['accent1']};margin-bottom:12px;box-shadow:0 2px 8px rgba(0,0,0,.1);">
                                    <div style="font-size:12px;color:#0f172a;font-weight:700;margin-bottom:8px;">{pair}</div>
                                    <div style="font-size:20px;font-weight:800;color:#0f172a;text-align:right;">{fmt_rate(rate)}</div>
                                    <div style="font-size:10px;color:#1e293b;opacity:.7;margin-top:6px;">Exchange Rate</div>
                                    {change_info}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                # Summary metrics
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
                    # Date range selector
                    c1, c2 = st.columns(2)
                    with c1:
                        start_date = st.date_input("From Date",
                                                 value=df_fx["date"].min().date(),
                                                 key="fx_start_date")
                    with c2:
                        end_date = st.date_input("To Date",
                                               value=df_fx["date"].max().date(),
                                               key="fx_end_date")

                    # Filter data
                    fx_filtered = df_fx[
                        (df_fx["date"].dt.date >= start_date) &
                        (df_fx["date"].dt.date <= end_date)
                    ].copy()

                    if not fx_filtered.empty:
                        # Currency pair selector
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
                                    pio.templates["brand"].layout.font.family = "Inter"

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
                                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                            except Exception:
                                # Fallback to streamlit line chart
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
                    # Calculate volatility metrics
                    volatility_stats = df_fx.groupby("currency_pair").agg({
                        "change_pct": ["std", "mean", "min", "max"],
                        "rate": "last"
                    }).round(4)

                    volatility_stats.columns = ["Volatility (%)", "Avg Change (%)", "Min Change (%)", "Max Change (%)", "Current Rate"]
                    volatility_stats = volatility_stats.reset_index()
                    volatility_stats = volatility_stats.rename(columns={"currency_pair": "Currency Pair"})

                    # Sort by volatility
                    volatility_stats = volatility_stats.sort_values("Volatility (%)", ascending=False)

                    # Display table
                    st.dataframe(
                        styled_dataframe(volatility_stats,
                                        num_cols=["Volatility (%)", "Avg Change (%)", "Min Change (%)", "Max Change (%)", "Current Rate"],
                                        decimals=4),
                        use_container_width=True,
                        height=400
                    )

                    # Volatility chart
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
                            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                        except Exception:
                            st.bar_chart(volatility_stats.set_index("Currency Pair")["Volatility (%)"])
                else:
                    st.info("Volatility analysis requires historical rate changes.")

            else:  # Table View
                st.subheader("📋 Exchange Rate Data Table")

                # Filters
                col1, col2 = st.columns(2)
                with col1:
                    if "currency_pair" in df_fx.columns:
                        available_pairs = ["All"] + sorted(df_fx["currency_pair"].unique())
                        selected_pair = st.selectbox("Filter by Currency Pair", available_pairs, key="fx_table_pair")

                with col2:
                    if "date" in df_fx.columns:
                        date_range = st.number_input("Last N days", min_value=1, max_value=365, value=30, key="fx_date_range")

                # Apply filters
                display_data = df_fx.copy()

                if "date" in display_data.columns:
                    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=date_range)
                    display_data = display_data[display_data["date"] >= cutoff_date]

                if selected_pair != "All":
                    display_data = display_data[display_data["currency_pair"] == selected_pair]

                if not display_data.empty:
                    # Prepare table
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

                    # Select columns to display
                    display_cols = ["Currency Pair", "Rate"]
                    if "Date" in table_data.columns:
                        display_cols.append("Date")
                    if "Change" in table_data.columns:
                        display_cols.append("Change")
                    if "Change %" in table_data.columns:
                        display_cols.append("Change %")

                    display_cols = [col for col in display_cols if col in table_data.columns]
                    table_show = table_data[display_cols].sort_values("Date" if "Date" in display_cols else "Currency Pair", ascending=False)

                    # Apply styling
                    num_cols = [col for col in ["Rate", "Change", "Change %"] if col in table_show.columns]
                    styled_table = styled_dataframe(table_show, num_cols=num_cols, decimals=4)

                    # Color code changes
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
        pass

    # ---- Reports tab ----
    with tab_reports:
        st.markdown('<span class="section-chip">📊 Complete Report Export</span>', unsafe_allow_html=True)
        st.info("Download a complete Excel report containing all dashboard data across multiple sheets.")

        # Generate and download button
        excel_data = generate_complete_report(
            df_by_bank, df_pay_approved, df_pay_released, df_lc, df_lc_paid,
            df_fm, df_cvp, df_fx, total_balance, approved_sum, lc_next4_sum, banks_cnt
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

def generate_complete_report(df_by_bank, df_pay_approved, df_pay_released, df_lc, df_lc_paid, df_fm, df_cvp, df_fx, total_balance, approved_sum, lc_next4_sum, banks_cnt):
    """Generate a complete Excel report with multiple sheets."""
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Summary Sheet
        summary_data = pd.DataFrame({
            'Metric': ['Total Balance', 'Approved Payments', 'LCR & STL Due (Next 4 Days)', 'Active Banks'],
            'Value': [total_balance, approved_sum, lc_next4_sum, banks_cnt]
        })
        summary_data.to_excel(writer, sheet_name='Summary KPIs', index=False)

        # Bank Balances
        if not df_by_bank.empty:
            df_by_bank.to_excel(writer, sheet_name='Bank Balances', index=False)

        # Supplier Payments - Approved
        if not df_pay_approved.empty:
            df_pay_approved.to_excel(writer, sheet_name='Supplier Payments Approved', index=False)

        # Supplier Payments - Released
        if not df_pay_released.empty:
            df_pay_released.to_excel(writer, sheet_name='Supplier Payments Released', index=False)

        # Settlements - Pending
        if not df_lc.empty:
            df_lc.to_excel(writer, sheet_name='Settlements Pending', index=False)

        # Settlements - Paid
        if not df_lc_paid.empty:
            df_lc_paid.to_excel(writer, sheet_name='Settlements Paid', index=False)

        # Fund Movement
        if not df_fm.empty:
            df_fm.to_excel(writer, sheet_name='Fund Movement', index=False)

        # Branch CVP
        if not df_cvp.empty:
            df_cvp.to_excel(writer, sheet_name='Branch CVP', index=False)

        # Exchange Rates
        if not df_fx.empty:
            df_fx.to_excel(writer, sheet_name='Exchange Rates', index=False)

    processed_data = output.getvalue()
    return processed_data

if __name__ == "__main__":
    main()
