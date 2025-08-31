Of course! This is a great question. Changing the "total look" of the dashboard involves a combination of color, layout, and component styling. I'll guide you through a complete visual makeover, from simple tweaks to a more dramatic redesign.

We will create a new, professional "Corporate Blue & Gray" theme. This will involve:
1.  **Creating a New Color Palette.**
2.  **Overhauling the CSS** for a cleaner, more modern look.
3.  **Redesigning the Key Components** like the info cards (`.dash-card`) and section headers.
4.  **Changing the Page Background Color.**

Below, I'll provide the code to completely transform your dashboard's appearance.

---

### The "Corporate Blue & Gray" Makeover

Here is what the new design philosophy will be:
*   **Background:** A very light gray to make the white components "pop."
*   **Cards:** Clean, white cards with a subtle top border and a more defined shadow, making them feel like they are floating.
*   **Headers:** Instead of "chips," we'll use full-width section headers with a clean line underneath.
*   **Color Scheme:** A professional palette of blues, grays, and a touch of green for positive indicators.

### Step 1: Add the New Color Palette

First, we need to add our new theme to the `PALETTES` dictionary. Find the `PALETTES` dictionary in your code and add the `"Corporate Blue"` entry.

**Replace your `PALETTES` dictionary with this one:**

```python
# ----------------------------
# Theme Palettes
# ----------------------------
PALETTES = {
    "Corporate Blue": {
        "accent1": "#00529B", "accent2": "#003A70", "pos": "#1E3A8A", "neg": "#D92D20",
        "card_best": "#F0F5FF", "card_good": "#E6F7FF", "card_ok": "#FAFAFA",
        "card_low": "#FFFFFF", "card_neg": "#FFF1F0", "heading_bg": "#F8F9FA"
    },
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
# Set the default theme to our new one
if "palette_name" not in st.session_state:
    st.session_state["palette_name"] = "Corporate Blue"
```

### Step 2: Replace the Entire CSS Block

This is the most important part. We are completely changing the styles for the cards, headers, background, and more.

**Find the entire CSS block that starts with `st.markdown(f"""<style>...""")` and replace it with this new, improved version:**

```python
# --- Overhauled CSS for a new "Corporate Blue & Gray" look ---
st.markdown(f"""
<style>
  /* --- 1. Global & Background --- */
  /* Set a light gray background for the main content area */
  .main > div {{
    background-color: #F8F9FA;
  }}

  /* --- 2. Custom Components --- */
  .top-gradient {{
    height: 6px; /* Make the top bar slimmer */
    background: linear-gradient(90deg, {THEME['accent1']} 0%, {THEME['accent2']} 100%);
    border-radius: 0; /* Full width */
  }}

  /* New section header style: a clean line instead of a chip */
  .section-header {{
    padding: 8px 0;
    margin-bottom: 24px;
    border-bottom: 2px solid {THEME['heading_bg']};
    color: {THEME['accent2']};
    font-size: 1.25rem; /* Larger font */
    font-weight: 700;
  }}

  /* --- 3. Card Redesign --- */
  .dash-card {{
    background-color: #FFFFFF !important; /* Force white background */
    border: 1px solid #EAECEF; /* Thin border */
    border-top: 4px solid {THEME['accent1']}; /* Accent on top */
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05); /* Softer shadow */
    transition: transform .2s ease, box-shadow .2s ease;
    padding: 24px; /* More padding */
  }}
  .dash-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
  }}

  /* --- 4. Streamlit Component Tweaks --- */
  /* Tabs styling for the new theme */
  [data-testid="stTabs"] button[role="tab"] {{
    border-radius: 8px 8px 0 0 !important; /* Top radius only */
    margin-right: 4px !important;
    font-weight: 600 !important;
    background: #EAECEF;
    color: #555;
    border: 1px solid #DEE2E6;
    border-bottom: none;
  }}
  [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
    background: #FFFFFF;
    color: {THEME['accent1']};
    border-color: #DEE2E6;
  }}

  /* Sidebar styling */
  [data-testid="stSidebar"] {{
      background-color: #FFFFFF;
      border-right: 1px solid #EAECEF;
  }}
</style>
""", unsafe_allow_html=True)
```

### Step 3: Update the Section Header Class

Because we changed the CSS class from `.section-chip` to `.section-header`, we need to update the HTML in the main part of the app.

**Do a "Find and Replace"** in your code editor:
*   **Find:** `class='section-chip'`
*   **Replace with:** `class='section-header'`

This will change lines like this:
`st.markdown('<span class="section-chip">💡 Quick Insights & Recommendations</span>', unsafe_allow_html=True)`

Into this:
`st.markdown('<div class="section-header">💡 Quick Insights & Recommendations</div>', unsafe_allow_html=True)`

I have already done this for you in the complete code below.

---

### Complete Code with "Corporate Blue" Makeover

Here is your full `app.py` file with all the changes integrated. Simply copy and paste this entire block to replace your existing code.

```python
# app.py — Enhanced Treasury Dashboard (Corporate Blue Theme)
# - Total visual overhaul with a new "Corporate Blue & Gray" theme.
# - Redesigned cards, headers, and background.
# - Sidebar cleaned up (Controls/Theme hidden) and new "Accepted Export LC" KPI added.
# - "Export LC" tab moved after "Supplier Payments" and a "Status" filter added.
# - Fixed bug where rows with no "SUBMITTED DATE" were excluded.
# - Fixed bug where tab focus jumped on filter change by adding stable keys.

import io
import time
import logging
import os
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
APP_FONT = os.getenv("APP_FONT", "Poppins") # Switched to a more modern font

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

set_app_font()

# ----------------------------
# Theme Palettes
# ----------------------------
PALETTES = {
    "Corporate Blue": {
        "accent1": "#00529B", "accent2": "#003A70", "pos": "#1E3A8A", "neg": "#D92D20",
        "card_best": "#F0F5FF", "card_good": "#E6F7FF", "card_ok": "#FAFAFA",
        "card_low": "#FFFFFF", "card_neg": "#FFF1F0", "heading_bg": "#EAECEF"
    },
    "Indigo":  {"accent1":"#3b5bfd","accent2":"#2f2fb5","pos":"#0f172a","neg":"#b91c1c",
                "card_best":"#e0e7ff","card_good":"#fce7f3","card_ok":"#e0f2fe",
                "card_low":"#ecfdf5","card_neg":"#fee2e2","heading_bg":"#eef4ff"},
}
# Set the default theme to our new one
if "palette_name" not in st.session_state:
    st.session_state["palette_name"] = "Corporate Blue"
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

# --- Overhauled CSS for a new "Corporate Blue & Gray" look ---
st.markdown(f"""
<style>
  /* --- 1. Global & Background --- */
  /* Set a light gray background for the main content area */
  .main > div {{
    background-color: #F8F9FA;
  }}

  /* --- 2. Custom Components --- */
  .top-gradient {{
    height: 6px; /* Make the top bar slimmer */
    background: linear-gradient(90deg, {THEME['accent1']} 0%, {THEME['accent2']} 100%);
    border-radius: 0; /* Full width */
  }}

  /* New section header style: a clean line instead of a chip */
  .section-header {{
    padding: 8px 0;
    margin-bottom: 24px;
    border-bottom: 2px solid {THEME['heading_bg']};
    color: {THEME['accent2']};
    font-size: 1.25rem; /* Larger font */
    font-weight: 700;
  }}

  /* --- 3. Card Redesign --- */
  .dash-card {{
    background-color: #FFFFFF !important; /* Force white background */
    border: 1px solid #EAECEF; /* Thin border */
    border-top: 4px solid {THEME['accent1']}; /* Accent on top */
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05); /* Softer shadow */
    transition: transform .2s ease, box-shadow .2s ease;
    padding: 24px; /* More padding */
  }}
  .dash-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
  }}

  /* --- 4. Streamlit Component Tweaks --- */
  /* Tabs styling for the new theme */
  [data-testid="stTabs"] button[role="tab"] {{
    border-radius: 8px 8px 0 0 !important; /* Top radius only */
    margin-right: 4px !important;
    font-weight: 600 !important;
    background: #EAECEF;
    color: #555;
    border: 1px solid #DEE2E6;
    border-bottom: none;
  }}
  [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
    background: #FFFFFF;
    color: {THEME['accent1']};
    border-color: #DEE2E6;
  }}

  /* Sidebar styling */
  [data-testid="stSidebar"] {{
      background-color: #FFFFFF;
      border-right: 1px solid #EAECEF;
  }}
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
    """Reads all sheets from an Excel file URL and combines them."""
    try:
        response = http_session.get(url, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()
        excel_content = io.BytesIO(response.content)
        
        # Read all sheets from the Excel file into a dictionary of DataFrames
        all_sheets = pd.read_excel(excel_content, sheet_name=None, engine='openpyxl')
        
        # Combine sheets, adding a 'branch' column from the sheet name
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
    st.markdown(f"<div class='section-header'>{title}</div>", unsafe_allow_html=True)
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
                <div class="dash-card" style="padding:{pad};border-radius:{radius};margin-bottom:12px;">
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
                    <div class="dash-card" style="text-align:center;padding:20px;margin-bottom:12px;">
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

        # Direct column mapping for Amount SAR and Status
        amount_col = None
        status_col = None
        
        # Find amount column - look for column D which should be "amount sar" after lowercase
        for col in d.columns:
            col_lower = str(col).strip().lower()
            if "amount" in col_lower and "sar" in col_lower:
                amount_col = col
                break
        
        # If not found, try other patterns
        if not amount_col:
            amount_col = next((c for c in d.columns if "balance" in c and "due" in c), None) or \
                         next((c for c in d.columns if "currently" in c and "due" in c), None) or \
                         next((c for c in d.columns if "balance" in c and "settlement" in c), None) or \
                         next((c for c in ["amount(sar)", "amount sar", "amount", "value"] if c in d.columns), None) or \
                         next((c for c in d.columns if "amount" in c), None)

        # Find status column - look for column G
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

        # Separate pending and closed/paid settlements using SUMIF logic
        df_pending = pd.DataFrame()
        df_paid = pd.DataFrame()
        
        if status_col:
            # SUMIF: if STATUS = "CLOSED" then include in paid, if STATUS = "PENDING" then include in pending
            df_pending = out[out["status"].str.upper().str.strip() == "PENDING"].copy()
            df_paid = out[out["status"].str.upper().str.strip() == "CLOSED"].copy()
        else:
            # If no status column, treat all as pending
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
        
        # Find date column
        date_col = None
        for col in d.columns:
            if any(term in col for term in ["date", "time", "updated"]):
                date_col = col
                break
        
        if not date_col:
            return pd.DataFrame()
            
        # Look for currency columns (USD, EUR, AED, QAR, etc.)
        currency_cols = []
        for col in d.columns:
            if col != date_col and col.upper() in ['USD', 'EUR', 'AED', 'QAR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD']:
                currency_cols.append(col)
        
        if not currency_cols:
            # Fallback: try to detect numeric columns that might be currencies
            for col in d.columns:
                if col != date_col and len(col) <= 4 and col.upper() == col:
                    # Check if column has numeric data
                    sample_vals = d[col].dropna().head(5)
                    if not sample_vals.empty:
                        numeric_count = sum(1 for val in sample_vals if pd.notna(_to_number(val)))
                        if numeric_count >= len(sample_vals) * 0.8:  # 80% numeric
                            currency_cols.append(col)
        
        if not currency_cols:
            return pd.DataFrame()
        
        # Convert to long format
        result_rows = []
        
        for _, row in d.iterrows():
            date_val = pd.to_datetime(row[date_col], errors="coerce")
            if pd.isna(date_val):
                continue
                
            for curr_col in currency_cols:
                rate_val = _to_number(row[curr_col])
                if pd.notna(rate_val) and rate_val > 0:
                    currency_pair = f"{curr_col.upper()}/SAR"  # Assuming SAR as base
                    result_rows.append({
                        "currency_pair": currency_pair,
                        "rate": rate_val,
                        "date": date_val
                    })
        
        if not result_rows:
            return pd.DataFrame()
            
        out = pd.DataFrame(result_rows)
        
        # Add change calculation if we have historical data
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
    """Parse and clean the combined Export LC data."""
    try:
        if df.empty: return pd.DataFrame()
        
        d = cols_lower(df)
        
        required_cols = ['submitted date', 'branch', 'value (sar)']
        if not all(col in d.columns for col in required_cols):
            logger.warning(f"Export LC sheet missing required columns. Found: {d.columns.tolist()}")
            return pd.DataFrame()
            
        d = d.rename(columns={
            'applicant': 'applicant',
            'l/c no.': 'lc_no',
            'issuing bank': 'issuing_bank',
            'advising bank': 'advising_bank',
            'reference no.': 'reference_no',
            'benefecery branch': 'beneficiary_branch',
            'invoice no.': 'invoice_no',
            'submitted date': 'submitted_date',
            'value (sar)': 'value_sar',
            'payment term (days)': 'payment_term_days',
            'maturity date': 'maturity_date',
            'status': 'status',
            'remarks': 'remarks',
            'branch': 'branch'
        })
        
        d['submitted_date'] = pd.to_datetime(d['submitted_date'], errors='coerce')
        d['maturity_date'] = pd.to_datetime(d['maturity_date'], errors='coerce')
        d['value_sar'] = d['value_sar'].apply(_to_number)
        
        # MODIFICATION: Do not drop rows with missing dates, only essential values
        out = d.dropna(subset=['value_sar', 'branch'])
        
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
        
        # Export LC Proceeds
        if not df_export_lc.empty:
            df_export_lc.to_excel(writer, sheet_name='Export LC Proceeds', index=False)
        
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
    
    # Load new Export LC data
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
    
    # New KPI: Accepted Export LC Sum
    accepted_export_lc_sum = 0.0
    if not df_export_lc.empty and 'status' in df_export_lc.columns:
        mask = df_export_lc['status'].astype(str).str.strip().str.upper() == 'ACCEPTED'
        accepted_export_lc_sum = float(df_export_lc.loc[mask, 'value_sar'].sum())

    # Sidebar
    render_sidebar({}, total_balance, approved_sum, lc_next4_sum, banks_cnt, accepted_export_lc_sum)
    # Density tokens
    pad = "12px" if st.session_state.get("compact_density", False) else "24px"
    radius = "8px"
    shadow = "0 4px 12px rgba(0,0,0,0.05)"

    # ===== Quick Insights =====
    st.markdown('<div class="section-header">💡 Quick Insights & Recommendations</div>', unsafe_allow_html=True)
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

        st.markdown('<div class="section-header">📅 Month-to-Date — Detailed Insights</div>', unsafe_allow_html=True)

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
                st.dataframe(style_right(topn, num_cols=num_cols), use_container_width=True, height=320)
            else:
                st.info("No bank balances available.")

        # 2) LCR & STL Settlements this month - Enhanced Visual Design
        st.markdown('<div class="section-header">📅 LCR & STL Settlements — Overview</div>', unsafe_allow_html=True)
        st.markdown('<div style="background:#f1f5f9;padding:12px;border-radius:8px;border-left:4px solid #3b82f6;margin-bottom:16px;"><small>📊 <strong>Metrics show ALL settlements</strong> | 📈 <strong>Chart & table show current month only</strong></small></div>', unsafe_allow_html=True)
        
        if df_lc.empty and df_lc_paid.empty:
            st.info("No LCR & STL data.")
        else:
            lc_m = df_lc[(df_lc["settlement_date"] >= month_start) & (df_lc["settlement_date"] <= month_end)].copy() if not df_lc.empty else pd.DataFrame()
            lc_paid_m = df_lc_paid[(df_lc_paid["settlement_date"] >= month_start) & (df_lc_paid["settlement_date"] <= month_end)].copy() if not df_lc_paid.empty else pd.DataFrame()
            
            if lc_m.empty and lc_paid_m.empty:
                st.write("No LCR & STL for this month.")
            else:
                # Calculate metrics with specific criteria - ALL DATA (not just current month)
                # Use all settlement data, not just current month
                all_pending = df_lc.copy() if not df_lc.empty else pd.DataFrame()
                all_paid = df_lc_paid.copy() if not df_lc_paid.empty else pd.DataFrame()
                
                # Total Due: Sum of all amounts in column D (ALL settlements)
                total_due = (all_pending["amount"].sum() if not all_pending.empty else 0.0) + \
                           (all_paid["amount"].sum() if not all_paid.empty else 0.0)
                
                # Current Due: Sum of column D where Status="PENDING" AND Remark is not empty
                if not all_pending.empty:
                    # Check for non-empty remarks - look for rows with actual text content
                    current_due_mask = (all_pending["status"].str.upper().str.strip() == "PENDING") & \
                                      (all_pending["remark"].notna()) & \
                                      (all_pending["remark"].astype(str).str.strip() != "") & \
                                      (all_pending["remark"].astype(str).str.strip() != "-") & \
                                      (all_pending["remark"].astype(str).str.strip().str.lower() != "nan")
                    current_due = all_pending.loc[current_due_mask, "amount"].sum()
                else:
                    current_due = 0.0
                
                # Paid: Sum of column D where Status="CLOSED"
                paid_amount = all_paid["amount"].sum() if not all_paid.empty else 0.0
                
                # Balance Due: Sum of column D where Status="PENDING" AND Remark is EMPTY
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
                
                # Completion rate based on Total Due
                completion_rate = (paid_amount / total_due * 100) if total_due > 0 else 0
                
                # Enhanced KPI Cards with new criteria
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Due", fmt_number_only(total_due))
                with col2:
                    st.metric("Current Due", fmt_number_only(current_due))
                with col3:
                    st.metric("Paid", fmt_number_only(paid_amount))
                with col4:
                    st.metric("Balance Due", fmt_number_only(balance_due))
                
                # Enhanced Chart Section
                if not lc_m.empty:
                    lc_m["week"] = lc_m["settlement_date"].dt.isocalendar().week.astype(int)
                    weekly = lc_m.groupby("week", as_index=False)["amount"].sum().sort_values("week")
                    
                    chart_col1, chart_col2 = st.columns([2, 1])
                    with chart_col1:
                        st.markdown("##### Weekly Settlement Schedule (Current Month)")
                        try:
                            import plotly.graph_objects as go
                            fig = go.Figure(go.Bar(
                                x=[f"W{w}" for w in weekly["week"]],
                                y=weekly["amount"],
                                text=[f"{v/1000:.0f}K" for v in weekly["amount"]],
                                textposition="outside"
                            ))
                            fig.update_layout(height=350, margin=dict(l=20,r=20,t=20,b=40))
                            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
                        except Exception:
                            st.bar_chart(weekly.set_index("week")["amount"])
                    with chart_col2:
                        st.markdown("##### Key Insights")
                        if not weekly.empty:
                            peak_week = weekly.loc[weekly["amount"].idxmax()]
                            st.metric("Peak Week", f"Week {int(peak_week['week'])}")
                            st.metric("Peak Amount", fmt_number_only(peak_week['amount']))

    # ---- Bank tab ----
    with tab_bank:
        st.markdown('<div class="section-header">🏦 Bank Balance</div>', unsafe_allow_html=True)
        if df_by_bank.empty:
            st.info("No balances found.")
        else:
            view = st.radio("", options=["Cards", "List", "Mini Cards", "Table"],
                            index=0, horizontal=True, label_visibility="collapsed", key="bank_view")
            df_bal_view = df_by_bank.copy().sort_values("balance", ascending=False)
            if view == "Cards":
                cols = st.columns(4)
                for i, row in df_bal_view.iterrows():
                    with cols[int(i) % 4]:
                        bal = row.get('balance', np.nan); after = row.get('after_settlement', np.nan)
                        amt_color = THEME["amount_color"]["neg"] if pd.notna(bal) and bal < 0 else THEME["amount_color"]["pos"]
                        after_html = ""
                        if pd.notna(after):
                            as_pos = after >= 0
                            badge_bg = THEME["badge"]["pos_bg"] if as_pos else THEME["badge"]["neg_bg"]
                            badge_color = "#065f46" if as_pos else THEME["amount_color"]["neg"]
                            after_html = (f'<div style="display:inline-block; padding:6px 10px; border-radius:8px; '
                                          f'background:{badge_bg}; color:{badge_color}; font-weight:800; margin-top:10px;">'
                                          f'After: {fmt_currency(after)}</div>')
                        st.markdown(
                            f"""
                            <div class="dash-card">
                                <div style="display:flex;align-items:center;margin-bottom:12px;">
                                    <span style="font-size:13px;font-weight:700;color:#1e293b;">{row['bank']}</span>
                                </div>
                                <div style="font-size:24px;font-weight:900;color:{amt_color};text-align:right;">{fmt_currency(bal)}</div>
                                <div style="font-size:10px;color:#1e293b;opacity:.7;margin-top:6px;">Available Balance</div>
                                {after_html}
                            </div>
                            """, unsafe_allow_html=True)
            elif view == "List":
                display_as_list(df_bal_view, "bank", "balance", "")
            elif view == "Mini Cards":
                display_as_mini_cards(df_bal_view, "bank", "balance", pad=pad, radius=radius, shadow=shadow)
            else:
                table = df_bal_view.copy()
                rename_map = {"bank": "Bank", "balance": "Balance"}
                if "after_settlement" in table.columns:
                    rename_map["after_settlement"] = "After Settlement"
                table = table.rename(columns=rename_map)
                num_cols = [c for c in ["Balance", "After Settlement"] if c in table.columns]
                st.dataframe(style_right(table, num_cols=num_cols), use_container_width=True, height=360)

        st.markdown('<div class="section-header">📈 Liquidity Trend Analysis</div>', unsafe_allow_html=True)
        if df_fm.empty:
            st.info("No liquidity data available.")
        else:
            try:
                import plotly.graph_objects as go
                latest_liquidity = df_fm.iloc[-1]["total_liquidity"]
                c1, c2 = st.columns([3, 1])
                with c1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_fm["date"], y=df_fm["total_liquidity"], mode='lines+markers', line=dict(width=3, color=THEME["accent1"]), marker=dict(size=6)))
                    fig.update_layout(title="Total Liquidity Trend",
                                      height=400, margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
                with c2:
                    st.markdown("##### Liquidity Metrics")
                    st.metric("Current", fmt_number_only(latest_liquidity))
                    st.markdown("###### Statistics (30d)")
                    last30 = df_fm.tail(30)
                    st.write(f"**Max:** {fmt_number_only(last30['total_liquidity'].max())}")
                    st.write(f"**Min:** {fmt_number_only(last30['total_liquidity'].min())}")
                    st.write(f"**Avg:** {fmt_number_only(last30['total_liquidity'].mean())}")
            except Exception:
                st.error("❌ Unable to display liquidity trend analysis")
                st.line_chart(df_fm.set_index("date")["total_liquidity"])

        st.markdown('<div class="section-header">🏢 Collection vs Payments — by Branch</div>', unsafe_allow_html=True)
        if df_cvp.empty:
            st.info("No data in 'Collection vs Payments by Branch'.")
        else:
            cvp_view = st.radio("", options=["Bars", "Table"], index=0, horizontal=True, label_visibility="collapsed", key="cvp_view")
            cvp_sorted = df_cvp.sort_values("net", ascending=False).reset_index(drop=True)
            if cvp_view == "Bars":
                try:
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    fig.add_bar(name="Collection", x=cvp_sorted["branch"], y=cvp_sorted["collection"], marker_color=THEME["accent1"])
                    fig.add_bar(name="Payments", x=cvp_sorted["branch"], y=cvp_sorted["payments"], marker_color=THEME["heading_bg"])
                    fig.update_layout(barmode="group", height=420, margin=dict(l=20, r=20, t=30, b=80))
                    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
                except Exception:
                    st.bar_chart(cvp_sorted.set_index("branch")[["collection", "payments"]])
            else:
                tbl = cvp_sorted.rename(columns={"branch": "Branch", "collection": "Collection", "payments": "Payments", "net": "Net"})
                styled = style_right(tbl, num_cols=["Collection", "Payments", "Net"])
                def _net_red(val):
                    try: return f'color:{THEME["neg"]};font-weight:700;' if float(val) < 0 else ''
                    except Exception: return ''
                styled = styled.applymap(_net_red, subset=["Net"])
                st.dataframe(styled, use_container_width=True, height=420)

    # ---- Settlements tab ----
    with tab_settlements:
        st.markdown('<div class="section-header">📅 LCR & STL Settlements</div>', unsafe_allow_html=True)
        
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
                cc1, cc2, cc3 = st.columns(3)
                with cc1: st.metric(f"Total {status_label} Amount", fmt_number_only(view_data["amount"].sum()))
                with cc2: st.metric(f"Number of {status_label}", len(view_data))
                
                if status_label == "Pending":
                    with cc3: st.metric("Urgent (2 days)", len(view_data[view_data["settlement_date"] <= today0 + pd.Timedelta(days=2)]))
                    
                    viz = view_data.copy()
                    viz["Settlement Date"] = viz["settlement_date"].dt.strftime(config.DATE_FMT)
                    viz["Days Until Due"] = (viz["settlement_date"] - today0).dt.days
                    rename = {"reference": "Reference", "bank": "Bank", "type": "Type", "status": "Status", "remark": "Remark", "amount": "Amount"}
                    viz = viz.rename(columns={k: v for k, v in rename.items() if k in viz.columns})
                    cols = ["Reference", "Bank", "Type", "Status", "Settlement Date", "Amount", "Days Until Due", "Remark"]
                    cols = [c for c in cols if c in viz.columns]
                    show = viz[cols].sort_values("Settlement Date")
                    
                    def _highlight(row):
                        if "Days Until Due" in row:
                            if row["Days Until Due"] <= 2: return [f'background-color: {THEME["card_neg"]}'] * len(row)
                            if row["Days Until Due"] <= 7: return ['background-color: #FEF9C3'] * len(row)
                        return [''] * len(row)
                    styled = style_right(show, num_cols=["Amount"]).apply(_highlight, axis=1)
                    st.dataframe(styled, use_container_width=True, height=400)
                else:
                    viz_paid = view_data.copy()
                    viz_paid["Settlement Date"] = viz_paid["settlement_date"].dt.strftime(config.DATE_FMT)
                    rename = {"reference": "Reference", "bank": "Bank", "type": "Type", "status": "Status", "remark": "Remark", "amount": "Amount"}
                    viz_paid = viz_paid.rename(columns={k: v for k, v in rename.items() if k in viz_paid.columns})
                    cols_paid = ["Reference", "Bank", "Type", "Status", "Settlement Date", "Amount", "Remark"]
                    cols_paid = [c for c in cols_paid if c in viz_paid.columns]
                    show_paid = viz_paid[cols_paid].sort_values("Settlement Date", ascending=False)
                    st.dataframe(style_right(show_paid, num_cols=["Amount"]), use_container_width=True, height=400)
            else:
                st.info("No settlements match the selected criteria.")
        
        tab_pending, tab_paid = st.tabs(["Pending", "Paid"])
        with tab_pending: render_settlements_tab(df_lc, "Pending", "pending")
        with tab_paid: render_settlements_tab(df_lc_paid, "Paid", "paid")

    # ---- Supplier Payments tab ----
    with tab_payments:
        st.markdown('<div class="section-header">💰 Supplier Payments</div>', unsafe_allow_html=True)
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
                c1, c2, c3 = st.columns(3)
                with c1: st.metric(f"Total {status_label} Amount", fmt_number_only(view_data["amount"].sum()))
                with c2: st.metric("Number of Payments", len(view_data))
                with c3: st.metric("Average Payment", fmt_number_only(view_data["amount"].mean()))
                
                show_cols = [c for c in ["bank", "supplier", "currency", "amount", "status"] if c in view_data.columns]
                v = view_data[show_cols].rename(columns={"bank": "Bank", "supplier": "Supplier", "currency": "Currency",
                                                            "amount": "Amount", "status": "Status"})
                st.dataframe(style_right(v, num_cols=["Amount"]), use_container_width=True, height=360)
            else:
                st.info("No payments match the selected criteria.")
        tab_approved, tab_released = st.tabs(["Approved", "Released"])
        with tab_approved: render_payments_tab(df_pay_approved, "Approved", "approved")
        with tab_released: render_payments_tab(df_pay_released, "Released", "released")

    # ---- Export LC tab ----
    with tab_export_lc:
        st.markdown('<div class="section-header">🚢 Export LC Proceeds</div>', unsafe_allow_html=True)
        if df_export_lc.empty:
            st.info("No Export LC data found or the file is invalid.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                branches = sorted(df_export_lc["branch"].unique())
                selected_branches = st.multiselect("Filter by Branch", options=branches, default=branches, key="export_lc_branch_filter")
                
                if 'status' in df_export_lc.columns:
                    statuses = sorted(df_export_lc["status"].dropna().astype(str).unique())
                    selected_statuses = st.multiselect("Filter by Status", options=statuses, default=statuses, key="export_lc_status_filter")
                else:
                    selected_statuses = []

            with col2:
                min_date = df_export_lc["submitted_date"].dropna().min().date()
                max_date = df_export_lc["submitted_date"].dropna().max().date()
                start_date_filter = st.date_input("From Submitted Date", value=min_date, key="export_lc_start_date")
                end_date_filter = st.date_input("To Submitted Date", value=max_date, key="export_lc_end_date")

            # Apply branch and status filters
            filtered_df = df_export_lc[df_export_lc["branch"].isin(selected_branches)]
            if 'status' in filtered_df.columns and selected_statuses:
                filtered_df = filtered_df[filtered_df["status"].isin(selected_statuses)]

            # Apply date filter but keep rows with no date
            date_mask = (filtered_df["submitted_date"].dt.date >= start_date_filter) & (filtered_df["submitted_date"].dt.date <= end_date_filter)
            no_date_mask = filtered_df["submitted_date"].isna()
            filtered_df = filtered_df[date_mask | no_date_mask].copy()
            
            # Display metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Value (SAR)", fmt_number_only(filtered_df['value_sar'].sum()))
            m2.metric("Total LCs", len(filtered_df))
            m3.metric("Average Value (SAR)", fmt_number_only(filtered_df['value_sar'].mean()))

            # Display table
            display_cols = {
                'branch': 'Branch', 'applicant': 'Applicant', 'lc_no': 'LC No.',
                'submitted_date': 'Submitted Date', 'maturity_date': 'Maturity Date',
                'value_sar': 'Value (SAR)', 'status': 'Status', 'remarks': 'Remarks'
            }
            cols_to_show = {k: v for k, v in display_cols.items() if k in filtered_df.columns}
            table_view = filtered_df[list(cols_to_show.keys())].rename(columns=cols_to_show)
            
            # Safely format date columns
            for date_col in ['Submitted Date', 'Maturity Date']:
                if date_col in table_view.columns:
                    table_view[date_col] = pd.to_datetime(table_view[date_col]).dt.strftime('%Y-%m-%d')
            
            st.dataframe(
                style_right(table_view, num_cols=['Value (SAR)']),
                use_container_width=True, height=500)

    # ---- Exchange Rates tab ----
    with tab_fx:
        st.markdown('<div class="section-header">💱 Exchange Rates</div>', unsafe_allow_html=True)
        if df_fx.empty:
            st.info("No exchange rate data available.")
        else:
            latest_fx = df_fx.groupby("currency_pair").last().reset_index()
            if not latest_fx.empty:
                cols = st.columns(min(4, len(latest_fx)))
                for i, row in latest_fx.iterrows():
                    with cols[int(i) % min(4, len(latest_fx))]:
                        pair = row["currency_pair"]
                        rate = row["rate"]
                        change_pct = row.get("change_pct", np.nan)
                        st.metric(label=pair, value=fmt_rate(rate), delta=f"{change_pct:.2f}%" if pd.notna(change_pct) else None)

    # ---- Facility Report tab ----
    with tab_facility:
        pass # Intentionally empty

    # ---- Reports tab ----
    with tab_reports:
        st.markdown('<div class="section-header">📊 Complete Report Export</div>', unsafe_allow_html=True)
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
```
