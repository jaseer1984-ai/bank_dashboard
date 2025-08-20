# app.py — Enhanced Treasury Dashboard (Themed, Tabs)
# - "Remaining in Month" shows Balance Due from Settlements sheet
# - Comma-separated numeric formatting (with decimals where needed)
# - Plotly toolbars hidden
# - All other features preserved

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
      .stDataFrame, .stDataFrame * {{ font-variant-numeric: tabular-nums; }}
      [data-testid="stDecoration"], [data-testid="stStatusWidget"], [data-testid="stToolbar"] {{ display: none !important; }}
    </style>
    """.format(font_q=family.replace(" ", "+"), font=family)
    st.markdown(css, unsafe_allow_html=True)

set_app_font()

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
    """Format plain numbers with comma separation and custom decimals."""
    try:
        if pd.isna(v): return "N/A"
        return f"{float(v):,.{decimals}f}"
    except Exception:
        return str(v)

def fmt_number_only(v) -> str:
    return fmt_number(v, 0)

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
    except Exception:
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

def parse_settlements(df: pd.DataFrame) -> pd.DataFrame:
    try:
        d = cols_lower(df)

        # --- find key columns (broader matching) ---
        bank_col = next((c for c in d.columns if "bank" in c), None)

        # date: settlement date / maturity date / due date / generic date
        date_col = next((c for c in d.columns if "settlement" in c and "date" in c), None) or \
                   next((c for c in d.columns if "maturity" in c and "new" not in c), None) or \
                   next((c for c in d.columns if "due" in c and "date" in c), None) or \
                   next((c for c in d.columns if c.strip().lower() == "date"), None)

        # amount: balance due / currently due / balance settlement / amount(sar) / amount / value
        amount_col = next((c for c in d.columns if "balance" in c and "due" in c), None) or \
                     next((c for c in d.columns if "currently" in c and "due" in c), None) or \
                     next((c for c in d.columns if "balance" in c and "settlement" in c), None) or \
                     next((c for c in ["amount(sar)", "amount", "value"] if c in d.columns), None)

        # optional columns
        status_col = next((c for c in d.columns if "status" in c), None)
        type_col   = next((c for c in d.columns if "type" in c), None)
        remark_col = next((c for c in d.columns if "remark" in c), None)
        ref_col    = next((c for c in d.columns if any(t in c for t in ["a/c", "ref", "account", "reference"])), None)

        if not all([bank_col, date_col, amount_col]):
            return pd.DataFrame()

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

        # Filter to pending only when a real status column exists and contains 'pending'
        if status_col:
            has_pending = out["status"].str.contains("pending", case=False, na=False).any()
            if has_pending:
                out = out[out["status"].str.contains("pending", case=False, na=False)]

        return out.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


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

def parse_exchange_rate(df: pd.DataFrame) -> pd.DataFrame:
    try:
        d = df.copy()
        d.columns = [str(c).strip().strip('"').strip("'") for c in d.columns]
        has_date_col = any("date" in c.lower() for c in d.columns)
        if has_date_col:
            date_col = next(c for c in d.columns if "date" in c.lower())
            value_cols = [c for c in d.columns if c != date_col]
            long_df = d.melt(id_vars=[date_col], value_vars=value_cols,
                             var_name="currency", value_name="rate")
            long_df = long_df.rename(columns={date_col: "date"})
            long_df["date"] = pd.to_datetime(long_df["date"], errors="coerce", dayfirst=True)
            long_df["currency"] = long_df["currency"].astype(str).str.strip()
            long_df["rate"] = long_df["rate"].map(_to_number)
        else:
            cur_col = next((c for c in d.columns if "currency" in c.lower()), d.columns[0])
            candidate_cols = [c for c in d.columns if c != cur_col]
            parsed = pd.to_datetime(pd.Index(candidate_cols), errors="coerce", dayfirst=True)
            date_cols = [col for col, dt_ in zip(candidate_cols, parsed) if pd.notna(dt_)]
            if not date_cols: return pd.DataFrame()
            long_df = d.melt(id_vars=[cur_col], value_vars=date_cols,
                             var_name="date", value_name="rate").rename(columns={cur_col: "currency"})
            long_df["date"] = pd.to_datetime(long_df["date"], errors="coerce", dayfirst=True)
            long_df["currency"] = long_df["currency"].astype(str).str.strip()
            long_df["rate"] = long_df["rate"].map(_to_number)
        long_df = long_df.dropna(subset=["currency", "date", "rate"])
        long_df = long_df[long_df["currency"] != ""]
        long_df = long_df.sort_values(["currency", "date"]).reset_index(drop=True)
        return long_df
    except Exception:
        return pd.DataFrame()

def extract_balance_due_value(df_raw: pd.DataFrame) -> float:
    """
    Robustly extract 'Balance Due' value from the Settlements sheet dump (any layout).
    Looks for a cell containing 'balance due' and returns the first numeric in the same row (prefer to the right).
    Falls back to a column header containing 'balance due'.
    """
    if df_raw.empty:
        return np.nan
    try:
        d = df_raw.copy()

        # 1) Find any cell text containing "balance due"
        mask = d.applymap(lambda x: isinstance(x, str) and ("balance due" in x.strip().lower()))
        if mask.any().any():
            coords = np.argwhere(mask.values)
            r, c = coords[0]  # first match
            row_vals = d.iloc[r].apply(_to_number)

            # Prefer first numeric cell to the right of the label
            after = row_vals.iloc[c+1:]
            cand = after[after.notna()]
            if not cand.empty:
                return float(cand.iloc[0])

            # Otherwise any numeric in the row (last one)
            row_nums = row_vals[row_vals.notna()]
            if not row_nums.empty:
                return float(row_nums.iloc[-1])

        # 2) Maybe it's a column header named "Balance Due"
        col = next((col for col in d.columns if isinstance(col, str) and "balance due" in col.strip().lower()), None)
        if col:
            series = d[col].apply(_to_number).dropna()
            if not series.empty:
                return float(series.iloc[-1])

        # 3) Row label in one column, number in another
        for idx, row in d.iterrows():
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
def render_sidebar(data_status, total_balance, approved_sum, lc_next4_sum, banks_cnt):
    with st.sidebar:
        st.markdown("### 🔄 Refresh")
        if st.button("Refresh Now", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        # FX toggle (kept)
        if "show_fx" not in st.session_state:
            st.session_state["show_fx"] = False
        if st.button("📈 Exchange Rates", use_container_width=True, help="Show/Hide exchange rate chart"):
            st.session_state["show_fx"] = not st.session_state["show_fx"]
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
        _kpi("LC DUE (NEXT 4 DAYS)", lc_next4_sum, THEME["heading_bg"], THEME["accent1"], "#92400E")
        _kpi("ACTIVE BANKS", banks_cnt, THEME["heading_bg"], THEME["accent2"], "#9F1239")

        st.markdown("---")
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
    df_lc = parse_settlements(df_lc_raw)
    balance_due_value = extract_balance_due_value(df_lc_raw)  # <<-- Balance Due from sheet

    df_fm_raw = read_csv(LINKS["Fund Movement"])
    df_fm = parse_fund_movement(df_fm_raw)

    df_cvp_raw = read_csv(LINKS["COLLECTION_BRANCH"])
    df_cvp = parse_branch_cvp(df_cvp_raw)

    df_fx_raw = read_csv(LINKS["EXCHANGE_RATE"])
    df_fx = parse_exchange_rate(df_fx_raw)

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

    # Sidebar
    render_sidebar({}, total_balance, approved_sum, lc_next4_sum, banks_cnt)

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
            insights.append({"type": "error","title": "Urgent LC Settlements",
                             "content": f"{len(urgent7)} LC settlements due within 7 days totaling {fmt_number_only(urgent7['amount'].sum())}."})
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
    tab_overview, tab_bank, tab_settlements, tab_payments, tab_fx, tab_facility = st.tabs(
        ["Overview", "Bank", "Settlements", "Supplier Payments", "Exchange Rate", "Facility Report"]
    )

    # ---- Overview: Monthly detailed insights ----
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

                    # Chart with date-only ticks
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
                # include After Settlement when available
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

        # 2) LC Settlements this month
        st.subheader("LC Settlements — This Month")
        if df_lc.empty:
            st.info("No LC data.")
        else:
            lc_m = df_lc[(df_lc["settlement_date"] >= month_start) & (df_lc["settlement_date"] <= month_end)].copy()
            if lc_m.empty:
                st.write("No LC due this month.")
            else:
                lc_m["week"] = lc_m["settlement_date"].dt.isocalendar().week.astype(int)
                weekly = lc_m.groupby("week", as_index=False)["amount"].sum().sort_values("week")
                k1, k2, k3 = st.columns(3)
                with k1: st.metric("Current Due", fmt_number_only(lc_m["amount"].sum()))
                with k2: st.metric("# of LCs", len(lc_m))
                with k3:
                    # >>> Show Balance Due from sheet <<<
                    st.metric("Remaining in Month", fmt_number(balance_due_value, 2))
                try:
                    import plotly.io as pio, plotly.graph_objects as go
                    if "brand" not in pio.templates:
                        pio.templates["brand"] = pio.templates["plotly_white"]
                        pio.templates["brand"].layout.colorway = [THEME["accent1"], THEME["accent2"], "#64748b", "#94a3b8"]
                        pio.templates["brand"].layout.font.family = APP_FONT
                    fig = go.Figure(go.Bar(x=weekly["week"], y=weekly["amount"]))
                    fig.update_layout(template="brand", height=280, margin=dict(l=20,r=20,t=10,b=10),
                                      xaxis_title="ISO Week", yaxis_title="Amount (SAR)", showlegend=False)
                    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
                except Exception:
                    st.bar_chart(weekly.set_index("week")["amount"])
                detail = lc_m[["bank","settlement_date","amount","type","remark"]].rename(
                    columns={"bank":"Bank","settlement_date":"Due Date","amount":"Amount","type":"Type","remark":"Remark"})
                detail["Due Date"] = detail["Due Date"].dt.strftime(config.DATE_FMT)
                st.dataframe(style_right(detail, num_cols=["Amount"]), use_container_width=True, height=280)

        st.markdown("---")

        # 3) FX MTD
        st.subheader("FX — Month-to-Date Change")
        if df_fx.empty:
            st.info("No FX data.")
        else:
            fx_m = df_fx[(df_fx["date"] >= month_start) & (df_fx["date"] <= month_end)].copy()
            if fx_m.empty:
                st.write("No FX quotes for this month.")
            else:
                focus = [c for c in ["USD","AED","EUR","QAR"] if c in fx_m["currency"].unique()] or fx_m["currency"].unique()[:4].tolist()
                rows = []
                for ccy in focus:
                    sub = fx_m[fx_m["currency"] == ccy].sort_values("date")
                    if sub.empty: continue
                    first = sub.iloc[0]["rate"]; last = sub.iloc[-1]["rate"]
                    chg = last - first; pct = (chg / first * 100.0) if first else np.nan
                    rows.append({"Currency": ccy, "Start": first, "Latest": last, "Δ": chg, "Δ%": pct})
                if rows:
                    fx_tbl = pd.DataFrame(rows)
                    st.dataframe(style_right(fx_tbl, num_cols=["Start","Latest","Δ","Δ%"], decimals=4),
                                 use_container_width=True, height=220)
                try:
                    import plotly.io as pio, plotly.graph_objects as go
                    if "brand" not in pio.templates:
                        pio.templates["brand"] = pio.templates["plotly_white"]
                        pio.templates["brand"].layout.colorway = [THEME["accent1"], THEME["accent2"], "#64748b", "#94a3b8"]
                        pio.templates["brand"].layout.font.family = APP_FONT
                    chart_curr = st.multiselect("Currencies to plot", sorted(fx_m["currency"].unique()), default=focus, key="fx_m_plot")
                    if chart_curr:
                        fig = go.Figure()
                        for cur in chart_curr:
                            s = fx_m[fx_m["currency"] == cur]
                            fig.add_trace(go.Scatter(x=s["date"].dt.normalize(), y=s["rate"], mode="lines+markers", name=cur))
                        fig.update_layout(template="brand", height=300, margin=dict(l=20,r=20,t=10,b=10),
                                          xaxis_title=None, yaxis_title="Rate (SAR)")
                        fig.update_xaxes(tickformat="%b %d", rangeslider_visible=False, rangeselector=None)
                        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
                except Exception:
                    pass

        st.markdown("---")
        st.subheader("Branches — Net Position (Snapshot)")
        if df_cvp.empty:
            st.info("No branch CVP data.")
        else:
            snap = df_cvp.sort_values("net", ascending=False).rename(
                columns={"branch":"Branch","collection":"Collection","payments":"Payments","net":"Net"})
            st.dataframe(style_right(snap, num_cols=["Collection","Payments","Net"]), use_container_width=True, height=300)

        st.caption(f"Period: {month_start.strftime('%Y-%m-%d')} → {month_end.strftime('%Y-%m-%d')}  •  Today: {today0_local.strftime('%Y-%m-%d')}")

    # ---- Bank tab: Bank balance, Liquidity chart, CVP by Branch ----
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
        st.markdown('<span class="section-chip">📅 LC Settlements — Pending</span>', unsafe_allow_html=True)
        if df_lc.empty:
            st.info("No LC (Pending) data. Ensure the sheet has the required columns.")
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
                    styled = style_right(show, num_cols=["Amount"]).apply(_highlight, axis=1)
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
                else:
                    cards = lc_view.groupby("bank", as_index=False)["amount"].sum().rename(columns={"amount": "balance"})
                    display_as_mini_cards(cards, "bank", "balance", pad=pad, radius=radius, shadow=shadow)
                urgent_lcs = lc_view[lc_view["settlement_date"] <= today0 + pd.Timedelta(days=3)]
                if not urgent_lcs.empty:
                    st.warning(f"⚠️ {len(urgent_lcs)} LC(s) due within 3 days!")
                    for _, lc in urgent_lcs.iterrows():
                        days_left = (lc["settlement_date"] - today0).days
                        st.write(f"• {lc['bank']} - {fmt_number_only(lc['amount'])} - {days_left} day(s) left)")

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

    # ---- Exchange Rate tab ----
    with tab_fx:
        st.markdown('<span class="section-chip">💱 Exchange Rate — Variation</span>', unsafe_allow_html=True)
        if df_fx.empty:
            st.info("No exchange rate data.")
        else:
            all_curr = sorted(df_fx["currency"].unique().tolist())
            default_pick = [c for c in ["USD", "AED", "EUR", "QAR"] if c in all_curr] or all_curr[:3]
            col1, col2 = st.columns([2, 1])
            with col1:
                pick_curr = st.multiselect("Currencies", all_curr, default=default_pick, key="fx_curr_tab")
            dmin = df_fx["date"].min().date(); dmax = df_fx["date"].max().date()
            view_fx = df_fx[(df_fx["currency"].isin(pick_curr)) &
                            (df_fx["date"].dt.date >= dmin) & (df_fx["date"].dt.date <= dmax)].copy()
            if view_fx.empty:
                st.info("No data for the selected filters.")
            else:
                try:
                    import plotly.io as pio, plotly.graph_objects as go
                    if "brand" not in pio.templates:
                        pio.templates["brand"] = pio.templates["plotly_white"]
                        pio.templates["brand"].layout.colorway = [THEME["accent1"], THEME["accent2"], "#64748b", "#94a3b8"]
                        pio.templates["brand"].layout.font.family = APP_FONT
                        pio.templates["brand"].layout.paper_bgcolor = "white"
                        pio.templates["brand"].layout.plot_bgcolor = "white"
                    chart_type = st.selectbox("Chart type", ["Line", "Area", "Step", "Bar"], index=0, key="fx_chart_type_tab")
                    view_fx["date_only"] = view_fx["date"].dt.normalize()
                    fig = go.Figure()
                    for cur in pick_curr:
                        sub = view_fx[view_fx["currency"] == cur]
                        if sub.empty: continue
                        if chart_type == "Bar":
                            fig.add_trace(go.Bar(x=sub["date_only"], y=sub["rate"], name=cur))
                        else:
                            line_shape = "linear" if chart_type in ("Line", "Area") else "hv"
                            fig.add_trace(go.Scatter(x=sub["date_only"], y=sub["rate"], name=cur,
                                                     mode="lines+markers",
                                                     line=dict(shape=line_shape, width=2),
                                                     fill="tozeroy" if chart_type == "Area" else None))
                    fig.update_layout(template="brand", height=420, margin=dict(l=20, r=20, t=40, b=20),
                                      title="Daily Exchange Rate Variation",
                                      xaxis_title=None, yaxis_title="Rate (SAR)", showlegend=True)
                    fig.update_xaxes(tickformat="%b %d, %Y", showticklabels=True, rangeslider_visible=False, rangeselector=None)
                    fig.update_traces(hovertemplate="%{x|%b %d, %Y}<br>Rate: %{y}<extra>%{fullData.name}</extra>")
                    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
                except Exception:
                    st.line_chart(view_fx.pivot_table(index="date_only", columns="currency", values="rate"))

    # ---- Facility Report tab (placeholder) ----
    with tab_facility:
        st.markdown('<span class="section-chip">🏗️ Facility Report</span>', unsafe_allow_html=True)
        st.info("Hook this tab to your facilities dataset (limits, utilizations, expiries). Once you share the sheet/GID, I’ll wire it up.")

    st.markdown("<hr style='margin: 8px 0 16px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center; opacity:0.8; font-size:12px;'>Powered By <strong>Jaseer Pykkarathodi</strong></div>", unsafe_allow_html=True)

    if st.session_state.get("auto_refresh"):
        interval = int(st.session_state.get("auto_interval", 120))
        with st.status(f"Auto refreshing in {interval}s…", expanded=False):
            time.sleep(interval)
        st.rerun()

if __name__ == "__main__":
    main()


