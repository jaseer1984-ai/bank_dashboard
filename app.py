# app.py — Full Treasury Dashboard (Google Sheets)
# - Brand: Issam Kabbani & Partners – Unitech + small logo 'ikk_logo.png'
# - Top-right Refresh button
# - KPI numbers right-aligned (no "(sum)" / "• Pending")
# - Bank Balances as cards (4 per row). No tables for balances
# - Supplier Payments: Approved List only (Amount right aligned)
# - LC Settlements: Pending only, date filter hidden, amounts right aligned
# - Liquidity Trend chart + Quick Insights
# - Footer: Created By Jaseer Pykarathodi

import io
from datetime import datetime
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Treasury Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------
COMPANY_NAME = "Issam Kabbani & Partners – Unitech"
LOGO_FILE = "ikk_logo.png"   # put this PNG in the same folder as app.py
DATE_FMT = "%Y-%m-%d"
TZ = "Asia/Riyadh"

# Google Sheets file id & GIDs
FILE_ID = "1371amvaCbejUWVJI_moWdIchy5DF1lPO"
LINKS = {
    "BANK BALANCE":      f"https://docs.google.com/spreadsheets/d/{FILE_ID}/export?format=csv&gid=860709395",
    "SUPPLIER PAYMENTS": f"https://docs.google.com/spreadsheets/d/{FILE_ID}/export?format=csv&gid=20805295",
    "SETTLEMENTS":       f"https://docs.google.com/spreadsheets/d/{FILE_ID}/export?format=csv&gid=978859477",
    "Fund Movement":     f"https://docs.google.com/spreadsheets/d/{FILE_ID}/export?format=csv&gid=66055663",
}

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def _to_number(x):
    """Convert '1,234.56' or '10%' to float; return NaN on failure."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace(",", "")
    if s.endswith("%"):
        s = s[:-1]
    try:
        return float(s)
    except Exception:
        return np.nan


@st.cache_data(ttl=300)
def read_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=25)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def cols_lower(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in df.columns]
    return out


def fmt_num(v) -> str:
    try:
        return f"{float(v):,.2f}"
    except Exception:
        return str(v)

# -----------------------------------------------------------------------------
# PARSERS
# -----------------------------------------------------------------------------
def parse_bank_balance(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Timestamp]]:
    """
    Return (by_bank, latest_date)
      by_bank: DataFrame with columns ['bank','balance']
      latest_date: date or None

    Handles two layouts:
      A) 'bank' + ('amount' or 'amount(sar)') columns
      B) bank names in one column + date columns for balances (picks latest date)
    """
    c = cols_lower(df)

    # Case A: bank + amount column
    if "bank" in c.columns and ("amount" in c.columns or "amount(sar)" in c.columns):
        amt_col = "amount" if "amount" in c.columns else "amount(sar)"
        out = pd.DataFrame({
            "bank": c["bank"].astype(str).str.strip(),
            "balance": c[amt_col].map(_to_number),
        }).dropna(subset=["bank", "balance"])
        latest_date = None
        by_bank = out.groupby("bank", as_index=False)["balance"].sum()
        return by_bank, latest_date

    # Case B: bank col + date headers
    raw = df.copy().dropna(how="all").dropna(axis=1, how="all")

    bank_col = None
    for col in raw.columns:
        if raw[col].dtype == object and (raw[col].dropna().astype(str).str.strip() != "").sum() >= 3:
            bank_col = col
            break
    if bank_col is None:
        raise ValueError("BANK BALANCE: could not detect bank column.")

    parsed = pd.to_datetime(pd.Index(raw.columns), errors="coerce", dayfirst=False)
    date_cols = [col for col, d in zip(raw.columns, parsed) if pd.notna(d)]
    if not date_cols:
        raise ValueError("BANK BALANCE: no date columns in header.")

    date_map = {col: pd.to_datetime(col, errors="coerce", dayfirst=False) for col in date_cols}
    latest_col = max(date_cols, key=lambda c: date_map[c])

    s = raw[bank_col].astype(str).str.strip()
    mask = s.ne("") & ~s.str.contains("available|total", case=False, na=False)
    sub = raw.loc[mask, [bank_col, latest_col]].copy()
    sub.columns = ["bank", "balance"]
    sub["balance"] = sub["balance"].astype(str).str.replace(",", "", regex=False).map(_to_number)
    sub["bank"] = sub["bank"].str.replace(r"\s*-\s*.*$", "", regex=True).str.strip()

    latest_date = date_map[latest_col].date()
    by_bank = sub.dropna(subset=["balance"]).groupby("bank", as_index=False)["balance"].sum()
    return by_bank, latest_date


def parse_supplier_payments(df: pd.DataFrame) -> pd.DataFrame:
    """Return only Approved rows with columns: bank, supplier, currency, amount, status"""
    d = cols_lower(df).rename(columns={
        "supplier name": "supplier",
        "amount(sar)": "amount_sar",
        "order/sh/branch": "order_branch",
    })

    if "bank" not in d.columns or "status" not in d.columns:
        return pd.DataFrame()

    status_norm = d["status"].astype(str).str.strip().str.lower()
    approved_mask = status_norm.str.contains("approved")
    d = d.loc[approved_mask].copy()
    if d.empty:
        return pd.DataFrame()

    amt_col = "amount_sar" if "amount_sar" in d.columns else ("amount" if "amount" in d.columns else None)
    if amt_col is None:
        return pd.DataFrame()

    d["amount"] = d[amt_col].map(_to_number)
    d["bank"] = d["bank"].astype(str).str.strip()

    keep = [c for c in ["bank", "supplier", "currency", "amount", "status"] if c in d.columns]
    out = d[keep].dropna(subset=["amount"]).copy()
    if "status" in out.columns:
        out["status"] = out["status"].astype(str).str.title()
    return out


def parse_settlements(df: pd.DataFrame) -> pd.DataFrame:
    """Return LC rows that are Pending only."""
    d = cols_lower(df)
    bank = next((c for c in d.columns if c.startswith("bank")), None)

    # date column preference
    date_col = None
    for cand in d.columns:
        if "maturity" in cand and "new" not in cand:
            date_col = cand
            break
    if date_col is None:
        for cand in d.columns:
            if "new" in cand and "maturity" in cand:
                date_col = cand
                break

    # amount precedence
    amount_col = None
    for cand in d.columns:
        if "balance" in cand and "settlement" in cand:
            amount_col = cand
            break
    if amount_col is None:
        for cand in d.columns:
            if "currently" in cand and "due" in cand:
                amount_col = cand
                break
    if amount_col is None:
        amount_col = "amount(sar)" if "amount(sar)" in d.columns else \
                     ("amount" if "amount" in d.columns else None)

    status_col = next((c for c in d.columns if "status" in c), None)
    type_col   = next((c for c in d.columns if "type" in c), None)
    remark_col = next((c for c in d.columns if "remark" in c), None)

    if not bank or not amount_col or not date_col:
        return pd.DataFrame()

    out = pd.DataFrame({
        "bank": d[bank].astype(str).str.strip(),
        "settlement_date": pd.to_datetime(d[date_col], errors="coerce"),
        "amount": d[amount_col].map(_to_number),
        "status": d[status_col].astype(str).str.title().str.strip() if status_col else "",
        "type": d[type_col].astype(str).str.upper().str.strip() if type_col else "",
        "remark": d[remark_col].astype(str).str.strip() if remark_col else "",
        "description": "",
    })
    out = out.dropna(subset=["bank", "amount", "settlement_date"])
    out = out[out["status"].str.lower() == "pending"].copy()
    return out


def parse_fund_movement(df: pd.DataFrame) -> pd.DataFrame:
    """Return liquidity series with columns: date, total_liquidity."""
    d = cols_lower(df)
    date_col = "date" if "date" in d.columns else None
    liq_col = next((c for c in d.columns if ("total" in c and "liquidity" in c)), None)
    if not date_col or not liq_col:
        return pd.DataFrame()
    out = pd.DataFrame({
        "date": pd.to_datetime(d[date_col], errors="coerce"),
        "total_liquidity": d[liq_col].map(_to_number),
    }).dropna()
    return out.sort_values("date")

# -----------------------------------------------------------------------------
# UI PRIMITIVES
# -----------------------------------------------------------------------------
def kpi_card(title, value, bg="#EEF2FF", border="#C7D2FE", text="#111827"):
    """Right-aligned KPI card."""
    st.markdown(
        f"""
        <div style="
            background:{bg};border:1px solid {border};
            border-radius:12px;padding:14px 16px;
            box-shadow:0 1px 6px rgba(0,0,0,.04);">
            <div style="font-size:12px;color:#374151;text-transform:uppercase;letter-spacing:.08em">{title}</div>
            <div style="font-size:28px;font-weight:800;color:{text};margin-top:4px;text-align:right;">
                {fmt_num(value) if isinstance(value,(int,float,np.integer,np.floating)) else value}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def bank_card(bank: str, balance: float):
    """Visual bank card — no charts."""
    st.markdown(
        f"""
        <div style="
            background:#ffffff; border:1px solid #e5e7eb;
            border-radius:14px; padding:14px 16px;
            box-shadow:0 1px 6px rgba(0,0,0,.06);
            height:120px; display:flex; flex-direction:column; justify-content:center;">
            <div style="font-size:14px; color:#4b5563; margin-bottom:6px;">{bank}</div>
            <div style="font-size:22px; font-weight:700; color:#111827; text-align:right;">
                {fmt_num(balance)} <span style="font-size:12px; color:#6b7280;">SAR</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------------------------------------------------------
# HEADER (logo + title + top-right refresh)
# -----------------------------------------------------------------------------
h1, h2, h3 = st.columns([0.06, 0.74, 0.20])
with h1:
    try:
        st.image(LOGO_FILE, width=40)
    except Exception:
        st.write("")
with h2:
    st.markdown(
        f"<h2 style='margin:0; padding-top:4px'>{COMPANY_NAME} — Treasury Dashboard</h2>",
        unsafe_allow_html=True
    )
with h3:
    st.caption(datetime.now().strftime("Last refresh: %Y-%m-%d %H:%M:%S"))
    if st.button("Refresh", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.markdown("")

# -----------------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------------
notes = []

try:
    df_bal_raw = read_csv(LINKS["BANK BALANCE"])
except Exception as e:
    notes.append(f"BANK BALANCE load error: {e}")
    df_bal_raw = pd.DataFrame()

try:
    df_pay_raw = read_csv(LINKS["SUPPLIER PAYMENTS"])
except Exception as e:
    notes.append(f"SUPPLIER PAYMENTS load error: {e}")
    df_pay_raw = pd.DataFrame()

try:
    df_lc_raw = read_csv(LINKS["SETTLEMENTS"])
except Exception as e:
    notes.append(f"SETTLEMENTS load error: {e}")
    df_lc_raw = pd.DataFrame()

try:
    df_fm_raw = read_csv(LINKS["Fund Movement"])
except Exception as e:
    notes.append(f"Fund Movement load error: {e}")
    df_fm_raw = pd.DataFrame()

# -----------------------------------------------------------------------------
# PARSE DATA
# -----------------------------------------------------------------------------
df_by_bank, bal_date = pd.DataFrame(), None
if not df_bal_raw.empty:
    try:
        df_by_bank, bal_date = parse_bank_balance(df_bal_raw)
    except Exception as e:
        notes.append(f"BANK BALANCE parse: {e}")

df_pay = pd.DataFrame()
if not df_pay_raw.empty:
    try:
        df_pay = parse_supplier_payments(df_pay_raw)
    except Exception as e:
        notes.append(f"SUPPLIER PAYMENTS parse: {e}")

df_lc = pd.DataFrame()
if not df_lc_raw.empty:
    try:
        df_lc = parse_settlements(df_lc_raw)
    except Exception as e:
        notes.append(f"SETTLEMENTS parse: {e}")

df_fm = pd.DataFrame()
if not df_fm_raw.empty:
    try:
        df_fm = parse_fund_movement(df_fm_raw)
    except Exception as e:
        notes.append(f"Fund Movement parse: {e}")

for n in notes:
    st.warning(n)

# -----------------------------------------------------------------------------
# KPIs
# -----------------------------------------------------------------------------
total_balance = df_by_bank["balance"].sum() if not df_by_bank.empty else 0.0
banks_cnt     = df_by_bank["bank"].nunique() if not df_by_bank.empty else 0

today0 = pd.Timestamp.now(tz=TZ).normalize().tz_localize(None)
next4  = today0 + pd.Timedelta(days=3)

lc_next4_sum = (
    df_lc.loc[df_lc["settlement_date"].between(today0, next4), "amount"].sum()
    if not df_lc.empty else 0.0
)
approved_sum  = df_pay["amount"].sum() if not df_pay.empty else 0.0

k1, k2, k3, k4 = st.columns(4)
with k1:
    kpi_card("Total Balance", total_balance, bg="#E6F0FF", border="#C7D8FE", text="#1E3A8A")
with k2:
    kpi_card("Approved Payments", approved_sum, bg="#E9FFF2", border="#C7F7DD", text="#065F46")
with k3:
    kpi_card("LC due (next 4 days)", lc_next4_sum, bg="#FFF7E6", border="#FDE9C8", text="#92400E")
with k4:
    kpi_card("Banks", banks_cnt, bg="#FFF1F2", border="#FBD5D8", text="#9F1239")

st.markdown("---")

# -----------------------------------------------------------------------------
# BANK BALANCES — CARDS (NO TABLES/CHARTS)
# -----------------------------------------------------------------------------
st.subheader("Bank Balances — Cards")
if bal_date:
    st.caption(f"As of {bal_date}")

if df_by_bank.empty:
    st.info("No balances found.")
else:
    cards_df = df_by_bank.sort_values("balance", ascending=False).reset_index(drop=True)
    cols_per_row = 4
    for i in range(0, len(cards_df), cols_per_row):
        row = cards_df.iloc[i:i+cols_per_row]
        cols = st.columns(cols_per_row)
        for col, (_, rec) in zip(cols, row.iterrows()):
            with col:
                bank_card(str(rec["bank"]), float(rec["balance"]))

st.markdown("---")

# -----------------------------------------------------------------------------
# SUPPLIER PAYMENTS — APPROVED ONLY
# -----------------------------------------------------------------------------
st.header("Approved List")
if df_pay.empty:
    st.info("No Approved payments in sheet.")
else:
    banks = sorted(df_pay["bank"].dropna().unique())
    sel = st.multiselect("Filter by Bank", banks, default=banks, label_visibility="collapsed")

    view = df_pay[df_pay["bank"].isin(sel)].copy()
    if view.empty:
        st.info("No rows after filter.")
    else:
        # Sum table by bank (Amount right-aligned)
        st.markdown("**Sum by Bank**")
        grp = view.groupby("bank", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
        grp = grp.rename(columns={"bank": "Bank", "amount": "Amount"})
        grp_style = grp.style.format({"Amount": "{:,.2f}"}).set_properties(subset=["Amount"], **{"text-align": "right"})
        st.dataframe(grp_style, use_container_width=True, height=220)

        # Detail rows (Approved) — Amount right aligned
        st.markdown("**Approved List (rows)**")
        show_cols = [c for c in ["bank", "supplier", "currency", "amount", "status"] if c in view.columns]
        v = view[show_cols].rename(columns={
            "bank": "Bank", "supplier": "Supplier", "currency": "Curr", "amount": "Amount", "status": "Status"
        })
        v_style = v.style.format({"Amount": "{:,.2f}"}).set_properties(subset=["Amount"], **{"text-align": "right"})
        st.dataframe(v_style, use_container_width=True, height=360)

st.markdown("---")

# -----------------------------------------------------------------------------
# LC SETTLEMENTS — PENDING ONLY (NO SECOND LIST, NO DATE INPUT)
# -----------------------------------------------------------------------------
st.header("LC Settlements — Pending Only")
if df_lc.empty:
    st.info("No LC (Pending) data. Ensure sheet has Bank, Maturity Date/New Maturity Date, and an Amount column.")
else:
    # full available date window; hide controls
    dmin, dmax = df_lc["settlement_date"].min().date(), df_lc["settlement_date"].max().date()
    d1, d2 = pd.to_datetime(dmin), pd.to_datetime(dmax) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    lc_view = df_lc.loc[df_lc["settlement_date"].between(d1, d2)].copy()

    c1, c2 = st.columns(2)
    with c1:
        kpi_card("LC Amount (filtered sum)", lc_view["amount"].sum(), bg="#FFF7E6", border="#FDE9C8", text="#92400E")
    with c2:
        kpi_card("LC Items (count)", len(lc_view), bg="#E6F0FF", border="#C7D8FE", text="#1E3A8A")

    if lc_view.empty:
        st.info("No LC items in range.")
    else:
        viz = lc_view.copy()
        viz["settlement_date"] = pd.to_datetime(viz["settlement_date"]).dt.strftime(DATE_FMT)
        viz = viz.rename(columns={
            "bank": "Bank", "type": "Type", "status": "Status",
            "settlement_date": "Settlement Date", "amount": "Amount",
            "remark": "Remark", "description": "Description",
        })
        viz = viz[["Bank", "Type", "Status", "Settlement Date", "Amount", "Remark", "Description"]]
        viz_style = viz.style.format({"Amount": "{:,.2f}"}).set_properties(subset=["Amount"], **{"text-align": "right"})
        st.dataframe(viz_style, use_container_width=True, height=380)

st.markdown("---")

# -----------------------------------------------------------------------------
# LIQUIDITY TREND
# -----------------------------------------------------------------------------
st.header("Liquidity Trend")
if df_fm.empty:
    st.info("No Fund Movement data (need Date + Total Liquidity).")
else:
    try:
        import plotly.express as px
        fig = px.line(df_fm, x="date", y="total_liquidity", title="Total Liquidity Over Time")
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=360, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.line_chart(df_fm.set_index("date")["total_liquidity"])

# -----------------------------------------------------------------------------
# QUICK INSIGHTS
# -----------------------------------------------------------------------------
st.header("Quick Insights")
ins = []
if not df_by_bank.empty:
    top = df_by_bank.sort_values("balance", ascending=False).head(1)
    if not top.empty:
        ins.append(f"Top balance: **{top.iloc[0]['bank']}** ({fmt_num(top.iloc[0]['balance'])}).")
if not df_pay.empty:
    byb = df_pay.groupby("bank")["amount"].sum().sort_values(ascending=False)
    if len(byb) > 0:
        ins.append(f"Highest approved payments bank: **{byb.index[0]}** ({fmt_num(byb.iloc[0])}).")
if not df_lc.empty:
    today0 = pd.Timestamp.now(tz=TZ).normalize().tz_localize(None)
    next4 = today0 + pd.Timedelta(days=3)
    next4_sum = df_lc.loc[df_lc['settlement_date'].between(today0, next4), 'amount'].sum()
    ins.append(f"Pending LC due in next 4 days: **{fmt_num(next4_sum)}**.")
if not df_fm.empty:
    latest = df_fm.dropna().sort_values("date").iloc[-1]["total_liquidity"]
    ins.append(f"Latest total liquidity: **{fmt_num(latest)}**.")

if ins:
    for i in ins:
        st.markdown(f"- {i}")
else:
    st.info("Insights will appear once data is available.")

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:grey;'>Created By <b>Jaseer Pykarathodi</b></p>",
    unsafe_allow_html=True,
)
