# Treasury Dashboard — Google Sheets
# - Supplier Payments: parser fix + Approved-only
# - Bank Balances: table view (no cards)
# - LC Settlements: Pending-only + Remark panel
# File: app.py

import io, time, re
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Treasury Dashboard", layout="wide")

# ---------------------------------------------------------------------------
# YOUR GOOGLE SHEET (same file id, different gids per tab)
# ---------------------------------------------------------------------------
FILE_ID = "1371amvaCbejUWVJI_moWdIchy5DF1lPO"
LINKS = {
    "BANK BALANCE":      f"https://docs.google.com/spreadsheets/d/{FILE_ID}/export?format=csv&gid=860709395",
    "SUPPLIER PAYMENTS": f"https://docs.google.com/spreadsheets/d/{FILE_ID}/export?format=csv&gid=20805295",
    "SETTLEMENTS":       f"https://docs.google.com/spreadsheets/d/{FILE_ID}/export?format=csv&gid=978859477",
    "Fund Movement":     f"https://docs.google.com/spreadsheets/d/{FILE_ID}/export?format=csv&gid=66055663",
}

COMPANY = "Issam Kaabani Partners"
DATE_FMT = "%Y-%m-%d"
TZ = "Asia/Riyadh"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_number(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().replace(",", "")
    if s.endswith("%"): s = s[:-1]
    try: return float(s)
    except Exception: return np.nan

@st.cache_data(ttl=300)
def read_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=25)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

def cols_lower(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in df.columns]
    return out

def fmt_num(v):
    try: return f"{float(v):,.2f}"
    except: return str(v)

def kpi_card(title, value, subtitle="", bg="#EEF2FF", border="#C7D2FE", text="#111827"):
    st.markdown(
        f"""
        <div style="
            background:{bg};border:1px solid {border};
            border-radius:12px;padding:12px 14px;
            box-shadow:0 1px 6px rgba(0,0,0,.04);">
            <div style="font-size:12px;color:#374151;text-transform:uppercase;letter-spacing:.08em">{title}</div>
            <div style="font-size:26px;font-weight:800;color:{text};margin-top:2px">
                {fmt_num(value) if isinstance(value,(int,float,np.integer,np.floating)) else value}
            </div>
            <div style="font-size:12px;color:#6B7280;margin-top:2px">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
def parse_bank_balance(df: pd.DataFrame):
    """
    A) If columns 'Bank' and ('Amount' or 'Amount(SAR)') exist -> sum by bank.
    B) Otherwise, bank name column + date headers -> pick latest date column.
    """
    c = cols_lower(df)
    # Case A: simple bank/amount table
    if "bank" in c.columns and ("amount" in c.columns or "amount(sar)" in c.columns):
        amt_col = "amount" if "amount" in c.columns else "amount(sar)"
        out = pd.DataFrame({
            "bank": c["bank"].astype(str).str.strip(),
            "balance": c[amt_col].map(_to_number)
        }).dropna()
        latest_date = None
        by_bank = out.groupby("bank", as_index=False)["balance"].sum()
        return by_bank, latest_date

    # Case B: header contains dates
    raw = df.copy().dropna(how="all").dropna(axis=1, how="all")
    bank_col = None
    for col in raw.columns:
        if raw[col].dtype == object and (raw[col].dropna().astype(str).str.strip() != "").sum() >= 3:
            bank_col = col; break
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
    by_bank = sub.dropna().groupby("bank", as_index=False)["balance"].sum()
    return by_bank, latest_date

def parse_supplier_payments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supplier Payments:
      Bank, Supplier Name, Currency, Amount or Amount(SAR), Status
    Keep ONLY rows where Status contains 'Approved' (case-insensitive).
    """
    d = cols_lower(df).rename(columns={
        "supplier name": "supplier",
        "amount(sar)": "amount_sar",
        "order/sh/branch": "order_branch"
    })

    # Ensure columns exist
    if "bank" not in d.columns or "status" not in d.columns:
        return pd.DataFrame()

    # Robust status filtering (fix for Series.strip error)
    status_norm = d["status"].astype(str).str.strip().str.lower()
    approved_mask = status_norm.str.contains("approved")  # only "Approved"
    d = d.loc[approved_mask].copy()
    if d.empty:
        return pd.DataFrame()

    # Choose amount column
    amt_col = "amount_sar" if "amount_sar" in d.columns else ("amount" if "amount" in d.columns else None)
    if amt_col is None:
        return pd.DataFrame()

    d["amount"] = d[amt_col].map(_to_number)
    d["bank"] = d["bank"].astype(str).str.strip()

    # Keep tidy columns
    keep = [c for c in ["bank","supplier","currency","amount","status"] if c in d.columns]
    out = d[keep].dropna(subset=["amount"]).copy()
    # Re-title case the status for display
    if "status" in out.columns:
        out["status"] = out["status"].astype(str).str.title()
    return out

def parse_settlements(df: pd.DataFrame) -> pd.DataFrame:
    """
    LC Settlements (Pending only) with REMARK support.
    - Date column: prefer 'maturity date', else 'new maturity date'
    - Amount column priority: 'balance for settlement' > 'currently due' > 'amount(sar)' > 'amount'
    - Keep only 'status' == 'Pending'
    """
    d = cols_lower(df)

    # Bank
    bank = next((c for c in d.columns if c.startswith("bank")), None)

    # Date (priority)
    date_col = None
    for cand in d.columns:
        if "maturity" in cand and "new" not in cand:   # 'maturity date'
            date_col = cand; break
    if date_col is None:
        for cand in d.columns:
            if "new" in cand and "maturity" in cand:  # 'new maturity date'
                date_col = cand; break

    # Amount (priority)
    amount_col = None
    for cand in d.columns:
        if "balance" in cand and "settlement" in cand:
            amount_col = cand; break
    if amount_col is None:
        for cand in d.columns:
            if "currently" in cand and "due" in cand:
                amount_col = cand; break
    if amount_col is None:
        amount_col = "amount(sar)" if "amount(sar)" in d.columns else ("amount" if "amount" in d.columns else None)

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
        "description": ""
    })
    out = out.dropna(subset=["bank","amount","settlement_date"])
    out = out[out["status"].str.lower() == "pending"].copy()
    return out

def parse_fund_movement(df: pd.DataFrame) -> pd.DataFrame:
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

# ---------------------------------------------------------------------------
# Header & Refresh
# ---------------------------------------------------------------------------
st.title(f"{COMPANY} — Treasury Dashboard")
st.caption(pd.Timestamp.now(tz=TZ).strftime("Last refresh: %Y-%m-%d %H:%M:%S %Z"))

st.sidebar.header("Refresh")
if st.sidebar.button("Refresh now"):
    st.cache_data.clear()
    try: st.rerun()
    except: st.experimental_rerun()

auto = st.sidebar.checkbox("Auto refresh", value=False)
sec = st.sidebar.number_input("Interval (seconds)", 15, 600, 60, 15)
if auto:
    time.sleep(int(sec))
    try: st.rerun()
    except: st.experimental_rerun()

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
notes = []
try: df_bal_raw = read_csv(LINKS["BANK BALANCE"])
except Exception as e: notes.append(f"BANK BALANCE load error: {e}"); df_bal_raw = pd.DataFrame()

try: df_pay_raw = read_csv(LINKS["SUPPLIER PAYMENTS"])
except Exception as e: notes.append(f"SUPPLIER PAYMENTS load error: {e}"); df_pay_raw = pd.DataFrame()

try: df_lc_raw = read_csv(LINKS["SETTLEMENTS"])
except Exception as e: notes.append(f"SETTLEMENTS load error: {e}"); df_lc_raw = pd.DataFrame()

try: df_fm_raw = read_csv(LINKS["Fund Movement"])
except Exception as e: notes.append(f"Fund Movement load error: {e}"); df_fm_raw = pd.DataFrame()

# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------
df_by_bank, bal_date = pd.DataFrame(), None
if not df_bal_raw.empty:
    try: df_by_bank, bal_date = parse_bank_balance(df_bal_raw)
    except Exception as e: notes.append(f"BANK BALANCE parse: {e}")

df_pay = pd.DataFrame()
if not df_pay_raw.empty:
    try: df_pay = parse_supplier_payments(df_pay_raw)   # Approved ONLY
    except Exception as e: notes.append(f"SUPPLIER PAYMENTS parse: {e}")

df_lc = pd.DataFrame()
if not df_lc_raw.empty:
    try: df_lc = parse_settlements(df_lc_raw)           # Pending ONLY + REMARK
    except Exception as e: notes.append(f"SETTLEMENTS parse: {e}")

df_fm = pd.DataFrame()
if not df_fm_raw.empty:
    try: df_fm = parse_fund_movement(df_fm_raw)
    except Exception as e: notes.append(f"Fund Movement parse: {e}")

if notes:
    for n in notes: st.warning(n)

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
total_balance = df_by_bank["balance"].sum() if not df_by_bank.empty else 0.0
banks_cnt = df_by_bank["bank"].nunique() if not df_by_bank.empty else 0

today = pd.Timestamp.now(tz=TZ).normalize().tz_localize(None)
next4 = today + pd.Timedelta(days=3)

lc_next4_sum = (
    df_lc.loc[df_lc["settlement_date"].between(today, next4), "amount"].sum()
    if not df_lc.empty else 0.0
)
approved_sum = df_pay["amount"].sum() if not df_pay.empty else 0.0

c1, c2, c3, c4 = st.columns(4)
# ▼▼ change: removed "As of {bal_date}" subtitle
with c1: kpi_card("Total Balance", total_balance, "", bg="#E6F0FF", border="#C7D8FE", text="#1E3A8A")
with c2: kpi_card("Approved Payments (sum)", approved_sum, bg="#E9FFF2", border="#C7F7DD", text="#065F46")
with c3: kpi_card("LC due (next 4 days) • Pending", lc_next4_sum, bg="#FFF7E6", border="#FDE9C8", text="#92400E")
with c4: kpi_card("Banks", banks_cnt, bg="#FFF1F2", border="#FBD5D8", text="#9F1239")

# ---------------------------------------------------------------------------
# Bank Balances — TABLE VIEW
# ---------------------------------------------------------------------------
# ▼▼ change: removed "(Table)" wording
st.markdown("### Bank Balances")
if df_by_bank.empty:
    st.info("No balances found.")
else:
    df_bal_table = df_by_bank.copy()
    df_bal_table = df_bal_table.sort_values("balance", ascending=False)
    total = df_bal_table["balance"].sum()
    df_bal_table["share_%"] = (df_bal_table["balance"] / total * 100).round(2)
    df_bal_table["balance"] = df_bal_table["balance"].map(fmt_num)
    st.dataframe(df_bal_table.rename(columns={"bank":"Bank","balance":"Balance","share_%":"Share %"}),
                 use_container_width=True, height=320)
st.markdown("---")

# ---------------------------------------------------------------------------
# Supplier Payments (Approved only)
# ---------------------------------------------------------------------------
st.header("Approved Supplier Payments")
if df_pay.empty:
    st.info("No Approved payments in sheet.")
else:
    colf1, _ = st.columns([1.4, 1])
    with colf1:
        banks = sorted(df_pay["bank"].dropna().unique())
        pick_banks = st.multiselect("Filter by Bank", banks, default=banks)

    view = df_pay[df_pay["bank"].isin(pick_banks)].copy()

    grp = view.groupby("bank", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
    st.markdown("**Sum by Bank (Approved)**")
    st.dataframe(grp.assign(amount=grp["amount"].map(fmt_num)).rename(columns={"bank":"Bank","amount":"Amount"}),
                 use_container_width=True, height=220)

    st.markdown("**Approved rows**")
    show_cols = [c for c in ["bank","supplier","currency","amount","status"] if c in view.columns]
    v = view.copy(); v["amount"] = v["amount"].map(fmt_num)
    st.dataframe(v[show_cols].rename(columns={"bank":"Bank","supplier":"Supplier","currency":"Curr","amount":"Amount","status":"Status"}),
                 use_container_width=True, height=360)

st.markdown("---")

# ---------------------------------------------------------------------------
# LC Settlements — PENDING ONLY (with REMARKS)
# ---------------------------------------------------------------------------
# ▼▼ change: header text without "— Pending Only"
st.header("LC Settlements")
if df_lc.empty:
    st.info("No LC (Pending) data. Ensure sheet has Bank, Maturity Date/New Maturity Date, and any of: Balance for Settlement / Currently Due / Amount(SAR).")
else:
    # Filters
    l1, l2, l3 = st.columns([1.2, 1.2, 1])
    with l1:
        banks = sorted(df_lc["bank"].dropna().unique())
        sel_banks = st.multiselect("Bank", banks, default=banks)
    with l2:
        dmin, dmax = df_lc["settlement_date"].min().date(), df_lc["settlement_date"].max().date()
        rng = st.date_input("Settlement date range", (dmin, dmax))
    with l3:
        pass

    d1, d2 = pd.to_datetime(rng[0]), pd.to_datetime(rng[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    mask = df_lc["bank"].isin(sel_banks) & df_lc["settlement_date"].between(d1, d2)
    lc_view = df_lc.loc[mask].copy()

    lc_sum = lc_view["amount"].sum(); lc_cnt = len(lc_view)
    cc1, cc2 = st.columns(2)
    with cc1: kpi_card("LC Amount (filtered sum)", lc_sum, bg="#FFF7E6", border="#FDE9C8", text="#92400E")
    with cc2: kpi_card("LC Items (count)", lc_cnt, bg="#E6F0FF", border="#C7D8FE", text="#1E3A8A")

    if not lc_view.empty:
        viz = lc_view.copy()
        viz["settlement_date"] = pd.to_datetime(viz["settlement_date"]).dt.strftime(DATE_FMT)
        viz["amount"] = viz["amount"].map(fmt_num)
        show = [c for c in ["bank","type","status","settlement_date","amount","remark","description"] if c in viz.columns]
        st.dataframe(viz[show].rename(columns={
            "bank":"Bank","type":"Type","status":"Status","settlement_date":"Settlement Date","amount":"Amount","remark":"Remark","description":"Description"
        }).sort_values("Settlement Date"),
        use_container_width=True, height=360)

        # Remarks panel — rows where remark has text
        remarks = lc_view.loc[lc_view["remark"].astype(str).str.strip() != ""].copy()
        if not remarks.empty:
            st.subheader("Remarks")
            remarks = remarks.assign(
                settlement_date=remarks["settlement_date"].dt.strftime(DATE_FMT),
                amount=remarks["amount"].map(fmt_num)
            )[["settlement_date","bank","amount","remark"]].sort_values("settlement_date")
            remarks = remarks.rename(columns={"settlement_date":"Settlement Date","bank":"Bank","amount":"Amount","remark":"Remark"})
            st.dataframe(remarks, use_container_width=True, height=220)

st.markdown("---")

# ---------------------------------------------------------------------------
# Liquidity Trend
# ---------------------------------------------------------------------------
st.header("Liquidity Trend")
if df_fm.empty:
    st.info("No Fund Movement data (need Date + Total Liquidity).")
else:
    try:
        import plotly.express as px
        fig = px.line(df_fm, x="date", y="total_liquidity", title="Total Liquidity Over Time")
        fig.update_layout(margin=dict(l=20,r=20,t=50,b=20), height=360, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.line_chart(df_fm.set_index("date")["total_liquidity"])

# ---------------------------------------------------------------------------
# Quick Insights
# ---------------------------------------------------------------------------
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
    next4_sum = df_lc.loc[df_lc['settlement_date'].between(today0, next4),'amount'].sum()
    ins.append(f"Pending LC due in next 4 days: **{fmt_num(next4_sum)}**.")
if ins:
    for i in ins: st.markdown(f"- {i}")
else:
    st.info("Insights will appear once data is available.")
