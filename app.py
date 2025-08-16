# app.py — Treasury Dashboard (Google Sheets CSV ONLY, no currency symbol)
# - Always reads published-to-web CSVs from Google Sheets
# - No data-source selector is shown
# - No SAR symbol; amounts shown as plain numbers
# - Includes refresh controls, stale-bank highlighting, quick insights

import io, time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Treasury Dashboard", layout="wide")

# ======================
# CONFIG (EDIT THESE)
# ======================
COMPANY_NAME = "Issam Kaabani Partners"
APP_TITLE = f"{COMPANY_NAME} — Treasury Dashboard"
DATE_FMT = "%Y-%m-%d"
TZ = "Asia/Riyadh"

# 👉 Put your Google Sheet CSV links here.
# You can use one link (auto-detects which table it is),
# or provide separate links for each section.
CSV_URLS = [
    # Your single link (auto-detects):
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vRNHOjAIFJLwopq80pyrB8FY55pnQSkPLhu0kKYrMsOrHEqQmnL8ZKfMLUSxgaGww/pub?output=csv",
    # If you later publish more tabs as CSV, add them here as new strings.
]

# ======================
# HELPERS (formatting & UI)
# ======================
def fmt_money(v):
    """Show plain numbers (no currency symbol)."""
    try:
        return f"{float(v):,.2f}"
    except Exception:
        return str(v)

def light_card(title: str, value, sub: str | None = None, bg="#F8FAFF", border="#E6ECFF"):
    st.markdown(f"""
    <div style="
        background:{bg}; border:1px solid {border};
        border-radius:16px; padding:16px 20px;
        box-shadow:0 2px 10px rgba(0,0,0,.04); height:108px">
      <div style="font-size:12px;color:#556;text-transform:uppercase;letter-spacing:.08em">{title}</div>
      <div style="font-size:28px;font-weight:700;margin-top:6px">
        {fmt_money(value) if isinstance(value,(int,float,np.integer,np.floating)) else value}
      </div>
      <div style="font-size:12px;color:#667;margin-top:4px">{sub or ""}</div>
    </div>
    """, unsafe_allow_html=True)

def bank_cards(by_bank_balances: pd.DataFrame, stale_banks: set[str] | None = None, cols_per_row=4):
    if by_bank_balances.empty:
        st.warning("No balances found.")
        return
    stale_banks = stale_banks or set()
    pastel = ["#F8FAFF", "#FFF8FA", "#F8FFF9", "#FFFDF8"]
    borders = ["#E6ECFF", "#FFE0EA", "#DFF3E8", "#F2EAD3"]
    recs = by_bank_balances.sort_values("balance", ascending=False).to_dict("records")
    for i in range(0, len(recs), cols_per_row):
        row = recs[i:i+cols_per_row]
        cols = st.columns(len(row))
        for j, (col, rec) in enumerate(zip(cols, row)):
            with col:
                idx = (i + j) % len(pastel)
                comment = "📝 not updated this refresh" if rec["bank"] in stale_banks else None
                st.markdown(f"""
                <div style="
                    background:{pastel[idx]}; border:1px solid {borders[idx]};
                    border-radius:16px; padding:16px 20px;
                    box-shadow:0 2px 10px rgba(0,0,0,.04);">
                  <div style="font-size:12px;color:#556;text-transform:uppercase;letter-spacing:.08em">{rec["bank"]}</div>
                  <div style="font-size:28px;font-weight:700;margin-top:6px">{fmt_money(rec["balance"])}</div>
                  <div style="font-size:12px;color:#667;margin-top:4px">Available balance</div>
                  {f'<div style="font-size:12px;color:#999;margin-top:6px">{comment}</div>' if comment else ''}
                </div>
                """, unsafe_allow_html=True)

# Compatibility helper for rerun (Streamlit >=1.26 uses st.rerun)
try:
    _rerun = st.rerun
except AttributeError:
    _rerun = st.experimental_rerun

# ======================
# LOADERS
# ======================
@st.cache_data(ttl=300)
def load_csv_url(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

def _cols_lower(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out

def detect_table_kind(df: pd.DataFrame) -> str:
    """
    Return one of {"UNT A","SUPPLIER PAYMENTS","SETTLEMENTS","Fund Movement","UNKNOWN"}.
    """
    d = _cols_lower(df)
    cols = set(d.columns)
    if {"date"} <= cols and any("total liquidity" in c for c in d.columns): return "Fund Movement"
    if any(c.startswith("bank") for c in d.columns) and \
       (any("new maturity" in c for c in d.columns) or any("maturity date" in c for c in d.columns)) and \
       any(("amount (sar" in c) or (c == "amount") or ("amount_sar" in c) for c in d.columns): return "SETTLEMENTS"
    if any(c.startswith("bank") for c in d.columns) and \
       any("status" in c for c in d.columns) and \
       any(c == "amount" or "amount (sar" in c or "amount_sar" in c for c in d.columns): return "SUPPLIER PAYMENTS"
    parsed_dates = pd.to_datetime(pd.Index(df.columns), errors="coerce")
    if pd.notna(parsed_dates).sum() >= 1:
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        if obj_cols: return "UNT A"
    return "UNKNOWN"

# ======================
# PARSERS (match your sheets)
# ======================
def parse_unt_a(df: pd.DataFrame) -> tuple[pd.DataFrame, datetime | None]:
    df = df.copy().dropna(how="all").dropna(axis=1, how="all")
    bank_col = None
    for c in df.columns:
        if df[c].dtype == object and (df[c].dropna().astype(str).str.strip() != "").sum() >= 3:
            bank_col = c; break
    if bank_col is None: raise ValueError('[UNT A] Could not detect "Bank" column.')

    parsed_dates = pd.to_datetime(pd.Index(df.columns), errors="coerce")
    date_cols = [col for col, d in zip(df.columns, parsed_dates) if pd.notna(d)]
    if not date_cols: raise ValueError('[UNT A] No date columns found in header.')
    date_map = {col: pd.to_datetime(col, errors="coerce") for col in date_cols}
    latest_col = max(date_cols, key=lambda c: date_map[c])

    bank_series = df[bank_col].astype(str).str.strip()
    mask_bank = bank_series.ne("") & ~bank_series.str.contains("available|total", case=False, na=False)
    sub = df.loc[mask_bank, [bank_col, latest_col]].copy()
    sub.columns = ["bank", "balance"]
    sub["balance"] = pd.to_numeric(sub["balance"], errors="coerce"); sub = sub.dropna(subset=["balance"])
    sub["bank"] = (sub["bank"].str.replace(r"\s*-\s*.*$", "", regex=True)
                              .str.replace(r"\s*CURRENT A/C.*$", "", regex=True)
                              .str.strip())
    by_bank = sub.groupby("bank", as_index=False)["balance"].sum().sort_values("balance", ascending=False)
    latest_date = date_map[latest_col].date() if pd.notna(date_map[latest_col]) else None
    return by_bank, latest_date

def parse_payments(df: pd.DataFrame) -> pd.DataFrame:
    d = _cols_lower(df)
    col_bank = next((c for c in d.columns if c.startswith("bank")), None)
    col_amt_sar = next((c for c in d.columns if "amount(sar" in c or "amount (sar" in c or "amount_sar" in c), None)
    col_amt = col_amt_sar or next((c for c in d.columns if c == "amount"), None)
    col_status = next((c for c in d.columns if "status" in c), None)
    col_due = next((c for c in d.columns if "due" in c or "payment date" in c or "expected" in c or "schedule" in c), None)
    col_date = next((c for c in d.columns if c == "date"), None)
    if not col_bank or not col_status or not col_amt:
        raise ValueError('[SUPPLIER PAYMENTS] Need Bank, Status, and Amount/Amount(SAR).')
    out = pd.DataFrame({
        "bank": d[col_bank].astype(str).str.strip(),
        "status_raw": d[col_status].astype(str).str.strip(),
        "amount": pd.to_numeric(d[col_amt], errors="coerce"),
        "date": pd.to_datetime(d[col_date], errors="coerce") if col_date else pd.NaT,
        "due_date": pd.to_datetime(d[col_due], errors="coerce") if col_due else pd.NaT,
        "description": ""
    })
    s = out["status_raw"].str.lower()
    out["status"] = np.select([s.str.contains("release"), s.str.contains("reject")], ["Approved", "Rejected"], default="Pending")
    out = out.drop(columns=["status_raw"]).dropna(subset=["bank", "amount"])
    return out

def parse_settlements(df: pd.DataFrame) -> pd.DataFrame:
    d = _cols_lower(df)
    col_bank = next((c for c in d.columns if c.startswith("bank")), None)
    col_amt_sar = next((c for c in d.columns if "amount(sar" in c or "amount (sar" in c or "amount_sar" in c), None)
    col_amt = col_amt_sar or next((c for c in d.columns if c == "amount"), None)
    col_maturity_new = next((c for c in d.columns if "new maturity" in c), None)
    col_maturity = col_maturity_new or next((c for c in d.columns if "maturity date" in c), None)
    col_type = next((c for c in d.columns if "type" in c), None)
    col_status = next((c for c in d.columns if "status" in c), None)
    if not col_bank or not col_amt or not col_maturity:
        raise ValueError('[SETTLEMENTS] Need Bank, Amount/Amount(SAR), and (New) Maturity Date.')
    out = pd.DataFrame({
        "bank": d[col_bank].astype(str).str.strip(),
        "settlement_date": pd.to_datetime(d[col_maturity], errors="coerce"),
        "amount": pd.to_numeric(d[col_amt], errors="coerce"),
        "type": d[col_type].astype(str).str.upper().str.strip() if col_type else "",
        "status": d[col_status].astype(str).str.title().str.strip() if col_status else "",
        "description": ""
    }).dropna(subset=["bank", "amount", "settlement_date"])
    return out

def parse_fund_movement(df: pd.DataFrame) -> pd.DataFrame:
    d = _cols_lower(df)
    col_date = next((c for c in d.columns if c == "date"), None)
    col_total = next((c for c in d.columns if "total liquidity" in c), None)
    if not col_date or not col_total:
        raise ValueError('[Fund Movement] Need Date and Total Liquidity.')
    out = pd.DataFrame({
        "date": pd.to_datetime(d[col_date], errors="coerce"),
        "total_liquidity": pd.to_numeric(d[col_total], errors="coerce")
    }).dropna().sort_values("date")
    return out

# ======================
# HEADER (no data-source UI)
# ======================
st.title(APP_TITLE)

# Header chips (last refresh only)
last_refresh = pd.Timestamp.now(tz=TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
st.markdown(f"""
<span style="display:inline-block;background:#F1F5FF;border:1px solid #E0E7FF;color:#1E3A8A;
             padding:4px 10px;border-radius:999px;font-size:12px;">Last refresh: {last_refresh}</span>
""", unsafe_allow_html=True)

# ======================
# REFRESH (keep in sidebar)
# ======================
st.sidebar.header("Refresh")
if st.sidebar.button("Refresh now"):
    st.cache_data.clear()
    _rerun()

auto = st.sidebar.checkbox("Auto refresh", value=False)
interval = st.sidebar.number_input("Interval (seconds)", min_value=15, max_value=600, value=60, step=15)
if auto:
    time.sleep(int(interval))
    _rerun()

# Preferences
st.sidebar.header("Preferences")
stale_flag_enabled = st.sidebar.checkbox("Highlight non-updated banks on refresh", value=True)

# ======================
# FETCH & DETECT TABLES (CSV ONLY)
# ======================
dfs_raw = {}
errors = []
for u in CSV_URLS:
    try:
        dfu = load_csv_url(u.strip())
        kind = detect_table_kind(dfu)
        if kind in {"UNT A","SUPPLIER PAYMENTS","SETTLEMENTS","Fund Movement"}:
            dfs_raw[kind] = dfu if kind not in dfs_raw else pd.concat([dfs_raw[kind], dfu], ignore_index=True)
        else:
            errors.append(f"Could not auto-detect table type for: {u}")
    except Exception as e:
        errors.append(f"CSV error for {u}: {e}")

if errors:
    with st.expander("Notes / parsing messages"):
        for m in errors: st.write("• " + m)

if not dfs_raw:
    st.info("No CSV data loaded. Ensure your Google Sheet is 'Publish to the web → CSV' and the URL(s) are in CSV_URLS.")
    st.stop()

# ======================
# PARSE
# ======================
by_bank, latest_balance_date = pd.DataFrame(), None
payments = pd.DataFrame()
settlements = pd.DataFrame()
fund_mv = pd.DataFrame()

if "UNT A" in dfs_raw:
    by_bank, latest_balance_date = parse_unt_a(dfs_raw["UNT A"])
if "SUPPLIER PAYMENTS" in dfs_raw:
    payments = parse_payments(dfs_raw["SUPPLIER PAYMENTS"])
if "SETTLEMENTS" in dfs_raw:
    settlements = parse_settlements(dfs_raw["SETTLEMENTS"])
if "Fund Movement" in dfs_raw:
    fund_mv = parse_fund_movement(dfs_raw["Fund Movement"])

# ======================
# STALE BANK DETECTION
# ======================
stale_banks: set[str] = set()
if stale_flag_enabled and not by_bank.empty:
    if "prev_by_bank" not in st.session_state:
        st.session_state.prev_by_bank = {}
    prev = st.session_state.prev_by_bank
    current = {r["bank"]: float(r["balance"]) for r in by_bank.to_dict("records")}
    tol = 0.0001
    for b, bal in current.items():
        if b in prev and abs(prev[b] - bal) <= tol:
            stale_banks.add(b)
    st.session_state.prev_by_bank = current

# ======================
# KPIs
# ======================
total_balance = by_bank["balance"].sum() if not by_bank.empty else 0.0
bank_cnt = by_bank["bank"].nunique() if not by_bank.empty else 0

now_local = pd.Timestamp.now(tz=TZ).normalize()
up4_end = now_local + pd.Timedelta(days=3)
lc_up4 = settlements.loc[
    (settlements["settlement_date"] >= now_local.tz_localize(None)) &
    (settlements["settlement_date"] <= up4_end.tz_localize(None))
] if not settlements.empty else pd.DataFrame()
lc_up4_sum = lc_up4["amount"].sum() if not lc_up4.empty else 0.0

c1, c2, c3, c4 = st.columns(4)
with c1: light_card("Total Balance", total_balance, f"As of {latest_balance_date}" if latest_balance_date else None, bg="#F8FAFF", border="#E6ECFF")
with c2: light_card("Pending Payments", payments.loc[payments["status"]=="Pending", "amount"].sum() if not payments.empty else 0.0, bg="#FFF8FA", border="#FFE0EA")
with c3: light_card("LC due (next 4 days)", lc_up4_sum, bg="#F8FFF9", border="#DFF3E8")
with c4: light_card("Banks", bank_cnt, bg="#FFFDF8", border="#F2EAD3")

# ======================
# BANK BALANCES
# ======================
st.markdown("### Bank Balances (Cards)")
if not by_bank.empty:
    bank_cards(by_bank, stale_banks=stale_banks)
    if stale_flag_enabled and stale_banks:
        st.caption("📝 Banks marked 'not updated this refresh' kept the same balance as last time you refreshed.")
else:
    st.info("No balances detected for 'UNT A' data.")

st.markdown("---")

# ======================
# PAYMENTS
# ======================
st.header("Approved / Pending / Rejected Payments")
if payments.empty:
    st.info("No payments data detected.")
else:
    p1, p2, p3, p4 = st.columns([1.2, 1.2, 1.2, 1])
    with p1:
        pay_banks = sorted(payments["bank"].unique())
        pay_bank_sel = st.multiselect("Bank (Payments)", pay_banks, default=pay_banks)
    with p2:
        status_opts = ["Approved", "Pending", "Rejected"]
        status_sel = st.multiselect("Status", status_opts, default=status_opts)
    with p3:
        if payments["due_date"].notna().any():
            pmin = payments["due_date"].dropna().min().date()
            pmax = payments["due_date"].dropna().max().date()
            pay_rng = st.date_input("Due Date range", (pmin, pmax))
            pd1, pd2 = pd.to_datetime(pay_rng[0]), pd.to_datetime(pay_rng[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            mdate = payments["due_date"].between(pd1, pd2)
        elif payments["date"].notna().any():
            pmin = payments["date"].dropna().min().date()
            pmax = payments["date"].dropna().max().date()
            pay_rng = st.date_input("Date range", (pmin, pmax))
            pd1, pd2 = pd.to_datetime(pay_rng[0]), pd.to_datetime(pay_rng[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            mdate = payments["date"].between(pd1, pd2)
        else:
            pay_rng = None; mdate = True
    with p4:
        if st.button("Clear payment filters"):
            pay_bank_sel, status_sel = pay_banks, status_opts

    mpay = (payments["bank"].isin(pay_bank_sel)) & (payments["status"].isin(status_sel)) & (mdate if isinstance(mdate, pd.Series) else True)
    df_pay_f = payments.loc[mpay].copy()

    s_app = df_pay_f.loc[df_pay_f["status"]=="Approved", "amount"].sum()
    s_pen = df_pay_f.loc[df_pay_f["status"]=="Pending", "amount"].sum()
    s_rej = df_pay_f.loc[df_pay_f["status"]=="Rejected", "amount"].sum()
    s_cnt = len(df_pay_f)

    pc1, pc2, pc3, pc4 = st.columns(4)
    with pc1: light_card("Approved (sum)", s_app, bg="#F8FFF9", border="#DFF3E8")
    with pc2: light_card("Pending (sum)", s_pen, bg="#FFF8FA", border="#FFE0EA")
    with pc3: light_card("Rejected (sum)", s_rej, bg="#FFF8FA", border="#FFE0EA")
    with pc4: light_card("Payments (count)", s_cnt, bg="#F8FAFF", border="#E6ECFF")

    show_cols = [c for c in ["date", "bank", "status", "amount", "due_date", "description"] if c in df_pay_f.columns]
    viz = df_pay_f.copy()
    for c in ["date", "due_date"]:
        if c in viz.columns:
            viz[c] = pd.to_datetime(viz[c], errors="coerce").dt.strftime(DATE_FMT)
    if "amount" in viz.columns:
        viz["amount"] = viz["amount"].map(fmt_money)
    st.dataframe(
        viz[show_cols].sort_values(by=[c for c in ["due_date","date"] if c in show_cols], ascending=True),
        use_container_width=True, height=320
    )

st.markdown("---")

# ======================
# LC SETTLEMENTS
# ======================
st.header("LC Settlements")
if settlements.empty:
    st.info("No LC data detected.")
else:
    l1, l2, l3, l4 = st.columns([1.2, 1.2, 1.2, 1])
    with l1:
        lc_banks = sorted(settlements["bank"].unique())
        lc_bank_sel = st.multiselect("Bank (LC)", lc_banks, default=lc_banks)
    with l2:
        types = sorted([t for t in settlements["type"].unique() if isinstance(t, str) and t.strip() != ""])
        lc_type_sel = st.multiselect("Type", types if types else [], default=types if types else [])
    with l3:
        statuses = sorted([s for s in settlements["status"].unique() if isinstance(s, str) and s.strip() != ""])
        lc_status_sel = st.multiselect("Status", statuses if statuses else [], default=statuses if statuses else [])
    with l4:
        lmin = settlements["settlement_date"].min().date()
        lmax = settlements["settlement_date"].max().date()
        lc_rng = st.date_input("Settlement Date range", (lmin, lmax))

    lm1, lm2 = pd.to_datetime(lc_rng[0]), pd.to_datetime(lc_rng[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    m = settlements["bank"].isin(lc_bank_sel) & settlements["settlement_date"].between(lm1, lm2)
    if lc_type_sel:   m &= settlements["type"].isin(lc_type_sel)
    if lc_status_sel: m &= settlements["status"].isin(lc_status_sel)

    df_lc_f = settlements.loc[m].copy()

    lc_sum = df_lc_f["amount"].sum()
    lc_cnt = len(df_lc_f)
    lc1c, lc2c = st.columns(2)
    with lc1c: light_card("LC Amount (filtered sum)", lc_sum, bg="#FFFDF8", border="#F2EAD3")
    with lc2c: light_card("LC Items (count)", lc_cnt, bg="#F8FAFF", border="#E6ECFF")

    viz = df_lc_f.copy()
    viz["settlement_date"] = pd.to_datetime(viz["settlement_date"]).dt.strftime(DATE_FMT)
    viz["amount"] = viz["amount"].map(fmt_money)
    show_cols = [c for c in ["bank", "type", "status", "settlement_date", "amount", "description"] if c in viz.columns]
    st.dataframe(viz[show_cols].sort_values("settlement_date"), use_container_width=True, height=320)

    st.subheader("Coming 4 Days Settlements")
    start = now_local.tz_localize(None); end = (now_local + pd.Timedelta(days=3)).tz_localize(None)
    upcoming = settlements.loc[settlements["bank"].isin(lc_bank_sel) & settlements["settlement_date"].between(start, end)].copy()
    up_sum = upcoming["amount"].sum() if not upcoming.empty else 0.0
    uc1, uc2 = st.columns(2)
    with uc1: light_card("Upcoming 4-day Amount", up_sum, bg="#F8FFF9", border="#DFF3E8")
    with uc2: light_card("Upcoming 4-day Items", int(len(upcoming)), bg="#FFF8FA", border="#FFE0EA")
    if not upcoming.empty:
        upcoming["settlement_date"] = pd.to_datetime(upcoming["settlement_date"]).dt.strftime(DATE_FMT)
        upcoming["amount"] = upcoming["amount"].map(fmt_money)
        show_cols2 = [c for c in ["bank", "type", "status", "settlement_date", "amount", "description"] if c in upcoming.columns]
        st.dataframe(upcoming[show_cols2].sort_values("settlement_date"), use_container_width=True)

st.markdown("---")

# ======================
# LIQUIDITY TREND
# ======================
st.header("Liquidity Trend")
if fund_mv.empty:
    st.info("No Fund Movement data detected (need Date, Total Liquidity).")
else:
    try:
        import plotly.express as px
        fig = px.line(fund_mv, x="date", y="total_liquidity", title="Total Liquidity Over Time")
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=380, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.line_chart(fund_mv.set_index("date")["total_liquidity"])

# ======================
# QUICK INSIGHTS
# ======================
st.header("Quick Insights")
ins = []
if not by_bank.empty:
    top3 = by_bank.head(3)
    if not top3.empty:
        ins.append("Top balances: " + ", ".join([f"**{r.bank}** ({fmt_money(r.balance)})" for _, r in top3.iterrows()]))
if not payments.empty and "due_date" in payments.columns:
    today0 = now_local.tz_localize(None)
    nxt7 = payments.loc[payments["due_date"].between(today0, today0 + pd.Timedelta(days=7))]
    if not nxt7.empty:
        by_bank_p = nxt7.groupby("bank")["amount"].sum().sort_values(ascending=False)
        ins.append(f"Payments due next 7 days — top bank: **{by_bank_p.index[0]}** ({fmt_money(by_bank_p.iloc[0])})")
if not settlements.empty:
    lc7 = settlements.loc[settlements["settlement_date"].between(now_local.tz_localize(None), (now_local + pd.Timedelta(days=7)).tz_localize(None))]
    if not lc7.empty:
        by_bank_lc = lc7.groupby("bank")["amount"].sum().sort_values(ascending=False)
        ins.append(f"LC due next 7 days — top bank: **{by_bank_lc.index[0]}** ({fmt_money(by_bank_lc.iloc[0])})")
if ins:
    for tip in ins: st.markdown(f"- {tip}")
else:
    st.info("Insights will appear after data loads.")

# ======================
# FOOTER
# ======================
st.caption("Data source: Google Sheets (CSV). Use the sidebar to refresh. Light cards • No currency symbol • Stale-bank detector • Quick Insights")
