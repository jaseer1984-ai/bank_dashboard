# Treasury Dashboard — Google Sheets ONLY (force-mapped), colored bank cards, no currency symbol

import io, time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Treasury Dashboard", layout="wide")

# -------------------------
# YOUR GOOGLE SHEET LINKS
# (force-map each section)
# -------------------------
RAW_LINKS = {
    # You shared these four; two are pubhtml (we convert them to CSV automatically)
    "UNT A": "https://docs.google.com/spreadsheets/d/e/2PACX-1vRNHOjAIFJLwopq80pyrB8FY55pnQSkPLhu0kKYrMsOrHEqQmnL8ZKfMLUSxgaGww/pubhtml",  # BANK BALANCE
    "SETTLEMENTS": "https://docs.google.com/spreadsheets/d/e/2PACX-1vRNHOjAIFJLwopq80pyrB8FY55pnQSkPLhu0kKYrMsOrHEqQmnL8ZKfMLUSxgaGww/pub?output=csv",
    "SUPPLIER PAYMENTS": "https://docs.google.com/spreadsheets/d/e/2PACX-1vRNHOjAIFJLwopq80pyrB8FY55pnQSkPLhu0kKYrMsOrHEqQmnL8ZKfMLUSxgaGww/pub?output=csv",
    "Fund Movement": "https://docs.google.com/spreadsheets/d/e/2PACX-1vRNHOjAIFJLwopq80pyrB8FY55pnQSkPLhu0kKYrMsOrHEqQmnL8ZKfMLUSxgaGww/pubhtml",
}

COMPANY_NAME = "Issam Kaabani Partners"
APP_TITLE = f"{COMPANY_NAME} — Treasury Dashboard"
DATE_FMT = "%Y-%m-%d"
TZ = "Asia/Riyadh"

# ---- helpers ----
def to_csv_url(u: str) -> str:
    # Accepts pubhtml or pub?output=csv; normalizes to CSV
    if u.endswith("pubhtml"):
        return u.rsplit("/", 1)[0] + "/pub?output=csv"
    return u

def fmt_money(v):
    # Plain numbers (no SAR)
    try:
        return f"{float(v):,.2f}"
    except Exception:
        return str(v)

def colored_card(title: str, value, subtitle: str = "", bg="#EEF2FF", border="#C7D2FE", text="#111827"):
    st.markdown(f"""
    <div style="
      background:{bg}; border:1px solid {border};
      border-radius:18px; padding:18px 20px;
      box-shadow:0 2px 10px rgba(0,0,0,.05);">
      <div style="font-size:12px;color:#374151;text-transform:uppercase;letter-spacing:.08em">{title}</div>
      <div style="font-size:30px;font-weight:800;color:{text};margin-top:6px">
        {fmt_money(value) if isinstance(value,(int,float,np.integer,np.floating)) else value}
      </div>
      <div style="font-size:12px;color:#6B7280;margin-top:4px">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

def bank_cards_colorful(by_bank: pd.DataFrame, stale: set[str] | None = None):
    if by_bank.empty:
        st.info("No balances detected.")
        return
    stale = stale or set()
    # Nice palette (bg, border, text)
    palette = [
        ("#E6FFF2", "#C7F7DD", "#065F46"),  # green
        ("#E6F0FF", "#C7D8FE", "#1E3A8A"),  # indigo
        ("#FFF1F2", "#FBD5D8", "#9F1239"),  # rose
        ("#FFF7E6", "#FDE9C8", "#92400E"),  # amber
        ("#E6FBFF", "#C7F3FE", "#075985"),  # sky
        ("#F3E8FF", "#E9D5FF", "#6B21A8"),  # purple
    ]
    records = by_bank.sort_values("balance", ascending=False).to_dict("records")
    for i in range(0, len(records), 4):
        row = records[i:i+4]
        cols = st.columns(len(row))
        for j, (col, rec) in enumerate(zip(cols, row)):
            with col:
                bg, bd, tx = palette[(i + j) % len(palette)]
                note = "📝 not updated this refresh" if rec["bank"] in stale else ""
                st.markdown(f"""
                <div style="
                  background:{bg}; border:1px solid {bd};
                  border-radius:18px; padding:18px 20px;
                  box-shadow:0 2px 10px rgba(0,0,0,.05);">
                  <div style="font-size:12px;color:#374151;text-transform:uppercase;letter-spacing:.08em">{rec["bank"]}</div>
                  <div style="font-size:30px;font-weight:800;color:{tx};margin-top:6px">{fmt_money(rec["balance"])}</div>
                  <div style="font-size:12px;color:#6B7280;margin-top:4px">Available balance</div>
                  {f'<div style="font-size:12px;color:#9CA3AF;margin-top:6px">{note}</div>' if note else ''}
                </div>
                """, unsafe_allow_html=True)

# rerun compatibility
try:
    _rerun = st.rerun
except AttributeError:
    _rerun = st.experimental_rerun

# ---- loaders ----
@st.cache_data(ttl=300)
def load_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=25)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

def _cols_lower(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out

# ---- parsers ----
def parse_unt_a(df: pd.DataFrame):
    df = df.copy().dropna(how="all").dropna(axis=1, how="all")
    bank_col = None
    for c in df.columns:
        if df[c].dtype == object and (df[c].dropna().astype(str).str.strip() != "").sum() >= 3:
            bank_col = c; break
    if bank_col is None:
        raise ValueError('[UNT A] Could not detect "Bank" column.')

    parsed_dates = pd.to_datetime(pd.Index(df.columns), errors="coerce")
    date_cols = [col for col, d in zip(df.columns, parsed_dates) if pd.notna(d)]
    if not date_cols:
        raise ValueError('[UNT A] No date columns found in header.')
    date_map = {col: pd.to_datetime(col, errors="coerce") for col in date_cols}
    latest_col = max(date_cols, key=lambda c: date_map[c])

    bank_series = df[bank_col].astype(str).str.strip()
    mask = bank_series.ne("") & ~bank_series.str.contains("available|total", case=False, na=False)
    sub = df.loc[mask, [bank_col, latest_col]].copy()
    sub.columns = ["bank", "balance"]
    sub["balance"] = pd.to_numeric(sub["balance"], errors="coerce")
    sub = sub.dropna(subset=["balance"])
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
    return out.drop(columns=["status_raw"]).dropna(subset=["bank", "amount"])

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
    })
    return out.dropna(subset=["bank", "amount", "settlement_date"])

def parse_fund_movement(df: pd.DataFrame) -> pd.DataFrame:
    d = _cols_lower(df)
    col_date = next((c for c in d.columns if c == "date"), None)
    col_total = next((c for c in d.columns if "total liquidity" in c), None)
    if not col_date or not col_total:
        raise ValueError('[Fund Movement] Need Date and Total Liquidity.')
    out = pd.DataFrame({
        "date": pd.to_datetime(d[col_date], errors="coerce"),
        "total_liquidity": pd.to_numeric(d[col_total], errors="coerce")
    }).dropna()
    return out.sort_values("date")

# -------------------------
# HEADER + REFRESH
# -------------------------
st.title(APP_TITLE)
last_refresh = pd.Timestamp.now(tz=TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
st.markdown(
    f'<span style="display:inline-block;background:#E5EDFF;border:1px solid #D6E0FF;color:#1E3A8A;padding:4px 10px;border-radius:999px;font-size:12px;">Last refresh: {last_refresh}</span>',
    unsafe_allow_html=True
)

st.sidebar.header("Refresh")
if st.sidebar.button("Refresh now"):
    st.cache_data.clear()
    _rerun()
auto = st.sidebar.checkbox("Auto refresh", value=False)
interval = st.sidebar.number_input("Interval (seconds)", 15, 600, 60, 15)
if auto:
    time.sleep(int(interval))
    _rerun()

st.sidebar.header("Preferences")
stale_flag_enabled = st.sidebar.checkbox("Highlight non-updated banks on refresh", value=True)

# -------------------------
# LOAD + PARSE (force-map)
# -------------------------
dfs_raw, errors = {}, []
for kind, raw in RAW_LINKS.items():
    url = to_csv_url(raw)
    try:
        dfs_raw[kind] = load_csv(url)
    except Exception as e:
        errors.append(f"{kind} CSV error: {e}")

if errors:
    with st.expander("Notes / parsing messages"):
        for m in errors: st.write("• " + m)

by_bank, latest_balance_date = pd.DataFrame(), None
payments = pd.DataFrame()
settlements = pd.DataFrame()
fund_mv = pd.DataFrame()

if "UNT A" in dfs_raw:
    try: by_bank, latest_balance_date = parse_unt_a(dfs_raw["UNT A"])
    except Exception as e: st.warning(f"UNT A parse: {e}")
if "SUPPLIER PAYMENTS" in dfs_raw:
    try: payments = parse_payments(dfs_raw["SUPPLIER PAYMENTS"])
    except Exception as e: st.warning(f"Payments parse: {e}")
if "SETTLEMENTS" in dfs_raw:
    try: settlements = parse_settlements(dfs_raw["SETTLEMENTS"])
    except Exception as e: st.warning(f"Settlements parse: {e}")
if "Fund Movement" in dfs_raw:
    try: fund_mv = parse_fund_movement(dfs_raw["Fund Movement"])
    except Exception as e: st.warning(f"Fund Movement parse: {e}")

# -------------------------
# STALE BANK DETECTION
# -------------------------
stale_banks = set()
if stale_flag_enabled and not by_bank.empty:
    if "prev_by_bank" not in st.session_state:
        st.session_state.prev_by_bank = {}
    prev = st.session_state.prev_by_bank
    current = {r["bank"]: float(r["balance"]) for r in by_bank.to_dict("records")}
    tol = 1e-6
    for b, bal in current.items():
        if b in prev and abs(prev[b] - bal) <= tol:
            stale_banks.add(b)
    st.session_state.prev_by_bank = current

# -------------------------
# TOP CARDS
# -------------------------
total_balance = by_bank["balance"].sum() if not by_bank.empty else 0.0
bank_cnt = by_bank["bank"].nunique() if not by_bank.empty else 0

now_local = pd.Timestamp.now(tz=TZ).normalize()
end4 = now_local + pd.Timedelta(days=3)
lc_next4 = settlements.loc[
    (settlements["settlement_date"] >= now_local.tz_localize(None)) &
    (settlements["settlement_date"] <= end4.tz_localize(None))
] if not settlements.empty else pd.DataFrame()
lc_next4_sum = lc_next4["amount"].sum() if not lc_next4.empty else 0.0
pending_sum = payments.loc[payments["status"]=="Pending", "amount"].sum() if not payments.empty else 0.0

k1, k2, k3, k4 = st.columns(4)
with k1: colored_card("Total Balance", total_balance, f"As of {latest_balance_date}", bg="#E6F0FF", border="#C7D8FE", text="#1E3A8A")
with k2: colored_card("Pending Payments", pending_sum, bg="#FFECEF", border="#FBD5D8", text="#9F1239")
with k3: colored_card("LC due (next 4 days)", lc_next4_sum, bg="#E9FFF2", border="#C7F7DD", text="#065F46")
with k4: colored_card("Banks", bank_cnt, bg="#FFF7E6", border="#FDE9C8", text="#92400E")

# -------------------------
# BANK BALANCES
# -------------------------
st.markdown("### Bank Balances (Cards)")
if not by_bank.empty:
    bank_cards_colorful(by_bank, stale=stale_banks)
    if stale_flag_enabled and stale_banks:
        st.caption("📝 Banks marked 'not updated this refresh' kept the same balance as last time you refreshed.")
else:
    st.info("No balances detected for 'UNT A' data.")
st.markdown("---")

# -------------------------
# PAYMENTS
# -------------------------
st.header("Approved / Pending / Rejected Payments")
if payments.empty:
    st.info("No payments data detected.")
else:
    p1, p2, p3, p4 = st.columns([1.2, 1.2, 1.2, 1])
    with p1:
        banks = sorted(payments["bank"].dropna().astype(str).unique())
        pay_bank_sel = st.multiselect("Bank", banks, default=banks)
    with p2:
        status_opts = ["Approved", "Pending", "Rejected"]
        status_sel = st.multiselect("Status", status_opts, default=status_opts)
    with p3:
        if payments["due_date"].notna().any():
            pmin = payments["due_date"].dropna().min().date()
            pmax = payments["due_date"].dropna().max().date()
            rng = st.date_input("Due Date range", (pmin, pmax))
            d1, d2 = pd.to_datetime(rng[0]), pd.to_datetime(rng[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            mdate = payments["due_date"].between(d1, d2)
        elif payments["date"].notna().any():
            pmin = payments["date"].dropna().min().date()
            pmax = payments["date"].dropna().max().date()
            rng = st.date_input("Date range", (pmin, pmax))
            d1, d2 = pd.to_datetime(rng[0]), pd.to_datetime(rng[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            mdate = payments["date"].between(d1, d2)
        else:
            mdate = True
    with p4:
        if st.button("Clear payment filters"):
            pay_bank_sel, status_sel = banks, status_opts

    mpay = (payments["bank"].astype(str).isin(pay_bank_sel)) & (payments["status"].isin(status_sel)) & (mdate if isinstance(mdate, pd.Series) else True)
    df_pay = payments.loc[mpay].copy()

    s_app = df_pay.loc[df_pay["status"]=="Approved", "amount"].sum()
    s_pen = df_pay.loc[df_pay["status"]=="Pending", "amount"].sum()
    s_rej = df_pay.loc[df_pay["status"]=="Rejected", "amount"].sum()
    s_cnt = len(df_pay)

    c1, c2, c3, c4 = st.columns(4)
    with c1: colored_card("Approved (sum)", s_app, bg="#E9FFF2", border="#C7F7DD", text="#065F46")
    with c2: colored_card("Pending (sum)", s_pen, bg="#FFECEF", border="#FBD5D8", text="#9F1239")
    with c3: colored_card("Rejected (sum)", s_rej, bg="#FFF1F2", border="#FBD5D8", text="#9F1239")
    with c4: colored_card("Payments (count)", s_cnt, bg="#E6F0FF", border="#C7D8FE", text="#1E3A8A")

    show_cols = [c for c in ["date","bank","status","amount","due_date","description"] if c in df_pay.columns]
    viz = df_pay.copy()
    for c in ["date","due_date"]:
        if c in viz.columns:
            viz[c] = pd.to_datetime(viz[c], errors="coerce").dt.strftime(DATE_FMT)
    if "amount" in viz.columns:
        viz["amount"] = viz["amount"].map(fmt_money)
    st.dataframe(viz[show_cols].sort_values(by=[c for c in ["due_date","date"] if c in show_cols]), use_container_width=True, height=320)

st.markdown("---")

# -------------------------
# LC SETTLEMENTS
# -------------------------
st.header("LC Settlements")
if settlements.empty:
    st.info("No LC data detected.")
else:
    l1, l2, l3, l4 = st.columns([1.2, 1.2, 1.2, 1])
    with l1:
        banks = sorted(settlements["bank"].dropna().astype(str).unique())
        lc_bank_sel = st.multiselect("Bank", banks, default=banks)
    with l2:
        types = sorted([t for t in settlements["type"].unique() if isinstance(t,str) and t.strip()])
        lc_type_sel = st.multiselect("Type", types, default=types)
    with l3:
        stats = sorted([s for s in settlements["status"].unique() if isinstance(s,str) and s.strip()])
        lc_status_sel = st.multiselect("Status", stats, default=stats)
    with l4:
        lmin = settlements["settlement_date"].min().date()
        lmax = settlements["settlement_date"].max().date()
        lc_rng = st.date_input("Settlement Date range", (lmin, lmax))

    lm1, lm2 = pd.to_datetime(lc_rng[0]), pd.to_datetime(lc_rng[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    mask = settlements["bank"].astype(str).isin(lc_bank_sel) & settlements["settlement_date"].between(lm1, lm2)
    if lc_type_sel:   mask &= settlements["type"].isin(lc_type_sel)
    if lc_status_sel: mask &= settlements["status"].isin(lc_status_sel)

    df_lc = settlements.loc[mask].copy()
    lc_sum = df_lc["amount"].sum(); lc_cnt = len(df_lc)
    cc1, cc2 = st.columns(2)
    with cc1: colored_card("LC Amount (filtered sum)", lc_sum, bg="#FFF7E6", border="#FDE9C8", text="#92400E")
    with cc2: colored_card("LC Items (count)", lc_cnt, bg="#E6F0FF", border="#C7D8FE", text="#1E3A8A")

    viz = df_lc.copy()
    viz["settlement_date"] = pd.to_datetime(viz["settlement_date"]).dt.strftime(DATE_FMT)
    viz["amount"] = viz["amount"].map(fmt_money)
    show_cols = [c for c in ["bank","type","status","settlement_date","amount","description"] if c in viz.columns]
    st.dataframe(viz[show_cols].sort_values("settlement_date"), use_container_width=True, height=320)

    # Coming 4 days
    st.subheader("Coming 4 Days Settlements")
    start = now_local.tz_localize(None); end = (now_local + pd.Timedelta(days=3)).tz_localize(None)
    upcoming = settlements.loc[settlements["bank"].astype(str).isin(lc_bank_sel) & settlements["settlement_date"].between(start, end)].copy()
    up_sum = upcoming["amount"].sum() if not upcoming.empty else 0.0
    uc1, uc2 = st.columns(2)
    with uc1: colored_card("Upcoming 4-day Amount", up_sum, bg="#E9FFF2", border="#C7F7DD", text="#065F46")
    with uc2: colored_card("Upcoming 4-day Items", int(len(upcoming)), bg="#FFECEF", border="#FBD5D8", text="#9F1239")
    if not upcoming.empty:
        upcoming["settlement_date"] = pd.to_datetime(upcoming["settlement_date"]).dt.strftime(DATE_FMT)
        upcoming["amount"] = upcoming["amount"].map(fmt_money)
        show2 = [c for c in ["bank","type","status","settlement_date","amount","description"] if c in upcoming.columns]
        st.dataframe(upcoming[show2].sort_values("settlement_date"), use_container_width=True)

st.markdown("---")

# -------------------------
# LIQUIDITY TREND
# -------------------------
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

# -------------------------
# QUICK INSIGHTS
# -------------------------
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

st.caption("Data source: Google Sheets (CSV). Use the sidebar to refresh. Colored bank cards • No currency symbol • Stale-bank detector • Quick Insights")
