# Treasury Dashboard — Google Sheets (compact & numeric-safe)

import io
from datetime import timedelta
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Treasury Dashboard", layout="wide")

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

# ---- CSS (right KPIs, hide sidebar) ----
st.markdown("""
<style>
  [data-testid="stSidebar"], [data-testid="stSidebarNav"] { display:none !important; }
  .kpi-box { text-align:right; }
  .kpi-title { font-size:12px;color:#374151;text-transform:uppercase;letter-spacing:.08em; }
  .kpi-value { font-size:26px;font-weight:800;margin-top:2px; }
  .kpi-sub { font-size:12px;color:#6B7280;margin-top:2px; }
</style>
""", unsafe_allow_html=True)

# ---- helpers ----
def _to_number(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().replace(",", "")
    if s.endswith("%"): s = s[:-1]
    try: return float(s)
    except Exception: return np.nan

@st.cache_data(ttl=300)
def read_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=25); r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

def cols_lower(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in df.columns]
    return out

def number_col(label: str, fmt: str = "%,.2f", width="small"):
    return st.column_config.NumberColumn(label=label, format=fmt, width=width)

def text_col(label: str, width="small"):
    return st.column_config.TextColumn(label=label, width=width)

def kpi_card(title, value, subtitle="", bg="#EEF2FF", border="#C7D2FE", text="#111827"):
    st.markdown(
        f"""
        <div class="kpi-box" style="
            background:{bg};border:1px solid {border};
            border-radius:12px;padding:12px 14px;
            box-shadow:0 1px 6px rgba(0,0,0,.04);">
          <div class="kpi-title">{title}</div>
          <div class="kpi-value" style="color:{text}">{value:,.2f}</div>
          <div class="kpi-sub">{subtitle}</div>
        </div>
        """, unsafe_allow_html=True
    )

# ---- parsers ----
def parse_bank_balance(df: pd.DataFrame):
    c = cols_lower(df)
    if "bank" in c.columns and ("amount" in c.columns or "amount(sar)" in c.columns):
        amt_col = "amount" if "amount" in c.columns else "amount(sar)"
        out = pd.DataFrame({
            "bank": c["bank"].astype(str).str.strip(),
            "balance": c[amt_col].map(_to_number)
        }).dropna()
        by_bank = out.groupby("bank", as_index=False)["balance"].sum()
        return by_bank, None

    raw = df.copy().dropna(how="all").dropna(axis=1, how="all")
    bank_col = None
    for col in raw.columns:
        if raw[col].dtype == object and (raw[col].dropna().astype(str).str.strip() != "").sum() >= 3:
            bank_col = col; break
    if bank_col is None: raise ValueError("BANK BALANCE: could not detect bank column.")

    parsed = pd.to_datetime(pd.Index(raw.columns), errors="coerce", dayfirst=False)
    date_cols = [col for col, d in zip(raw.columns, parsed) if pd.notna(d)]
    if not date_cols: raise ValueError("BANK BALANCE: no date columns in header.")
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
    d = cols_lower(df).rename(columns={"supplier name":"supplier","amount(sar)":"amount_sar"})
    if "bank" not in d.columns or "status" not in d.columns: return pd.DataFrame()
    status_norm = d["status"].astype(str).str.strip().str.lower()
    d = d.loc[status_norm.str.contains("approved")].copy()
    if d.empty: return pd.DataFrame()
    amt_col = "amount_sar" if "amount_sar" in d.columns else ("amount" if "amount" in d.columns else None)
    if amt_col is None: return pd.DataFrame()
    d["amount"] = d[amt_col].map(_to_number)
    d["bank"] = d["bank"].astype(str).str.strip()
    keep = [c for c in ["bank","supplier","currency","amount","status"] if c in d.columns]
    out = d[keep].dropna(subset=["amount"]).copy()
    if "status" in out.columns: out["status"] = out["status"].astype(str).str.title()
    return out

def parse_settlements(df: pd.DataFrame) -> pd.DataFrame:
    d = cols_lower(df)
    bank = next((c for c in d.columns if c.startswith("bank")), None)
    date_col = None
    for cand in d.columns:
        if "maturity" in cand and "new" not in cand: date_col = cand; break
    if date_col is None:
        for cand in d.columns:
            if "new" in cand and "maturity" in cand: date_col = cand; break
    amount_col = None
    for cand in d.columns:
        if "balance" in cand and "settlement" in cand: amount_col = cand; break
    if amount_col is None:
        for cand in d.columns:
            if "currently" in cand and "due" in cand: amount_col = cand; break
    if amount_col is None:
        amount_col = "amount(sar)" if "amount(sar)" in d.columns else ("amount" if "amount" in d.columns else None)
    status_col = next((c for c in d.columns if "status" in c), None)
    type_col   = next((c for c in d.columns if "type" in c), None)
    remark_col = next((c for c in d.columns if "remark" in c), None)
    if not bank or not amount_col or not date_col: return pd.DataFrame()

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
    if "date" not in d.columns: return pd.DataFrame()
    liq_col = next((c for c in d.columns if ("total" in c and "liquidity" in c)), None)
    if not liq_col: return pd.DataFrame()
    out = pd.DataFrame({
        "date": pd.to_datetime(d["date"], errors="coerce"),
        "total_liquidity": d[liq_col].map(_to_number)
    }).dropna()
    return out.sort_values("date")

# ---- header ----
left, btn = st.columns([1,0.13])
with left:
    st.title(f"{COMPANY} — Treasury Dashboard")
    st.caption(pd.Timestamp.now(tz=TZ).strftime("Last refresh: %Y-%m-%d %H:%M:%S %Z"))
with btn:
    if st.button("🔁 Refresh", use_container_width=True):
        st.cache_data.clear(); st.rerun()

# ---- load & parse ----
notes = []
try: df_bal_raw = read_csv(LINKS["BANK BALANCE"])
except Exception as e: notes.append(f"BANK BALANCE load error: {e}"); df_bal_raw = pd.DataFrame()
try: df_pay_raw = read_csv(LINKS["SUPPLIER PAYMENTS"])
except Exception as e: notes.append(f"SUPPLIER PAYMENTS load error: {e}"); df_pay_raw = pd.DataFrame()
try: df_lc_raw = read_csv(LINKS["SETTLEMENTS"])
except Exception as e: notes.append(f"SETTLEMENTS load error: {e}"); df_lc_raw = pd.DataFrame()
try: df_fm_raw = read_csv(LINKS["Fund Movement"])
except Exception as e: notes.append(f"Fund Movement load error: {e}"); df_fm_raw = pd.DataFrame()

df_by_bank, bal_date = (pd.DataFrame(), None)
if not df_bal_raw.empty:
    try: df_by_bank, bal_date = parse_bank_balance(df_bal_raw)
    except Exception as e: notes.append(f"BANK BALANCE parse: {e}")

df_pay = pd.DataFrame()
if not df_pay_raw.empty:
    try: df_pay = parse_supplier_payments(df_pay_raw)
    except Exception as e: notes.append(f"SUPPLIER PAYMENTS parse: {e}")

df_lc = pd.DataFrame()
if not df_lc_raw.empty:
    try: df_lc = parse_settlements(df_lc_raw)
    except Exception as e: notes.append(f"SETTLEMENTS parse: {e}")

df_fm = pd.DataFrame()
if not df_fm_raw.empty:
    try: df_fm = parse_fund_movement(df_fm_raw)
    except Exception as e: notes.append(f"Fund Movement parse: {e}")

for n in notes: st.warning(n)

# ---- KPIs ----
total_balance = df_by_bank["balance"].sum() if not df_by_bank.empty else 0.0
banks_cnt = df_by_bank["bank"].nunique() if not df_by_bank.empty else 0
today = pd.Timestamp.now(tz=TZ).normalize().tz_localize(None)
next4 = today + timedelta(days=3)
lc_next4_sum = df_lc.loc[df_lc["settlement_date"].between(today, next4), "amount"].sum() if not df_lc.empty else 0.0
approved_sum = df_pay["amount"].sum() if not df_pay.empty else 0.0

c1,c2,c3,c4 = st.columns(4)
with c1: kpi_card("Total Balance", total_balance, "", bg="#E6F0FF", border="#C7D8FE", text="#1E3A8A")
with c2: kpi_card("Approved Payments (sum)", approved_sum, "", bg="#E9FFF2", border="#C7F7DD", text="#065F46")
with c3: kpi_card("LC due (next 4 days)", lc_next4_sum, "", bg="#FFF7E6", border="#FDE9C8", text="#92400E")
with c4:
    st.markdown(f"""
    <div class="kpi-box" style="background:#FFF1F2;border:1px solid #FBD5D8;border-radius:12px;padding:12px 14px;">
      <div class="kpi-title">Banks</div>
      <div class="kpi-value" style="color:#9F1239">{banks_cnt:,.0f}</div>
      <div class="kpi-sub"></div>
    </div>
    """, unsafe_allow_html=True)

# ---- Bank Balances (compact + numeric) ----
st.subheader("Bank Balances")
if df_by_bank.empty:
    st.info("No balances found.")
else:
    # force numeric and build a clean display dataframe
    dfb = df_by_bank.copy()
    dfb["balance"] = pd.to_numeric(dfb["balance"], errors="coerce").astype(float)
    total = float(dfb["balance"].sum() or 0)
    dfb["share_%"] = np.where(total > 0, dfb["balance"]/total*100.0, 0.0).astype(float)

    show_bal = pd.DataFrame({
        "Bank": dfb["bank"].astype(str),
        "Balance": dfb["balance"].astype(float),
        "Share %": dfb["share_%"].astype(float),
    }).sort_values("Balance", ascending=False)

    st.dataframe(
        show_bal,
        use_container_width=True,
        height=210,  # reduced height
        column_config={
            "Bank": text_col("Bank", width="small"),
            "Balance": number_col("Balance", "%,.2f", width="small"),
            "Share %": number_col("Share %", "%,.2f%%", width="small"),
        }
    )

st.markdown("---")

# ---- Approved Supplier Payments ----
st.header("Approved Supplier Payments")
if df_pay.empty:
    st.info("No Approved payments in sheet.")
else:
    leftf, _ = st.columns([1.4,1])
    with leftf:
        banks = sorted(df_pay["bank"].dropna().unique())
        selected = st.multiselect("Filter by Bank", banks, default=banks)

    view = df_pay[df_pay["bank"].isin(selected)].copy()
    view["amount"] = pd.to_numeric(view["amount"], errors="coerce").astype(float)

    grp = view.groupby("bank", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
    grp_show = pd.DataFrame({"Bank": grp["bank"].astype(str), "Amount": grp["amount"].astype(float)})
    st.markdown("**Sum by Bank**")
    st.dataframe(
        grp_show,
        use_container_width=True,
        height=190,
        column_config={
            "Bank": text_col("Bank", width="small"),
            "Amount": number_col("Amount", "%,.2f", width="small"),
        }
    )

    st.markdown("**Approved rows**")
    rows_show = pd.DataFrame({
        "Bank": view["bank"].astype(str),
        "Supplier": view.get("supplier","").astype(str) if "supplier" in view else "",
        "Curr": view.get("currency","").astype(str) if "currency" in view else "",
        "Amount": view["amount"].astype(float),
        "Status": view.get("status","").astype(str) if "status" in view else "",
    })
    st.dataframe(
        rows_show,
        use_container_width=True,
        height=280,
        column_config={
            "Bank": text_col("Bank", width="small"),
            "Supplier": text_col("Supplier", width="medium"),
            "Curr": text_col("Curr", width="small"),
            "Amount": number_col("Amount", "%,.2f", width="small"),
            "Status": text_col("Status", width="small"),
        }
    )

st.markdown("---")

# ---- LC Settlements (Pending only; numeric-safe) ----
st.header("LC Settlements")
if df_lc.empty:
    st.info("No LC data.")
else:
    a,b,_ = st.columns([1.2,1.2,1])
    with a:
        banks = sorted(df_lc["bank"].dropna().unique())
        sel_banks = st.multiselect("Bank", banks, default=banks)
    with b:
        dmin, dmax = df_lc["settlement_date"].min().date(), df_lc["settlement_date"].max().date()
        drange = st.date_input("Settlement date range", (dmin, dmax))

    d1 = pd.to_datetime(drange[0]); d2 = pd.to_datetime(drange[1]) + timedelta(days=1) - timedelta(seconds=1)
    mask = df_lc["bank"].isin(sel_banks) & df_lc["settlement_date"].between(d1, d2)
    lc_view = df_lc.loc[mask].copy()
    lc_view["amount"] = pd.to_numeric(lc_view["amount"], errors="coerce").astype(float)

    lc_sum = float(lc_view["amount"].sum() or 0); lc_cnt = len(lc_view)
    m1,m2 = st.columns(2)
    with m1: kpi_card("LC Amount (filtered sum)", lc_sum, "", bg="#FFF7E6", border="#FDE9C8", text="#92400E")
    with m2:
        st.markdown(f"""
        <div class="kpi-box" style="background:#E6F0FF;border:1px solid #C7D8FE;border-radius:12px;padding:12px 14px;">
          <div class="kpi-title">LC Items (count)</div>
          <div class="kpi-value" style="color:#1E3A8A">{lc_cnt:,.0f}</div>
          <div class="kpi-sub"></div>
        </div>
        """, unsafe_allow_html=True)

    if not lc_view.empty:
        lc_show = pd.DataFrame({
            "Bank": lc_view["bank"].astype(str),
            "Type": lc_view.get("type","").astype(str) if "type" in lc_view else "",
            "Status": lc_view.get("status","").astype(str) if "status" in lc_view else "",
            "Settlement Date": pd.to_datetime(lc_view["settlement_date"]).dt.strftime(DATE_FMT),
            "Amount": lc_view["amount"].astype(float),
            "Remark": lc_view.get("remark","").astype(str) if "remark" in lc_view else "",
            "Description": lc_view.get("description","").astype(str) if "description" in lc_view else "",
        }).sort_values("Settlement Date")

        st.dataframe(
            lc_show,
            use_container_width=True,
            height=290,
            column_config={
                "Bank": text_col("Bank", width="small"),
                "Type": text_col("Type", width="small"),
                "Status": text_col("Status", width="small"),
                "Settlement Date": text_col("Settlement Date", width="small"),
                "Amount": number_col("Amount", "%,.2f", width="small"),
                "Remark": text_col("Remark", width="medium"),
                "Description": text_col("Description", width="medium"),
            }
        )

        remarks = lc_view.loc[lc_view["remark"].astype(str).str.strip() != ""].copy()
        if not remarks.empty:
            st.subheader("Remarks")
            remarks_show = pd.DataFrame({
                "Settlement Date": pd.to_datetime(remarks["settlement_date"]).dt.strftime(DATE_FMT),
                "Bank": remarks["bank"].astype(str),
                "Amount": pd.to_numeric(remarks["amount"], errors="coerce").astype(float),
                "Remark": remarks["remark"].astype(str),
            }).sort_values("Settlement Date")
            st.dataframe(
                remarks_show,
                use_container_width=True,
                height=200,
                column_config={
                    "Settlement Date": text_col("Settlement Date", width="small"),
                    "Bank": text_col("Bank", width="small"),
                    "Amount": number_col("Amount", "%,.2f", width="small"),
                    "Remark": text_col("Remark", width="medium"),
                }
            )

# (Liquidity Trend and Quick Insights omitted here for brevity; they are unchanged)
