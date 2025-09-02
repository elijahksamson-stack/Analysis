# above the start of the split layout, and below the tickers entry box. Please use plotly to plot the price graph on one graph for each of the securities in the list for the last 1y: 

# fundtech_app.py
# Streamlit app: Fundamentals + Technical Rank composite scoring with liquidity check + correlation heatmap
# Run: streamlit run fundtech_app.py

import io
import math
import re
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# ---- UI CONFIG -----------
# ==========================
st.set_page_config(page_title="Fund+Tech Rank", page_icon="📊", layout="wide")

st.title("📊 Fundamentals + Technical Rank — Scoring + Liquidity + Correlation")
st.caption("Composite scoring table, liquidity check, and correlation heatmap of price changes.")

# ==========================
# ---- Fundamentals Scraper ----------
# ==========================

_KEEP_METRICS = [
    "Market Cap","P/E","Forward P/E","PEG","P/S","P/B","EV/EBITDA","EPS (ttm)",
    "Insider Own","Shs Float","Short Interest","Avg Volume","Beta","ATR","SMA20","SMA50","SMA200",
    "52W High","52W Low","RSI (14)","Gross Margin","Oper. Margin","Profit Margin","ROE","ROA",
    "Sales","Income","Debt/Eq","Current Ratio","Quick Ratio","LT Debt/Eq","Cash/sh","P/FCF",
    "Inst Own","Inst Trans","Float Short","Target Price","Perf YTD","Sales past 5Y","EPS next Y","EPS next 5Y",
    "Close Price","Recom","Sales Y/Y TTM"
]

_HEADERS = {"User-Agent": "Mozilla/5.0"}

@st.cache_data(show_spinner=False, ttl=600)
def _requests_custom(url: str) -> Optional[str]:
    try:
        r = requests.get(url, headers=_HEADERS, timeout=10)
        if r.status_code == 200:
            return r.text
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=False, ttl=600)
def get_finviz_metrics(symbol: str) -> pd.DataFrame:
    url = f"https://finviz.com/quote.ashx?t={symbol}&p=d"
    html = _requests_custom(url)
    if html is None:
        return pd.DataFrame()
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="js-snapshot-table snapshot-table2 screener_snapshot-table-body")
    metrics = []
    if table:
        for row in table.find_all("tr"):
            cols = row.find_all("td")
            if len(cols) % 2 == 0:
                for i in range(0, len(cols), 2):
                    metric_name = cols[i].text.strip()
                    metric_value = cols[i + 1].text.strip()
                    metrics.append({"Metric": metric_name, "Value": metric_value})
    cp = soup.find("strong", class_="quote-price_wrapper_price")
    if cp:
        metrics.append({"Metric": "Close Price", "Value": cp.text.strip().replace(",", "")})
    df = pd.DataFrame(metrics)
    if df.empty:
        return df
    return df[df["Metric"].isin(_KEEP_METRICS)].reset_index(drop=True)

# ---------- Converters ----------
def _convert_market_cap(t: Optional[str]):
    if not t: return np.nan
    try:
        t = str(t).replace(",", "").strip().upper()
        if t.endswith("T"): return float(t[:-1]) * 1e12
        if t.endswith("B"): return float(t[:-1]) * 1e9
        if t.endswith("M"): return float(t[:-1]) * 1e6
        if t.endswith("K"): return float(t[:-1]) * 1e3
        return float(t)
    except Exception:
        return np.nan

def _convert_percent(s):
    if not s: return np.nan
    try:
        return float(str(s).replace("%", "").strip()) / 100.0
    except Exception:
        return np.nan

def _convert_float(s):
    try:
        return float(str(s).replace(",", "").strip())
    except Exception:
        return np.nan

# ---------- Normalize row ----------
def normalize_fundamentals_row(ticker: str, df_metrics: pd.DataFrame) -> Dict:
    v = {m: None for m in _KEEP_METRICS}
    for _, r in df_metrics.iterrows():
        v[r["Metric"]] = r["Value"]
    row = {"Ticker": ticker}
    row["Market Cap"] = _convert_market_cap(v["Market Cap"])
    row["Forward P/E"] = _convert_float(v["Forward P/E"])
    row["P/E"] = _convert_float(v["P/E"])
    row["Close Price"] = _convert_float(v["Close Price"])
    row["Avg Volume"] = _convert_market_cap(v["Avg Volume"])
    return row

@st.cache_data(show_spinner=True, ttl=900)
def fetch_fundamentals(tickers: List[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        dfm = get_finviz_metrics(t)
        if dfm.empty: continue
        rows.append(normalize_fundamentals_row(t, dfm))
    return pd.DataFrame(rows)

# ==========================
# ---- Technical Rank ----------
# ==========================
def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _roc(s: pd.Series, n: int) -> pd.Series:
    return s.pct_change(n) * 100.0

def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_dn = down.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / roll_dn.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))

def technical_rank_series(close: pd.Series) -> pd.Series:
    c = close.dropna().astype(float)
    if c.empty: return pd.Series(dtype=float)
    sma200 = c.rolling(200).mean()
    sma50  = c.rolling(50).mean()
    longtermma  = 0.30 * 100.0 * (c - sma200) / sma200
    longtermroc = 0.30 * _roc(c, 125)
    midtermma   = 0.15 * 100.0 * (c - sma50) / sma50
    midtermroc  = 0.15 * _roc(c, 20)
    ema12, ema26 = _ema(c, 12), _ema(c, 26)
    ppo = 100.0 * (ema12 - ema26) / ema26
    sig = _ema(ppo, 9)
    ppo_hist = ppo - sig
    slope  = (ppo_hist - ppo_hist.shift(8)) / 3.0
    stPpo  = 0.05 * 100.0 * slope
    stRsi  = 0.05 * _rsi(c, 14)
    trank = (longtermma + longtermroc + midtermma + midtermroc + stPpo + stRsi)
    return trank.clip(lower=0, upper=100)

def compute_trank_for_list(tickers: List[str], period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    rows = []
    for t in [x.strip().upper() for x in tickers if x and x.strip()]:
        try:
            df = yf.download(t, period=period, interval=interval, progress=False, group_by="column")
            if df is None or df.empty:
                rows.append({"Ticker": t, "TRank": np.nan, "Close": np.nan, "AvgVol20": np.nan})
                continue
            close = df.get("Close", df.get("Adj Close")).astype(float).dropna()
            vol = df.get("Volume").astype(float).dropna() if "Volume" in df.columns else None
            tr = technical_rank_series(close)
            latest_tr = float(tr.dropna().iloc[-1]) if not tr.dropna().empty else np.nan
            last_close = float(close.iloc[-1]) if not close.empty else np.nan
            avg20 = float(vol.rolling(20).mean().iloc[-1]) if vol is not None and len(vol) >= 20 else np.nan
            rows.append({"Ticker": t, "TRank": latest_tr, "Close": last_close, "AvgVol20": avg20})
        except Exception:
            rows.append({"Ticker": t, "TRank": np.nan, "Close": np.nan, "AvgVol20": np.nan})
    return pd.DataFrame(rows)

# ==========================
# ---- Scoring Method & Liquidity ----------
# ==========================
def zscore_percentiles(df: pd.DataFrame, metrics: List[str], lower_is_better: List[str]) -> pd.DataFrame:
    out = pd.DataFrame({"Ticker": df["Ticker"]})
    for m in metrics:
        col = df[m]
        vals = -col if m in lower_is_better else col
        mu, sd = vals.mean(skipna=True), vals.std(skipna=True, ddof=1)
        z = (vals - mu) / sd if sd and np.isfinite(sd) and sd != 0 else pd.Series([np.nan]*len(vals), index=vals.index)
        pct = 0.5 * (1.0 + (z / math.sqrt(2)).apply(math.erf))
        out[m + "_pct"] = pct
    out["FundamentalScore"] = out[[c for c in out.columns if c.endswith("_pct")]].mean(axis=1, skipna=True)
    return out

# ==========================
# ---- UI Controls ---------
# ==========================
st.markdown("**Enter Tickers**")
tickers_text = st.text_area("Tickers", value="AAPL, MSFT, NVDA, AMZN, GOOGL", height=100)
tickers = sorted(set([t.strip().upper() for t in re.split(r"[\s,;]+", tickers_text) if t.strip()]))
st.write(f"**Parsed {len(tickers)} tickers**: {', '.join(tickers)}")

run_btn = st.button("Run Scoring", type="primary")

# ==========================
# ---- Pipeline ------------
# ==========================
if run_btn:
    with st.spinner("Fetching fundamentals…"):
        fdf = fetch_fundamentals(tickers)
    if fdf.empty:
        st.error("No fundamentals fetched."); st.stop()

    with st.spinner("Scoring fundamentals…"):
        metrics_to_use = ["P/E","Forward P/E","ROE","ROA","Beta","Debt/Eq","Gross Margin","Oper. Margin","Profit Margin"]
        lower_to_use = ["P/E","Forward P/E","Debt/Eq","Beta"]
        fscore = zscore_percentiles(fdf, [m for m in metrics_to_use if m in fdf.columns], lower_to_use)

    with st.spinner("Computing Technical Rank…"):
        tr = compute_trank_for_list(tickers)

    combo = fdf[["Ticker"]].merge(fscore[["Ticker","FundamentalScore"]], on="Ticker", how="left").merge(tr[["Ticker","TRank","Close","AvgVol20"]], on="Ticker", how="left")
    combo["TRank_0to1"] = combo["TRank"] / 100.0
    combo["Score"] = 0.7 * combo["FundamentalScore"] + 0.3 * combo["TRank_0to1"]

    # Liquidity check
    invest_amt = 30_000_000
    threshold_pct = 15.0
    combo["DollarVol20"] = combo["AvgVol20"] * combo["Close"]
    combo["PctOfDaily$Vol"] = (invest_amt / combo["DollarVol20"]) * 100.0
    combo["IlliquidFlag"] = combo["PctOfDaily$Vol"] > threshold_pct

    sorted_combo = combo.loc[~combo["IlliquidFlag"]].sort_values("Score", ascending=False).reset_index(drop=True)
    display_df = sorted_combo[["Ticker","FundamentalScore","TRank","Score"]].rename(columns={"TRank":"TechnicalScore"})

    illiquid = combo.loc[combo["IlliquidFlag"]].copy()

    # Split layout: left = tables, right = heatmap
    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Scoring Method")
        st.dataframe(display_df, use_container_width=True)

        if not illiquid.empty:
            st.subheader("⚠️ Illiquid Stocks")
            st.caption(f"Flagged where $30,000,000 > {threshold_pct:.0f}% of daily $ volume (20‑day avg vol × close).")
            cols = [c for c in ["Ticker","Close","AvgVol20","DollarVol20","PctOfDaily$Vol","Score"] if c in illiquid.columns]
            st.dataframe(illiquid[cols].sort_values("PctOfDaily$Vol", ascending=False).reset_index(drop=True), use_container_width=True)
        else:
            st.subheader("✅ No illiquid stocks under current parameters")

    with right_col:
        st.subheader("Correlation Heatmap of Daily % Changes")
        try:
            data = yf.download(tickers, period="6mo", interval="1d", progress=False, auto_adjust=False)
            # Fallback: if Adj Close not available, use Close
            if "Adj Close" in data.columns:
                px = data["Adj Close"]
            else:
                px = data["Close"]
            returns = px.pct_change().dropna()
            corr = returns.corr()
    
            import matplotlib.pyplot as plt
            import seaborn as sns
    
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Heatmap generation failed: {e}")
