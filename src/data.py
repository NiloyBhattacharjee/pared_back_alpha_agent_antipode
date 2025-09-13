# simple_loader.py
"""
Minimal data loader:
- Prices: REST (financialdatasets.ai) -> synthetic fallback
- News: local JSONs -> synthetic fallback
- Facts: local CSV -> tiny default table

ENV required for REST:
  FINANCIALDATASETS_API_KEY=...     (your key)
  FD_PRICES_URL=https://api.financialdatasets.ai/prices   (recommended)

Usage:
  df = load_prices(as_of=pd.Timestamp("2025-09-10"), lookback_days=200, forward_days=63)
  news = load_news(as_of, lookback_days=60)
  facts = load_facts()
"""

import os
from datetime import datetime, timedelta
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import requests

from . import UNIVERSE  # ["AAPL","MSFT","NVDA","TSLA"]

# --------------------------
# Helpers
# --------------------------
def _bdays(end: pd.Timestamp, lookback: int, fwd: int) -> pd.DatetimeIndex:
    end_with_fwd = pd.bdate_range(end=end, periods=1)[0] + pd.offsets.BDay(fwd)
    start = end_with_fwd - pd.offsets.BDay(lookback + 30)
    return pd.bdate_range(start=start, end=end_with_fwd)

def _clip_forward(df: pd.DataFrame, as_of: pd.Timestamp, forward_days: int) -> pd.DataFrame:
    end_limit = pd.bdate_range(end=as_of, periods=1)[0] + pd.offsets.BDay(forward_days)
    return df[df["date"] <= pd.to_datetime(end_limit.date())].copy()

# --------------------------
# PRICES (REST -> synthetic)
# --------------------------
def _fetch_prices_rest_one(ticker: str, start_date: str, end_date: str,
                           url: str, api_key: str, session: Optional[requests.Session]=None) -> pd.DataFrame:
    s = session or requests.Session()
    headers = {"X-API-KEY": api_key}
    params = {
        "ticker": ticker,
        "interval": "day",
        "interval_multiplier": 1,
        "start_date": start_date,
        "end_date": end_date,
        "limit": 5000,
    }
    r = s.get(url, headers=headers, params=params, timeout=20)
    r.raise_for_status()
    js = r.json()
    rows = js.get("prices", js.get("data", js.get("results", js)))
    if not isinstance(rows, list) or len(rows) == 0:
        raise ValueError("No price rows")
    df = pd.DataFrame(rows)

    # date/price column detection
    date_col = next((c for c in ("time","date","timestamp","Date") if c in df.columns), None)
    price_col = next((c for c in ("adj_close","close","adjusted_close","Adj Close","Close") if c in df.columns), None)
    if not date_col or not price_col:
        raise ValueError(f"Missing date/price columns in API response. cols={list(df.columns)[:10]}")

    df["date"] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
    if price_col != "adj_close":
        df = df.rename(columns={price_col: "adj_close"})
    df = df[["date", "adj_close"]]
    df["ticker"] = ticker
    return df

def load_prices(as_of: datetime, lookback_days: int = 200, forward_days: int = 21) -> pd.DataFrame:
    """
    Returns long-format prices with columns: date, ticker, adj_close.
    Tries REST (requires env vars) then falls back to synthetic.
    """
    as_of = pd.Timestamp(as_of)
    start_iso = (as_of - timedelta(days=int(lookback_days * 1.5))).date().isoformat()
    end_iso   = (as_of + timedelta(days=int(forward_days * 2))).date().isoformat()

    url = os.getenv("FD_PRICES_URL", "https://api.financialdatasets.ai/prices")
    api_key = os.getenv("FINANCIALDATASETS_API_KEY")
    use_fd = os.getenv("USE_FINANCIALDATASETS", "1") != "0"

    # --- REST path ---
    if use_fd and api_key:
        frames = []
        try:
            with requests.Session() as s:
                for t in UNIVERSE:
                    frames.append(_fetch_prices_rest_one(t, start_iso, end_iso, url, api_key, s))
            df = pd.concat(frames, ignore_index=True).sort_values(["date","ticker"]).reset_index(drop=True)
            df = _clip_forward(df, as_of, forward_days)
            df.attrs["source"] = "api"
            return df
        except Exception as e:
            print(f"[REST] falling back due to: {e}")

    # --- CACHE path (offline support; required fallback) ---
    here = os.path.dirname(os.path.dirname(__file__))
    cache_path = os.path.join(here, "data", "prices_cache.csv")
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path)
            # Normalize
            if "date" not in df.columns or "ticker" not in df.columns:
                raise ValueError("prices_cache.csv must have columns: date,ticker,adj_close")
            # Determine price col
            px_col = None
            for c in ("adj_close", "close", "price", "Adj Close", "Close"):
                if c in df.columns:
                    px_col = c
                    break
            if px_col is None:
                raise ValueError("prices_cache.csv must include an adjusted close column (e.g., adj_close)")
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            df = df.rename(columns={px_col: "adj_close"})
            df = df[df["ticker"].isin(UNIVERSE)]
            # trim to lookback + forward buffer around as_of
            start_lim = pd.to_datetime(as_of.date()) - pd.Timedelta(days=int(lookback_days * 1.5))
            end_lim = pd.to_datetime(as_of.date()) + pd.Timedelta(days=int(forward_days * 2))
            df = df[(df["date"] >= start_lim) & (df["date"] <= end_lim)]
            if not df.empty:
                df = df.sort_values(["date","ticker"]).reset_index(drop=True)
                df = _clip_forward(df, as_of, forward_days)
                df.attrs["source"] = "cache"
                return df
        except Exception as e:
            raise RuntimeError(
                f"Failed to load prices from cache at {cache_path}: {e}. Synthetic fallback is disabled."
            )

    # --- No cache and API not used: fail clearly (synthetic disabled) ---
    raise RuntimeError(
        "No API data and no cache found at data/prices_cache.csv. "
        "Create a cache online with `python run.py --as-of YYYY-MM-DD --forward-days N --write-cache`, "
        "or configure the API via FINANCIALDATASETS_API_KEY. Synthetic fallback has been disabled per configuration."
    )

def generate_synthetic_prices(universe: List[str], as_of: datetime,
                              lookback_days: int = 300, forward_days: int = 21, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = _bdays(pd.Timestamp(as_of), lookback_days, forward_days)
    drift_annual = {"AAPL":0.10, "MSFT":0.12, "NVDA":0.20, "TSLA":0.18}
    vol_annual   = {"AAPL":0.25, "MSFT":0.22, "NVDA":0.45, "TSLA":0.55}
    start_px     = {"AAPL":150.0, "MSFT":300.0, "NVDA":450.0, "TSLA":250.0}
    dt = 1/252
    rows = []
    for t in universe:
        mu, sg, p0 = drift_annual[t], vol_annual[t], start_px[t]
        daily_mu, daily_sigma = mu*dt, sg*np.sqrt(dt)
        prices = [p0]
        for _ in range(1, len(days)):
            prices.append(prices[-1] * np.exp(rng.normal(daily_mu, daily_sigma)))
        rows.append(pd.DataFrame({"date": days, "ticker": t, "adj_close": prices}))
    return pd.concat(rows, ignore_index=True).sort_values(["date","ticker"]).reset_index(drop=True)

# --------------------------
# NEWS (JSON dir -> synthetic)
# --------------------------
POS_WORDS = {"beat","strong","growth","record","surge","upgrade","optimism","win","expand","raise","partnership"}
NEG_WORDS = {"miss","weak","decline","loss","downgrade","probe","delay","risk","recall","investigation","antitrust"}

def load_news(as_of: datetime, lookback_days: int = 120) -> pd.DataFrame:
    as_of = pd.Timestamp(as_of)
    here = os.path.dirname(os.path.dirname(__file__))
    news_dir = os.path.join(here, "data", "news")
    rows = []
    try:
        if os.path.isdir(news_dir):
            import json
            for fn in os.listdir(news_dir):
                if fn.lower().endswith(".json"):
                    with open(os.path.join(news_dir, fn), "r", encoding="utf-8") as f:
                        items = json.load(f)
                    ticker = os.path.splitext(fn)[0].upper()
                    for it in items:
                        d = pd.to_datetime(it.get("date"))
                        if pd.isna(d): 
                            continue
                        if as_of - pd.Timedelta(days=lookback_days) <= d <= as_of:
                            title = (it.get("title","") + " " + it.get("snippet","")).strip()
                            rows.append({"date": d, "ticker": ticker, "headline": title})
        if rows:
            return pd.DataFrame(rows).sort_values(["date","ticker"]).reset_index(drop=True)
    except Exception as e:
        print(f"[NEWS] falling back due to: {e}")
    # synthetic news fallback
    return generate_synthetic_news(UNIVERSE, as_of, lookback_days)

def generate_synthetic_news(universe: List[str], as_of: datetime, lookback_days: int = 120, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.date_range(end=as_of, periods=min(lookback_days, 120))
    rows = []
    for d in days:
        for t in universe:
            for _ in range(int(rng.integers(0, 3))):   # 0..2 headlines/day/ticker
                pos_ct = int(rng.integers(0, 3))
                neg_ct = int(rng.integers(0, 3))
                words = []
                if pos_ct: words += list(rng.choice(list(POS_WORDS), size=pos_ct, replace=False))
                if neg_ct: words += list(rng.choice(list(NEG_WORDS), size=neg_ct, replace=False))
                headline = f"{t} {' '.join(words)}" if words else f"{t} reports update"
                rows.append({"date": pd.to_datetime(d.date()), "ticker": t, "headline": headline})
    return pd.DataFrame(rows).sort_values(["date","ticker"]).reset_index(drop=True)

# --------------------------
# FACTS (CSV -> tiny default)
# --------------------------
def load_facts() -> pd.DataFrame:
    here = os.path.dirname(os.path.dirname(__file__))
    facts_dir = os.path.join(here, "data", "facts")
    rows = []
    # Preferred: per-ticker JSON files with optional 'notes'
    try:
        if os.path.isdir(facts_dir):
            import json
            for fn in os.listdir(facts_dir):
                if fn.lower().endswith(".json"):
                    with open(os.path.join(facts_dir, fn), "r", encoding="utf-8") as f:
                        obj = json.load(f)
                    if isinstance(obj, dict) and obj.get("ticker"):
                        rows.append(obj)
        if rows:
            df = pd.DataFrame(rows)
            # Keep only required tickers and known fields; notes are ignored in scoring
            keep_cols = [
                "ticker",
                "revenue_growth_pct",
                "operating_margin_pct",
                "gross_margin_pct",
                "margin_trend",
                "leverage_ratio",
                "capex_intensity_pct",
                "cf_stability",
            ]
            cols = [c for c in keep_cols if c in df.columns]
            df = df[cols]
            return df[df["ticker"].isin(UNIVERSE)].reset_index(drop=True)
    except Exception as e:
        print(f"[FACTS] JSON load failed; falling back to CSV/default: {e}")

    # CSV path (simple and auditable)
    csv_path = os.path.join(here, "data", "facts.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df[df["ticker"].isin(UNIVERSE)].reset_index(drop=True)

    # tiny default table (documented approximations)
    return pd.DataFrame([
        {"ticker":"AAPL", "revenue_growth_pct":12, "operating_margin_pct":29, "margin_trend":+1.2, "leverage_ratio":0.3, "capex_intensity_pct":5, "cf_stability":0.9},
        {"ticker":"MSFT", "revenue_growth_pct":11, "operating_margin_pct":42, "margin_trend": 0.0, "leverage_ratio":0.2, "capex_intensity_pct":7, "cf_stability":0.95},
        {"ticker":"NVDA", "revenue_growth_pct":35, "operating_margin_pct":55, "margin_trend":+2.5, "leverage_ratio":0.1, "capex_intensity_pct":6, "cf_stability":0.8},
        {"ticker":"TSLA", "revenue_growth_pct":10, "operating_margin_pct":12, "margin_trend": 0.0, "leverage_ratio":0.6, "capex_intensity_pct":9, "cf_stability":0.7},
    ])
