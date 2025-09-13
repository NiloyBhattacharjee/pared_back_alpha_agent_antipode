from __future__ import annotations

import os
import pandas as pd


def _normalize_prices_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a canonical (date,ticker,adj_close) DataFrame.

    - Ensures timezone-naive datetime
    - Uppercases ticker
    - Drops duplicates on (ticker,date)
    - Rounds adj_close to 6 decimals
    - Sorts by date then ticker for stable output
    """
    df = df.copy()
    # identify price column
    price_col = None
    for c in ("adj_close", "close", "price", "Adj Close", "Close"):
        if c in df.columns:
            price_col = c
            break
    if price_col is None:
        raise ValueError("cache write: missing price column (adj_close/close/price)")
    # build canonical
    out = df[["date", "ticker", price_col]].rename(columns={price_col: "adj_close"}).copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out["ticker"] = out["ticker"].astype(str).str.upper()
    out["adj_close"] = pd.to_numeric(out["adj_close"], errors="coerce").round(6)
    out = out.dropna(subset=["date", "ticker", "adj_close"])  # basic hygiene
    out = out.drop_duplicates(subset=["ticker", "date"], keep="last")
    out = out.sort_values(["date", "ticker"]).reset_index(drop=True)
    return out


def write_prices_cache(prices: pd.DataFrame, path: str = "data/prices_cache.csv") -> str:
    """Write normalized prices (date,ticker,adj_close) to a cache CSV.

    Behavior
    - Merges with existing cache if present (idempotent, de-duplicated)
    - Produces stable ordering and formatting across runs
    - Does not write unless explicitly called
    """
    if "date" not in prices.columns or "ticker" not in prices.columns:
        raise ValueError("prices must contain 'date' and 'ticker' columns")

    new_norm = _normalize_prices_frame(prices)

    # Merge with existing cache if present
    if os.path.exists(path):
        try:
            existing = pd.read_csv(path)
            exist_norm = _normalize_prices_frame(existing)
            merged = pd.concat([exist_norm, new_norm], ignore_index=True)
            merged = merged.drop_duplicates(subset=["ticker", "date"], keep="last")
            merged = merged.sort_values(["date", "ticker"]).reset_index(drop=True)
        except Exception as e:
            # If existing cache is malformed, fall back to writing the new data only
            merged = new_norm
    else:
        merged = new_norm

    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Use consistent newline and float formatting for cross-platform stability
    merged.to_csv(path, index=False, float_format="%.6f", lineterminator="\n")
    return path
