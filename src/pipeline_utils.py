from __future__ import annotations

import os
import pandas as pd


def write_prices_cache(prices: pd.DataFrame, path: str = "data/prices_cache.csv") -> str:
    """Write normalized prices (date,ticker,adj_close) to a cache CSV.

    Shared between CLI and notebooks to avoid duplication.
    """
    cols = {c.lower() for c in prices.columns}
    df = prices.copy()
    if "date" not in cols or "ticker" not in cols:
        raise ValueError("prices must contain 'date' and 'ticker' columns")
    price_col = None
    for c in ("adj_close", "close", "price"):
        if c in df.columns:
            price_col = c
            break
        if c.capitalize() in df.columns:
            price_col = c.capitalize()
            break
    if price_col is None:
        raise ValueError("prices must contain an 'adj_close' (or close/price) column")

    out = df[["date", "ticker", price_col]].rename(columns={price_col: "adj_close"}).copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)
    return path

