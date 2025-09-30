"""
Train a simple price-based ML model to predict forward active return.

Inputs
- Reads canonical prices from data/prices_cache.csv (date,ticker,adj_close)
- Uses rolling features (momentum, volatility, price/MA deviation)

Target
- For each (date=t, ticker=i): y_i(t) = fwd_ret_i(t) - mean_j fwd_ret_j(t)
  where fwd_ret_i(t) = adj_close_{t+F}/adj_close_t - 1 and F=forward_days

Output
- Saves a scikit-learn pipeline to artifacts/price_model.pkl

Usage
  python scripts/train_price_ml.py --forward-days 21
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import pandas as pd


def _load_cache(path: str = "data/prices_cache.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"date", "ticker", "adj_close"}.issubset(df.columns):
        raise ValueError("prices_cache.csv must contain columns: date,ticker,adj_close")
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    feats: List[pd.DataFrame] = []
    for t, g in df.groupby("ticker"):
        g = g.copy()
        px = g["adj_close"].astype(float).values
        dates = g["date"].values
        s = pd.Series(px, index=pd.to_datetime(dates))
        r = s.pct_change()

        out = pd.DataFrame(index=s.index)
        # Momentum-style windowed returns
        for w in (5, 21, 63, 126):
            out[f"ret_{w}"] = (1 + r).rolling(w).apply(np.prod, raw=True) - 1.0
        # 12-1 momentum
        out["ret_12m_ex1m"] = (1 + r.shift(21)).rolling(252 - 21).apply(np.prod, raw=True) - 1.0
        # Realized vols
        for w in (21, 63):
            out[f"vol_{w}"] = r.rolling(w).std(ddof=0) * np.sqrt(252)
        # Price vs MAs
        for w in (20, 63, 126):
            ma = s.rolling(w).mean()
            out[f"dev_ma_{w}"] = (s / ma) - 1.0

        out["ticker"] = t
        feats.append(out)
    X = pd.concat(feats).reset_index().rename(columns={"index": "date"})
    return X


def _build_dataset(df: pd.DataFrame, forward_days: int) -> pd.DataFrame:
    # Features aligned to t
    X = _build_features(df)
    # Forward return aligned to t
    fwd = (
        df.sort_values(["ticker", "date"])\
          .assign(fwd=lambda d: d.groupby("ticker")["adj_close"].shift(-forward_days) / d["adj_close"] - 1.0)
    )[["date", "ticker", "fwd"]]
    # Benchmark (equal-weight) per date
    bench = fwd.groupby("date")["fwd"].mean().rename("bench").reset_index()
    y = fwd.merge(bench, on="date", how="inner")
    y["y_active"] = y["fwd"] - y["bench"]

    ds = X.merge(y[["date", "ticker", "y_active"]], on=["date", "ticker"], how="inner")
    # Drop rows without full forward window or with NaNs
    ds = ds.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return ds


def train(ds: pd.DataFrame, out_path: str = "artifacts/price_model.pkl") -> None:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error
    import joblib

    # Order by date to respect temporal structure
    ds = ds.sort_values("date").reset_index(drop=True)
    feature_cols = [c for c in ds.columns if c not in {"date", "ticker", "y_active"}]
    X = ds[feature_cols].values
    y = ds["y_active"].values

    alphas = [0.1, 1.0, 10.0]
    tscv = TimeSeriesSplit(n_splits=5)
    best_alpha, best_score = None, float("inf")
    for a in alphas:
        mse_vals = []
        for train_idx, val_idx in tscv.split(X):
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=a, random_state=7)),
            ])
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[val_idx])
            mse_vals.append(mean_squared_error(y[val_idx], pred))
        avg_mse = float(np.mean(mse_vals)) if mse_vals else float("inf")
        if avg_mse < best_score:
            best_alpha, best_score = a, avg_mse

    final = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=best_alpha or 1.0, random_state=7)),
    ])
    final.fit(X, y)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(final, out_path)
    print(f"Saved model to {out_path} (alpha={best_alpha}, n={len(ds)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward-days", type=int, default=21)
    ap.add_argument("--cache", default="data/prices_cache.csv")
    ap.add_argument("--out", default="artifacts/price_model.pkl")
    args = ap.parse_args()

    df = _load_cache(args.cache)
    ds = _build_dataset(df, args.forward_days)
    if ds.empty:
        raise SystemExit("Dataset is empty. Ensure data/prices_cache.csv has enough history and forward window.")
    train(ds, args.out)


if __name__ == "__main__":
    main()

