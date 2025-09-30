"""
Train a simple fundamentals-based ML model to predict forward active return.

Notes
- Fundamentals are typically lower-frequency; this script pairs each date in
  the cache with the current fundamentals snapshot per ticker (repeated over
  time), to provide enough samples for a basic linear model. For real use,
  supply historical fundamentals over time.

Output
- Saves a scikit-learn pipeline to artifacts/fund_model.pkl

Usage
  python scripts/train_fund_ml.py --forward-days 21
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from src.data import load_facts


def _load_cache(path: str = "data/prices_cache.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"date", "ticker", "adj_close"}.issubset(df.columns):
        raise ValueError("prices_cache.csv must contain columns: date,ticker,adj_close")
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def _build_dataset(df: pd.DataFrame, facts: pd.DataFrame, forward_days: int) -> pd.DataFrame:
    # Forward return aligned to t
    fwd = (
        df.sort_values(["ticker", "date"])\
          .assign(fwd=lambda d: d.groupby("ticker")["adj_close"].shift(-forward_days) / d["adj_close"] - 1.0)
    )[["date", "ticker", "fwd"]]
    bench = fwd.groupby("date")["fwd"].mean().rename("bench").reset_index()
    y = fwd.merge(bench, on="date", how="inner")
    y["y_active"] = y["fwd"] - y["bench"]

    # Repeat fundamentals across dates (best-effort until time-series fundamentals are added)
    facts = facts.copy()
    cols = [
        "revenue_growth_pct",
        "operating_margin_pct",
        "gross_margin_pct",
        "margin_trend",
        "leverage_ratio",
        "capex_intensity_pct",
        "cf_stability",
    ]
    for c in cols:
        if c not in facts.columns:
            facts[c] = 0.0

    # Join per (date,ticker)
    ds = y.merge(facts, on="ticker", how="left")
    ds = ds.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return ds


def train(ds: pd.DataFrame, out_path: str = "artifacts/fund_model.pkl") -> None:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error
    import joblib

    ds = ds.sort_values("date").reset_index(drop=True)
    feature_cols = [
        c
        for c in ds.columns
        if c
        not in {
            "date",
            "ticker",
            "y_active",
            "fwd",
            "bench",
        }
    ]
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
    ap.add_argument("--out", default="artifacts/fund_model.pkl")
    args = ap.parse_args()

    df = _load_cache(args.cache)
    facts = load_facts()
    ds = _build_dataset(df, facts, args.forward_days)
    if ds.empty:
        raise SystemExit("Dataset is empty. Ensure cache and facts exist.")
    train(ds, args.out)


if __name__ == "__main__":
    main()

