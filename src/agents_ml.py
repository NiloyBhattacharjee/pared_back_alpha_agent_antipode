from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

from .agents import ValuationMomentumAgent, FundamentalAgent, _to_rating


def _try_load_model(path: str):
    try:
        import joblib  # type: ignore

        return joblib.load(path)
    except Exception:
        return None


@dataclass
class PriceMLAgent:
    """Price/features ML agent.

    Attempts to load a regression model from `model_path` that maps engineered
    price features to an expected forward active return. If the model is not
    available, it gracefully falls back to the rule-based
    ValuationMomentumAgent score so the pipeline remains usable offline.
    """

    model_path: str = "artifacts/price_model.pkl"
    lookback_days: int = 252
    momentum_window: int = 63
    value_window: int = 126

    def _features(self, prices: pd.DataFrame, as_of: datetime) -> pd.DataFrame:
        df = prices.copy()
        df = df[df["date"] <= pd.to_datetime(as_of.date())]
        df = df.sort_values(["ticker", "date"]).copy()

        feats: List[pd.DataFrame] = []
        for t, g in df.groupby("ticker"):
            g = g.tail(self.lookback_days).copy()
            g = g.set_index("date")
            px = g["adj_close"].astype(float)
            r = px.pct_change()

            def dev_ma(w: int):
                ma = px.rolling(w).mean()
                return (px / ma) - 1.0

            out = pd.DataFrame(index=px.index)
            # momentum returns
            for w in (5, 21, 63, 126):
                out[f"ret_{w}"] = (1 + r).rolling(w).apply(np.prod, raw=True) - 1.0
            # 12-1 momentum (skip recent month)
            if len(r) >= 252:
                past = (1 + r.shift(21)).rolling(252 - 21).apply(np.prod, raw=True) - 1.0
                out["ret_12m_ex1m"] = past
            # vols
            for w in (21, 63):
                out[f"vol_{w}"] = r.rolling(w).std(ddof=0) * np.sqrt(252)
            # valuation proxies
            for w in (20, 63, self.value_window):
                out[f"dev_ma_{w}"] = dev_ma(w)

            out["ticker"] = t
            feats.append(out.tail(1))  # last row as current snapshot

        X = pd.concat(feats, ignore_index=False)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X.index.name = "date"
        # Use ticker as index for prediction alignment
        tickers = X["ticker"].values
        X = X.drop(columns=["ticker"]) if "ticker" in X.columns else X
        X.index = pd.Index(tickers, name="ticker")
        return X

    def score(self, prices: pd.DataFrame, as_of: datetime) -> pd.Series:
        model = _try_load_model(self.model_path)
        if model is None:
            # Fallback to rule-based valuation/momentum composite
            rule = ValuationMomentumAgent(
                lookback_days=max(self.lookback_days, self.value_window + 30),
                momentum_window=self.momentum_window,
                value_window=self.value_window,
            )
            return rule.score(prices, as_of)

        X = self._features(prices, as_of)
        try:
            pred = model.predict(X)
            return pd.Series(pred, index=X.index)
        except Exception:
            # If model inference fails, fall back to rule score
            rule = ValuationMomentumAgent(
                lookback_days=max(self.lookback_days, self.value_window + 30),
                momentum_window=self.momentum_window,
                value_window=self.value_window,
            )
            return rule.score(prices, as_of)

    def rate(self, prices: pd.DataFrame, as_of: datetime) -> Tuple[pd.Series, pd.Series]:
        s = self.score(prices, as_of)
        # Conservative thresholds for an ML expected-active-return style score
        ratings = s.apply(lambda x: _to_rating(x, 0.1, -0.1))
        return s, ratings


@dataclass
class FundamentalMLAgent:
    """Fundamentals ML agent.

    Loads a model from `model_path` mapping fundamental features to an expected
    active return. If the model is not available, falls back to the
    rule-based FundamentalAgent score.
    """

    model_path: str = "artifacts/fund_model.pkl"

    def _features(self, facts: pd.DataFrame) -> pd.DataFrame:
        # For now, assume input `facts` has the required columns. In training,
        # match the exact preprocessing used. Here we forward-fill missing with 0.
        cols = [
            "revenue_growth_pct",
            "operating_margin_pct",
            "gross_margin_pct",
            "margin_trend",
            "leverage_ratio",
            "capex_intensity_pct",
            "cf_stability",
        ]
        X = facts.copy()
        for c in cols:
            if c not in X.columns:
                X[c] = 0.0
        X = X[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X.index = facts["ticker"].astype(str)
        return X

    def score(self, facts: pd.DataFrame) -> pd.Series:
        model = _try_load_model(self.model_path)
        if model is None:
            return FundamentalAgent().score(facts)
        X = self._features(facts)
        try:
            pred = model.predict(X)
            return pd.Series(pred, index=X.index)
        except Exception:
            return FundamentalAgent().score(facts)

    def rate(self, facts: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        s = self.score(facts)
        # Keep thresholds aligned with rule-based fundamentals for familiarity
        ratings = s.apply(lambda x: _to_rating(x, 0.3, -0.3))
        return s, ratings

