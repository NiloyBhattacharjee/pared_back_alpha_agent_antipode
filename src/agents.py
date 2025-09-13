# This file defines three agent classes for generating stock ratings based on different data sources and methodologies:
# 1. MomentumAgent: Uses historical price data to compute a momentum score for each ticker.
#    - Calculates trailing return over a window, normalized by volatility.
#    - Returns both the score and a rating ('BUY', 'HOLD', 'SELL') based on thresholds.
# 2. NewsSentimentAgent: Analyzes news headlines for each ticker over a lookback period.
#    - Scores headlines based on the presence of positive and negative words.
#    - Aggregates sentiment scores and converts them to ratings.
# 3. FundamentalAgent: Scores companies based on fundamental metrics: revenue growth, margins, margin trend, leverage, and cash flow stability.
#    - Each metric is normalized and weighted.
#    - Produces a composite score and a rating.
# All agents provide `score` and `rate` methods.
# The `score` method computes a numeric score per ticker.
# The `rate` method converts scores to ratings using the `_to_rating` helper.

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

from .data import POS_WORDS, NEG_WORDS


Rating = str  # 'BUY' | 'HOLD' | 'SELL'


def _to_rating(x: float, pos_th: float = 0.5, neg_th: float = -0.5) -> Rating:
    if x >= pos_th:
        return "BUY"
    if x <= neg_th:
        return "SELL"
    return "HOLD"


@dataclass
class MomentumAgent:
    """Price momentum agent (transparent, leakage-safe).

    Signal
    - Compute trailing daily returns over a lookback window
    - Score = total return over last `window` days divided by realized volatility
      (annualized denom); akin to a simple Sharpe-like ratio
    - Higher is better
    """
    lookback_days: int = 126  # ~6 months of business days
    window: int = 63  # 3 months for signal

    def score(self, prices: pd.DataFrame, as_of: datetime) -> pd.Series:
        """Return a per-ticker momentum score using prices up to `as_of`.

        Math
        - r_t = pct_change of adjusted close
        - R = product(1 + r_t) over last `window` - 1 (total return)
        - vol = std(r_t over last `window`) * sqrt(252)
        - score = R / vol (NaN if insufficient data)
        """
        df = prices.copy()
        df = df[df["date"] <= pd.to_datetime(as_of.date())]
        df = df.sort_values(["ticker", "date"])  # ensure order
        # trailing return and volatility
        def _calc(g: pd.DataFrame) -> float:
            g = g.tail(self.lookback_days)
            g = g.set_index("date")["adj_close"].pct_change().dropna()
            if len(g) < self.window + 5:
                return np.nan
            r = (1 + g.tail(self.window)).prod() - 1.0
            vol = g.tail(self.window).std(ddof=0) * np.sqrt(252)
            if vol == 0 or np.isnan(vol):
                return np.nan
            return float(r) / float(vol)

        scores = df.groupby("ticker").apply(_calc)
        return scores

    def rate(self, prices: pd.DataFrame, as_of: datetime) -> Tuple[pd.Series, pd.Series]:
        """Map scores to BUY/HOLD/SELL with thresholds (+0.5/-0.5)."""
        s = self.score(prices, as_of)
        ratings = s.apply(lambda x: _to_rating(x, 0.5, -0.5))
        return s, ratings


@dataclass
class ValuationMomentumAgent:
    """
    Price-only agent combining momentum and a simple valuation proxy.

    - Momentum: 63-day total return divided by realized volatility (annualized denom).
    - Valuation proxy: negative of deviation from a trailing moving average
      (current price vs. MA over a longer window). If price is below its
      moving average, it is considered relatively "cheaper".

    The two components are z-scored across the universe and combined.
    Ratings map from the composite score using thresholds similar to momentum.
    """
    lookback_days: int = 189   # ensure enough history for MA window
    momentum_window: int = 63  # ~3 months
    value_window: int = 126    # ~6 months MA
    w_momentum: float = 0.6
    w_value: float = 0.4

    def _per_ticker_components(self, g: pd.DataFrame, as_of: datetime) -> Tuple[float, float]:
        g = g[g["date"] <= pd.to_datetime(as_of.date())]
        g = g.sort_values("date").tail(self.lookback_days).copy()
        # momentum
        ret = g["adj_close"].pct_change().dropna()
        mom = np.nan
        if len(ret) >= self.momentum_window + 5:
            r = (1 + ret.tail(self.momentum_window)).prod() - 1.0
            vol = ret.tail(self.momentum_window).std(ddof=0) * np.sqrt(252)
            if vol and not np.isnan(vol):
                mom = float(r) / float(vol)
        # value proxy (price vs moving average)
        val = np.nan
        if len(g) >= self.value_window:
            px = float(g["adj_close"].iloc[-1])
            ma = float(g["adj_close"].rolling(self.value_window).mean().iloc[-1])
            if ma:
                # deviation from MA; negative means cheap -> invert sign
                dev = (px / ma) - 1.0
                val = -dev
        return mom, val

    def score(self, prices: pd.DataFrame, as_of: datetime) -> pd.Series:
        """Combine z-scored momentum and valuation proxies into one signal.

        - Momentum leg: as in MomentumAgent
        - Valuation leg: negative deviation from a 126-day moving average
          (below MA considered relatively cheap)
        - Combine with weights and skip missing components by reweighting
        """
        df = prices.copy()
        # compute components per ticker
        comps = df.groupby("ticker").apply(lambda g: pd.Series(self._per_ticker_components(g, as_of), index=["mom", "val"]))
        # z-score across available tickers for each component
        def zscore(s: pd.Series) -> pd.Series:
            mu = s.mean(skipna=True)
            sd = s.std(ddof=0, skipna=True)
            if sd == 0 or np.isnan(sd):
                return pd.Series([np.nan] * len(s), index=s.index)
            return (s - mu) / sd

        z_m = zscore(comps["mom"]) if "mom" in comps.columns else pd.Series(dtype=float)
        z_v = zscore(comps["val"]) if "val" in comps.columns else pd.Series(dtype=float)

        # combine with reweighting when one leg is missing
        out = {}
        for t in comps.index:
            parts = []
            weights = []
            if not np.isnan(z_m.get(t, np.nan)):
                parts.append(float(z_m[t])); weights.append(self.w_momentum)
            if not np.isnan(z_v.get(t, np.nan)):
                parts.append(float(z_v[t])); weights.append(self.w_value)
            if not weights:
                out[t] = np.nan
            else:
                w = np.array(weights)
                w = w / w.sum()
                out[t] = float(np.dot(np.array(parts), w))
        return pd.Series(out)

    def rate(self, prices: pd.DataFrame, as_of: datetime) -> Tuple[pd.Series, pd.Series]:
        """Map composite scores to BUY/HOLD/SELL with thresholds (+0.5/-0.5)."""
        s = self.score(prices, as_of)
        ratings = s.apply(lambda x: _to_rating(x, 0.5, -0.5))
        return s, ratings


@dataclass
class NewsSentimentAgent:
    """Headline sentiment agent with a transparent lexicon and optional VADER.

    - Per-headline lexicon score: (pos - neg) / (pos + neg) using words in
      `POS_WORDS` and `NEG_WORDS`
    - If VADER is installed, blend: 0.5*lexicon + 0.5*VADER compound
    - Per-ticker sentiment: mean across headlines in the lookback, restricted to `date <= as_of`
    """
    lookback_days: int = 60
    _vader: Optional[object] = None

    def __post_init__(self):
        # Optional VADER sentiment analyzer; if unavailable, remain None
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
            self._vader = SentimentIntensityAnalyzer()
        except Exception:
            self._vader = None

    def _headline_score(self, text: str) -> float:
        t = text.lower()
        pos = sum(1 for w in POS_WORDS if w in t)
        neg = sum(1 for w in NEG_WORDS if w in t)
        if pos == 0 and neg == 0:
            lex = 0.0
        else:
            lex = (pos - neg) / (pos + neg)
        if self._vader is not None:
            try:
                comp = float(self._vader.polarity_scores(text).get("compound", 0.0))
                return 0.5 * lex + 0.5 * comp
            except Exception:
                return lex
        return lex

    def score(self, news: pd.DataFrame, as_of: datetime) -> pd.Series:
        """Average headline sentiment per ticker over the trailing window."""
        if news.empty:
            return pd.Series(dtype=float)
        df = news.copy()
        df = df[df["date"] <= pd.to_datetime(as_of.date())]
        start = pd.to_datetime(as_of.date()) - timedelta(days=self.lookback_days)
        df = df[df["date"] >= start]
        if df.empty:
            return pd.Series(dtype=float)
        df["sent"] = df["headline"].astype(str).apply(self._headline_score)
        scores = df.groupby("ticker")["sent"].mean()
        return scores

    def rate(self, news: pd.DataFrame, as_of: datetime) -> Tuple[pd.Series, pd.Series]:
        """Map sentiment to BUY/HOLD/SELL using +/-0.1 thresholds."""
        s = self.score(news, as_of)
        ratings = s.apply(lambda x: _to_rating(x, 0.1, -0.1))
        return s, ratings


@dataclass
class FundamentalAgent:
    """Fundamental quality composite (clarity-first).

    Inputs (per ticker)
    - revenue_growth_pct (higher better)
    - operating_margin_pct or gross_margin_pct (higher better)
    - margin_trend (higher better)
    - leverage_ratio (lower better)
    - capex_intensity_pct (lower better)
    - cf_stability (higher better)

    Score = weighted sum of z-scored features (see weights below).
    """
    # expects df with: ticker, revenue_growth_pct, operating_margin_pct|gross_margin_pct, margin_trend,
    # leverage_ratio (lower better), capex_intensity_pct (lower better), cf_stability (0..1)
    w_growth: float = 0.30
    w_opmargin: float = 0.30
    w_trend: float = 0.15
    w_leverage: float = 0.10
    w_capex: float = 0.05
    w_cf: float = 0.10

    def score(self, facts: pd.DataFrame) -> pd.Series:
        """Compute weighted z-score composite per ticker from `facts`."""
        df = facts.copy()
        # Normalize features to ~[-1, +1]
        def nz(x):
            return (x - x.mean()) / (x.std(ddof=0) + 1e-9)

        g = nz(df["revenue_growth_pct"])  # higher better
        # Prefer operating margin; fallback to gross margin if needed
        if "operating_margin_pct" in df.columns and df["operating_margin_pct"].notna().any():
            margin = nz(df["operating_margin_pct"])  # higher better
        else:
            margin = nz(df["gross_margin_pct"])      # higher better
        t = nz(df["margin_trend"])                   # higher better
        lev = -nz(df["leverage_ratio"])              # lower leverage is better
        capex = -nz(df.get("capex_intensity_pct", pd.Series([0]*len(df))))  # lower capex intensity is better
        cf = nz(df["cf_stability"])                  # higher better
        s = (
            self.w_growth * g
            + self.w_opmargin * margin
            + self.w_trend * t
            + self.w_leverage * lev
            + self.w_capex * capex
            + self.w_cf * cf
        )
        s.index = df["ticker"].values
        return s

    def rate(self, facts: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Map composite quality score to BUY/HOLD/SELL using +/-0.3 thresholds."""
        s = self.score(facts)
        ratings = s.apply(lambda x: _to_rating(x, 0.3, -0.3))
        return s, ratings
