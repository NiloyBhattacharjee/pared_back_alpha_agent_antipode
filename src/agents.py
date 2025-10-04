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

        scores = df.groupby("ticker").apply(_calc, include_groups=False)
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
        comps = df.groupby("ticker").apply(lambda g: pd.Series(self._per_ticker_components(g, as_of), index=["mom", "val"]), include_groups=False)
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
    """Fundamental quality composite with configurable features, normalization, and weighting.

    Backward-compatible defaults preserve the prior behavior: simple z-scores and
    fixed weights for six core features. You can optionally enable:
    - Expanded features (R&D intensity, FCF margin, ROIC, revenue volatility / earnings stability,
      debt-to-equity) when available in `facts`.
    - Alternative normalization: rank-based, winsorized z-scores, or industry-relative z-scores.
    - Config-driven weights (dict or YAML/JSON file).
    - Per-ticker reweighting to handle missing fields robustly.
    - Optional helper to learn weights from training data via OLS/PCA.

    Expected base columns (as before):
    - ticker
    - revenue_growth_pct (higher better)
    - operating_margin_pct or gross_margin_pct (higher better)
    - margin_trend (higher better)
    - leverage_ratio (lower better)  [or debt_to_equity as alternative]
    - capex_intensity_pct (lower better)
    - cf_stability (higher better)
    Optional extra columns (used if present):
    - rd_expense, revenue -> rd_intensity = rd_expense / revenue (higher often better)
    - free_cash_flow, revenue -> fcf_margin = FCF / revenue (higher better)
    - fcf_margin_trend (higher better)
    - roic (higher better) or components to compute it externally
    - revenue_volatility (lower better) or earnings_stability (higher better)
    - debt_to_equity (lower better) if leverage_ratio missing
    - sector or industry (string) for industry-relative z-scores
    """
    # --- Base weights (unchanged defaults) ---
    # expects df with: ticker, revenue_growth_pct, operating_margin_pct|gross_margin_pct, margin_trend,
    # leverage_ratio (lower better), capex_intensity_pct (lower better), cf_stability (0..1)
    w_growth: float = 0.30
    w_opmargin: float = 0.30
    w_trend: float = 0.15
    w_leverage: float = 0.10
    w_capex: float = 0.05
    w_cf: float = 0.10

    # --- New configuration knobs (all optional; defaults preserve old behavior) ---
    norm_method: str = "z"              # one of: "z", "winsor_z", "rank", "industry_z"
    winsor_alpha: float = 0.05          # used only for winsor_z
    group_col: Optional[str] = None     # e.g., "sector" or "industry" for industry_z
    weights_path: Optional[str] = None  # YAML/JSON mapping feature->weight
    weights: Optional[Dict[str, float]] = None  # programmatic weights override
    reweight_missing: bool = True       # reweight present features per ticker
    coverage_reweight: bool = False     # downweight features with low coverage across universe

    # Extra features weights (only used if provided and column(s) available)
    extra_feature_weights: Optional[Dict[str, float]] = None

    # Learned weights storage (set by fit_weights_* helpers)
    _learned_weights: Optional[Dict[str, float]] = None

    # ------------------ helpers ------------------
    def _load_weight_overrides(self) -> Optional[Dict[str, float]]:
        if self.weights is not None:
            return dict(self.weights)
        if not self.weights_path:
            return None
        try:
            import os, json
            path = self.weights_path
            if not os.path.exists(path):
                return None
            # Try YAML first then JSON
            try:
                import yaml  # type: ignore

                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
            except Exception:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            if isinstance(data, dict):
                return {str(k): float(v) for k, v in data.items() if v is not None}
        except Exception:
            return None
        return None

    def _zscore(self, s: pd.Series) -> pd.Series:
        mu = s.mean(skipna=True)
        sd = s.std(ddof=0, skipna=True)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.nan, index=s.index)
        return (s - mu) / sd

    def _winsorize(self, s: pd.Series, alpha: float) -> pd.Series:
        lo = s.quantile(alpha)
        hi = s.quantile(1 - alpha)
        return s.clip(lower=lo, upper=hi)

    def _rank_norm(self, s: pd.Series) -> pd.Series:
        # Rank -> z-like by z-scoring ranks (robust to outliers; preserves ordering)
        r = s.rank(method="average", na_option="keep")
        return self._zscore(r)

    def _industry_z(self, s: pd.Series, groups: pd.Series) -> pd.Series:
        if groups is None or groups.empty:
            return self._zscore(s)
        def _z(g: pd.Series) -> pd.Series:
            return self._zscore(g)
        z = s.groupby(groups).transform(_z)
        # If any groups had zero variance, those positions may be NaN; backfill with global z
        z_glob = self._zscore(s)
        return z.where(~z.isna(), z_glob)

    def _normalize(self, s: pd.Series, df: pd.DataFrame) -> pd.Series:
        s = s.replace([np.inf, -np.inf], np.nan)
        if self.norm_method == "z":
            return self._zscore(s)
        elif self.norm_method == "winsor_z":
            return self._zscore(self._winsorize(s, self.winsor_alpha))
        elif self.norm_method == "rank":
            return self._rank_norm(s)
        elif self.norm_method == "industry_z":
            grp = df[self.group_col] if (self.group_col and self.group_col in df.columns) else pd.Series(index=df.index, dtype=object)
            return self._industry_z(s, grp)
        else:
            # fallback to z
            return self._zscore(s)

    def _base_weights(self) -> Dict[str, float]:
        # Build base weights dict consistent with previous implementation
        return {
            "revenue_growth_pct": self.w_growth,
            "margin": self.w_opmargin,
            "margin_trend": self.w_trend,
            "leverage_effect": self.w_leverage,
            "capex_intensity_pct": self.w_capex,
            "cf_stability": self.w_cf,
        }

    def _collect_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        # Core features (preserve legacy behavior)
        feats: Dict[str, pd.Series] = {}
        # revenue growth
        if "revenue_growth_pct" in df.columns:
            feats["revenue_growth_pct"] = df["revenue_growth_pct"].astype(float)
        # margin: prefer operating, else gross
        if "operating_margin_pct" in df.columns and df["operating_margin_pct"].notna().any():
            feats["margin"] = df["operating_margin_pct"].astype(float)
        elif "gross_margin_pct" in df.columns:
            feats["margin"] = df["gross_margin_pct"].astype(float)
        # margin trend
        if "margin_trend" in df.columns:
            feats["margin_trend"] = df["margin_trend"].astype(float)
        # leverage: prefer explicit leverage_ratio; else debt_to_equity
        if "leverage_ratio" in df.columns:
            lev = df["leverage_ratio"].astype(float)
        elif "debt_to_equity" in df.columns:
            lev = df["debt_to_equity"].astype(float)
        else:
            lev = pd.Series(np.nan, index=df.index)
        # store as effect where higher is better (invert sign later via normalization path)
        feats["leverage_raw"] = lev
        # capex intensity (lower better)
        if "capex_intensity_pct" in df.columns:
            feats["capex_intensity_pct"] = df["capex_intensity_pct"].astype(float)
        # cash flow stability
        if "cf_stability" in df.columns:
            feats["cf_stability"] = df["cf_stability"].astype(float)

        # --- Optional expanded features (only if columns exist) ---
        # R&D intensity = R&D / Revenue
        if "rd_expense" in df.columns and "revenue" in df.columns:
            with np.errstate(divide='ignore', invalid='ignore'):
                rd_int = (df["rd_expense"].astype(float) / df["revenue"].astype(float)).replace([np.inf, -np.inf], np.nan)
            feats["rd_intensity"] = rd_int
        # FCF margin (level) or trend if available
        if "free_cash_flow" in df.columns and "revenue" in df.columns:
            with np.errstate(divide='ignore', invalid='ignore'):
                fcf_margin = (df["free_cash_flow"].astype(float) / df["revenue"].astype(float)).replace([np.inf, -np.inf], np.nan)
            feats["fcf_margin"] = fcf_margin
        if "fcf_margin_trend" in df.columns:
            feats["fcf_margin_trend"] = df["fcf_margin_trend"].astype(float)
        # ROIC
        if "roic" in df.columns:
            feats["roic"] = df["roic"].astype(float)
        # Revenue volatility (lower better) or earnings stability (higher better)
        if "revenue_volatility" in df.columns:
            feats["revenue_volatility"] = df["revenue_volatility"].astype(float)
        if "earnings_stability" in df.columns:
            feats["earnings_stability"] = df["earnings_stability"].astype(float)

        return feats

    def _feature_weights(self, df: pd.DataFrame) -> Dict[str, float]:
        # Start with base weights mapping to collected feature keys
        w = self._base_weights()
        # Map leverage to internal key
        # Use a single key 'leverage_effect' so downstream logic is unified
        if "leverage_raw" in df.columns:
            pass  # placeholder for readability

        # Inject any extra features weights if provided
        if self.extra_feature_weights:
            for k, v in self.extra_feature_weights.items():
                w[str(k)] = float(v)

        # External overrides: learned > explicit weights arg > file
        overrides = self._learned_weights or self._load_weight_overrides()
        if overrides:
            for k, v in overrides.items():
                w[str(k)] = float(v)
        return w

    def score(self, facts: pd.DataFrame) -> pd.Series:
        """Compute a weighted composite per ticker from `facts`.

        Defaults: z-scores + fixed six-feature weights (legacy behavior).
        If `norm_method`/`weights`/`weights_path`/`extra_feature_weights` are provided,
        they modify behavior as described above.
        """
        if facts.empty:
            return pd.Series(dtype=float)
        df = facts.copy()
        if "ticker" not in df.columns:
            raise ValueError("facts must include a 'ticker' column")
        # Keep a stable alignment index
        idx = df.index

        feats_raw = self._collect_features(df)
        if not feats_raw:
            return pd.Series(dtype=float)

        # Build a working DataFrame of features for normalization
        F = pd.DataFrame(feats_raw)

        # Normalize each feature. For features where lower is better, invert sign after normalization
        normed: Dict[str, pd.Series] = {}
        for name, s in F.items():
            if name == "leverage_raw":
                z = self._normalize(s.astype(float), df)
                normed["leverage_effect"] = -z  # lower leverage -> higher score
            elif name == "capex_intensity_pct" or name == "revenue_volatility":
                z = self._normalize(s.astype(float), df)
                normed[name] = -z  # lower better
            else:
                z = self._normalize(s.astype(float), df)
                normed[name] = z  # higher better

        Z = pd.DataFrame(normed)

        # Assemble weights
        w_map = self._feature_weights(Z)

        # Optionally downweight features with poor coverage across tickers
        if self.coverage_reweight:
            coverage = (~Z.isna()).mean(axis=0)  # fraction non-NaN per feature
            for k in list(w_map.keys()):
                if k in coverage.index:
                    w_map[k] = float(w_map[k]) * float(coverage[k])

        # Compute per-ticker weighted sum with reweighting over available features
        out = {}
        for i, row in Z.iterrows():
            parts = []
            weights = []
            for k, v in w_map.items():
                if k not in Z.columns:
                    continue
                val = row.get(k)
                if pd.isna(val):
                    continue
                parts.append(float(val))
                weights.append(float(v))
            if not weights:
                out[df.loc[i, "ticker"]] = np.nan
            else:
                w = np.array(weights)
                if self.reweight_missing:
                    if w.sum() == 0:
                        out[df.loc[i, "ticker"]] = np.nan
                        continue
                    w = w / w.sum()
                out[df.loc[i, "ticker"]] = float(np.dot(np.array(parts), w))

        return pd.Series(out)

    def rate(self, facts: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Map composite quality score to BUY/HOLD/SELL using +/-0.3 thresholds."""
        s = self.score(facts)
        ratings = s.apply(lambda x: _to_rating(x, 0.3, -0.3))
        return s, ratings

    # ------------------ optional learning ------------------
    def fit_weights_ols(self, X: pd.DataFrame, y: pd.Series, feature_map: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Fit linear weights via OLS on provided training matrix.

        Parameters
        - X: DataFrame with columns matching feature names used by this agent
             (e.g., 'revenue_growth_pct','margin','margin_trend','leverage_effect',...).
        - y: Series target to fit (e.g., forward returns or realized outperformance)
        - feature_map: optional mapping from external column names to agent feature names.

        Returns a dict of learned weights and stores them internally for future scoring.
        """
        if feature_map:
            X2 = X.rename(columns=feature_map)
        else:
            X2 = X.copy()
        # Keep only columns known to the agent to avoid leakage from unknowns
        valid_cols = set([
            "revenue_growth_pct","margin","margin_trend","leverage_effect",
            "capex_intensity_pct","cf_stability","rd_intensity","fcf_margin",
            "fcf_margin_trend","roic","revenue_volatility","earnings_stability",
        ])
        cols = [c for c in X2.columns if c in valid_cols]
        if not cols:
            raise ValueError("No valid feature columns found for OLS fit.")
        X3 = X2[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        yv = y.reindex(X3.index).astype(float).fillna(0.0)
        try:
            # Minimal OLS using numpy (no external deps)
            A = X3.values
            w, *_ = np.linalg.lstsq(A, yv.values, rcond=None)
            learned = {c: float(w[i]) for i, c in enumerate(cols)}
            self._learned_weights = learned
            return learned
        except Exception as e:
            raise RuntimeError(f"OLS weight fit failed: {e}")

    def fit_weights_pca(self, X: pd.DataFrame, n_components: int = 1, feature_map: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Derive weights from the first principal component loading (unsupervised).

        Returns a weight dict normalized to sum of absolute values = 1.
        """
        if feature_map:
            X2 = X.rename(columns=feature_map)
        else:
            X2 = X.copy()
        valid_cols = [
            "revenue_growth_pct","margin","margin_trend","leverage_effect",
            "capex_intensity_pct","cf_stability","rd_intensity","fcf_margin",
            "fcf_margin_trend","roic","revenue_volatility","earnings_stability",
        ]
        cols = [c for c in X2.columns if c in valid_cols]
        if not cols:
            raise ValueError("No valid feature columns found for PCA fit.")
        X3 = X2[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        # Standardize features (z-score) before PCA
        X3 = (X3 - X3.mean()) / (X3.std(ddof=0) + 1e-9)
        # Power method for first PC (simple and dependency-free)
        M = np.cov(X3.T, ddof=0)
        v = np.ones(len(cols)) / np.sqrt(len(cols))
        for _ in range(50):
            v_new = M @ v
            nrm = np.linalg.norm(v_new)
            if nrm == 0:
                break
            v = v_new / nrm
        loadings = v  # approximate first eigenvector
        # Normalize to sum|w|=1 for interpretability
        s_abs = np.sum(np.abs(loadings)) + 1e-12
        learned = {c: float(loadings[i] / s_abs) for i, c in enumerate(cols)}
        self._learned_weights = learned
        return learned

