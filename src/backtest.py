"""
Backtest (single-period, leakage-safe)

Notation
- P_i(d): adjusted close for ticker i on date d
- r_i(d): per-ticker daily return = P_i(d)/P_i(d-1) - 1
- r_bench(d): equal-weight average of r_i(d) across the fixed universe
- B: set of consensus BUY tickers at as_of (weights w_i = 1/|B|)
- r_port(d): equal-weight portfolio return = sum_{i in B} w_i * r_i(d)
- r_active(d): r_port(d) - r_bench(d)
- cum_x(d): cumulative product of (1 + r_x) over the forward window
- sharpe_proxy: mean(r_active) / std(r_active) using population std (ddof=0)

Leakage controls
- Agents compute scores only with data up to and including as_of
- Forward returns begin the next business day after as_of
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def equal_weight_benchmark(prices: pd.DataFrame, as_of: datetime, forward_days: int = 21) -> pd.DataFrame:
    """Equal-weight benchmark over the forward window.

    - Window: business days from the next business day after `as_of` for `forward_days` days
    - Per-ticker return: pct_change of adjusted close (no rebasing/anchoring needed)
    - Aggregation: simple mean across tickers by date
    """
    df = prices.copy().sort_values(["ticker", "date"])  # ensure order
    start_date = pd.bdate_range(start=pd.to_datetime(as_of.date()) + pd.offsets.BDay(1), periods=1)[0]
    end_date = pd.bdate_range(start=start_date, periods=forward_days)[-1]
    fwd = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
    if fwd.empty:
        return pd.DataFrame(columns=["benchmark_return"])  # nothing in window
    fwd["ret"] = fwd.groupby("ticker")["adj_close"].pct_change()
    fwd = fwd.dropna(subset=["ret"])  # first day per ticker has NaN
    if fwd.empty:
        return pd.DataFrame(columns=["benchmark_return"])  # no second point per ticker
    bench = fwd.groupby("date")["ret"].mean().rename("benchmark_return").to_frame()
    return bench


@dataclass
class Backtester:
    forward_days: int = 21

    def _portfolio_weights(self, final_ratings: pd.Series) -> Dict[str, float]:
        buys = [t for t, r in final_ratings.items() if r == "BUY"]
        if not buys:
            return {}  # will trigger benchmark fallback
        w = 1.0 / len(buys)
        return {t: w for t in buys}

    def run(self, prices: pd.DataFrame, as_of: datetime, final_ratings: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run the backtest for a single decision date.

        Steps
        1) Build forward business-day index strictly after `as_of` of length `forward_days`.
        2) Compute equal-weight benchmark returns across all four tickers.
        3) Compute portfolio returns as the equal-weight average across BUY names.
           If there are no BUYs, the portfolio tracks the benchmark.
        4) Join series on overlapping dates (no reindexing/filling), then compute:
           - active = portfolio - benchmark
           - cumulative curves via cumulative product of (1 + returns)
           - a simple Sharpe-style proxy = mean(active) / std(active) (population std)
        """
        df = prices.copy().sort_values(["ticker", "date"])  # ensure order
        # define forward window strictly after as_of
        start_date = pd.bdate_range(start=pd.to_datetime(as_of.date()) + pd.offsets.BDay(1), periods=1)[0]
        forward_index = pd.bdate_range(start=start_date, periods=self.forward_days)

        # portfolio weights at as_of
        w = self._portfolio_weights(final_ratings)

        # build benchmark
        bench = equal_weight_benchmark(prices, as_of, self.forward_days)

        # compute portfolio returns (simple equal-weight across BUYs)
        if not w:
            # fallback: track benchmark if no BUYs
            port = bench.rename(columns={"benchmark_return": "portfolio_return"})
        else:
            sub = df[df["ticker"].isin(w.keys())].copy()
            sub = sub[(sub["date"] >= forward_index[0]) & (sub["date"] <= forward_index[-1])]
            if sub.empty:
                port = pd.DataFrame(columns=["portfolio_return"])  # nothing in window
            else:
                sub["ret"] = sub.groupby("ticker")["adj_close"].pct_change()
                sub = sub.dropna(subset=["ret"])  # drop first day per ticker
                if sub.empty:
                    port = pd.DataFrame(columns=["portfolio_return"])  # need at least 2 points per ticker
                else:
                    sub["weight"] = sub["ticker"].map(w)
                    port = sub.groupby("date").apply(lambda g: (g["ret"] * g["weight"]).sum(), include_groups=False)
                    if isinstance(port, pd.Series):
                        port = port.rename("portfolio_return").to_frame()
                    else:
                        port = port.to_frame(name="portfolio_return")

        # Join on dates where both series exist; keep it simple (no reindex, no fills)
        perf = bench.join(port, how="inner").sort_index()
        if perf.empty:
            # Fallback: if nothing overlaps, mirror the benchmark to produce a valid curve
            perf = bench.copy()
            if not perf.empty:
                perf["portfolio_return"] = perf["benchmark_return"]
            else:
                empty = pd.DataFrame(columns=["benchmark_return", "portfolio_return", "active_return", "cum_benchmark", "cum_portfolio"])  # noqa: E501
                summary = pd.DataFrame({
                    "metric": ["buy_count", "sharpe_proxy", "total_active"],
                    "value": [float(len([x for x in final_ratings.values if x == "BUY"])), 0.0, 0.0],
                })
                return empty, summary
        perf["active_return"] = perf["portfolio_return"] - perf["benchmark_return"]
        perf["cum_benchmark"] = (1 + perf["benchmark_return"]).cumprod()
        perf["cum_portfolio"] = (1 + perf["portfolio_return"]).cumprod()

        # summary
        mean_active = perf["active_return"].mean()
        vol_active = perf["active_return"].std(ddof=0)
        sharpe_proxy = (mean_active / vol_active) if vol_active > 0 else 0.0
        summary = pd.DataFrame(
            {
                "metric": ["buy_count", "sharpe_proxy", "total_active"],
                "value": [float(len([x for x in final_ratings.values if x == "BUY"])), float(sharpe_proxy), float(perf["active_return"].sum())],
            }
        )
        return perf, summary

