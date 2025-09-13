"""Consensus coordinator: weighted blend + simple tie-breaks.

Rules
- Convert individual agent scores to a combined score via a weighted average.
- Reweight if any component is NaN (missing).
- Map combined score to BUY/HOLD/SELL using +/-0.5 thresholds.
- When borderline (HOLD), apply majority vote; if tied, resolve by priority:
  Fundamentals > Momentum > News.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


RATING_TO_NUM = {"SELL": -1.0, "HOLD": 0.0, "BUY": 1.0}
NUM_TO_RATING = lambda x: "BUY" if x >= 0.5 else ("SELL" if x <= -0.5 else "HOLD")


@dataclass
class Coordinator:
    """Blend agent signals and derive a final rating per ticker."""
    w_fund: float = 0.4
    w_momo: float = 0.4
    w_news: float = 0.2

    def combine(
        self,
        fundamentals: pd.Series,
        momentum: pd.Series,
        news: pd.Series,
        fund_ratings: pd.Series,
        momo_ratings: pd.Series,
        news_ratings: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Combine numeric scores and ratings into a consensus view.

        - Weighted average of numeric scores (reweights to skip missing legs)
        - Final rating via threshold on the combined score, with a transparent tie-break
        """
        tickers = sorted(set(fundamentals.index) | set(momentum.index) | set(news.index))
        rows = []
        scores = {}
        for t in tickers:
            s_f = float(fundamentals.get(t, np.nan))
            s_m = float(momentum.get(t, np.nan))
            s_n = float(news.get(t, np.nan))
            # weighted average (skip NaNs by reweighting)
            vals = []
            weights = []
            for s, w in ((s_f, self.w_fund), (s_m, self.w_momo), (s_n, self.w_news)):
                if not np.isnan(s):
                    vals.append(s)
                    weights.append(w)
            if not weights:
                wavg = 0.0
            else:
                w = np.array(weights)
                w = w / w.sum()
                wavg = float(np.dot(np.array(vals), w))
            scores[t] = wavg
            rows.append(
                {
                    "ticker": t,
                    "fund_score": s_f,
                    "momo_score": s_m,
                    "news_score": s_n,
                    "fund_rating": fund_ratings.get(t, "HOLD"),
                    "momo_rating": momo_ratings.get(t, "HOLD"),
                    "news_rating": news_ratings.get(t, "HOLD"),
                }
            )
        df = pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
        cons = pd.Series(scores)

        # final rating with tie‑breaks favoring Fundamentals > Momentum > News
        final_rating = []
        for _, r in df.iterrows():
            base = NUM_TO_RATING(cons[r["ticker"]])
            if base == "HOLD":
                # tie‑breaks among HOLD: majority vote with priority
                votes = [r["fund_rating"], r["momo_rating"], r["news_rating"]]
                if votes.count("BUY") >= 2:
                    base = "BUY"
                elif votes.count("SELL") >= 2:
                    base = "SELL"
                else:
                    # single BUY/SELL wins in priority order
                    if r["fund_rating"] != "HOLD":
                        base = r["fund_rating"]
                    elif r["momo_rating"] != "HOLD":
                        base = r["momo_rating"]
                    elif r["news_rating"] != "HOLD":
                        base = r["news_rating"]
            final_rating.append(base)
        df["final_score"] = df["ticker"].map(cons.to_dict())
        df["final_rating"] = final_rating
        return df, cons
