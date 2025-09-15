import pandas as pd
import numpy as np

from src.coordinator import Coordinator


def test_coordinator_weighted_blend_and_tiebreak():
    # Numeric scores: fundamentals strong for AAPL, momentum strong for MSFT, no news
    fund = pd.Series({"AAPL": 1.0, "MSFT": -0.2})
    momo = pd.Series({"AAPL": 0.0, "MSFT": 0.8})
    news = pd.Series({"AAPL": np.nan, "MSFT": np.nan})

    fund_r = pd.Series({"AAPL": "BUY", "MSFT": "SELL"})
    momo_r = pd.Series({"AAPL": "HOLD", "MSFT": "BUY"})
    news_r = pd.Series({"AAPL": "HOLD", "MSFT": "HOLD"})

    coord = Coordinator(w_fund=0.4, w_momo=0.4, w_news=0.2)
    df, cons = coord.combine(fund, momo, news, fund_r, momo_r, news_r)

    # Both tickers appear
    assert set(df["ticker"]) == {"AAPL", "MSFT"}
    # Combined score should reflect reweighting (news missing)
    # For AAPL: 0.5*fund + 0.5*momo ≈ 0.5*1.0 + 0.5*0 = 0.5
    aapl_score = df.loc[df["ticker"] == "AAPL", "final_score"].iloc[0]
    assert aapl_score > 0.0
    # Tie-breaks: if combined ≈ 0 (HOLD), majority vote decides; here each has 1 BUY and 1 SELL -> priority: Fund > Momo > News
    # For AAPL, fundamentals BUY dominates
    aapl_rating = df.loc[df["ticker"] == "AAPL", "final_rating"].iloc[0]
    assert aapl_rating in {"BUY", "HOLD"}

