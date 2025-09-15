from datetime import datetime, timedelta
import pandas as pd

from src.agents import MomentumAgent, ValuationMomentumAgent, NewsSentimentAgent, FundamentalAgent


def test_momentum_agent_score_basic():
    # Construct a simple price series with upward drift for AAPL and flat for MSFT
    dates = pd.bdate_range(end=pd.Timestamp("2024-01-31"), periods=20)
    aapl_px = pd.Series(range(100, 120), index=dates).astype(float)
    msft_px = pd.Series([200.0] * 20, index=dates)
    df = pd.concat(
        [
            pd.DataFrame({"date": aapl_px.index, "ticker": "AAPL", "adj_close": aapl_px.values}),
            pd.DataFrame({"date": msft_px.index, "ticker": "MSFT", "adj_close": msft_px.values}),
        ],
        ignore_index=True,
    )
    agent = MomentumAgent(lookback_days=20, window=10)
    s = agent.score(df, as_of=datetime(2024, 1, 31))
    assert "AAPL" in s.index and "MSFT" in s.index
    # AAPL should have higher momentum score than MSFT (flat)
    assert s["AAPL"] > s["MSFT"] or pd.isna(s["MSFT"])  # tolerate NaN for flat volatility edge-case


def test_valuation_momentum_agent_value_leg_effect():
    # Two tickers with same returns but one below its 126d MA (cheaper)
    dates = pd.bdate_range(end=pd.Timestamp("2024-01-31"), periods=190)
    # both linearly increasing; tweak TSLA last price lower to be below MA
    aapl = pd.Series(range(100, 290), index=dates).astype(float)
    tsla = pd.Series(range(100, 290), index=dates).astype(float)
    tsla.iloc[-1] = tsla.iloc[-1] * 0.95  # price below MA -> more "value"
    df = pd.concat(
        [
            pd.DataFrame({"date": aapl.index, "ticker": "AAPL", "adj_close": aapl.values}),
            pd.DataFrame({"date": tsla.index, "ticker": "TSLA", "adj_close": tsla.values}),
        ],
        ignore_index=True,
    )
    agent = ValuationMomentumAgent(momentum_window=10, w_momentum=0.1, w_value=0.9)
    s = agent.score(df, as_of=datetime(2024, 1, 31))
    # TSLA should score higher due to positive value leg
    assert s["TSLA"] > s["AAPL"]


def test_news_sentiment_positive_vs_negative():
    # Same ticker, two headlines: one positive, one negative
    as_of = datetime(2024, 9, 30)
    news = pd.DataFrame(
        [
            {"date": pd.to_datetime("2024-09-10"), "ticker": "NVDA", "headline": "NVDA strong record growth upgrade"},
            {"date": pd.to_datetime("2024-09-12"), "ticker": "NVDA", "headline": "NVDA miss weak decline loss"},
        ]
    )
    agent = NewsSentimentAgent(lookback_days=60)
    s = agent.score(news, as_of=as_of)
    # Score should be around 0 (one positive, one negative), but finite
    assert "NVDA" in s.index
    assert s["NVDA"] <= 1.0 and s["NVDA"] >= -1.0


def test_fundamental_agent_ordering():
    facts = pd.DataFrame(
        [
            {"ticker": "AAPL", "revenue_growth_pct": 10, "operating_margin_pct": 30, "margin_trend": 0.5, "leverage_ratio": 0.4, "capex_intensity_pct": 5, "cf_stability": 0.9},
            {"ticker": "MSFT", "revenue_growth_pct": 8, "operating_margin_pct": 28, "margin_trend": 0.3, "leverage_ratio": 0.6, "capex_intensity_pct": 6, "cf_stability": 0.8},
        ]
    )
    agent = FundamentalAgent()
    s = agent.score(facts)
    # AAPL should generally rank above MSFT given stronger inputs
    assert s["AAPL"] > s["MSFT"]
