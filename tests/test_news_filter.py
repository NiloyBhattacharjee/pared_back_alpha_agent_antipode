from datetime import datetime

import pandas as pd

from src.agents import NewsSentimentAgent


def test_news_date_filtering_excludes_future_items():
    # Two headlines for the same ticker: one before as_of (negative), one after as_of (positive)
    news = pd.DataFrame(
        [
            {
                "date": pd.to_datetime("2025-07-01"),
                "ticker": "AAPL",
                "headline": "AAPL reports loss and decline",
            },
            {
                "date": pd.to_datetime("2025-07-10"),
                "ticker": "AAPL",
                "headline": "AAPL strong record growth upgrade",
            },
        ]
    )
    as_of = datetime(2025, 7, 5)

    agent = NewsSentimentAgent(lookback_days=60)
    scores = agent.score(news, as_of)

    # Only the 2025-07-01 item is within the window (<= as_of)
    assert "AAPL" in scores.index
    # That item contains negative lexicon words -> score should be negative
    assert scores["AAPL"] < 0

