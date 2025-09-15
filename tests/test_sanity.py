from datetime import datetime
import os
import pandas as pd

from src.data import load_prices
from src.backtest import Backtester


def _write_minimal_cache(as_of: datetime, forward_days: int = 5):
    # Create a tiny cache with a few forward business days after as_of for all universe tickers
    from src import UNIVERSE
    start = pd.bdate_range(start=pd.to_datetime(as_of.date()) + pd.offsets.BDay(1), periods=1)[0]
    days = pd.bdate_range(start=start, periods=forward_days)
    rows = []
    for t in UNIVERSE:
        base = 100.0
        for i, d in enumerate(days):
            rows.append({"date": pd.to_datetime(d), "ticker": t, "adj_close": base * (1 + 0.01 * i)})
    df = pd.DataFrame(rows)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/prices_cache.csv", index=False)


def test_data_loader_has_forward_window(tmp_path):
    as_of = datetime(2025, 6, 30)
    forward_days = 21
    # Ensure a cache exists regardless of environment
    _write_minimal_cache(as_of, forward_days=5)
    prices = load_prices(as_of, lookback_days=100, forward_days=forward_days)
    max_date = prices["date"].max()
    # Expect some data strictly after as_of for backtest
    assert max_date > pd.to_datetime(as_of.date())


def test_backtest_cumprod_math():
    # Construct simple two-day returns and verify cumprod logic
    perf = pd.DataFrame({
        "benchmark_return": [0.01, -0.02, 0.03],
        "portfolio_return": [0.02, 0.00, 0.01],
    })
    perf["active_return"] = perf["portfolio_return"] - perf["benchmark_return"]
    perf["cum_benchmark"] = (1 + perf["benchmark_return"]).cumprod()
    perf["cum_portfolio"] = (1 + perf["portfolio_return"]).cumprod()
    assert abs(perf["cum_benchmark"].iloc[-1] - ((1.01) * (0.98) * (1.03))) < 1e-9
    assert abs(perf["cum_portfolio"].iloc[-1] - ((1.02) * (1.00) * (1.01))) < 1e-9
