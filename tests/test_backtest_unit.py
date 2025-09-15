from datetime import datetime
import pandas as pd

from src.backtest import equal_weight_benchmark, Backtester


def make_prices(as_of: datetime) -> pd.DataFrame:
    # Build three forward business days after as_of
    start = pd.bdate_range(start=pd.to_datetime(as_of.date()) + pd.offsets.BDay(1), periods=1)[0]
    days = pd.bdate_range(start=start, periods=3)
    rows = []
    # AAPL: flat then up 10% then flat
    aapl = [100.0, 110.0, 110.0]
    # MSFT: flat then up 5% then flat
    msft = [200.0, 210.0, 210.0]
    for d, p in zip(days, aapl):
        rows.append({"date": pd.to_datetime(d), "ticker": "AAPL", "adj_close": p})
    for d, p in zip(days, msft):
        rows.append({"date": pd.to_datetime(d), "ticker": "MSFT", "adj_close": p})
    return pd.DataFrame(rows)


def test_equal_weight_benchmark_returns_mean():
    as_of = datetime(2024, 1, 1)
    prices = make_prices(as_of)
    bench = equal_weight_benchmark(prices, as_of, forward_days=3)
    # First forward day has no pct_change -> starts from second day
    assert not bench.empty
    # On the second day: AAPL +10%, MSFT +5% -> mean = 7.5%
    # Locate the second business day
    second_day = bench.index.min()
    assert abs(float(bench.loc[second_day, "benchmark_return"]) - 0.075) < 1e-9


def test_backtester_tracks_benchmark_when_no_buys():
    as_of = datetime(2024, 1, 1)
    prices = make_prices(as_of)
    # No BUYs
    final_ratings = pd.Series({"AAPL": "HOLD", "MSFT": "HOLD"})
    bt = Backtester(forward_days=3)
    perf, summary = bt.run(prices, as_of, final_ratings)
    assert not perf.empty
    # Portfolio equals benchmark when there are no BUYs
    assert (perf["portfolio_return"] == perf["benchmark_return"]).all()
    # Summary has expected fields
    metrics = set(summary["metric"])  # type: ignore
    assert {"buy_count", "sharpe_proxy", "total_active"}.issubset(metrics)

