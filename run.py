import argparse
from datetime import datetime
import os

import pandas as pd

from src.data import load_prices, load_news, load_facts
from src.agents import MomentumAgent, NewsSentimentAgent, FundamentalAgent, ValuationMomentumAgent
from src.coordinator import Coordinator
from src.backtest import Backtester
from src import UNIVERSE
import matplotlib.pyplot as plt
from src.pipeline_utils import write_prices_cache



def main():
    parser = argparse.ArgumentParser(description="Run Antipode multi-agent pipeline")
    parser.add_argument("--as-of", required=False, default=None, help="As-of date YYYY-MM-DD; defaults to today")
    parser.add_argument("--forward-days", type=int, default=21, help="Forward holding window in business days")
    parser.add_argument("--write-cache", action="store_true", help="Write fetched prices to data/prices_cache.csv for offline runs")
    parser.add_argument(
        "--strict-as-of",
        action="store_true",
        help="Fail if not enough forward business days exist after as_of (disables auto-backshift/fallback)",
    )
    args = parser.parse_args()

    as_of = datetime.strptime(args.as_of, "%Y-%m-%d").replace(hour=0, minute=0, second=0, microsecond=0) if args.as_of else datetime.today()

    # Data: fetch tight window (lookback sized for indicators + buffer; include forward days)
    # Valuation/Momentum (price-only) agent
    momo = ValuationMomentumAgent()
    # Ensure enough lookback for momentum plus some buffer
    lookback_days = max(momo.lookback_days + 30, 150)
    prices = load_prices(as_of, lookback_days=lookback_days, forward_days=args.forward_days)
    try:
        src = prices.attrs.get("source", "unknown")
        print(f"Price source: {src}")
    except Exception:
        pass
    # If using live API near present, there may be no (or too few) forward prices strictly after as_of.
    # In strict mode, fail fast; otherwise, auto-backshift to fit the requested window.
    as_of_bd = pd.to_datetime(as_of.date())
    max_dt = pd.to_datetime(prices["date"]).max() if not prices.empty else as_of_bd
    start_next = pd.bdate_range(start=as_of_bd + pd.offsets.BDay(1), periods=1)[0]
    avail_forward_idx = pd.bdate_range(start=start_next, end=max_dt)
    if args.strict_as_of:
        if max_dt <= as_of_bd or len(avail_forward_idx) < args.forward_days:
            need = args.forward_days
            have = max(0, len(avail_forward_idx))
            print(
                f"Strict as-of: only {have} forward business days available after {as_of_bd.date()} (need {need}). "
                f"Aborting. Choose an earlier --as-of or reduce --forward-days."
            )
            raise SystemExit(2)
    else:
        # Auto-backshift behavior
        if max_dt <= as_of_bd:
            allowed_start = pd.bdate_range(end=max_dt, periods=args.forward_days)[0]
            as_of_eff = allowed_start - pd.offsets.BDay(1)
            print(f"No forward prices after as_of={as_of_bd.date()}. Using as_of={as_of_eff.date()} for backtest.")
            as_of = as_of_eff.to_pydatetime()
            as_of_bd = pd.to_datetime(as_of.date())
            start_next = pd.bdate_range(start=as_of_bd + pd.offsets.BDay(1), periods=1)[0]
            avail_forward_idx = pd.bdate_range(start=start_next, end=max_dt)
        if len(avail_forward_idx) < args.forward_days:
            allowed_start = pd.bdate_range(end=max_dt, periods=args.forward_days)[0]
            as_of_eff = allowed_start - pd.offsets.BDay(1)
            if pd.to_datetime(as_of_eff.date()) != as_of_bd:
                print(
                    f"Only {len(avail_forward_idx)} forward business days available after {as_of_bd.date()}. "
                    f"Using as_of={as_of_eff.date()} to fit {args.forward_days} days."
                )
                as_of = as_of_eff.to_pydatetime()
    news = load_news(as_of)
    facts = load_facts()

    # Agents
    news_agent = NewsSentimentAgent()
    fund = FundamentalAgent()

    momo_s, momo_r = momo.rate(prices, as_of)
    news_s, news_r = news_agent.rate(news, as_of)
    fund_s, fund_r = fund.rate(facts)

    # Coordinator
    coord = Coordinator()
    picks_df, final_scores = coord.combine(fund_s, momo_s, news_s, fund_r, momo_r, news_r)

    # Backtest
    bt = Backtester(forward_days=args.forward_days)
    try:
        perf, summary = bt.run(prices, as_of, picks_df.set_index("ticker")["final_rating"]) 
    except RuntimeError as e:
        if args.strict_as_of:
            raise
        # Partial-window fallback: if some forward days exist but the strict guard tripped, rerun with available days
        as_of_bd = pd.to_datetime(as_of.date())
        start_next = pd.bdate_range(start=as_of_bd + pd.offsets.BDay(1), periods=1)[0]
        max_dt = pd.to_datetime(prices["date"]).max() if not prices.empty else as_of_bd
        avail_forward_idx = pd.bdate_range(start=start_next, end=max_dt)
        n = len(avail_forward_idx)
        if n <= 0:
            raise
        print(f"Partial-window fallback: using {n} forward business days after {as_of_bd.date()}.")
        bt = Backtester(forward_days=n)
        perf, summary = bt.run(prices, as_of, picks_df.set_index("ticker")["final_rating"]) 

    # Outputs
    os.makedirs("outputs", exist_ok=True)
    picks_out = picks_df.copy()
    picks_out.to_csv("outputs/picks.csv", index=False)

    perf_out = perf.copy()
    perf_out.index.name = "date"
    perf_path = "outputs/performance.csv"
    # Write performance table first
    perf_out.to_csv(perf_path)
    # Append a blank line and then the summary block (metric,value)
    with open(perf_path, "a", newline="") as f:
        f.write("\n")
        f.write("metric,value\n")
    summary.to_csv(perf_path, mode="a", index=False, header=False)
    # Also write separate summary for easy parsing
    summary.to_csv("outputs/performance_summary.csv", index=False)

    # Chart
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(perf.index, perf["cum_portfolio"], label="Portfolio")
    ax.plot(perf.index, perf["cum_benchmark"], label="Benchmark")
    ax.set_title("Growth of $1 (Forward Window)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig("outputs/equity_curve.png", dpi=150)
    plt.close(fig)

    # Optional: write cache for future offline runs
    if args.write_cache:
        cache_path = write_prices_cache(prices)
        if cache_path:
            print(f"Wrote price cache to {cache_path}")

    print("Done. Outputs saved in outputs/.")


if __name__ == "__main__":
    main()
