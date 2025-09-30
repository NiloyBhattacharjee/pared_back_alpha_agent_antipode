from __future__ import annotations

"""
LangGraph runner that mirrors run.py as a graph:

Nodes
- load_prices_initial -> maybe_backshift -> load_news_facts
- score_momo (in parallel)
- score_news (in parallel)
- score_fund (in parallel)
- combine -> backtest -> write_outputs -> END

This file is optional and requires the `langgraph` package to run.
It does not affect normal CLI or tests.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import TypedDict, Dict, Optional

import os
import pandas as pd
import numpy as np

from langgraph.graph import StateGraph, END

from .data import load_prices, load_news, load_facts
from .agents import MomentumAgent, ValuationMomentumAgent, NewsSentimentAgent, FundamentalAgent
from .coordinator import Coordinator
from .backtest import Backtester


class State(TypedDict, total=False):
    # Config
    as_of: datetime
    forward_days: int
    strict_as_of: bool
    write_cache: bool
    outputs_dir: str
    momo_agent: str       # 'rule' | 'ml'
    momo_rule: str        # 'val' | 'simple'
    fund_agent: str       # 'rule' | 'ml'
    news_agent: str       # 'lexicon' | 'langchain'

    # Data
    prices: pd.DataFrame
    news: pd.DataFrame
    facts: pd.DataFrame

    # Scores/ratings
    momo_score: pd.Series
    news_score: pd.Series
    fund_score: pd.Series
    momo_rating: pd.Series
    news_rating: pd.Series
    fund_rating: pd.Series

    # Outputs
    picks: pd.DataFrame
    perf: pd.DataFrame
    summary: pd.DataFrame


def _select_momo_agent(s: State):
    if s.get("momo_agent", "rule") == "ml":
        try:
            from .agents_ml import PriceMLAgent  # type: ignore

            return PriceMLAgent()
        except Exception:
            return ValuationMomentumAgent()
    # rule-based
    if s.get("momo_rule", "val") == "simple":
        return MomentumAgent()
    return ValuationMomentumAgent()


def _select_news_agent(s: State):
    if s.get("news_agent", "lexicon") == "langchain":
        try:
            from .agents_langchain import LangChainNewsAgent  # type: ignore

            return LangChainNewsAgent()
        except Exception:
            return NewsSentimentAgent()
    return NewsSentimentAgent()


def load_prices_initial(state: State) -> State:
    as_of = state["as_of"]
    forward_days = state.get("forward_days", 21)
    # ensure enough lookback for ValuationMomentumAgent by default
    _momo = _select_momo_agent(state)
    lookback_days = 180
    if isinstance(_momo, ValuationMomentumAgent):
        lookback_days = max(_momo.lookback_days + 30, 150)
    prices = load_prices(as_of, lookback_days=lookback_days, forward_days=forward_days)
    try:
        src = prices.attrs.get("source", "unknown")
        print(f"Price source: {src}")
    except Exception:
        pass
    return {"prices": prices}


def maybe_backshift(state: State) -> State:
    as_of = pd.Timestamp(state["as_of"]).to_pydatetime()
    forward_days = state.get("forward_days", 21)
    strict = state.get("strict_as_of", False)
    prices = state["prices"]

    as_of_bd = pd.to_datetime(as_of.date())
    max_dt = pd.to_datetime(prices["date"]).max() if not prices.empty else as_of_bd
    start_next = pd.bdate_range(start=as_of_bd + pd.offsets.BDay(1), periods=1)[0]
    avail_forward_idx = pd.bdate_range(start=start_next, end=max_dt)
    if strict:
        if max_dt <= as_of_bd or len(avail_forward_idx) < forward_days:
            need = forward_days
            have = max(0, len(avail_forward_idx))
            print(
                f"Strict as-of: only {have} forward business days available after {as_of_bd.date()} (need {need}). Aborting."
            )
            raise SystemExit(2)
        return {}
    # Auto-backshift as in run.py
    changed = False
    if max_dt <= as_of_bd:
        allowed_start = pd.bdate_range(end=max_dt, periods=forward_days)[0]
        as_of_eff = allowed_start - pd.offsets.BDay(1)
        print(f"No forward prices after as_of={as_of_bd.date()}. Using as_of={as_of_eff.date()} for backtest.")
        as_of = as_of_eff.to_pydatetime()
        changed = True
    else:
        if len(avail_forward_idx) < forward_days:
            allowed_start = pd.bdate_range(end=max_dt, periods=forward_days)[0]
            as_of_eff = allowed_start - pd.offsets.BDay(1)
            if pd.to_datetime(as_of_eff.date()) != as_of_bd:
                print(
                    f"Only {len(avail_forward_idx)} forward business days available after {as_of_bd.date()}. "
                    f"Using as_of={as_of_eff.date()} to fit {forward_days} days."
                )
                as_of = as_of_eff.to_pydatetime()
                changed = True
    if not changed:
        return {}
    # reload prices with adjusted as_of
    _momo = _select_momo_agent(state)
    lookback_days = 180
    if isinstance(_momo, ValuationMomentumAgent):
        lookback_days = max(_momo.lookback_days + 30, 150)
    prices = load_prices(as_of, lookback_days=lookback_days, forward_days=state.get("forward_days", 21))
    return {"as_of": as_of, "prices": prices}


def load_news_facts(state: State) -> State:
    as_of = state["as_of"]
    return {"news": load_news(as_of), "facts": load_facts()}


def score_momo(state: State) -> State:
    agent = _select_momo_agent(state)
    s, r = agent.rate(state["prices"], state["as_of"])  # type: ignore
    return {"momo_score": s, "momo_rating": r}


def score_news(state: State) -> State:
    agent = _select_news_agent(state)
    s, r = agent.rate(state["news"], state["as_of"])  # type: ignore
    return {"news_score": s, "news_rating": r}


def score_fund(state: State) -> State:
    agent = FundamentalAgent()
    s, r = agent.rate(state["facts"])  # type: ignore
    return {"fund_score": s, "fund_rating": r}


def combine(state: State) -> State:
    # wait until all three scores are present
    for k in ("momo_score", "news_score", "fund_score", "momo_rating", "news_rating", "fund_rating"):
        if k not in state:
            return {}
    df, _ = Coordinator().combine(
        state["fund_score"],
        state["momo_score"],
        state["news_score"],
        state["fund_rating"],
        state["momo_rating"],
        state["news_rating"],
    )
    return {"picks": df}


def backtest(state: State) -> State:
    if "picks" not in state:
        return {}
    bt = Backtester(forward_days=state.get("forward_days", 21))
    perf, summary = bt.run(state["prices"], state["as_of"], state["picks"].set_index("ticker")["final_rating"])  # type: ignore
    return {"perf": perf, "summary": summary}


def write_outputs(state: State) -> State:
    if "perf" not in state or "summary" not in state or "picks" not in state:
        return {}
    outdir = state.get("outputs_dir", "outputs")
    os.makedirs(outdir, exist_ok=True)

    picks_out = state["picks"].copy()
    picks_out.to_csv(os.path.join(outdir, "picks.csv"), index=False)

    perf = state["perf"].copy()
    perf.index.name = "date"
    perf_path = os.path.join(outdir, "performance.csv")
    perf.to_csv(perf_path)
    with open(perf_path, "a", newline="") as f:
        f.write("\n")
        f.write("metric,value\n")
    state["summary"].to_csv(perf_path, mode="a", index=False, header=False)
    state["summary"].to_csv(os.path.join(outdir, "performance_summary.csv"), index=False)

    # chart
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(perf.index, perf["cum_portfolio"], label="Portfolio")
        ax.plot(perf.index, perf["cum_benchmark"], label="Benchmark")
        ax.set_title("Growth of $1 (Forward Window)")
        ax.legend(); ax.grid(True, alpha=0.3); fig.autofmt_xdate(); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "equity_curve.png"), dpi=150)
        plt.close(fig)
    except Exception:
        pass
    print(f"Done. Outputs saved in {outdir}/.")
    return {}


def build_graph():
    g = StateGraph(State)
    g.add_node("load_prices_initial", load_prices_initial)
    g.add_node("maybe_backshift", maybe_backshift)
    g.add_node("load_news_facts", load_news_facts)
    g.add_node("score_momo", score_momo)
    g.add_node("score_news", score_news)
    g.add_node("score_fund", score_fund)
    g.add_node("combine", combine)
    g.add_node("backtest", backtest)
    g.add_node("write_outputs", write_outputs)

    g.set_entry_point("load_prices_initial")
    g.add_edge("load_prices_initial", "maybe_backshift")
    g.add_edge("maybe_backshift", "load_news_facts")
    # fan-out
    g.add_edge("load_news_facts", "score_momo")
    g.add_edge("load_news_facts", "score_news")
    g.add_edge("load_news_facts", "score_fund")
    # join
    g.add_edge("score_momo", "combine")
    g.add_edge("score_news", "combine")
    g.add_edge("score_fund", "combine")
    g.add_edge("combine", "backtest")
    g.add_edge("backtest", "write_outputs")
    g.add_edge("write_outputs", END)
    return g.compile()

