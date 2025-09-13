# Antipode Guide

This guide explains how to set up the environment, run the code, how the components work, and the simple math used throughout. It also summarizes the key changes introduced in this iteration.

## Quick Setup (venv)

- Windows PowerShell
  - `python -m venv .venv`
  - `.\.venv\Scripts\Activate.ps1`
  - `python -m pip install -r requirements.txt`

- macOS/Linux
  - `python3 -m venv .venv`
  - `source .venv/bin/activate`
  - `pip install -r requirements.txt`

Run the CLI: `python run.py --as-of 2025-07-01 --forward-days 63`

## Data Sources and Fallbacks

`src/data.py:load_prices` fetches prices in the following order:
1) REST API (`financialdatasets.ai`) when `FINANCIALDATASETS_API_KEY` is set and `USE_FINANCIALDATASETS != 0`.
2) Local cache CSV at `data/prices_cache.csv` (offline-friendly).
3) Deterministic synthetic data for reproducible demos.

The loader tags the result with `df.attrs["source"]` as `api`, `cache`, or `synthetic`.

Create a cache file for offline runs:
- CLI: `python run.py --as-of 2025-07-01 --forward-days 21 --write-cache`

## Running

- CLI: `python run.py --as_of YYYY-MM-DD --forward-days N [--write-cache]`
  - Writes outputs to `outputs/` (`picks.csv`, `performance.csv`, `performance_summary.csv`, `equity_curve.png`).
  - If the API has insufficient forward days, the runner may backshift `as_of` and will print the adjusted date.

## Components and Math (Concise)

Backtest math (leakage-safe):
- Per-ticker daily return on date `d`: `r_i(d) = P_i(d) / P_i(d-1) - 1`.
- Benchmark (equal-weight): `r_bench(d) = mean_i r_i(d)` across all four tickers.
- Portfolio (equal-weight across BUY set `B`): `r_port(d) = sum_{i in B} w_i r_i(d)`, `w_i = 1/|B|`.
- Active return: `r_active(d) = r_port(d) - r_bench(d)`.
- Cumulative curves: `cum_x(d) = ∏ (1 + r_x)` over the forward window.
- Sharpe-style proxy: `mean(r_active) / std(r_active)` (population std, ddof=0).

Leakage controls:
- Agents compute scores using data filtered to `date <= as_of`.
- Forward returns start strictly after `as_of`.

Code references:
- Benchmark: `src/backtest.py:27`
- Backtest runner: `src/backtest.py:59`
- Coordinator: `src/coordinator.py:22` (class), `src/coordinator.py:29` (combine)

Agents:
- Momentum (class `MomentumAgent`)
  - Score = total return over last 63 trading days ÷ realized volatility (annualized denom); a simple Sharpe-like ratio.
  - Code: `src/agents.py:49` (score), `src/agents.py:76` (rate)

- Valuation/Momentum (class `ValuationMomentumAgent`)
  - Momentum leg as above; valuation leg = negative deviation from a 126-day moving average (below MA treated as relatively cheaper).
  - Z-score both legs across the universe, blend with weights (default 0.6/0.4), and map to BUY/HOLD/SELL via thresholds.
  - Code: `src/agents.py:124` (score), `src/agents.py:163` (rate)

- News/Sentiment (class `NewsSentimentAgent`)
  - Per-headline lexicon score `(pos - neg)/(pos + neg)` using transparent wordlists; optionally blended 50/50 with VADER’s compound score if package is installed.
  - Per-ticker score = average of headline scores within the lookback window (dates `<= as_of`).
  - Code: `src/agents.py:170` (class), `src/agents.py:206` (score), `src/agents.py:220` (rate)

- Fundamental/Quality (class `FundamentalAgent`)
  - Weighted z-score composite of: revenue growth, operating (or gross) margin, margin trend, leverage (lower is better), capex intensity (lower is better), and CF stability.
  - Code: `src/agents.py:250` (score), `src/agents.py:278` (rate)

## Data Files

- News/Sentiment: place curated items at `data/news/<TICKER>.json` with fields `title`, `snippet`, `date` (ISO). The agent uses only dates `<= as_of` within the lookback.
- Facts: per-ticker JSONs at `data/facts/<TICKER>.json` (preferred) or a consolidated `data/facts.csv`. See `src/data.py:load_facts` for loader behavior.
- Price cache (offline): `data/prices_cache.csv` with columns `date,ticker,adj_close`.

## What Was Done (This Iteration)

- Added `ValuationMomentumAgent` and integrated it into the runner.
- Added transparent news sentiment with local JSONs and optional VADER blending.
- Added per-ticker factsheets (JSON preferred; CSV fallback) and clarified quality weights.
- Simplified the backtest and documented the math and leakage controls inline.
- Implemented a cache fallback for prices and `--write-cache` in the CLI.
-- Removed notebook dependency; CLI remains the primary entrypoint.
- Removed an obsolete API probe script to keep the repo focused.

## Troubleshooting

- Recent `--as-of` dates may lack forward prices. The CLI will backshift as needed or use a partial forward window and will print what it used.
- If you see only zeros, ensure the forward window has data (pick an earlier `as_of` or use synthetic via `USE_FINANCIALDATASETS=0`).
- Outputs are saved to `outputs/` when running the CLI.
