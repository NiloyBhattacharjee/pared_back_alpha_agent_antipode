# Antipode – Math, Methods, and Assumptions

This document explains the math and logic used across the Antipode pipeline: data loading, agent signals, coordinator rules, and the backtest mechanics. File references point to exact locations for quick navigation.

## Notation and Conventions
- `P_i(d)`: adjusted close for ticker `i` on date `d`.
- `r_i(d) = P_i(d)/P_i(d-1) - 1`: simple daily return.
- Business days: pandas `BDay`, i.e., Monday–Friday excluding weekends.
- Population standard deviation: `std(ddof=0)`.

---

## Data Loading and Utilities

### Prices (REST → cache)
- Entrypoint: `load_prices` (src/data.py:63)
- Attempts REST first using env vars `FINANCIALDATASETS_API_KEY` and `FD_PRICES_URL`; then trims forward data to at most the requested window.
- Fallback: reads `data/prices_cache.csv` and normalizes to columns `(date, ticker, adj_close)`. Attribute `prices.attrs['source']` is set to `api` or `cache`.
- Helpers:
  - `_bdays` (src/data.py:30): builds a business‑day index that spans `lookback + buffer` before as‑of and `forward_days` after.
  - `_clip_forward` (src/data.py:36): restricts records to at most the next `forward_days` business days after `as_of`.
  - `_fetch_prices_rest_one` (src/data.py:43): REST call per ticker; detects date/price columns and returns normalized rows.

### News
- `load_news` (src/data.py:133): loads JSON files from `data/news/`. Each item: `{date, title, snippet}`. Dates are parsed; invalid dates are skipped.
- Rows kept if `as_of - lookback_days <= date <= as_of`. Headline text = `title + ' ' + snippet`.
- Synthetic fallback: `generate_synthetic_news` (src/data.py:156) samples 0–2 headlines/day/ticker with random positive/negative lexicon tokens.

### Fundamentals
- `load_facts` (src/data.py:170): prefers per‑ticker JSON files in `data/facts/`; otherwise falls back to `data/facts.csv`; or a small default table.
- Only the universe tickers are kept and only documented fields are used.

### Cache Utilities
- `_normalize_prices_frame` (src/pipeline_utils.py:6): returns canonical `(date, ticker, adj_close)` with timezone‑naive dates, uppercased tickers, de‑duplicated pairs, sorted.
- `write_prices_cache` (src/pipeline_utils.py:30): merges new data into `data/prices_cache.csv` idempotently and writes with stable formatting.

---

## Agent Signals and Ratings

### Rating Helper
- `_to_rating(x, pos_th, neg_th)` (src/agents.py:21): returns `BUY` if `x >= pos_th`, `SELL` if `x <= neg_th`, else `HOLD`.

### MomentumAgent
- Files: `MomentumAgent.score` (src/agents.py:45), `MomentumAgent.rate` (src/agents.py:68)
- Data window: use all data up to `as_of`; sort by (`ticker`, `date`).
- For each ticker:
  - Compute daily returns `r_t = pct_change(adj_close)`.
  - Momentum return over last `window=63` days: `R = ∏(1 + r_t) - 1`.
  - Realized volatility: `vol = std(r_t over last 63) * sqrt(252)`.
  - Score: `score = R / vol` if `vol > 0`, else NaN.
- Implementation: `df.groupby("ticker").apply(_calc, include_groups=False)`.
- Rating: map score with thresholds `+0.5/-0.5`.

### ValuationMomentumAgent
- Files: `ValuationMomentumAgent._per_ticker_components` (src/agents.py:102), `.score` (src/agents.py:125), `.rate` (src/agents.py:152)
- Components per ticker (using data up to `as_of`):
  - Momentum (as above) using `momentum_window=63`.
  - Valuation proxy: `val = -((px / MA_126) - 1)`, where `px` is last price and `MA_126` is 126‑day moving average. Below MA ⇒ cheaper ⇒ higher `val` due to negation.
- Cross‑sectional z‑scores across available tickers:
  - `z(x) = (x - mean(x)) / std(x)` (skip NaN; if `std=0`, return NaN vector).
- Composite score per ticker:
  - `s = w_momentum * z_m + w_value * z_v` with `w_momentum=0.6`, `w_value=0.4`.
  - If one leg is missing, weights are re‑normalized over existing legs.
- Rating: thresholds `+0.5/-0.5`.

### NewsSentimentAgent
- Files: `_headline_score` (src/agents.py:180), `.score` (src/agents.py:197), `.rate` (src/agents.py:208)
- Per‑headline lexicon score:
  - `POS_WORDS`/`NEG_WORDS` (src/data.py:122).
  - `lex = (pos_count - neg_count) / (pos_count + neg_count)` with 0 if denominator is 0.
  - Optional VADER blend (if installed): `0.5*lex + 0.5*vader_compound`.
- Per‑ticker score: mean of headline scores within lookback (default 60 days) and `date <= as_of`.
- Rating: thresholds `+0.1/-0.1`.

### FundamentalAgent
- Files: `.score` (src/agents.py:224), `.rate` (src/agents.py:255)
- Inputs per ticker: `revenue_growth_pct`, `operating_margin_pct` (or `gross_margin_pct`), `margin_trend`, `leverage_ratio`, `capex_intensity_pct`, `cf_stability`.
- Normalize each feature via z‑score; invert leverage and capex since lower is better: `-z(leverage)`, `-z(capex)`.
- Composite score:
  - `s = 0.30*g + 0.30*margin + 0.15*trend + 0.10*(-leverage) + 0.05*(-capex) + 0.10*cf`.
- Rating: thresholds `+0.3/-0.3`.
Where:
	•	g: Revenue growth z-score — Higher is better.
	•	margin: Operating margin or gross margin z-score — Indicates profitability.
	•	trend: Margin trend z-score — Are margins improving?
	•	leverage: Leverage ratio z-score (inverted) — Lower leverage is preferred.
	•	capex: CapEx intensity z-score (inverted) — Lower spending relative to revenue is preferred.
	•	cf: Cash flow stability z-score — More stable = better.
---

## Coordinator (Consensus)
- File: `Coordinator.combine` (src/coordinator.py:17)
- Numeric consensus per ticker: weighted average of available scores with re‑weighting to skip NaNs:
  - `cons = w_fund*fund + w_momo*momo + w_news*news`, with `(w_fund, w_momo, w_news) = (0.4, 0.4, 0.2)` normalized over present legs.
- Provisional rating from consensus score using thresholds `+0.5/-0.5`.
- HOLD tie‑breaks: majority vote among agent ratings; if still tied, priority order Fundamentals > Momentum > News.
- Output: table with component scores/ratings and `final_score`, plus the `consensus` series.

---

## Backtest Mechanics

### Benchmark
- File: `equal_weight_benchmark` (src/backtest.py:22)
- Window: business days from the next business day after `as_of` for `forward_days`.
- Per‑ticker forward return: `pct_change(adj_close)` over that forward window (first day per ticker is dropped).
- Daily benchmark return: equal‑weight average across tickers on each date.

### Portfolio and Performance
- File: `Backtester.run` (src/backtest.py:54)
- Forward business‑day index: starts at `next_bday(as_of)` with length `forward_days`.
- Weights: equal weight among tickers rated `BUY` at `as_of` (`w_i = 1/|B|`). If no BUYs, track the benchmark.
- Portfolio daily return: for BUY tickers only, `sum_i w_i * r_i(d)`.
- Join with benchmark on overlapping dates only (no fills).
- Active return: `r_active(d) = r_port(d) - r_bench(d)`.
- Cumulative curves: `cum_x(d) = ∏_{t<=d} (1 + r_x(t))` for `x ∈ {portfolio, benchmark}`.
- Sharpe‑style proxy: `mean(r_active) / std(r_active)` with population std (ddof=0); returns 0 if `std=0`.
- Fallbacks: if nothing overlaps, mirror benchmark; earlier in the pipeline (`run.py`) we may backshift `as_of` to ensure enough forward days.

---

## Pipeline Orchestration
- File: `run.py:main()` (run.py:1)
- Steps:
  - Parse args: `--as-of`, `--forward-days`, `--write-cache`, `--strict-as-of`.
  - Load prices for a sufficiently long lookback (driven by valuation/momentum windows) and the requested forward horizon; backshift `as_of` if near present and insufficient forward data (unless in strict mode).
  - Load news and facts; compute each agent’s scores and ratings.
  - Combine via `Coordinator` to produce picks; run `Backtester` for forward returns.
  - Write `outputs/`: `picks.csv`, `performance.csv` (daily and summary), `performance_summary.csv`, `equity_curve.png`.
  - Optional: `--write-cache` merges price data into `data/prices_cache.csv`.

---

## Assumptions and Design Choices
- Universe is fixed (AAPL, MSFT, NVDA, TSLA) and equal‑weighted on BUYs; no transaction costs or slippage.
- Signals are leakage‑aware: agents only look at information dated `<= as_of`; backtest starts the next business day.
- Simple, transparent thresholds for ratings; numeric composites rely on z‑scores for cross‑ticker comparability.
- Benchmark is an equal‑weight average of all universe tickers (not cap‑weighted).
- Population standard deviation `ddof=0` used intentionally for stable small‑sample behavior.
- Data hygiene: de‑duplicate `(ticker,date)`, timezone‑naive dates, stable CSV formatting.

---

## Outcomes and Diagnostics
- Per‑run artifacts:
  - `outputs/picks.csv`: agent scores/ratings and final consensus per ticker.
  - `outputs/performance.csv`: daily `benchmark_return`, `portfolio_return`, `active_return`, and cumulative series.
  - `outputs/performance_summary.csv`: `buy_count`, `sharpe_proxy`, `total_active`.
- `outputs/equity_curve.png`: “Growth of $1” for portfolio vs benchmark over the window.
### Output Columns
- `picks.csv`
  - `ticker`: symbol.
  - `fund_score` / `momo_score` / `news_score`: numeric agent scores (already normalized/composite; higher is better).
  - `fund_rating` / `momo_rating` / `news_rating`: each agent’s BUY/HOLD/SELL derived from its score.
  - `final_score`: weighted average of available scores (fund 0.4, momentum 0.4, news 0.2; weights re‑normalized if a leg is missing).
  - `final_rating`: consensus rating from `final_score` with HOLD tie‑breaks (majority; if still tied → Fundamentals > Momentum > News).

- `performance.csv`
  - `date`: business day in the forward window (starts next business day after `as_of`).
  - `benchmark_return`: equal‑weight daily return across the full universe.
  - `portfolio_return`: daily return of the fixed BUY basket (equal‑weighted).
  - `active_return`: `portfolio_return - benchmark_return`.
  - `cum_benchmark`: cumulative growth of $1 for the benchmark = cumprod(1 + `benchmark_return`).
  - `cum_portfolio`: cumulative growth of $1 for the portfolio = cumprod(1 + `portfolio_return`).
  - Note: A blank line is appended after the daily rows, followed by a small `metric,value` block identical to `performance_summary.csv`.

- `performance_summary.csv`
  - `metric`: summary metric name.
  - `value`: numeric value. Metrics included:
    - `buy_count`: number of BUY tickers held in the portfolio.
    - `sharpe_proxy`: mean(active) / std(active) using population std (ddof=0).
    - `total_active`: sum of daily active returns over the window.

---
---

## Future Improvements
- Portfolio construction
  - Add transaction costs, turnover constraints, and periodic rebalancing.
  - Risk‑aware weighting (e.g., inverse volatility, risk‑parity, beta‑neutralization).
  - Robustness to missing/late data with explicit as‑of data lags.
- Signals
  - Calibrate thresholds by cross‑validation; smooth momentum/valuation components.
  - Enrich news signal (better lexicon, topic filters, VADER/transformer blend with confidence weights).
  - Fundamentals: add growth quality metrics (R&D intensity, FCF margin trend) and industry‑neutral z‑scores.
- Backtest
  - Multi‑period panel backtest with rolling decision dates; confidence intervals via block bootstrap.
  - Add drawdown statistics and hit‑rate.
- Ops
  - Expand universe configuration, add CLI to pick subsets; include `.dockerignore` and dev Docker target with pytest tools.

---

## Quick References (Clickable)
- src/agents.py:45, src/agents.py:102, src/agents.py:125, src/agents.py:180, src/agents.py:224
- src/backtest.py:22, src/backtest.py:54
- src/coordinator.py:17
- src/data.py:30, src/data.py:36, src/data.py:43, src/data.py:63, src/data.py:133, src/data.py:170
- src/pipeline_utils.py:6, src/pipeline_utils.py:30
- run.py:1

---

## Algorithm

Below are concise algorithms for each major component. Pseudocode is implementation‑faithful to the Python code.

### MomentumAgent (per ticker)
```
Input: prices (date, ticker, adj_close), as_of, lookback_days, window
Filter: date <= as_of; sort by (ticker, date)
For each ticker group g:
  r = pct_change(g.adj_close).dropna()
  if len(r) < window + 5: return NaN
  R = product(1 + r.tail(window)) - 1
  vol = std(r.tail(window)) * sqrt(252)
  return NaN if vol <= 0 else R / vol
Map scores to BUY/HOLD/SELL via thresholds (+0.5 / -0.5)
```

### ValuationMomentumAgent (per ticker)
```
Input: prices, as_of
Filter g to date <= as_of; take last lookback_days rows
Momentum leg (as above) with momentum_window
Value leg: px = last(adj_close); ma = mean(adj_close over value_window)
           val = -((px / ma) - 1)  # below MA -> positive
Across tickers: z-score each component (skip NaNs; if std=0 -> NaNs)
Composite: s = w_momentum * z_m + w_value * z_v  (re-normalize weights over present legs)
Rating thresholds: +0.5 / -0.5
```

### NewsSentimentAgent
```
Input: news (date, ticker, headline), as_of, lookback_days
Filter: as_of - lookback_days <= date <= as_of
Per headline: lex = (pos_count - neg_count) / (pos_count + neg_count) or 0 if denom=0
If VADER available: score = 0.5*lex + 0.5*vader_compound; else score = lex
Per ticker: mean(headline_scores)
Rating thresholds: +0.1 / -0.1
```

### FundamentalAgent
```
Input: facts per ticker
Z-score columns: growth, margin(preferring operating), trend, leverage, capex, cf_stability
Invert: leverage, capex (lower is better)
Composite: 0.30*g + 0.30*margin + 0.15*trend - 0.10*z(leverage) - 0.05*z(capex) + 0.10*cf
Rating thresholds: +0.3 / -0.3
```

### Coordinator (Consensus)
```
Input: numeric scores (fund, momo, news) and categorical ratings
For each ticker:
  Take available numeric scores, re-normalize (w_fund, w_momo, w_news) over present legs
  final_score = dot(scores, weights)
  base = rating(final_score, +0.5/-0.5)
  If base == HOLD:
    majority vote among agent ratings; if tie -> priority Fundamentals > Momentum > News
Outputs: table with component scores/ratings and final_score, final_rating; consensus series
```

### Benchmark and Backtest
```
Benchmark:
  Window: next business day after as_of for forward_days
  Per ticker: r = pct_change(adj_close) within window; drop first per-ticker day
  By date: benchmark_return = mean(r across tickers)

Portfolio:
  BUY tickers from Coordinator at as_of
  Per date: portfolio_return = sum_i w_i * r_i(d), w_i = 1/|B|

Performance:
  Join on overlapping dates only
  active_return = portfolio_return - benchmark_return
  cum_benchmark = cumprod(1 + benchmark_return)
  cum_portfolio = cumprod(1 + portfolio_return)
  sharpe_proxy = mean(active_return) / std(active_return, ddof=0) or 0 if std=0
```

---

## Testing & Validation Methodology

### How to Run
- Local: `pytest -q`
- In Docker (image includes runtime deps): `docker run --rm -it antipode:latest python -m pytest -q`

### What the Suite Covers (unit tests)
- Agents (tests/test_agents_unit.py)
  - Momentum: higher score for upward drift vs flat; tolerates zero‑vol edge cases.
  - ValuationMomentum: value leg increases score when price < MA.
  - News: positive vs negative lexicon balances; scores bounded.
  - Fundamentals: ordering aligns with stronger inputs.
- Backtest (tests/test_backtest_unit.py)
  - Benchmark mean return equals cross‑sectional average.
  - Portfolio tracks benchmark when there are no BUYs; summary fields present.
- Coordinator (tests/test_coordinator_unit.py)
  - Reweighted blend when a leg is missing; HOLD tie‑breaks and priority.
- News filter (tests/test_news_filter.py)
  - Excludes future‑dated headlines; negative lexicon produces negative score.
- Pipeline utilities (tests/test_pipeline_utils.py)
  - Cache writer normalizes schema, merges idempotently, uppercases ticker, rounds prices.
- Sanity (tests/test_sanity.py)
  - Data loader produces forward dates after `as_of` when a tiny cache exists.
  - Cumulative product math verified on simple series.

### Test Principles Used
- Determinism: synthetic data uses fixed seeds; pandas operations use explicit sorting.
- Leakage safety: all agent tests filter to `date <= as_of`.
- Isolation: units (agents, backtest, coordinator) tested independently with small, crafted frames.
- Robustness: tolerate NaNs, zero‑volatility, malformed cache columns.

### Recommended Additions (next steps)
- Property tests for momentum invariants (scale invariance of returns, monotonicity with drift).
- Broader backtest checks: drawdown, turnover, and no‑lookahead under multiple decision dates.
- News/VADER integration tests gated by optional dependency.
- CLI workflow tests (subprocess) that verify outputs in `outputs/` for a fixed `--as-of` using cache data.

