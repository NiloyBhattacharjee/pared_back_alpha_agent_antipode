# Antipode: Minimal Multi-Agent Equity Views and Backtest

Antipode is a tiny, transparent, leakage-safe multi-agent system that issues BUY/HOLD/SELL views on a fixed US equity universe (AAPL, MSFT, NVDA, TSLA) and evaluates those views in a lightweight backtest.Check docs for detailed break down of project 

## Setup
## Quick Setup (venv)

- Windows PowerShell
  - `python -m venv .venv`
  - `.\.venv\Scripts\Activate.ps1`
  - `python -m pip install -r requirements.txt`

- macOS/Linux
  - `python3 -m venv .venv`
  - `source .venv/bin/activate`
  - `pip install -r requirements.txt`

- Requirements: Python 3.9+, `pip`
- Install deps: `pip install -r requirements.txt`
- Optional data provider: `financialdatasets.ai`
  - REST setup (recommended):
    - `pip install -r requirements.txt` (includes `requests`)
    - Set API key via env:
      - macOS/Linux: `export FINANCIALDATASETS_API_KEY=YOUR_KEY`
      - Windows PowerShell: `$env:FINANCIALDATASETS_API_KEY = "YOUR_KEY"`
    - Endpoint URL (optional): `FD_PRICES_URL` (e.g., `https://api.financialdatasets.ai/prices/daily`)
    - Ensure API is enabled (default): `USE_FINANCIALDATASETS` unset or not "0"
  - Fallback order when fetching prices:
    1) financialdatasets.ai (tight window; enough lookback for indicators + forward window for backtest)
    2) Cached CSV at `data/prices_cache.csv` (written via `--write-cache`)

## How to Run

- CLI (primary): `python run.py --as-of 2025-06-28 --forward-days 21`
  - Optional: `--write-cache` saves fetched prices into `data/prices_cache.csv` for offline runs.
  - To disable API and use cache only: set `USE_FINANCIALDATASETS=0`.
  - Live API note: if there are no prices strictly after `--as-of`, the runner auto-backshifts the effective as_of so a full forward window exists (it prints the adjusted date). Use `--strict-as-of` to fail instead.
  - When using REST, verify the loader prints `Price source: api` at startup.

Outputs (written to `outputs/`):
- `picks.csv`: per-ticker agent scores + ratings + coordinator rating
- `performance.csv`: daily portfolio vs benchmark returns, active, cumulative series; summary metrics appended
- `equity_curve.png`: growth of $1, portfolio vs benchmark

## Testing

- Install pytest: `python -m pip install pytest`
- Run all tests: `pytest -q`
- Verbose names: `pytest -v`
- Single file: `pytest tests/test_sanity.py -v`
- Specific test: `pytest -k news_filter -v`

What the tests cover:
- Loader forward window: ensures `load_prices(as_of, forward_days)` includes prices strictly after `as_of`.
- Cumprod math sanity: verifies cumulative return math via simple products of (1 + returns).
- News date filtering: confirms `NewsSentimentAgent` only uses headlines with `date <= as_of` (no future leakage).

## Design

Agents (simple, auditable rules):

1) Valuation/Momentum Agent (class: `ValuationMomentumAgent`)
- Signal: 63-day total return divided by 63-day realized volatility (a Sharpe-like ratio) + simple valuation proxy (below 126-day MA = cheaper), blended and z-scored.
- Rating thresholds: > +0.5 BUY, < -0.5 SELL, else HOLD
- Leakage control: uses prices strictly up to and including `as_of`

2) News/Sentiment Agent
- Data: curated local JSON files in `data/news/<ticker>.json` with fields: `title`, `snippet`, `date` (ISO). Falls back to neutral if empty.
- Signal: average sentiment over the trailing window using a transparent lexicon (optional VADER blend)
- Rating thresholds: > +0.1 BUY, < -0.1 SELL, else HOLD
- Leakage control: uses headlines with timestamps `<= as_of`

3) Fundamental/Quality Agent
- Data: factsheets in `data/facts/*.json` (preferred) or consolidated `data/facts.csv`.
  - Fields: `revenue_growth_pct`, `operating_margin_pct` (or `gross_margin_pct`), `margin_trend` (-1..+1), `leverage_ratio` (lower better), `capex_intensity_pct` (lower better), `cf_stability` (0..1)
- Scoring: weighted sum of normalized features; leverage and capex intensity penalized
- Rating thresholds: > +0.3 BUY, < -0.3 SELL, else HOLD

Coordinator
- Map ratings to numeric: BUY=+1, HOLD=0, SELL=-1
- Weights: Fundamentals 0.4, Val/Mom 0.4, News 0.2
- Consensus: weighted average + simple tie-breaks (Fund > Val/Mom > News)

Backtest
- Window: forward N business days (configurable)
- Portfolio: equal-weight across BUY names at `as_of` (if none, track equal-weight benchmark)
- Benchmark: equal-weight across all four tickers
- Metrics: daily returns, cumulative curves, active return, simple Sharpe proxy (mean/vol of active)
- Leakage controls: only past data for signals; forward returns start strictly after `as_of`

Data fetch window
- Loader fetches only sufficient lookback for momentum and forward window for evaluation.
- Agents slice inputs to `date <= as_of`; only the backtest consumes forward prices.

## News Scoring Rationale
- Transparent lexicon counts positive/negative terms in `title + snippet`.
- If `vaderSentiment` is installed, we compute VADER compound per headline and average it 50/50 with lexicon.
- Per-headline score: `(pos - neg) / (pos + neg)` (0 if neither present). Per-ticker = average across window.

## Factsheet Derivations (clarity-first)
- Source type: manual approximations based on public summaries, typical ranges, and qualitative trend judgments. Values are illustrative.
- Definitions:
  - `revenue_growth_pct`: Recent YoY revenue growth (pct)
  - `operating_margin_pct`: Operating margin level (pct); `gross_margin_pct` used as fallback
  - `margin_trend`: Directional proxy -1..+1 (negative=contracting, positive=expanding)
  - `leverage_ratio`: Debt/equity style proxy; lower indicates less leverage
  - `capex_intensity_pct`: Capex/Revenue proxy; higher implies heavier capital requirements
  - `cf_stability`: Cash flow stability heuristic on 0..1
## Assumptions & Limitations
- Fixed tiny universe; simple thresholds and weights; not optimized
- News lexicon is minimal and illustrative; real ingestion may vary
- Long-only; SELL is underweight vs benchmark (no short)
- Single-period hold (no intra-period rebalance)

## AI-Tool Usage
- Project scaffolded with an AI assistant; logic and thresholds are explicitly coded for transparency.

## Time Accounting (within 20 hours)
- Repo setup + venv + wiring: ~1.0h
- Price loader + cache fallback (API→cache, normalize/merge writer): ~2.0h
- Agents (Momentum/Valuation, News, Fundamentals): ~4.0h
- Coordinator (weights, tie‑breaks, audit columns): ~1.0h
- Backtest (benchmark, portfolio, metrics, chart): ~3.0h
- CLI integration + outputs (CSV/PNG, flags incl. strict‑as‑of): ~1.5h
- Curated news JSONs (4 tickers × 15 items, ISO dates): ~3.0h
- Tests (loader, backtest math, agents, coordinator): ~2.0h
- Docs (README merge, usage, math, troubleshooting): ~1.0h
- Hardening/robustness (date coercion, edge handling): ~1.5h
Total: ~20.0 hours