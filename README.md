# Antipode: Minimal Multi-Agent Equity Views and Backtest

Antipode is a tiny, transparent, leakage-safe multi-agent system that issues BUY/HOLD/SELL views on a fixed US equity universe (AAPL, MSFT, NVDA, TSLA) and evaluates those views in a lightweight backtest.

### Quick Setup (venv)

- Windows PowerShell
  - `python -m venv .venv`
  - `.\\.venv\\Scripts\\Activate.ps1`
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
      - Env var examples:
    - macOS/Linux: `export FINANCIALDATASETS_API_KEY=YOUR_KEY`
    - Windows PowerShell: `$env:FINANCIALDATASETS_API_KEY = "YOUR_KEY"`
    - Disable API: `export USE_FINANCIALDATASETS=0` or `$env:USE_FINANCIALDATASETS = "0"`
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

## Assumptions & Limitations
- Fixed tiny universe; simple thresholds and weights; not optimized
- News lexicon is minimal and illustrative; real ingestion may vary
- Long-only; SELL is underweight vs benchmark (no short)
- Single-period hold (no intra-period rebalance)

## AI-Tool Usage
- Project scaffolded with an AI assistant; logic and thresholds are explicitly coded for transparency.
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

### Data Sources and Fallbacks

`src/data.py:load_prices` fetches prices in the following order:
1) REST API (`financialdatasets.ai`) when `FINANCIALDATASETS_API_KEY` is set and `USE_FINANCIALDATASETS != 0`.
2) Local cache CSV at `data/prices_cache.csv` (offline-friendly).

The loader tags the result with `df.attrs["source"]` as `api` or `cache`.

Create a cache file for offline runs:
- CLI: `python run.py --as-of 2025-07-01 --forward-days 21 --write-cache`

### Running

- CLI: `python run.py --as-of YYYY-MM-DD --forward-days N [--write-cache]`
  - Writes outputs to `outputs/` (`picks.csv`, `performance.csv`, `performance_summary.csv`, `equity_curve.png`).
  - If the API has insufficient forward days, the runner may backshift `as_of` and will print the adjusted date. Use `--strict-as-of` to fail instead of shifting.

### Components and Math (Concise)

Backtest math (leakage-safe):
- Per-ticker daily return on date `d`: `r_i(d) = P_i(d) / P_i(d-1) - 1`.
- Benchmark (equal-weight): `r_bench(d) = mean_i r_i(d)` across all four tickers.
- Portfolio (equal-weight across BUY set `B`): `r_port(d) = sum_{i in B} w_i r_i(d)`, `w_i = 1/|B|`.
- Active return: `r_active(d) = r_port(d) - r_bench(d)`.
- Cumulative curves: `cum_x(d) = product (1 + r_x)` over the forward window.
- Sharpe-style proxy: `mean(r_active) / std(r_active)` (population std, ddof=0).

Leakage controls:
- Agents compute scores using data filtered to `date <= as_of`.
- Forward returns start strictly after `as_of`.

### Code references
- Benchmark: `src/backtest.py:27`
- Backtest runner: `src/backtest.py:59`
- Coordinator: `src/coordinator.py`
- Agents: `src/agents.py`

### Data Files

- News/Sentiment: place curated items at `data/news/<TICKER>.json` with fields `title`, `snippet`, `date` (ISO). The agent uses only dates `<= as_of` within the lookback.
- Facts: per-ticker JSONs at `data/facts/<TICKER>.json` (preferred) or a consolidated `data/facts.csv`. See `src/data.py:load_facts` for loader behavior.
- Price cache (offline): `data/prices_cache.csv` with columns `date,ticker,adj_close`.

### What Was Done (This Iteration)

- Added `ValuationMomentumAgent` and integrated it into the runner.
- Added transparent news sentiment with local JSONs and optional VADER blending.
- Added per-ticker factsheets (JSON preferred; CSV fallback) and clarified quality weights.
- Simplified the backtest and documented the math and leakage controls inline.
- Implemented a cache fallback for prices and `--write-cache` in the CLI.
- Removed notebook dependency; CLI remains the primary entrypoint.

### Troubleshooting

- Recent `--as-of` dates may lack forward prices. The CLI will backshift as needed or use a partial forward window and will print what it used (or fail with `--strict-as-of`).
- If you see only zeros, ensure the forward window has data (pick an earlier `as_of`) and that your news items fall within the lookback window.
