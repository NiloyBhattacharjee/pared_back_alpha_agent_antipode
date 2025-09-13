# Antipode: Minimal Multi‑Agent Equity Views and Backtest

Antipode is a tiny, transparent, leakage‑safe multi‑agent system that issues BUY/HOLD/SELL views on a fixed US equity universe (AAPL, MSFT, NVDA, TSLA) and evaluates those views in a lightweight backtest.

## Setup

- Requirements: Python 3.9+, `pip`
- Install deps: `pip install -r requirements.txt`
- Optional data provider: `financialdatasets.ai`
  - REST setup (recommended):
    - `pip install -r requirements.txt` (includes `requests`)
    - Set API key header via env: 
      - macOS/Linux: `export FINANCIALDATASETS_API_KEY=YOUR_KEY`
      - Windows PowerShell: `$env:FINANCIALDATASETS_API_KEY = "YOUR_KEY"`
    - Set the daily prices endpoint URL: 
      - `FD_PRICES_URL` (for example: `https://api.financialdatasets.ai/prices/daily`)
      - Optional param names if your endpoint differs: `FD_TICKER_PARAM` (default `ticker`), `FD_START_PARAM` (default `start_date`), `FD_END_PARAM` (default `end_date`)
      - Optional interval controls (if your endpoint supports them):
        - `FD_INTERVAL` (e.g., `day`, `week`)
        - `FD_INTERVAL_MULTIPLIER` (e.g., `1`, `5`)
    - Ensure API is enabled (default): `USE_FINANCIALDATASETS` unset or not "0"
    - Debugging (optional): set `FD_DEBUG=1` to print REST requests and parsing diagnostics.
  - SDK (if available in your env): previously attempted; may not be published on PyPI. REST path above is the primary.
  - Fallback order when fetching prices:
    1) financialdatasets.ai (tight window: enough lookback for indicators + forward window for backtest)
    2) Cached CSV at `data/prices_cache.csv` (written via `--write-cache`)
    3) Deterministic synthetic prices (no leakage into signals; forward window included only for evaluation)

## How to Run

- Notebook (recommended): open `notebooks/antipode.ipynb`, run all cells. It will:
  - Load/generate data up to the configurable `as_of` date
  - Run three agents (Momentum/Valuation, News/Sentiment, Fundamental/Quality)
  - Combine ratings via a transparent coordinator
  - Backtest a 21‑day forward hold with an equal‑weight benchmark
  - Save outputs in `outputs/`

- CLI: `python run.py --as-of 2025-06-28 --forward-days 21`
  - Optional: `--write-cache` will save the fetched prices into `data/prices_cache.csv` for offline runs.
  - To disable API and use cache/synthetic: set `USE_FINANCIALDATASETS=0`.
  - Live API note: if there are no prices strictly after `--as-of` (e.g., you choose a very recent date), the runner automatically backshifts the effective `as_of` so a full forward window exists. It prints the adjusted date for transparency.
  - When using REST, verify the loader prints `Price source: api` at startup.

Outputs (written to `outputs/`):
- `picks.csv`: per‑ticker agent scores + ratings + coordinator rating
- `performance.csv`: daily portfolio vs benchmark returns, active, cumulative series; summary metrics appended
- `equity_curve.png`: growth of $1, portfolio vs benchmark

## Testing

- Install pytest: `python -m pip install pytest`
- Run all tests: `pytest -q`
- Verbose names: `pytest -v`
- Single file: `pytest tests/test_sanity.py -v`
- Specific test: `pytest -k news_filter -v`

What the tests cover:
- Loader forward window: ensures `load_prices(as_of, forward_days)` includes prices strictly after `as_of` so the backtest has forward data.
- Cumprod math sanity: verifies cumulative return math via simple products of (1 + returns).
- News date filtering: confirms `NewsSentimentAgent` only uses headlines with `date <= as_of` (no future leakage).

## Design

Agents (simple, auditable rules):

1) Valuation/Momentum Agent (class: `ValuationMomentumAgent`)
- Signal: 63‑day total return divided by 63‑day realized volatility (a Sharpe‑like ratio)
- Scoring: continuous ratio; Rating thresholds: > +0.5 BUY, < −0.5 SELL, else HOLD
- Leakage control: uses prices strictly up to and including `as_of`

2) News/Sentiment Agent
- Data: curated local JSON files in `data/news/<ticker>.json` with fields: `title`, `snippet`, `date` (ISO). 5–15 recent items per ticker from reputable sources (paraphrased summaries). Falls back to synthetic headlines if files are missing.
- Signal: average sentiment over the trailing 30 calendar days
- Rating thresholds: > +0.1 BUY, < −0.1 SELL, else HOLD
- Leakage control: uses headlines with timestamps ≤ `as_of`

3) Fundamental/Quality Agent
- Data: compact factsheets in `data/facts/*.json` (preferred) or consolidated `data/facts.csv`.
  - Loader prefers per-ticker JSONs and will fall back to `data/facts.csv`, then to a tiny built-in default table if neither is present.
  - Fields: `revenue_growth_pct`, `operating_margin_pct` (or `gross_margin_pct` fallback), `margin_trend` (-1..+1), `leverage_ratio` (lower better), `capex_intensity_pct` (lower better), `cf_stability` (0..1)
- Scoring: weighted sum of normalized features; positive growth/margin/trend and stability are rewarded; leverage and capex intensity penalized.
- Rating thresholds: > +0.3 BUY, < −0.3 SELL, else HOLD

Coordinator
- Map ratings to numeric: BUY=+1, HOLD=0, SELL=−1
- Weights: Fundamentals 0.4, Momentum 0.4, News 0.2
- Consensus: weighted average → final rating via same thresholds as Momentum
- Tie‑break rule: if borderline, prefer Fundamentals, then Momentum, then News

Backtest
- Window: forward 21 business days (configurable)
- Portfolio: long‑only, equal‑weight across BUY names at `as_of` (if none, hold equal‑weight benchmark)
- Benchmark: equal‑weight across all four tickers
- Metrics: daily returns, cumulative curves, active return, simple Sharpe proxy (mean/vol of active)
- Leakage controls: only past data for signals; forward returns start strictly after `as_of`

Data fetch window
- Prices loader fetches only: (i) sufficient lookback for Momentum (63 trading days within a 126-day window) plus a small buffer, and (ii) forward business days for evaluation.
- Agents slice inputs to `date <= as_of`; only the backtest consumes forward prices.

## Assumptions & Limitations
- Fixed tiny universe; no corporate actions adjustments beyond using adjusted close in provider path (synthetic data is split‑free by construction)
- Simple thresholds and weights; not optimized
- News lexicon is minimal and illustrative; real news ingestion may vary
- Long‑only; SELL is treated as underweight vs benchmark, not outright short
- Single‑period hold (no intra‑period rebalance)

## AI‑Tool Usage
- This project was drafted with an AI coding assistant for scaffolding and code comments. Logic and thresholds are explicitly coded for transparency and reproducibility.
  - Env var examples:
    - macOS/Linux: `export FINANCIALDATASETS_API_KEY=YOUR_KEY`
    - Windows PowerShell: `$env:FINANCIALDATASETS_API_KEY = "YOUR_KEY"`
    - Disable API: `export USE_FINANCIALDATASETS=0` or `$env:USE_FINANCIALDATASETS = "0"`
## News Scoring Rationale
- Simple, transparent lexicon counts positive/negative terms in the concatenated `title + snippet`.
- If `vaderSentiment` is installed, we automatically compute the VADER compound score per headline and average it 50/50 with the lexicon score. Otherwise we use lexicon-only (default, fully reproducible).
- Per-headline score: `lexicon_score = (pos - neg) / (pos + neg)` with neutral→0 when counts=0; `final = 0.5*lexicon + 0.5*vader_compound` when VADER is available.
- Per-ticker score: average of headline scores over the lookback window.
## Factsheet Derivations (clarity-first)
- Source type: manual approximations based on public company summaries, typical ranges in recent filings/investor materials, and qualitative trend judgments. Values are illustrative for demo purposes and do not aim for point accuracy.
- Definitions:
  - `revenue_growth_pct`: Recent YoY revenue growth estimate in percent.
  - `operating_margin_pct`: Operating margin level in percent; `gross_margin_pct` used as fallback if op margin not provided.
  - `margin_trend`: Directional proxy of margin trajectory on a -1..+1 scale (negative=contracting, positive=expanding).
  - `leverage_ratio`: Debt/equity style proxy; lower indicates less leverage.
  - `capex_intensity_pct`: Capex/Revenue proxy; higher can indicate heavier capital requirements.
  - `cf_stability`: Cash flow stability heuristic on 0..1.
- Per‑ticker notes are embedded in `data/facts/<ticker>.json` under `notes` for transparency.
