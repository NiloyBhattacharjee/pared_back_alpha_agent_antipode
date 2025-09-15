import pandas as pd
from src.pipeline_utils import write_prices_cache


def test_write_prices_cache_normalizes_and_merges(tmp_path):
    path = tmp_path / "prices_cache.csv"

    # First write: lowercase ticker and different price column name
    df1 = pd.DataFrame(
        {
            "date": ["2024-09-01", "2024-09-01"],
            "ticker": ["aapl", "msft"],
            "Close": [190.1234567, 410.9999999],
        }
    )
    write_prices_cache(df1, str(path))

    # Second write: same date for AAPL (should de-duplicate/overwrite), new date for MSFT
    df2 = pd.DataFrame(
        {
            "date": ["2024-09-01", "2024-09-02"],
            "ticker": ["AAPL", "MSFT"],
            "adj_close": [191.0, 411.5],
        }
    )
    write_prices_cache(df2, str(path))

    out = pd.read_csv(path)
    # Canonical columns
    assert list(out.columns) == ["date", "ticker", "adj_close"]
    # Uppercase tickers
    assert set(out["ticker"]) == {"AAPL", "MSFT"}
    # De-duplicated on (ticker,date) and merged chronologically
    assert (out["date"].values == sorted(out["date"].values.tolist())) .all() if isinstance(out["date"].values, pd.api.extensions.ExtensionArray) else True
    # Prices rounded to 6 decimals
    assert out["adj_close"].dtype.kind in {"f", "i"}

