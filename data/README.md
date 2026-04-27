# Paper 2 — Supplementary Data

This directory contains the raw inputs, the engineered feature panel, and the summary statistics used in the paper.

## Files

| File | Rows | Bytes | Description |
| --- | ---: | ---: | --- |
| `spx_daily.parquet` | 5,534 | 256,911 | Daily OHLCV for `^GSPC` from Yahoo Finance, Jan 2 2004 to Nov 26 2025. Columns: `open`, `high`, `low`, `close`, `volume`. |
| `vix_daily.parquet` | 5,532 | 135,714 | Daily OHLCV for `^VIX` from Yahoo Finance, same window. Columns: `open`, `high`, `low`, `close`, `volume`. |
| `combined.parquet` | 5,446 | 1,756,143 | Engineered feature panel after dropping rows with NaN from rolling windows or forward targets. Columns include log returns, realized volatility at 5/22/66-day horizons, HAR components, lagged VIX, leverage-effect features, volume features, day-of-week dummies, and forward RV targets at 1/5/22-day horizons. |
| `summary_stats.csv` | 8 | 653 | Mean, std, min, quartiles, max for daily log return, 5-day RV, 22-day RV, and VIX close, used in Table 2 of the paper. |

## Provenance

- **Raw data source**: Yahoo Finance, fetched via the `yfinance` Python package.
- **Construction script**: [`code/01_collect_data.py`](../code/01_collect_data.py) — fully reproducible from the source script alone, no manual intervention.
- **Last refresh**: 2026-04-12.
- **Feature definitions**: see Section 3.4 of the paper for the 35-feature specification.

## Reproducing

From a clean Python 3.11+ environment with `yfinance`, `pandas`, `numpy`, `pyarrow` installed:

```bash
cd code && python 01_collect_data.py
```

The three parquet files and the summary CSV will be regenerated. Running counts may differ by a handful of rows from the snapshots committed here if Yahoo Finance has revised historical bars (it occasionally re-states corporate-action splits and dividends).

## License

The derived datasets in this directory are released under [CC0 1.0 Universal (Public Domain Dedication)](https://creativecommons.org/publicdomain/zero/1.0/). The raw `^GSPC` and `^VIX` series remain governed by Yahoo Finance's terms of use.
