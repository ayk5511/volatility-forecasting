"""
Paper 2: Volatility Forecasting — Data Collection
Downloads S&P 500, VIX, and treasury data from public sources.
All data is freely available via Yahoo Finance and FRED.

Usage: python 01_collect_data.py
Output: ../data/spx_daily.parquet, ../data/vix_daily.parquet, ../data/combined.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yfinance as yf
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

START = "2004-01-01"
END = "2025-12-31"


def download_spx():
    """Download S&P 500 daily OHLCV from Yahoo Finance."""
    print("Downloading S&P 500 (^GSPC)...")
    spx = yf.download("^GSPC", start=START, end=END, auto_adjust=True)
    spx.columns = spx.columns.droplevel(1) if isinstance(spx.columns, pd.MultiIndex) else spx.columns
    spx.columns = [c.lower() for c in spx.columns]
    spx.index.name = "date"
    print(f"  SPX: {len(spx)} rows, {spx.index[0].date()} to {spx.index[-1].date()}")
    return spx


def download_vix():
    """Download CBOE VIX from Yahoo Finance."""
    print("Downloading VIX (^VIX)...")
    vix = yf.download("^VIX", start=START, end=END, auto_adjust=True)
    vix.columns = vix.columns.droplevel(1) if isinstance(vix.columns, pd.MultiIndex) else vix.columns
    vix.columns = [c.lower() for c in vix.columns]
    vix.index.name = "date"
    print(f"  VIX: {len(vix)} rows, {vix.index[0].date()} to {vix.index[-1].date()}")
    return vix


def compute_realized_volatility(spx: pd.DataFrame) -> pd.DataFrame:
    """
    Compute realized volatility measures from daily close prices.

    Returns DataFrame with:
    - log_return: daily log return
    - rv_5d: 5-day realized volatility (annualized)
    - rv_22d: 22-day realized volatility (annualized)
    - rv_66d: 66-day realized volatility (annualized)
    - abs_return: absolute daily return (proxy for daily vol)
    - squared_return: squared daily return
    - range_vol: Parkinson range-based volatility estimator
    """
    df = pd.DataFrame(index=spx.index)

    # Log returns
    df["log_return"] = np.log(spx["close"] / spx["close"].shift(1))
    df["abs_return"] = df["log_return"].abs()
    df["squared_return"] = df["log_return"] ** 2

    # Parkinson range-based estimator (uses high/low)
    df["range_vol"] = np.sqrt(
        (1 / (4 * np.log(2))) * (np.log(spx["high"] / spx["low"])) ** 2
    )

    # Realized volatility at different horizons (annualized)
    for window in [5, 22, 66]:
        df[f"rv_{window}d"] = (
            df["squared_return"].rolling(window).mean().apply(np.sqrt) * np.sqrt(252)
        )

    # Forward realized volatility (what we are predicting)
    for horizon in [1, 5, 22]:
        df[f"rv_{horizon}d_fwd"] = (
            df["squared_return"]
            .shift(-horizon)
            .rolling(horizon)
            .mean()
            .apply(np.sqrt)
            * np.sqrt(252)
        )

    return df


def build_features(spx: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature set for volatility forecasting.
    All features are backward-looking (no look-ahead bias).
    """
    rv = compute_realized_volatility(spx)

    # Merge VIX
    rv["vix_close"] = vix["close"]
    rv["vix_close"] = rv["vix_close"].ffill()

    # Lagged realized volatility features
    for lag in [1, 2, 3, 5, 10, 22]:
        rv[f"rv_5d_lag{lag}"] = rv["rv_5d"].shift(lag)
        rv[f"abs_ret_lag{lag}"] = rv["abs_return"].shift(lag)

    # HAR components (Corsi 2009): daily, weekly, monthly RV
    rv["har_daily"] = rv["rv_5d"].shift(1)
    rv["har_weekly"] = rv["rv_5d"].rolling(5).mean().shift(1)
    rv["har_monthly"] = rv["rv_5d"].rolling(22).mean().shift(1)

    # VIX features (implied vol is a strong predictor)
    rv["vix_lag1"] = rv["vix_close"].shift(1) / 100  # scale to decimal
    rv["vix_lag5"] = rv["vix_close"].rolling(5).mean().shift(1) / 100
    rv["vix_change"] = rv["vix_close"].pct_change().shift(1)

    # Leverage effect: negative returns predict higher vol
    rv["neg_return_5d"] = rv["log_return"].clip(upper=0).rolling(5).sum().shift(1)
    rv["pos_return_5d"] = rv["log_return"].clip(lower=0).rolling(5).sum().shift(1)

    # Volume features (if available)
    if "volume" in spx.columns:
        rv["log_volume"] = np.log(spx["volume"] + 1)
        rv["volume_change"] = rv["log_volume"].pct_change().shift(1)
        rv["volume_ma_ratio"] = (
            rv["log_volume"] / rv["log_volume"].rolling(22).mean()
        ).shift(1)

    # Day-of-week dummies (Monday effect on vol)
    rv["dow"] = rv.index.dayofweek
    for d in range(5):
        rv[f"dow_{d}"] = (rv["dow"] == d).astype(int)
    rv.drop(columns=["dow"], inplace=True)

    # Month dummies (January effect, etc.)
    rv["month"] = rv.index.month

    return rv


def main():
    spx = download_spx()
    vix = download_vix()

    # Save raw data
    spx.to_parquet(DATA_DIR / "spx_daily.parquet")
    vix.to_parquet(DATA_DIR / "vix_daily.parquet")
    print(f"Saved raw data to {DATA_DIR}")

    # Build combined feature set
    combined = build_features(spx, vix)

    # Drop rows with NaN (from rolling windows and forward targets)
    initial_rows = len(combined)
    combined = combined.dropna()
    print(f"Combined dataset: {len(combined)} rows ({initial_rows - len(combined)} dropped for NaN)")
    print(f"Date range: {combined.index[0].date()} to {combined.index[-1].date()}")
    print(f"Features: {len(combined.columns)} columns")

    combined.to_parquet(DATA_DIR / "combined.parquet")
    print(f"Saved combined dataset to {DATA_DIR / 'combined.parquet'}")

    # Summary statistics for the paper
    stats = combined[["log_return", "rv_5d", "rv_22d", "vix_close"]].describe()
    stats.to_csv(DATA_DIR / "summary_stats.csv")
    print("\nSummary Statistics:")
    print(stats.round(4))


if __name__ == "__main__":
    main()
