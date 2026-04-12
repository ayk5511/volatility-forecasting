"""
Paper 2: Core models only (GARCH + HAR + LightGBM + XGBoost + Ensemble).
Neural nets skipped due to PyTorch/Python 3.14 compatibility issue.

Usage: python 03_run_core_models.py
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import json
import signal

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TRAIN_END = "2018-12-31"
VAL_END = "2021-12-31"
TEST_START = "2022-01-01"


def load_data():
    df = pd.read_parquet(DATA_DIR / "combined.parquet")
    print(f"Loaded {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
    return df


# === GARCH ===
def fit_garch(returns, test_start):
    from arch import arch_model
    results = {"GARCH": [], "EGARCH": [], "GJR-GARCH": []}
    dates = []
    test_returns = returns[test_start:]
    scaled = returns * 100
    refit = 44  # refit every ~2 months

    class Timeout(Exception):
        pass

    def handler(s, f):
        raise Timeout()

    for i in range(0, len(test_returns), refit):
        end = min(i + refit, len(test_returns))
        batch = test_returns.index[i:end]
        # Use all data up to AND including the batch for GARCH (it forecasts from fitted values)
        train = scaled[:batch[-1]]

        for name, vol, kwargs in [
            ("GARCH", "Garch", {"p": 1, "q": 1}),
            ("EGARCH", "EGARCH", {"p": 1, "q": 1}),
            ("GJR-GARCH", "GARCH", {"p": 1, "o": 1, "q": 1}),
        ]:
            try:
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(20)
                am = arch_model(train, vol=vol, **kwargs, dist="Normal", mean="Zero")
                res = am.fit(disp="off", show_warning=False, options={"maxiter": 150})
                # Get conditional volatility from fitted values
                cond_vol = res.conditional_volatility
                # Extract the batch period, convert from % daily to annualized
                batch_vol = cond_vol.loc[batch] * np.sqrt(252) / 100
                signal.alarm(0)
                results[name].extend(batch_vol.values[:len(batch)])
            except Exception as e:
                signal.alarm(0)
                results[name].extend([np.nan] * len(batch))

        dates.extend(batch)
        if i % 220 == 0:
            print(f"  GARCH: {i}/{len(test_returns)}")

    return {k: pd.Series(v, index=dates[:len(v)], name=k) for k, v in results.items()}


# === HAR-RV ===
def fit_har(df, target, test_start):
    from sklearn.linear_model import LinearRegression
    feats = ["har_daily", "har_weekly", "har_monthly"]
    test = df[test_start:]
    preds, dates = [], []

    for i in range(0, len(test), 66):
        end = min(i + 66, len(test))
        batch = test.iloc[i:end]
        train = df[:batch.index[0]]
        m = LinearRegression().fit(train[feats].values, train[target].values)
        p = m.predict(batch[feats].values)
        preds.extend(p)
        dates.extend(batch.index)

    return pd.Series(preds, index=dates[:len(preds)], name="HAR-RV")


# === Trees ===
def fit_trees(df, target, test_start):
    import lightgbm as lgb
    import xgboost as xgb

    fcols = [c for c in df.columns if c.startswith(("rv_5d_lag", "abs_ret_lag",
             "har_", "vix_", "neg_return", "pos_return", "range_vol",
             "volume", "dow_", "log_return")) and "_fwd" not in c]

    results = {"LightGBM": [], "XGBoost": []}
    dates = []
    test = df[test_start:]

    for i in range(0, len(test), 66):
        end = min(i + 66, len(test))
        batch = test.iloc[i:end]
        train = df[:batch.index[0]]
        val = df[TRAIN_END:VAL_END]

        X_tr = np.nan_to_num(train[fcols].values, nan=0, posinf=0, neginf=0)
        y_tr = np.nan_to_num(train[target].values, nan=0, posinf=0, neginf=0)
        X_va = np.nan_to_num(val[fcols].values, nan=0, posinf=0, neginf=0)
        y_va = np.nan_to_num(val[target].values, nan=0, posinf=0, neginf=0)
        X_te = np.nan_to_num(batch[fcols].values, nan=0, posinf=0, neginf=0)

        # LightGBM
        ds_tr = lgb.Dataset(X_tr, y_tr)
        ds_va = lgb.Dataset(X_va, y_va, reference=ds_tr)
        lgb_m = lgb.train(
            {"objective": "regression", "metric": "mse", "num_leaves": 31,
             "learning_rate": 0.05, "feature_fraction": 0.8, "bagging_fraction": 0.8,
             "bagging_freq": 5, "verbose": -1, "n_jobs": -1},
            ds_tr, 500, valid_sets=[ds_va],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
        )
        results["LightGBM"].extend(lgb_m.predict(X_te))

        # XGBoost
        xgb_m = xgb.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            early_stopping_rounds=30, eval_metric="rmse", verbosity=0, n_jobs=-1
        )
        xgb_m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        results["XGBoost"].extend(xgb_m.predict(X_te))

        dates.extend(batch.index)
        if i % 220 == 0:
            print(f"  Trees: {i}/{len(test)}")

    return {k: pd.Series(v, index=dates[:len(v)], name=k) for k, v in results.items()}, fcols


# === Metrics ===
def metrics(actual, predicted):
    from scipy.stats import pearsonr
    c = actual.index.intersection(predicted.index)
    a, p = actual.loc[c].values, predicted.loc[c].values
    mask = ~(np.isnan(a) | np.isnan(p))
    a, p = a[mask], p[mask]
    if len(a) == 0:
        return {k: np.nan for k in ["MSE","MAE","RMSE","QLIKE","R2","MZ_R2","N"]}
    mse = np.mean((a-p)**2)
    eps = 1e-8
    ratio = (a**2)/(p**2+eps)
    qlike = np.mean(ratio - np.log(ratio+eps) - 1)
    ss_res = np.sum((a-p)**2)
    ss_tot = np.sum((a-np.mean(a))**2)
    corr, _ = pearsonr(a, p)
    return {"MSE": mse, "MAE": np.mean(np.abs(a-p)), "RMSE": np.sqrt(mse),
            "QLIKE": qlike, "R2": 1-ss_res/ss_tot if ss_tot>0 else 0,
            "MZ_R2": corr**2, "N": len(a)}


def main():
    print("=" * 60)
    print("Paper 2: Volatility Forecasting (Core Models)")
    print("=" * 60)

    df = load_data()
    target = "rv_5d_fwd"
    all_fc = {}

    print("\n[1/3] GARCH family...")
    all_fc.update(fit_garch(df["log_return"], TEST_START))

    print("\n[2/3] HAR-RV...")
    all_fc["HAR-RV"] = fit_har(df, target, TEST_START)

    print("\n[3/3] LightGBM + XGBoost...")
    tree_r, fcols = fit_trees(df, target, TEST_START)
    all_fc.update(tree_r)

    # Ensemble
    print("\nBuilding ensemble...")
    ci = all_fc["LightGBM"].index.intersection(
        all_fc["HAR-RV"].index).intersection(all_fc["GARCH"].index)
    ens = (all_fc["LightGBM"].loc[ci] + all_fc["HAR-RV"].loc[ci] + all_fc["GARCH"].loc[ci]) / 3
    ens.name = "Ensemble"
    all_fc["Ensemble"] = ens

    # Evaluate
    actual = df[TEST_START:][target]
    all_m = {}

    print("\n" + "=" * 60)
    print("RESULTS (5-day forward RV, test: 2022+)")
    print("=" * 60)

    for name, preds in all_fc.items():
        m = metrics(actual, preds)
        all_m[name] = m
        print(f"\n{name}:")
        for k, v in m.items():
            print(f"  {k}: {int(v) if k=='N' and not np.isnan(v) else f'{v:.6f}'}")

    # Save
    mdf = pd.DataFrame(all_m).T
    mdf.to_csv(RESULTS_DIR / "metrics_5d.csv")

    fdf = pd.DataFrame(all_fc)
    fdf["actual"] = actual
    fdf.to_parquet(RESULTS_DIR / "forecasts_5d.parquet")

    with open(RESULTS_DIR / "metrics_5d.json", "w") as f:
        json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in all_m.items()}, f, indent=2)

    print(f"\nSaved to {RESULTS_DIR}")

    print("\n" + "=" * 60)
    print("RANKING (QLIKE, lower = better)")
    print("=" * 60)
    for i, (name, row) in enumerate(mdf.sort_values("QLIKE").iterrows()):
        print(f"  {i+1}. {name:<15} QLIKE={row['QLIKE']:.6f}  RMSE={row['RMSE']:.6f}  MZ_R²={row['MZ_R2']:.4f}")


if __name__ == "__main__":
    main()
