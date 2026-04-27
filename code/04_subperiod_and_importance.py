"""
Paper 2: Volatility Forecasting — Subperiod metrics + LightGBM feature importances.

Reads results/forecasts_5d.parquet, computes:
  1. Subperiod QLIKE/RMSE/MZ-R^2 by calendar year (2022 vs 2023-2025).
  2. Subperiod metrics by RV regime (75th-percentile split on actual 5-day RV).
  3. LightGBM feature importances from a single fit on the full training+validation
     window (2004-2021), used to fill the feature-importance table in Section 5.
  4. Plain-CSV export of the forecasts panel for the supplementary release.

Produces:
  results/metrics_subperiod.json
  results/feature_importance.json
  results/forecasts_5d.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

TRAIN_VAL_END = "2021-12-31"


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    from scipy.stats import pearsonr

    mask = ~(np.isnan(actual) | np.isnan(predicted))
    a, p = actual[mask], predicted[mask]
    if len(a) == 0:
        return {k: float("nan") for k in ["MSE", "MAE", "RMSE", "QLIKE", "R2", "MZ_R2", "N"]}
    mse = np.mean((a - p) ** 2)
    eps = 1e-8
    ratio = (a ** 2) / (p ** 2 + eps)
    qlike = np.mean(ratio - np.log(ratio + eps) - 1)
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    corr, _ = pearsonr(a, p)
    return {
        "MSE": float(mse),
        "MAE": float(np.mean(np.abs(a - p))),
        "RMSE": float(np.sqrt(mse)),
        "QLIKE": float(qlike),
        "R2": float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
        "MZ_R2": float(corr ** 2),
        "N": int(len(a)),
    }


def subperiod_metrics(fc: pd.DataFrame) -> dict:
    actual = fc["actual"].values
    models = [c for c in fc.columns if c != "actual"]
    out: dict = {"by_year": {}, "by_regime": {}}

    is_2022 = np.asarray(fc.index.year == 2022)
    is_2023_25 = np.asarray(fc.index.year >= 2023)
    out["by_year"]["2022_high_vol"] = {
        "N": int(is_2022.sum()),
        "models": {m: metrics(actual[is_2022], fc[m].values[is_2022]) for m in models},
    }
    out["by_year"]["2023_2025_lower_vol"] = {
        "N": int(is_2023_25.sum()),
        "models": {m: metrics(actual[is_2023_25], fc[m].values[is_2023_25]) for m in models},
    }

    threshold = np.nanpercentile(actual, 75)
    is_high = actual >= threshold
    out["by_regime"]["meta"] = {"threshold_q75": float(threshold)}
    out["by_regime"]["high_vol_top_quartile"] = {
        "N": int(is_high.sum()),
        "models": {m: metrics(actual[is_high], fc[m].values[is_high]) for m in models},
    }
    out["by_regime"]["lower_vol_bottom_3q"] = {
        "N": int((~is_high).sum()),
        "models": {m: metrics(actual[~is_high], fc[m].values[~is_high]) for m in models},
    }
    return out


def compute_feature_importance() -> dict:
    import lightgbm as lgb

    df = pd.read_parquet(DATA_DIR / "combined.parquet")
    target = "rv_5d_fwd"

    fcols = [
        c
        for c in df.columns
        if c.startswith(
            ("rv_5d_lag", "abs_ret_lag", "har_", "vix_", "neg_return", "pos_return", "range_vol", "volume", "dow_", "log_return")
        )
        and "_fwd" not in c
    ]

    train_val = df[:TRAIN_VAL_END].dropna(subset=[target])
    X = np.nan_to_num(train_val[fcols].values, nan=0, posinf=0, neginf=0)
    y = np.nan_to_num(train_val[target].values, nan=0, posinf=0, neginf=0)

    n_train = int(len(train_val) * 0.8)
    X_tr, y_tr = X[:n_train], y[:n_train]
    X_va, y_va = X[n_train:], y[n_train:]

    ds_tr = lgb.Dataset(X_tr, y_tr)
    ds_va = lgb.Dataset(X_va, y_va, reference=ds_tr)
    model = lgb.train(
        {
            "objective": "regression",
            "metric": "mse",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "n_jobs": -1,
        },
        ds_tr,
        num_boost_round=500,
        valid_sets=[ds_va],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )

    importance = model.feature_importance(importance_type="split")
    imp = sorted(zip(fcols, importance.tolist()), key=lambda x: -x[1])
    total = sum(importance) or 1
    return {
        "fit_window": f"2004-01-02 to {TRAIN_VAL_END}",
        "fit_rows": int(len(train_val)),
        "best_iteration": int(model.best_iteration or model.current_iteration()),
        "importance_type": "split",
        "n_features": len(fcols),
        "ranked": [
            {"rank": i + 1, "feature": f, "importance": int(v), "share": float(v) / float(total)}
            for i, (f, v) in enumerate(imp)
        ],
    }


def main():
    fc = pd.read_parquet(RESULTS_DIR / "forecasts_5d.parquet")
    print(f"Loaded forecasts: {len(fc)} rows, {fc.index[0].date()} to {fc.index[-1].date()}")

    sub = subperiod_metrics(fc)
    with open(RESULTS_DIR / "metrics_subperiod.json", "w") as f:
        json.dump(sub, f, indent=2)
    print("Wrote metrics_subperiod.json")

    print("Fitting LightGBM for feature importance...")
    fi = compute_feature_importance()
    with open(RESULTS_DIR / "feature_importance.json", "w") as f:
        json.dump(fi, f, indent=2)
    print(f"Wrote feature_importance.json (best_iteration={fi['best_iteration']})")

    csv_path = RESULTS_DIR / "forecasts_5d.csv"
    fc.to_csv(csv_path)
    print(f"Wrote {csv_path.name} ({csv_path.stat().st_size:,} bytes)")

    print("\nTop-10 features:")
    for r in fi["ranked"][:10]:
        print(f"  {r['rank']:2d}. {r['feature']:<22s}  importance={r['importance']:5d}  share={r['share']:.3f}")

    print("\n2022 (high-vol) QLIKE ranking:")
    yr_2022 = sub["by_year"]["2022_high_vol"]["models"]
    for name, m in sorted(yr_2022.items(), key=lambda x: x[1]["QLIKE"]):
        print(f"  {name:<12s}  QLIKE={m['QLIKE']:.4f}  RMSE={m['RMSE']:.4f}")

    print("\n2023-2025 (lower-vol) QLIKE ranking:")
    yr_other = sub["by_year"]["2023_2025_lower_vol"]["models"]
    for name, m in sorted(yr_other.items(), key=lambda x: x[1]["QLIKE"]):
        print(f"  {name:<12s}  QLIKE={m['QLIKE']:.4f}  RMSE={m['RMSE']:.4f}")


if __name__ == "__main__":
    main()
