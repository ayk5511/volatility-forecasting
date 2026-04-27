"""
Paper 2: Diebold-Mariano (1995) tests on QLIKE loss differentials.

For each ordered model pair (A, B) in the seven-model panel, computes
  d_t = L_QLIKE(actual_t, A_t) - L_QLIKE(actual_t, B_t)
and tests H_0: E[d] = 0 using a HAC-corrected (Newey-West) standard error
with bandwidth h-1 = 4 (rule-of-thumb for a 5-day forecast horizon).

A negative t-statistic means model A's QLIKE is lower (i.e. A is better).
A two-sided p-value < 0.05 indicates significant ranking.

Output: results/dm_tests.json
"""
import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = Path(__file__).parent.parent / "results"

H = 5  # forecast horizon, drives HAC bandwidth = H - 1


def qlike_loss(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    eps = 1e-8
    ratio = (actual ** 2) / (predicted ** 2 + eps)
    return ratio - np.log(ratio + eps) - 1


def newey_west_se(d: np.ndarray, q: int) -> float:
    """HAC long-run standard error of the mean with Bartlett kernel, bandwidth q."""
    d = d - d.mean()
    n = len(d)
    gamma0 = (d @ d) / n
    s = gamma0
    for k in range(1, q + 1):
        gk = (d[k:] @ d[:-k]) / n
        s += 2 * (1 - k / (q + 1)) * gk
    return float(np.sqrt(s / n)) if s > 0 else float("nan")


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, q: int = H - 1) -> dict:
    d = loss_a - loss_b
    mask = ~np.isnan(d)
    d = d[mask]
    if len(d) < 10:
        return {"n": len(d), "mean_diff": float("nan"), "t_stat": float("nan"), "p_value": float("nan")}
    se = newey_west_se(d, q)
    t = d.mean() / se if se and not np.isnan(se) and se > 0 else float("nan")
    p = 2 * (1 - stats.norm.cdf(abs(t))) if not np.isnan(t) else float("nan")
    return {
        "n": int(len(d)),
        "mean_diff": float(d.mean()),
        "t_stat": float(t),
        "p_value": float(p),
        "se": float(se),
    }


def main():
    fc = pd.read_parquet(RESULTS_DIR / "forecasts_5d.parquet")
    actual = fc["actual"].values
    models = [c for c in fc.columns if c != "actual"]
    losses = {m: qlike_loss(actual, fc[m].values) for m in models}

    results = {}
    for a, b in permutations(models, 2):
        results[f"{a} vs {b}"] = dm_test(losses[a], losses[b])

    with open(RESULTS_DIR / "dm_tests.json", "w") as f:
        json.dump(
            {
                "metadata": {
                    "loss_function": "QLIKE",
                    "forecast_horizon_days": H,
                    "hac_bandwidth": H - 1,
                    "test": "Diebold-Mariano (1995) two-sided",
                    "interpretation": "negative t_stat means A beats B; p_value < 0.05 = significant",
                },
                "pairs": results,
            },
            f,
            indent=2,
        )
    print(f"Wrote {RESULTS_DIR / 'dm_tests.json'}")

    print("\nKey pairs (full sample, two-sided p-values):")
    key_pairs = [
        ("GJR-GARCH", "GARCH"),
        ("EGARCH", "GARCH"),
        ("GJR-GARCH", "EGARCH"),
        ("Ensemble", "GARCH"),
        ("Ensemble", "GJR-GARCH"),
        ("Ensemble", "LightGBM"),
        ("LightGBM", "HAR-RV"),
        ("XGBoost", "LightGBM"),
        ("GJR-GARCH", "LightGBM"),
        ("GJR-GARCH", "HAR-RV"),
    ]
    print(f"  {'Pair':<32s} {'mean diff':>12s} {'t-stat':>8s} {'p-val':>8s}")
    for a, b in key_pairs:
        r = results[f"{a} vs {b}"]
        winner = a if r["mean_diff"] < 0 else b if not np.isnan(r["mean_diff"]) else "-"
        print(
            f"  {a + ' vs ' + b:<32s} {r['mean_diff']:>+12.5f} {r['t_stat']:>+8.3f} {r['p_value']:>8.4f}  ({'sig' if r['p_value'] < 0.05 else 'ns'}, {winner} wins)"
        )


if __name__ == "__main__":
    main()
