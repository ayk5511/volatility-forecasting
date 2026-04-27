"""
Paper 2: end-to-end audit script.

Re-derives every numerical claim in the paper from results/forecasts_5d.parquet
and compares against the JSON files. Reports any inconsistency, prints
recomputed bold-winners for every table, sanity-checks forecasts (NaN,
non-positive, ensemble identity), and re-runs the Diebold-Mariano test.

Run:  python code/06_audit.py

Exits 0 if all checks pass, 1 otherwise.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def fail(msg: str):
    print(f"  FAIL: {msg}")
    return 1


def passmsg(msg: str):
    print(f"  ok:   {msg}")
    return 0


def section(title: str):
    print(f"\n=== {title} ===")


def qlike_loss(actual, predicted, eps=1e-8):
    ratio = (actual ** 2) / (predicted ** 2 + eps)
    return ratio - np.log(ratio + eps) - 1


def newey_west_se(d, q):
    d = d - d.mean()
    n = len(d)
    gamma0 = (d @ d) / n
    s = gamma0
    for k in range(1, q + 1):
        gk = (d[k:] @ d[:-k]) / n
        s += 2 * (1 - k / (q + 1)) * gk
    return float(np.sqrt(s / n)) if s > 0 else float("nan")


def metrics(actual, predicted):
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    a, p = actual[mask], predicted[mask]
    if len(a) == 0:
        return None
    mse = np.mean((a - p) ** 2)
    eps = 1e-8
    ratio = (a ** 2) / (p ** 2 + eps)
    qlike = np.mean(ratio - np.log(ratio + eps) - 1)
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    corr, _ = stats.pearsonr(a, p)
    return {
        "MSE": float(mse),
        "MAE": float(np.mean(np.abs(a - p))),
        "RMSE": float(np.sqrt(mse)),
        "QLIKE": float(qlike),
        "MZ_R2": float(corr ** 2),
        "N": int(len(a)),
    }


def check_close(name, paper_val, computed_val, tol=1e-4):
    diff = abs(paper_val - computed_val)
    if diff < tol:
        return passmsg(f"{name}: paper={paper_val:.4f}  computed={computed_val:.4f}  diff={diff:.6f}")
    return fail(f"{name}: paper={paper_val:.4f}  computed={computed_val:.4f}  diff={diff:.6f}")


def main():
    fails = 0

    section("LOAD")
    fc = pd.read_parquet(RESULTS_DIR / "forecasts_5d.parquet")
    print(f"  forecasts: {len(fc)} rows  {fc.index[0].date()} -> {fc.index[-1].date()}")
    print(f"  columns: {fc.columns.tolist()}")
    actual = fc["actual"].values
    models = ["GARCH", "EGARCH", "GJR-GARCH", "HAR-RV", "LightGBM", "XGBoost", "Ensemble"]

    section("SANITY: no NaN, no negative, no infinity")
    for col in fc.columns:
        v = fc[col].values
        if np.any(np.isnan(v)):
            fails += fail(f"{col} has NaN")
        elif np.any(np.isinf(v)):
            fails += fail(f"{col} has inf")
        elif np.any(v < 0):
            fails += fail(f"{col} has negative values (count={np.sum(v < 0)})")
        elif np.any(v == 0):
            fails += fail(f"{col} has zero values (count={np.sum(v == 0)})")
        else:
            passmsg(f"{col}: min={v.min():.4f} max={v.max():.4f} mean={v.mean():.4f}")

    section("ENSEMBLE IDENTITY: ensemble = mean(LightGBM, HAR-RV, GARCH)")
    expected_ens = (fc["LightGBM"] + fc["HAR-RV"] + fc["GARCH"]) / 3
    diff = (fc["Ensemble"] - expected_ens).abs().max()
    if diff < 1e-9:
        passmsg(f"ensemble identity holds, max-abs-diff = {diff:.2e}")
    else:
        fails += fail(f"ensemble != mean of components, max-abs-diff = {diff:.6f}")

    section("FULL-SAMPLE METRICS vs metrics_5d.json")
    paper_metrics = json.load(open(RESULTS_DIR / "metrics_5d.json"))
    for m in models:
        c = metrics(actual, fc[m].values)
        for k in ["QLIKE", "RMSE", "MAE", "MZ_R2"]:
            check_close(f"{m}.{k}", paper_metrics[m][k], c[k]) if k in paper_metrics[m] else None

    section("FULL-SAMPLE: paper-text claims (abstract & body) vs computed")
    # Abstract claims:
    #   GJR-GARCH QLIKE 0.3447
    #   Ensemble QLIKE 0.3431
    paper_claims = {
        "GJR-GARCH QLIKE": ("GJR-GARCH", "QLIKE", 0.3447),
        "Ensemble QLIKE": ("Ensemble", "QLIKE", 0.3431),
        "GJR-GARCH RMSE": ("GJR-GARCH", "RMSE", 0.0734),
        "GJR-GARCH MZ_R2": ("GJR-GARCH", "MZ_R2", 0.3802),
        "XGBoost MAE": ("XGBoost", "MAE", 0.0470),
        "XGBoost QLIKE": ("XGBoost", "QLIKE", 0.3553),
        "LightGBM QLIKE": ("LightGBM", "QLIKE", 0.3632),
        "GARCH QLIKE": ("GARCH", "QLIKE", 0.3806),
        "EGARCH QLIKE": ("EGARCH", "QLIKE", 0.3748),
        "HAR-RV QLIKE": ("HAR-RV", "QLIKE", 0.4198),
    }
    for label, (model, metric, claimed) in paper_claims.items():
        c = metrics(actual, fc[model].values)
        if abs(c[metric] - claimed) > 5e-4:
            fails += fail(f"{label}: paper={claimed:.4f}  computed={c[metric]:.4f}")
        else:
            passmsg(f"{label} = {claimed:.4f} verified")

    section("BOLD-WINNERS for each table column (recomputed)")

    def bold_winner(col_name, vals_dict, lower_better=True):
        if lower_better:
            winner = min(vals_dict, key=vals_dict.get)
        else:
            winner = max(vals_dict, key=vals_dict.get)
        print(f"  {col_name:<25s} winner={winner:<12s} value={vals_dict[winner]:.4f}")
        return winner

    full_qlike = {m: metrics(actual, fc[m].values)["QLIKE"] for m in models}
    full_rmse = {m: metrics(actual, fc[m].values)["RMSE"] for m in models}
    full_mae = {m: metrics(actual, fc[m].values)["MAE"] for m in models}
    full_mzr2 = {m: metrics(actual, fc[m].values)["MZ_R2"] for m in models}
    print("Full sample (Tab. 4):")
    bold_winner("QLIKE", full_qlike, lower_better=True)
    bold_winner("RMSE", full_rmse, lower_better=True)
    bold_winner("MAE", full_mae, lower_better=True)
    bold_winner("MZ R^2", full_mzr2, lower_better=False)

    is_2022 = (fc.index.year == 2022)
    is_2023plus = (fc.index.year >= 2023)
    yr_2022_qlike = {m: metrics(actual[is_2022], fc[m].values[is_2022])["QLIKE"] for m in models}
    yr_other_qlike = {m: metrics(actual[is_2023plus], fc[m].values[is_2023plus])["QLIKE"] for m in models}
    print("Calendar split (Tab. 5):")
    bold_winner("2022 QLIKE", yr_2022_qlike, lower_better=True)
    bold_winner("2023-25 QLIKE", yr_other_qlike, lower_better=True)
    print(f"  2022 N = {is_2022.sum()},  2023+ N = {is_2023plus.sum()},  total = {is_2022.sum() + is_2023plus.sum()}")

    threshold = float(np.nanpercentile(actual, 75))
    is_high = actual >= threshold
    high_qlike = {m: metrics(actual[is_high], fc[m].values[is_high])["QLIKE"] for m in models}
    low_qlike = {m: metrics(actual[~is_high], fc[m].values[~is_high])["QLIKE"] for m in models}
    print("Regime split (Tab. 6):")
    bold_winner("High-vol QLIKE", high_qlike, lower_better=True)
    bold_winner("Lower-vol QLIKE", low_qlike, lower_better=True)
    print(f"  threshold (75th pctile of actual RV) = {threshold:.4f}")
    print(f"  high N = {is_high.sum()},  lower N = {(~is_high).sum()},  total = {is_high.sum() + (~is_high).sum()}")

    section("PAPER's BOLD CELL CLAIMS vs RECOMPUTED")
    # Tab. 4
    if min(full_qlike, key=full_qlike.get) != "Ensemble":
        fails += fail(f"Tab.4 QLIKE bold should be {min(full_qlike, key=full_qlike.get)}, paper bolds Ensemble")
    else:
        passmsg("Tab.4 QLIKE bold = Ensemble (matches paper)")
    if min(full_rmse, key=full_rmse.get) != "GJR-GARCH":
        fails += fail(f"Tab.4 RMSE bold should be {min(full_rmse, key=full_rmse.get)}, paper bolds GJR-GARCH")
    else:
        passmsg("Tab.4 RMSE bold = GJR-GARCH (matches paper)")
    if min(full_mae, key=full_mae.get) != "XGBoost":
        fails += fail(f"Tab.4 MAE bold should be {min(full_mae, key=full_mae.get)}, paper bolds XGBoost")
    else:
        passmsg("Tab.4 MAE bold = XGBoost (matches paper)")
    if max(full_mzr2, key=full_mzr2.get) != "GJR-GARCH":
        fails += fail(f"Tab.4 MZ_R2 bold should be {max(full_mzr2, key=full_mzr2.get)}, paper bolds GJR-GARCH")
    else:
        passmsg("Tab.4 MZ_R2 bold = GJR-GARCH (matches paper)")

    # Tab. 5
    paper_bold_2022 = "EGARCH"
    paper_bold_other = "XGBoost"
    paper_bold_full = "Ensemble"
    if min(yr_2022_qlike, key=yr_2022_qlike.get) != paper_bold_2022:
        fails += fail(f"Tab.5 2022 bold should be {min(yr_2022_qlike, key=yr_2022_qlike.get)}, paper bolds {paper_bold_2022}")
    else:
        passmsg(f"Tab.5 2022 bold = {paper_bold_2022} (matches paper)")
    if min(yr_other_qlike, key=yr_other_qlike.get) != paper_bold_other:
        fails += fail(f"Tab.5 2023-25 bold should be {min(yr_other_qlike, key=yr_other_qlike.get)}, paper bolds {paper_bold_other}")
    else:
        passmsg(f"Tab.5 2023-25 bold = {paper_bold_other} (matches paper)")

    # Tab. 6 (corrected after audit found bugs)
    paper_bold_high = "EGARCH"
    paper_bold_low = "Ensemble"
    if min(high_qlike, key=high_qlike.get) != paper_bold_high:
        fails += fail(f"Tab.6 high-vol bold should be {min(high_qlike, key=high_qlike.get)}, paper bolds {paper_bold_high}")
    else:
        passmsg(f"Tab.6 high-vol bold = {paper_bold_high} (matches paper)")
    if min(low_qlike, key=low_qlike.get) != paper_bold_low:
        fails += fail(f"Tab.6 lower-vol bold should be {min(low_qlike, key=low_qlike.get)}, paper bolds {paper_bold_low}")
    else:
        passmsg(f"Tab.6 lower-vol bold = {paper_bold_low} (matches paper)")

    section("DM TESTS (full sample) re-derive vs dm_tests.json")
    losses = {m: qlike_loss(actual, fc[m].values) for m in models}
    paper_dm = json.load(open(RESULTS_DIR / "dm_tests.json"))["pairs"]
    key_pairs = [
        ("GJR-GARCH", "GARCH"),
        ("GJR-GARCH", "EGARCH"),
        ("GJR-GARCH", "HAR-RV"),
        ("Ensemble", "GARCH"),
        ("Ensemble", "GJR-GARCH"),
        ("Ensemble", "LightGBM"),
        ("LightGBM", "HAR-RV"),
        ("XGBoost", "LightGBM"),
        ("EGARCH", "GARCH"),
    ]
    h = 5
    q = h - 1
    for a, b in key_pairs:
        d = losses[a] - losses[b]
        d = d[~np.isnan(d)]
        se = newey_west_se(d, q)
        t = d.mean() / se if se and not np.isnan(se) and se > 0 else float("nan")
        p = 2 * (1 - stats.norm.cdf(abs(t))) if not np.isnan(t) else float("nan")
        paper = paper_dm[f"{a} vs {b}"]
        if abs(t - paper["t_stat"]) > 0.01 or abs(p - paper["p_value"]) > 0.001:
            fails += fail(f"DM {a} vs {b}: recomputed t={t:.3f} p={p:.4f}  json t={paper['t_stat']:.3f} p={paper['p_value']:.4f}")
        else:
            passmsg(f"DM {a} vs {b}: t={t:+.3f}, p={p:.4f}")

    section("PROSE NUMBER CHECKS (subperiod values mentioned in §5.3 and §6)")
    prose_claims = {
        "§5.3 EGARCH 2022 = 0.2346": yr_2022_qlike["EGARCH"],
        "§5.3 GARCH 2022 = 0.2579": yr_2022_qlike["GARCH"],
        "§5.3 GJR-GARCH 2022 = 0.2597": yr_2022_qlike["GJR-GARCH"],
        "§5.3 XGBoost 2022 = 0.3148": yr_2022_qlike["XGBoost"],
        "§5.3 LightGBM 2022 = 0.3429": yr_2022_qlike["LightGBM"],
        "§5.3 Ensemble 2022 = 0.2616": yr_2022_qlike["Ensemble"],
        "§5.3 XGBoost 2023+ = 0.3693": yr_other_qlike["XGBoost"],
        "§5.3 LightGBM 2023+ = 0.3702": yr_other_qlike["LightGBM"],
        "§5.3 Ensemble 2023+ = 0.3712": yr_other_qlike["Ensemble"],
        "§5.3 GJR-GARCH 2023+ = 0.3740": yr_other_qlike["GJR-GARCH"],
        "§5.3 HAR-RV 2023+ = 0.4568": yr_other_qlike["HAR-RV"],
        "§6.2 EGARCH high-vol = 0.6252": high_qlike["EGARCH"],
        "§6.2 GJR-GARCH high-vol = 0.6282": high_qlike["GJR-GARCH"],
        "§6.2 GARCH high-vol = 0.6913": high_qlike["GARCH"],
        "§6.2 HAR-RV high-vol = 0.9015": high_qlike["HAR-RV"],
        "§6.2 LightGBM high-vol = 0.7276": high_qlike["LightGBM"],
        "§6.2 XGBoost high-vol = 0.6841": high_qlike["XGBoost"],
        "§6.2 Ensemble high-vol = 0.6850": high_qlike["Ensemble"],
        "§6.2 Ensemble lower-vol = 0.2291": low_qlike["Ensemble"],
        "§6.2 LightGBM lower-vol = 0.2418": low_qlike["LightGBM"],
        "§6.2 XGBoost lower-vol = 0.2457": low_qlike["XGBoost"],
        "§6.2 GJR-GARCH lower-vol = 0.2502": low_qlike["GJR-GARCH"],
        "§6.2 HAR-RV lower-vol = 0.2592": low_qlike["HAR-RV"],
        "§6.2 GARCH lower-vol = 0.2770": low_qlike["GARCH"],
        "§6.2 EGARCH lower-vol = 0.2913": low_qlike["EGARCH"],
        "§6.2 threshold q75 = 0.1905": float(np.nanpercentile(actual, 75)),
    }
    paper_vals = {
        "§5.3 EGARCH 2022 = 0.2346": 0.2346,
        "§5.3 GARCH 2022 = 0.2579": 0.2579,
        "§5.3 GJR-GARCH 2022 = 0.2597": 0.2597,
        "§5.3 XGBoost 2022 = 0.3148": 0.3148,
        "§5.3 LightGBM 2022 = 0.3429": 0.3429,
        "§5.3 Ensemble 2022 = 0.2616": 0.2616,
        "§5.3 XGBoost 2023+ = 0.3693": 0.3693,
        "§5.3 LightGBM 2023+ = 0.3702": 0.3702,
        "§5.3 Ensemble 2023+ = 0.3712": 0.3712,
        "§5.3 GJR-GARCH 2023+ = 0.3740": 0.3740,
        "§5.3 HAR-RV 2023+ = 0.4568": 0.4568,
        "§6.2 EGARCH high-vol = 0.6252": 0.6252,
        "§6.2 GJR-GARCH high-vol = 0.6282": 0.6282,
        "§6.2 GARCH high-vol = 0.6913": 0.6913,
        "§6.2 HAR-RV high-vol = 0.9015": 0.9015,
        "§6.2 LightGBM high-vol = 0.7276": 0.7276,
        "§6.2 XGBoost high-vol = 0.6841": 0.6841,
        "§6.2 Ensemble high-vol = 0.6850": 0.6850,
        "§6.2 Ensemble lower-vol = 0.2291": 0.2291,
        "§6.2 LightGBM lower-vol = 0.2418": 0.2418,
        "§6.2 XGBoost lower-vol = 0.2457": 0.2457,
        "§6.2 GJR-GARCH lower-vol = 0.2502": 0.2502,
        "§6.2 HAR-RV lower-vol = 0.2592": 0.2592,
        "§6.2 GARCH lower-vol = 0.2770": 0.2770,
        "§6.2 EGARCH lower-vol = 0.2913": 0.2913,
        "§6.2 threshold q75 = 0.1905": 0.1905,
    }
    for label in prose_claims:
        if abs(prose_claims[label] - paper_vals[label]) > 5e-4:
            fails += fail(f"{label}  computed={prose_claims[label]:.4f}")
        else:
            passmsg(label)

    section("FINAL")
    if fails == 0:
        print("\nALL CHECKS PASSED")
        return 0
    else:
        print(f"\n{fails} CHECK(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
