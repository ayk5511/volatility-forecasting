# Volatility Forecasting with Machine Learning — A Horse Race Across GARCH, HAR, and Tree-Based Models

This repository accompanies the paper:

> Khan, A. (2026). *Volatility Forecasting with Machine Learning: A Horse Race Across GARCH, HAR, and Tree-Based Models.* SSRN Working Paper.

It contains the LaTeX source, the data-collection and modeling scripts, the engineered feature panel, the per-day forecasts of all seven model configurations, and the full suite of metrics used to populate every table in the paper.

**Author:** Akram Khan
**ORCID:** [0009-0002-7521-8648](https://orcid.org/0009-0002-7521-8648)
**Contact:** [1819ak@gmail.com](mailto:1819ak@gmail.com)
**Companion paper:** Khan (2026), *Machine Learning in Quantitative Finance: A Systematic Review*, SSRN [6562398](https://ssrn.com/abstract=6562398) — this paper extends that survey by applying its Reproducibility Disclosure Score rubric to a concrete empirical case study.

## Headline result

We run seven volatility forecasting model configurations on S&P 500 realized volatility from 2004 through November 2025, with out-of-sample evaluation over January 2022 through November 2025 (980 trading days). On the full test sample:

| Model | QLIKE ↓ | RMSE ↓ | MZ R² ↑ |
|---|---:|---:|---:|
| **Ensemble** (LightGBM + HAR-RV + GARCH) | **0.3431** | 0.0738 | 0.3515 |
| GJR-GARCH | 0.3447 | **0.0734** | **0.3802** |
| XGBoost | 0.3553 | 0.0745 | 0.3638 |
| LightGBM | 0.3632 | 0.0742 | 0.3754 |
| EGARCH | 0.3748 | 0.0769 | 0.3097 |
| GARCH(1,1) | 0.3806 | 0.0796 | 0.2835 |
| HAR-RV | 0.4198 | 0.0779 | 0.2895 |

But the full-sample number hides a **subperiod reversal**:

| Model | 2022 (high vol, N=251) | 2023–2025 (lower vol, N=729) |
|---|---:|---:|
| **EGARCH** | **0.2346** | 0.4230 |
| GARCH(1,1) | 0.2579 | 0.4228 |
| GJR-GARCH | 0.2597 | 0.3740 |
| Ensemble | 0.2616 | 0.3712 |
| HAR-RV | 0.3123 | 0.4568 |
| XGBoost | 0.3148 | **0.3693** |
| LightGBM | 0.3429 | 0.3702 |

In 2022, the GARCH family wins and tree models trail. In 2023–2025, tree models lead and HAR-RV is last. The Diebold-Mariano test ([results/dm_tests.json](results/dm_tests.json)) shows the ensemble's full-sample edge over GJR-GARCH is **not** statistically significant (*p* = 0.90).

## Repository contents

```
volatility-forecasting/
├── README.md                              # This file
├── LICENSE                                # CC BY 4.0 (paper) + CC0 (data) + MIT (code)
├── .gitignore
├── code/
│   ├── 01_collect_data.py                 # Yahoo Finance pull + feature engineering
│   ├── 02_models.py                       # Reference model module
│   ├── 03_run_core_models.py              # Main runner (GARCH + HAR + LightGBM + XGBoost + Ensemble)
│   ├── 04_subperiod_and_importance.py     # Subperiod metrics + LightGBM split-importance
│   └── 05_dm_tests.py                     # Diebold-Mariano two-sided tests with HAC SE
├── data/
│   ├── README.md                          # Schema + provenance
│   ├── spx_daily.parquet                  # ^GSPC OHLCV from Yahoo Finance
│   ├── vix_daily.parquet                  # ^VIX OHLCV from Yahoo Finance
│   ├── combined.parquet                   # Engineered 35-feature panel + forward targets
│   └── summary_stats.csv                  # Table 2 in the paper
├── results/
│   ├── metrics_5d.csv                     # Table 4 in the paper, CSV form
│   ├── metrics_5d.json                    # Same metrics, JSON form
│   ├── metrics_subperiod.json             # Tables 5 & 6 raw values (year + RV-regime splits)
│   ├── feature_importance.json            # Table 7 raw values + full-feature ranking
│   ├── dm_tests.json                      # Diebold-Mariano p-values for all model pairs
│   ├── forecasts_5d.csv                   # Per-day forecasts (980 rows × 7 models + actual)
│   └── forecasts_5d.parquet               # Same panel in parquet
└── paper/
    ├── main.tex                           # Master LaTeX file
    ├── main.pdf                           # Compiled paper
    ├── sections/                          # 7 section files (intro through conclusion)
    ├── bib/references.bib                 # Bibliography
    └── submission-ssrn/
        ├── Khan_2026_Volatility_Forecasting.pdf
        └── SSRN_SUBMISSION_METADATA.txt   # Title, abstract, keywords, JEL codes
```

## Reproducing the paper

The full pipeline runs on a single laptop in under five minutes from a clean Python 3.11+ environment.

```bash
git clone https://github.com/ayk5511/volatility-forecasting.git
cd volatility-forecasting

python -m venv .venv && source .venv/bin/activate
pip install yfinance pandas numpy pyarrow scipy scikit-learn lightgbm xgboost arch

# Step 1: Download raw data + build feature panel.
python code/01_collect_data.py

# Step 2: Run all seven model configurations and dump forecasts + metrics.
python code/03_run_core_models.py

# Step 3: Compute subperiod metrics + LightGBM split-importance + CSV export.
python code/04_subperiod_and_importance.py

# Step 4: Diebold-Mariano significance tests.
python code/05_dm_tests.py
```

Steps 1–4 regenerate every numerical value reported in the paper. The Yahoo Finance pull may produce row counts that differ by a handful from the snapshots committed here if Yahoo has restated historical bars (it occasionally re-states corporate-action splits and dividends).

To recompile the paper:

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Reproducibility commitment

This paper aims to score **2** on the Reproducibility Disclosure Score (RDS) rubric proposed in [Khan (2026)](https://ssrn.com/abstract=6562398) — code public (+1) and data openly accessible (+1). Concretely:

- **Code public**: every Python script that touches the analysis is in `code/`, MIT-licensed.
- **Data accessible**: the raw inputs are obtainable for free via the included `01_collect_data.py` script; the engineered feature panel and all model outputs are committed to this repo under CC0.
- **Audit trail**: the `results/*.json` files contain not just headline metrics but also fit metadata (boosting-round count, HAC bandwidth, sample sizes by subperiod) so that any number in the paper can be traced back to a specific computation.
- **Issues**: corrections, extensions, and challenges to the rankings are welcomed via the [issue tracker](https://github.com/ayk5511/volatility-forecasting/issues).

## How to cite

```bibtex
@techreport{KhanVol2026,
  title  = {Volatility Forecasting with Machine Learning: A Horse Race Across GARCH, HAR, and Tree-Based Models},
  author = {Khan, Akram},
  institution = {SSRN Working Paper},
  year   = {2026},
  url    = {https://github.com/ayk5511/volatility-forecasting}
}
```

## License

See [LICENSE](LICENSE). In short: paper text is CC BY 4.0, derived datasets and model outputs are CC0, code is MIT.
