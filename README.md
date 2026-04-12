# Volatility Forecasting: A Horse Race Across GARCH, HAR, and Tree-Based Models

**Author:** Akram Khan  
**ORCID:** [0009-0002-7521-8648](https://orcid.org/0009-0002-7521-8648)

## Abstract

We conduct a comprehensive out-of-sample comparison of volatility forecasting methods on S&P 500 realized volatility from 2004 through 2025. Our horse race includes three GARCH variants (GARCH, EGARCH, GJR-GARCH), the HAR-RV model, gradient-boosted trees (LightGBM, XGBoost), and a simple equal-weight ensemble. The test period (2022-2025) covers the bear market, recovery, and rate normalization.

**Key finding:** GJR-GARCH is the best individual model (QLIKE = 0.3447), highlighting the dominant role of the leverage effect. A simple ensemble achieves the lowest QLIKE overall (0.3431).

## Results

| Model | QLIKE | RMSE | MZ R² |
|-------|-------|------|-------|
| **Ensemble** | **0.3431** | 0.0738 | 0.3515 |
| GJR-GARCH | 0.3447 | **0.0734** | **0.3802** |
| XGBoost | 0.3553 | 0.0745 | 0.3638 |
| LightGBM | 0.3632 | 0.0742 | 0.3754 |
| EGARCH | 0.3748 | 0.0769 | 0.3097 |
| GARCH | 0.3806 | 0.0796 | 0.2835 |
| HAR-RV | 0.4198 | 0.0779 | 0.2895 |

## Repository Contents

```
code/
  01_collect_data.py    # Download S&P 500 and VIX data from Yahoo Finance
  03_run_core_models.py # Run all models and generate results
paper/
  main.tex              # LaTeX source
  sections/             # Paper sections
  bib/references.bib    # Bibliography
  main.pdf              # Compiled paper (18 pages)
results/
  metrics_5d.csv        # Model comparison metrics
  metrics_5d.json       # Same in JSON format
```

## Reproducing the Results

```bash
# 1. Install dependencies
pip install pandas numpy yfinance arch lightgbm xgboost scikit-learn scipy

# 2. Download data (S&P 500 + VIX from Yahoo Finance, free)
cd code
python 01_collect_data.py

# 3. Run all models
python 03_run_core_models.py
```

**Requirements:** Python 3.10+, ~5 minutes runtime on a modern laptop.

## Data

All data is freely available from Yahoo Finance. No proprietary data is used. The scripts download data automatically.

- S&P 500 daily OHLCV (2004-2025): `^GSPC`
- CBOE VIX (2004-2025): `^VIX`

## Citation

```bibtex
@techreport{Khan2026vol,
  title={Volatility Forecasting with Machine Learning: A Horse Race Across GARCH, HAR, and Tree-Based Models},
  author={Khan, Akram},
  institution={SSRN},
  year={2026}
}
```

## License

MIT
