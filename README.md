# Investor sentiment and Bitcoin returns – data and code

Replication package for *"Yatırımcı duyarlılığı ile Bitcoin getirilerinin tahmini: Davranışsal finans ve makine öğrenmesi yaklaşımı"* (Predicting Bitcoin returns with investor sentiment: A behavioral finance and machine learning approach).

## Contents

| File | Description |
| --- | --- |
| `dataset_raw.csv` | Daily raw series, 31 Dec 2019 – 31 Dec 2024: BTC/USDT close and volume (Binance API, 00:00 UTC), Crypto Fear & Greed Index (Alternative.me), Google Trends "Bitcoin" worldwide (spliced daily series) |
| `dataset_final.csv` | Processed series used in all analyses (output of `analysis.py`) |
| `analysis.py` | Main analyses: unit roots, correlations, regressions (Tables 4–5), Granger tests (Table 7), RF/XGBoost (Tables 9–10) → `results.json` |
| `figs.py` | Figures 1–3 |
| `revision_analyses.py` | Robustness analyses added in revision (Tables 3, 6, 8, 10–11) → `rev_results.json` |
| `dow_check.py` | Granger/VAR with day-of-week dummies (Table 8) → `dow_results.json` |

## Reproduce

```bash
pip install -r requirements.txt
python analysis.py
python revision_analyses.py
python dow_check.py
python figs.py
```

Tested with Python 3.13, statsmodels 0.15, arch 8.0, scikit-learn 1.9, xgboost 2.1.4, pandas 3.0.
XGBoost 1.7.x produces slightly different XGBoost results; 2.x reproduces the reported values exactly.
All random procedures use `random_state = 42`.

## Data sources

Market data: Binance public API. Sentiment: Alternative.me Crypto Fear & Greed Index. Search interest: Google Trends.
Users should consult each provider's terms before redistributing the raw data.

## License

Code: MIT (see `LICENSE`).
