# SPA-MSJF: Multi-Task Financial Forecasting

## Files

| File | Description |
|------|-------------|
| `spa_msjf.py` | SPA-MSJF model architecture - private/shared LSTM encoders, SPA attention, task heads |
| `build_data.py` | Feature engineering, target computation, data loading pipeline |
| `train.py` | Training loop with early stopping, LR scheduling, evaluation |
| `tune.py` | Two-phase grid search over architecture and loss lambda weights |
| `build_sentiment.py` | GDELT scrape + FinBERT scoring pipeline for sentiment features |
| `test_evaluation.py` | Proposal test suite - architecture validation, regime metrics, MTL vs single-task |
| `setup.ps1` | One-command Windows/CUDA setup script |
| `requirements.txt` | Python dependencies |

---

## How To Run

```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 train.py
py -3.12 tune.py
py -3.12 build_sentiment.py
```

---

## Feature Engineering

Raw OHLCV replaced with 5 stationary engineered features per stock to fix train/val
distribution shift (AAPL went $15 in 2012 to $180 in 2023 - raw prices drift, these do not):

| Feature | Formula | Signal |
|---------|---------|--------|
| `log_return` | log(Close_t / Close_{t-1}) | Daily price direction |
| `hl_range` | (High - Low) / Close | Intraday volatility |
| `oc_return` | (Close - Open) / Open | Intraday momentum |
| `log_vol_change` | log(Vol_t / Vol_{t-1}) | Institutional activity |
| `momentum_5` | log(Close_t / Close_{t-5}) | 5-day trend |

Regression targets (return, volatility, Sharpe) are z-scored using train statistics only.

---

## Best Results (Price Features Only)

| Task | Metric | Result |
|------|--------|--------|
| Regime classification | Test accuracy | 96.8% |
| Direction (GOOG) | Val accuracy | 56.1% |
| Direction (avg) | Val accuracy | ~50% |
| Sharpe MSE (GOOG) | Val | 0.128 (normalised) |

Best hyperparams from `tune.py`: `private_hidden=32`, `shared_hidden=64`,
`spa_dim=32`, `dropout=0.1`, `lr=5e-4`, `weight_decay=1e-4`, `seq_len=60`.

---

## Sentiment Integration - Why It Did Not Help Yet

FinBERT sentiment was scraped from GDELT and fully wired in (`input_dim` auto-scales
5 to 9). Performance degraded across all configs:

| Config | Val Loss | Regime Test |
|--------|----------|-------------|
| Price only | 2.061 | 96.5% |
| Price + 3 sentiment features | 2.282 | 84.6% |
| Price + 4 sentiment + has_news flag | 2.295 | 92.6% |

Root cause: GDELT rate-limiting gave only 18.8% daily coverage (650 of 3453 days).
The LSTM sees neutral fill-in zeros 81.2% of the time and cannot distinguish real
sentiment from fill-in - sparse features act as noise, not signal. Even adding a
binary `has_news` flag only partially recovered regime accuracy.

The architecture is ready. Denser coverage from a paid API (Polygon.io, Benzinga)
would likely improve direction accuracy specifically, where sentiment is most predictive.

Sentiment signals when present are directionally correct:
- NVDA mean = +0.101 (positive - AI boom)
- META mean = -0.177 (negative - privacy controversies)
- GOOG mean = -0.123 (negative - antitrust)

---

## Data Split

| Split | Dates | Days |
|-------|-------|------|
| Train | 2012-05-18 to 2023-05-10 | 2762 |
| Val | 2023-05-11 to 2024-09-24 | 345 |
| Test | 2024-09-25 to 2026-02-11 | 345 |

Stocks: AAPL, GOOG, META, NVDA, TSLA