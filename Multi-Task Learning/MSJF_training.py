# Train MSJF Model 
from spa_msjf import SPAMSJF
import torch 
import sys
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix
from typing import Optional

# Set up device
# Use CPU/MPS if possible
device = None
if "google.colab" in sys.modules:
    # Running in Colab
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
else:
    # Not in Colab (e.g., Mac)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
STOCKS       = ["AAPL", "GOOG", "META", "NVDA", "TSLA"]
FEATURES     = ["Open", "High", "Low", "Close", "Volume"]

SEQ_LEN      = 30       # lookback window (trading days)
HORIZON      = 1        # predict 1 day ahead
VOL_WINDOW   = 20       # rolling window for volatility / Sharpe
DIR_THRESH   = 0.002    # ±0.2% → flat band for direction labels
NUM_REGIMES  = 4        # Bear or Bull Market
BATCH_SIZE   = 32
SPLIT        = (0.8, 0.1, 0.1)


# ─────────────────────────────────────────────
# 1. Load & align CSVs
# ─────────────────────────────────────────────
def load_and_align(data_dir: str) -> dict[str, pd.DataFrame]:
    """
    Load each CSV, parse dates, find the common date range across all
    5 stocks, and return aligned DataFrames indexed by date.
    """
    frames = {}
    for ticker in STOCKS:
        path = os.path.join(data_dir, f"Historical_Data_{ticker}.csv")
        df = pd.read_csv(path, parse_dates=["Date"])
        df = df.sort_values("Date").set_index("Date")
        df = df[FEATURES].dropna()
        frames[ticker] = df
 
    # Common date intersection
    common_idx = frames[STOCKS[0]].index
    for ticker in STOCKS[1:]:
        common_idx = common_idx.intersection(frames[ticker].index)
    common_idx = common_idx.sort_values()
 
    print(f"Common date range: {common_idx[0].date()} → {common_idx[-1].date()} "
          f"({len(common_idx)} trading days)")
 
    return {ticker: frames[ticker].loc[common_idx] for ticker in STOCKS}

data = load_and_align("../Finance Data")
print(data['AAPL'])

# ─────────────────────────────────────────────
# 2. Compute targets
# ─────────────────────────────────────────────

def compute_targets(frames: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    """
    For each stock compute:
        return      : 1-day forward log return
        volatility  : rolling 20-day realised vol of log returns
        sharpe      : rolling 20-day Sharpe (ann. approx)
        direction   : 0=down, 1=flat, 2=up  (based on forward return)
 
    Market-level:
        regime      : 2-state label (0=bear, 1=bull) via sign of 60d average
                      log return across all 5 stocks
 
    All arrays are shape (T,) or (T, K) and aligned to the common date index.
    """

    T = len(frames[0]) # Length of Time Series
    K = len(STOCKS) # Number of Stocks