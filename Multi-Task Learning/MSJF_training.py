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

# data = load_and_align("../Finance Data")
# print(data['AAPL'])

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
        regime      : 2-state label (0=bear, 1=bull) via sign of 60d average log return across all 5 stocks
 
    All arrays are shape (T,) or (T, K) and aligned to the common date index.
    """

    T = len(frames['AAPL']) # Length of Time Series
    K = len(STOCKS) # Number of Stocks

    # Define target values
    ret    = np.zeros((T, K), dtype=np.float32)
    vol    = np.zeros((T, K), dtype=np.float32)
    sharpe = np.zeros((T, K), dtype=np.float32)
    direc  = np.zeros((T, K), dtype=np.int64)

    # Iterate through Stocks
    for i, ticker in enumerate(STOCKS):
        close   = frames[ticker]["Close"].values.astype(np.float64) # Fetch Closing Values
        log_ret = np.concatenate([[0.0], np.diff(np.log(close))])  # Log Return Arrays: (T,)
 
        # 1-day forward log return (target at t = log_ret at t+1)
        fwd = np.roll(log_ret, -HORIZON)
        fwd[-HORIZON:] = np.nan
 
        # Rolling volatility & Sharpe over past VOL_WINDOW days
        roll_vol    = np.full(T, np.nan)
        roll_sharpe = np.full(T, np.nan)
        for t in range(VOL_WINDOW, T):
            window        = log_ret[t - VOL_WINDOW : t]
            rv            = window.std()
            roll_vol[t]   = rv
            roll_sharpe[t] = (window.mean() / rv * np.sqrt(252)) if rv > 1e-8 else 0.0
 
        ret[:, i]    = fwd.astype(np.float32)
        vol[:, i]    = roll_vol.astype(np.float32)
        sharpe[:, i] = roll_sharpe.astype(np.float32)
 
        # Direction: based on forward return
        d = np.ones(T, dtype=np.int64)          # flat
        d[fwd >  DIR_THRESH] = 2                # up
        d[fwd < -DIR_THRESH] = 0                # down
        direc[:, i] = d
 
    # ── Regime: 2-state bull / bear via 60-day momentum sign ──
    avg_close   = np.stack([frames[t]["Close"].values for t in STOCKS], axis=1).mean(axis=1) # Averaging Closing Price across stocks throughout time
    mkt_ret     = np.concatenate([[0.0], np.diff(np.log(avg_close))])
 
    momentum_60 = np.full(T, np.nan)
    for t in range(60, T):
        momentum_60[t] = mkt_ret[t - 60 : t].mean()
 
    # 0 = bear (negative 60d momentum), 1 = bull (positive 60d momentum)
    regime = np.zeros(T, dtype=np.int64)
    valid  = ~np.isnan(momentum_60)
    regime[valid & (momentum_60 >= 0)] = 1   # bull
    regime[valid & (momentum_60 <  0)] = 0   # bear
 
    return {
        "return":     ret,       # (T, K) float32
        "volatility": vol,       # (T, K) float32
        "sharpe":     sharpe,    # (T, K) float32
        "direction":  direc,     # (T, K) int64
        "regime":     regime,    # (T,)   int64
    }

# targets = compute_targets(data)

# ─────────────────────────────────────────────
# 3. Build raw data array + scale
# ─────────────────────────────────────────────

def build_arrays(frames: dict[str, pd.DataFrame]) -> tuple[np.ndarray, list[StandardScaler]]:
    """
    Stack OHLCV into (K, T, D) and fit one StandardScaler per stock
    on the training portion only (caller must slice before fitting).
    Returns unscaled (K, T, D) array and the list of scalers.
    """
    K = len(STOCKS) # Number of stocks
    T = len(frames['AAPL']) # Length of Time Series
    D = len(FEATURES) # Number of features
    data = np.zeros((K, T, D), dtype=np.float32) # define raw data

    # Populate Data
    for i, ticker in enumerate(STOCKS):
        data[i, :, :] = frames[ticker][FEATURES].values.astype(np.float32)

    # Construct Scalers
    scalers = [StandardScaler() for _ in range(K)]
    return data, scalers

# ─────────────────────────────────────────────
# 4. Chronological 80-10-10 split
# ─────────────────────────────────────────────
 
def split_indices(T: int, split: tuple = SPLIT) -> tuple[int, int]:
    """Returns (train_end, val_end) indices."""
    train_end = int(T * split[0])
    val_end   = int(T * (split[0] + split[1]))
    return train_end, val_end

# ─────────────────────────────────────────────
# 5. PyTorch Dataset
# ─────────────────────────────────────────────
class StockDataset(Dataset):
    """
    Sliding-window dataset. Each sample is:
        x       : (K, SEQ_LEN, D)  — input features, scaled
        targets : dict of tensors  — see compute_targets()
 
    NaN rows (start of vol/sharpe series) are excluded automatically.
    """
 
    def __init__(
        self,
        data: np.ndarray,           # (K, T, D) scaled
        targets: dict,              # from compute_targets()
        seq_len: int = SEQ_LEN,
        horizon: int = HORIZON,
    ):
        self.data    = torch.tensor(data,    dtype=torch.float32)
        self.seq_len = seq_len
        self.horizon = horizon
 
        self.targets = {
            "return":     torch.tensor(targets["return"],     dtype=torch.float32),
            "volatility": torch.tensor(targets["volatility"], dtype=torch.float32),
            "sharpe":     torch.tensor(targets["sharpe"],     dtype=torch.float32),
            "direction":  torch.tensor(targets["direction"],  dtype=torch.long),
            "regime":     torch.tensor(targets["regime"],     dtype=torch.long),
        }
 
        # Build valid indices: need seq_len lookback + horizon ahead,
        # and no NaN in any target at the prediction timestep
        self.valid = []
        T = data.shape[1]

        for t in range(seq_len, T - horizon + 1):
            pred_t = t + horizon - 1

            if pred_t >= T:
                continue

            # Skip if any regression target is NaN at this timestep
            if (np.isnan(targets["return"][pred_t]).any()  or
                np.isnan(targets["volatility"][pred_t]).any() or
                np.isnan(targets["sharpe"][pred_t]).any()):
                continue
            
            self.valid.append(t)
 
    def __len__(self) -> int:
        return len(self.valid)
 
    def __getitem__(self, idx: int):
        t      = self.valid[idx]
        pred_t = t + self.horizon - 1
        
        # x: (K, seq_len, D)
        x = self.data[:, t - self.seq_len : t, :]
        targets = {k: v[pred_t] for k, v in self.targets.items()}
        return x, targets

