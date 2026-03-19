# Train MSJF Model 
from spa_msjf import SPAMSJF
import torch
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix
from typing import Optional

def get_device() -> torch.device:
    """CUDA (NVIDIA) → MPS (Apple Silicon) → CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
STOCKS       = ["AAPL", "GOOG", "META", "NVDA", "TSLA"]
FEATURES     = ["Open", "High", "Low", "Close", "Volume"]  # raw CSV columns

# Engineered stationary input features (replaces raw OHLCV):
#   0. log_return     = log(Close_t / Close_{t-1})   — daily return
#   1. hl_range       = (High - Low) / Close          — normalised intraday range
#   2. oc_return      = (Close - Open) / Open          — open-to-close move
#   3. log_vol_change = log(Volume_t / Volume_{t-1})  — volume momentum
#   4. momentum_5     = log(Close_t / Close_{t-5})   — 5-day price momentum
# Raw OHLCV drifts over years (AAPL went $15→$180); these features don't.
INPUT_FEATURES = ["log_return", "hl_range", "oc_return", "log_vol_change", "momentum_5"]

SEQ_LEN      = 60       # lookback window — 60 days gives more temporal context
HORIZON      = 1        # predict 1 day ahead
VOL_WINDOW   = 20       # rolling window for volatility / Sharpe
DIR_THRESH   = 0.002    # ±0.2% → flat band for direction labels
NUM_REGIMES  = 2        # Bear or Bull Market
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
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
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

# ─────────────────────────────────────────────
# 3. Build raw data array + scale
# ─────────────────────────────────────────────

def build_arrays(frames: dict[str, pd.DataFrame]) -> tuple[np.ndarray, list[StandardScaler]]:
    """
    Compute 5 stationary engineered features per stock instead of raw OHLCV.
    Raw prices drift over years which causes train/val distribution shift;
    log-return-based features are stationary across the full 2012-2026 range.

    Returns:
        data    : (K, T, 5) float32 — unscaled engineered features
        scalers : list of K StandardScalers (fitted by caller on train only)
    """
    K = len(STOCKS)
    T = len(frames['AAPL'])
    D = len(INPUT_FEATURES)  # 5 engineered features
    data = np.zeros((K, T, D), dtype=np.float32)

    for i, ticker in enumerate(STOCKS):
        df     = frames[ticker]
        close  = df["Close"].values.astype(np.float64)
        high   = df["High"].values.astype(np.float64)
        low    = df["Low"].values.astype(np.float64)
        open_  = df["Open"].values.astype(np.float64)
        volume = df["Volume"].values.astype(np.float64)

        # 0. Daily log return — primary price signal, stationary
        log_ret = np.concatenate([[0.0], np.diff(np.log(np.where(close > 0, close, 1.0)))])

        # 1. Normalised intraday H-L range — measures daily volatility regime
        hl_range = (high - low) / np.where(close > 0, close, 1.0)

        # 2. Open-to-close return — captures intraday directional momentum
        oc_return = (close - open_) / np.where(open_ > 0, open_, 1.0)

        # 3. Log volume change — volume spikes signal institutional activity
        log_vol        = np.log(np.where(volume > 0, volume, 1.0))
        log_vol_change = np.concatenate([[0.0], np.diff(log_vol)])

        # 4. 5-day price momentum — short-term trend signal
        momentum_5       = np.zeros(T)
        momentum_5[5:]   = np.log(close[5:] / np.where(close[:-5] > 0, close[:-5], 1.0))

        data[i, :, 0] = log_ret.astype(np.float32)
        data[i, :, 1] = hl_range.astype(np.float32)
        data[i, :, 2] = oc_return.astype(np.float32)
        data[i, :, 3] = log_vol_change.astype(np.float32)
        data[i, :, 4] = momentum_5.astype(np.float32)

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

# ─────────────────────────────────────────────
# 6. Build Data-Loaders
# ─────────────────────────────────────────────
def build_dataloaders(
    data_dir: str,
    seq_len:    int   = SEQ_LEN,
    horizon:    int   = HORIZON,
    batch_size: int   = BATCH_SIZE,
    split:      tuple = SPLIT) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Full pipeline: load → align → compute targets → scale → split → Dataset → DataLoader.
 
    Returns:
        train_dl, val_dl, test_dl  — DataLoaders
        meta                       — dict with scalers, date_index, split_dates, stocks
    """

    # ── Load ──
    frames    = load_and_align(data_dir)
    date_idx  = frames['AAPL'].index
    T         = len(date_idx)

    # ── Targets (computed on unscaled closes) ──
    targets = compute_targets(frames)

    # ── Raw feature array ──
    data, scalers = build_arrays(frames)

    # ── Split indices ──
    train_end, val_end = split_indices(T, split)
    print(f"Split → train: [{date_idx[0].date()} – {date_idx[train_end-1].date()}] "
          f"({train_end} days)  |  "
          f"val: [{date_idx[train_end].date()} – {date_idx[val_end-1].date()}]  |  "
          f"test: [{date_idx[val_end].date()} – {date_idx[-1].date()}]")

    # ── Fit scalers on TRAIN only, transform all splits ──
    # Even though features are already stationary, StandardScaler removes
    # any remaining mean/variance differences across features.
    data_scaled = data.copy()
    for i in range(len(STOCKS)):
        scalers[i].fit(data[i, :train_end, :])
        data_scaled[i, :, :] = scalers[i].transform(data[i, :, :])

    # ── Normalise regression targets using TRAIN statistics only ──
    # Without this, Sharpe MSE runs 10-18 because the model predicts near
    # zero while true Sharpe spans roughly -3 to +3. Z-scoring brings all
    # regression targets to the same unit scale, stabilising joint loss.

    # Return: scale to unit variance (already near zero mean)
    ret_std = np.nanstd(targets["return"][:train_end]) + 1e-8
    targets["return"] = targets["return"] / ret_std

    # Volatility: strictly positive, scale to unit variance
    vol_std = np.nanstd(targets["volatility"][:train_end]) + 1e-8
    targets["volatility"] = targets["volatility"] / vol_std

    # Sharpe: clip outliers first then z-score
    sharpe_mean = np.nanmean(targets["sharpe"][:train_end])
    sharpe_std  = np.nanstd(targets["sharpe"][:train_end]) + 1e-8
    targets["sharpe"] = np.clip(targets["sharpe"], -5.0, 5.0)
    targets["sharpe"] = (targets["sharpe"] - sharpe_mean) / sharpe_std

    # Store normalisation stats in meta so predictions can be de-normalised later
    target_stats = {
        "ret_std":     ret_std,
        "vol_std":     vol_std,
        "sharpe_mean": sharpe_mean,
        "sharpe_std":  sharpe_std,
    }

    # ── Slice targets per split ──
    def slice_targets(start, end):
        return {k: v[start:end] for k, v in targets.items()}

    # Chronological Split of Data and targets
    train_data = data_scaled[:, :train_end, :]
    val_data   = data_scaled[:, train_end:val_end, :]
    test_data  = data_scaled[:, val_end:, :]

    train_targets = slice_targets(0, train_end)
    val_targets   = slice_targets(train_end, val_end)
    test_targets  = slice_targets(val_end, T)

    # ── Datasets ──
    train_ds = StockDataset(train_data, train_targets, seq_len, horizon)
    val_ds   = StockDataset(val_data,   val_targets,   seq_len, horizon)
    test_ds  = StockDataset(test_data,  test_targets,  seq_len, horizon)

    print(f"Dataset sizes → train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    # ── DataLoaders ──
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    meta = {
        "scalers":      scalers,
        "target_stats": target_stats,   # use these to de-normalise predictions
        "date_index":   date_idx,
        "split_dates": {
            "train": (date_idx[0],          date_idx[train_end - 1]),
            "val":   (date_idx[train_end],  date_idx[val_end - 1]),
            "test":  (date_idx[val_end],    date_idx[-1]),
        },
        "stocks":       STOCKS,
        "features":     INPUT_FEATURES,
        "seq_len":      seq_len,
        "horizon":      horizon,
        "T":            T,
        "train_end":    train_end,
        "val_end":      val_end,
    }
 
    return train_dl, val_dl, test_dl, meta

