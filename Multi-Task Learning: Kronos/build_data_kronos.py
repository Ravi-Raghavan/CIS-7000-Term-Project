import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Optional

from kronos_tokenizer_wrapper import KronosEncoderWrapper

def get_device() -> torch.device:
    """CUDA (NVIDIA) -> MPS (Apple Silicon) -> CPU."""
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

INPUT_FEATURES = ["log_return", "hl_range", "oc_return", "log_vol_change", "momentum_5"]

SEQ_LEN      = 60       # lookback window
HORIZON      = 1        # predict 1 day ahead
VOL_WINDOW   = 20       # rolling window for volatility / Sharpe
DIR_THRESH   = 0.002    # ±0.2%
NUM_REGIMES  = 2        # Bear or Bull Market
BATCH_SIZE   = 32
SPLIT        = (0.8, 0.1, 0.1)


# ─────────────────────────────────────────────
# 1. Load & align CSVs
# ─────────────────────────────────────────────
def load_and_align(data_dir: str) -> dict[str, pd.DataFrame]:
    frames = {}
    for ticker in STOCKS:
        path = os.path.join(data_dir, f"Historical_Data_{ticker}.csv")
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
        df = df.sort_values("Date").set_index("Date")
        df = df[FEATURES].dropna()
        frames[ticker] = df
 
    common_idx = frames[STOCKS[0]].index
    for ticker in STOCKS[1:]:
        common_idx = common_idx.intersection(frames[ticker].index)
    common_idx = common_idx.sort_values()
 
    print(f"Common date range: {common_idx[0].date()} -> {common_idx[-1].date()} "
          f"({len(common_idx)} trading days)")
 
    return {ticker: frames[ticker].loc[common_idx] for ticker in STOCKS}

# ─────────────────────────────────────────────
# 2. Compute targets
# ─────────────────────────────────────────────

def compute_targets(frames: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    T = len(frames['AAPL'])
    K = len(STOCKS)

    ret    = np.zeros((T, K), dtype=np.float32)
    vol    = np.zeros((T, K), dtype=np.float32)
    sharpe = np.zeros((T, K), dtype=np.float32)
    direc  = np.zeros((T, K), dtype=np.int64)

    for i, ticker in enumerate(STOCKS):
        close   = frames[ticker]["Close"].values.astype(np.float64)
        log_ret = np.concatenate([[0.0], np.diff(np.log(close))])
 
        fwd = np.roll(log_ret, -HORIZON)
        fwd[-HORIZON:] = np.nan
 
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
 
        d = np.ones(T, dtype=np.int64)
        d[fwd >  DIR_THRESH] = 2
        d[fwd < -DIR_THRESH] = 0
        direc[:, i] = d
 
    avg_close   = np.stack([frames[t]["Close"].values for t in STOCKS], axis=1).mean(axis=1)
    mkt_ret     = np.concatenate([[0.0], np.diff(np.log(avg_close))])
 
    momentum_60 = np.full(T, np.nan)
    for t in range(60, T):
        momentum_60[t] = mkt_ret[t - 60 : t].mean()
 
    regime = np.zeros(T, dtype=np.int64)
    valid  = ~np.isnan(momentum_60)
    regime[valid & (momentum_60 >= 0)] = 1
    regime[valid & (momentum_60 <  0)] = 0
 
    return {
        "return":     ret,
        "volatility": vol,
        "sharpe":     sharpe,
        "direction":  direc,
        "regime":     regime,
    }

# ─────────────────────────────────────────────
# 3. Build raw data array + scale
# ─────────────────────────────────────────────

def build_arrays(frames: dict[str, pd.DataFrame]) -> tuple[np.ndarray, list[StandardScaler]]:
    K = len(STOCKS)
    T = len(frames['AAPL'])
    D = len(INPUT_FEATURES)
    data = np.zeros((K, T, D), dtype=np.float32)

    for i, ticker in enumerate(STOCKS):
        df     = frames[ticker]
        close  = df["Close"].values.astype(np.float64)
        high   = df["High"].values.astype(np.float64)
        low    = df["Low"].values.astype(np.float64)
        open_  = df["Open"].values.astype(np.float64)
        volume = df["Volume"].values.astype(np.float64)

        log_ret = np.concatenate([[0.0], np.diff(np.log(np.where(close > 0, close, 1.0)))])
        hl_range = (high - low) / np.where(close > 0, close, 1.0)
        oc_return = (close - open_) / np.where(open_ > 0, open_, 1.0)

        log_vol        = np.log(np.where(volume > 0, volume, 1.0))
        log_vol_change = np.concatenate([[0.0], np.diff(log_vol)])

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
# 4. Build Kronos raw input
# ─────────────────────────────────────────────

def build_kronos_raw_input(frames: dict[str, pd.DataFrame]) -> np.ndarray:
    """Returns (K, T, 6): log1p-scaled [open, high, low, close, volume, amount]"""
    K = len(STOCKS)
    T = len(frames['AAPL'])
    data = np.zeros((K, T, 6), dtype=np.float32)
    
    for i, ticker in enumerate(STOCKS):
        df = frames[ticker]
        open_ = df["Open"].values.astype(np.float32)
        high = df["High"].values.astype(np.float32)
        low = df["Low"].values.astype(np.float32)
        close = df["Close"].values.astype(np.float32)
        volume = df["Volume"].values.astype(np.float32)
        amount = open_ * volume
        
        stack = np.stack([open_, high, low, close, volume, amount], axis=1)
        data[i] = np.clip(np.log1p(np.maximum(stack, 0.0)), -5.0, 5.0)
        
    return data

# ─────────────────────────────────────────────
# 5. Chronological 80-10-10 split
# ─────────────────────────────────────────────
 
def split_indices(T: int, split: tuple = SPLIT) -> tuple[int, int]:
    train_end = int(T * split[0])
    val_end   = int(T * (split[0] + split[1]))
    return train_end, val_end

# ─────────────────────────────────────────────
# 6. PyTorch Dataset (Modified for Kronos)
# ─────────────────────────────────────────────
class StockDataset(Dataset):
    def __init__(
        self,
        s1_tokens: np.ndarray,      # (K, T) int64
        s2_tokens: np.ndarray,      # (K, T) int64
        targets: dict,              # from compute_targets()
        seq_len: int = SEQ_LEN,
        horizon: int = HORIZON,
    ):
        self.s1 = torch.tensor(s1_tokens, dtype=torch.long)
        self.s2 = torch.tensor(s2_tokens, dtype=torch.long)
        self.seq_len = seq_len
        self.horizon = horizon
 
        self.targets = {
            "return":     torch.tensor(targets["return"],     dtype=torch.float32),
            "volatility": torch.tensor(targets["volatility"], dtype=torch.float32),
            "sharpe":     torch.tensor(targets["sharpe"],     dtype=torch.float32),
            "direction":  torch.tensor(targets["direction"],  dtype=torch.long),
            "regime":     torch.tensor(targets["regime"],     dtype=torch.long),
        }
 
        self.valid = []
        T = self.s1.shape[1]

        for t in range(seq_len, T - horizon + 1):
            pred_t = t + horizon - 1

            if pred_t >= T:
                continue

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
        
        # token windows: (K, seq_len)
        s1_window = self.s1[:, t - self.seq_len : t]
        s2_window = self.s2[:, t - self.seq_len : t]
        
        targets = {k: v[pred_t] for k, v in self.targets.items()}
        return (s1_window, s2_window), targets

# ─────────────────────────────────────────────
# 7. Build Data-Loaders
# ─────────────────────────────────────────────
def build_dataloaders(
    data_dir: str,
    seq_len:    int   = SEQ_LEN,
    horizon:    int   = HORIZON,
    batch_size: int   = BATCH_SIZE,
    split:      tuple = SPLIT,
    lora_path:  str   = None) -> tuple[DataLoader, DataLoader, DataLoader, dict]:

    frames    = load_and_align(data_dir)
    date_idx  = frames['AAPL'].index
    T         = len(date_idx)

    targets = compute_targets(frames)
    
    # We still need scalers logic for stats compatibility if needed, though we don't return `data` to models.
    data, scalers = build_arrays(frames)

    train_end, val_end = split_indices(T, split)
    print(f"Split -> train: [{date_idx[0].date()} - {date_idx[train_end-1].date()}] "
          f"({train_end} days)  |  "
          f"val: [{date_idx[train_end].date()} - {date_idx[val_end-1].date()}]  |  "
          f"test: [{date_idx[val_end].date()} - {date_idx[-1].date()}]")

    data_scaled = data.copy()
    for i in range(len(STOCKS)):
        scalers[i].fit(data[i, :train_end, :])
        data_scaled[i, :, :] = scalers[i].transform(data[i, :, :])

    ret_std = np.nanstd(targets["return"][:train_end]) + 1e-8
    targets["return"] = targets["return"] / ret_std

    vol_std = np.nanstd(targets["volatility"][:train_end]) + 1e-8
    targets["volatility"] = targets["volatility"] / vol_std

    sharpe_mean = np.nanmean(targets["sharpe"][:train_end])
    sharpe_std  = np.nanstd(targets["sharpe"][:train_end]) + 1e-8
    targets["sharpe"] = np.clip(targets["sharpe"], -5.0, 5.0)
    targets["sharpe"] = (targets["sharpe"] - sharpe_mean) / sharpe_std

    target_stats = {
        "ret_std":     ret_std,
        "vol_std":     vol_std,
        "sharpe_mean": sharpe_mean,
        "sharpe_std":  sharpe_std,
    }

    print("Encoding features with pre-trained Kronos Tokenizer ...")
    wrapper = KronosEncoderWrapper(device, lora_path=lora_path)
    kronos_raw = build_kronos_raw_input(frames)
    s1_tokens, s2_tokens = wrapper.precompute_tokens(kronos_raw)
    print(f"Kronos encoding done: s1_tokens={s1_tokens.shape}, s2_tokens={s2_tokens.shape}")

    def slice_targets(start, end):
        return {k: v[start:end] for k, v in targets.items()}

    train_s1 = s1_tokens[:, :train_end]
    train_s2 = s2_tokens[:, :train_end]
    val_s1   = s1_tokens[:, train_end:val_end]
    val_s2   = s2_tokens[:, train_end:val_end]
    test_s1  = s1_tokens[:, val_end:]
    test_s2  = s2_tokens[:, val_end:]

    train_targets = slice_targets(0, train_end)
    val_targets   = slice_targets(train_end, val_end)
    test_targets  = slice_targets(val_end, T)

    train_ds = StockDataset(train_s1, train_s2, train_targets, seq_len, horizon)
    val_ds   = StockDataset(val_s1, val_s2, val_targets, seq_len, horizon)
    test_ds  = StockDataset(test_s1, test_s2, test_targets, seq_len, horizon)

    print(f"Dataset sizes -> train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    meta = {
        "scalers":      scalers,
        "target_stats": target_stats,
        "date_index":   date_idx,
        "split_dates": {
            "train": (date_idx[0],          date_idx[train_end - 1]),
            "val":   (date_idx[train_end],  date_idx[val_end - 1]),
            "test":  (date_idx[val_end],    date_idx[-1]),
        },
        "stocks":       STOCKS,
        "features":     INPUT_FEATURES,
        "input_dim":    320,  # Kronos output dim
        "seq_len":      seq_len,
        "horizon":      horizon,
        "T":            T,
        "train_end":    train_end,
        "val_end":      val_end,
        "wrapper":      wrapper,
    }

    return train_dl, val_dl, test_dl, meta
