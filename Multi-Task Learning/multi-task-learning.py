"""
SPA-MSJF: Multi-series Jointly Forecasting with Shared-private Attention
Adapted for 5-stock multi-task financial forecasting.

Targets per stock:
  - Return          (regression, MSE)
  - Volatility      (regression, MSE)
  - Sharpe ratio    (regression, Huber)
  - Direction       (classification, CrossEntropy, 3 classes: down/flat/up)

Shared target (market-level):
  - Regime          (classification, CrossEntropy, N classes)

Architecture:
  Input (OHLCV x T) → Private LSTM (hidden=32) ──┐
                                                   ├→ SPA → [return, vol, sharpe, direction] heads
  All privates → Shared LSTM (hidden=64) ──────────┘
                                     └──────────────→ Regime head (shared, f_s only)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional


# ─────────────────────────────────────────────
# 1. Shared-Private Attention (SPA)
# ─────────────────────────────────────────────

class SPA(nn.Module):
    """
    Shared-Private Attention (paper Eq. 12).
    Learns scalar weights (w_s, w_k) for shared and private features,
    inspired by CAPM's Beta (market) / Alpha (idiosyncratic) decomposition.

    Both f_s and f_k are projected to a common dimension (attn_dim),
    then combined via learned scalar weights:

        w_s, w_k = softmax(Linear([proj_s(f_s); proj_k(f_k)]))
        f̃_k = w_s * proj_s(f_s) + w_k * proj_k(f_k)

    This follows the paper's Eq. 12 (element-wise weighted sum, not concat).
    """

    def __init__(self, shared_dim: int, private_dim: int, attn_dim: int = 32):
        super().__init__()
        # Project both to the same dimension so weighted sum is valid
        self.proj_s = nn.Linear(shared_dim, attn_dim)
        self.proj_k = nn.Linear(private_dim, attn_dim)
        self.attn = nn.Linear(2 * attn_dim, 2)
        self.attn_dim = attn_dim

    def forward(self, f_s: torch.Tensor, f_k: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_s: shared features,  shape (B, shared_dim)
            f_k: private features, shape (B, private_dim)
        Returns:
            f̃_k: combined features, shape (B, attn_dim)
        """
        ps = F.relu(self.proj_s(f_s))                    # (B, attn_dim)
        pk = F.relu(self.proj_k(f_k))                    # (B, attn_dim)

        combined = torch.cat([ps, pk], dim=-1)            # (B, 2 * attn_dim)
        weights = F.softmax(self.attn(combined), dim=-1)  # (B, 2)
        w_s = weights[:, 0:1]                             # (B, 1)
        w_k = weights[:, 1:2]                             # (B, 1)

        # Paper Eq. 12: element-wise weighted sum (same dim required)
        f_combined = w_s * ps + w_k * pk                  # (B, attn_dim)
        return f_combined


# ─────────────────────────────────────────────
# 2. Private Encoder (one per stock)
# ─────────────────────────────────────────────

class PrivateEncoder(nn.Module):
    """
    Single-stock LSTM encoder.
    Input:  (B, T, input_dim)  — OHLCV features over T timesteps
    Output: (B, hidden_dim)    — last hidden state
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim)
        _, (h_n, _) = self.lstm(x)   # h_n: (1, B, hidden_dim)
        return self.dropout(h_n.squeeze(0))  # (B, hidden_dim)


# ─────────────────────────────────────────────
# 3. Shared Encoder (sees all stocks)
# ─────────────────────────────────────────────

class SharedEncoder(nn.Module):
    """
    Encodes the concatenation of all stocks' raw time series to capture
    cross-stock (market-level) temporal patterns (paper Eq. 5).

    Input:  (B, K, T, input_dim) — raw time series for all stocks
    Output: (B, hidden_dim)

    The shared LSTM sees all stocks' features concatenated along the
    feature dimension at each timestep, so it models temporal dynamics
    across the full market, not just a single-step projection.
    """

    def __init__(self, num_stocks: int, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        # Input at each timestep: all K stocks' features concatenated
        self.lstm = nn.LSTM(
            input_size=num_stocks * input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, K, T, input_dim)
        B, K, T, D = x.shape
        # Concatenate all stocks' features at each timestep: (B, T, K*D)
        x = x.permute(0, 2, 1, 3).reshape(B, T, K * D)
        _, (h_n, _) = self.lstm(x)                 # h_n: (1, B, hidden_dim)
        return self.dropout(h_n.squeeze(0))         # (B, hidden_dim)


# ─────────────────────────────────────────────
# 4. Per-stock Task Heads
# ─────────────────────────────────────────────

class StockTaskHeads(nn.Module):
    """
    Four prediction heads for one stock, operating on SPA-combined features.
      - Return:     scalar regression
      - Volatility: scalar regression
      - Sharpe:     scalar regression
      - Direction:  3-class classification (down / flat / up)
    """

    def __init__(self, combined_dim: int, num_direction_classes: int = 3):
        super().__init__()
        self.return_head    = nn.Linear(combined_dim, 1)
        self.vol_head       = nn.Linear(combined_dim, 1)
        self.sharpe_head    = nn.Linear(combined_dim, 1)
        self.direction_head = nn.Linear(combined_dim, num_direction_classes)

    def forward(self, f_combined: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "return":    self.return_head(f_combined).squeeze(-1),     # (B,)
            "volatility": self.vol_head(f_combined).squeeze(-1),       # (B,)
            "sharpe":    self.sharpe_head(f_combined).squeeze(-1),     # (B,)
            "direction": self.direction_head(f_combined),              # (B, 3)
        }


# ─────────────────────────────────────────────
# 5. Regime Head (shared, reads f_s only)
# ─────────────────────────────────────────────

class RegimeHead(nn.Module):
    """
    Market regime classification using only the shared encoder output.
    Regime is a market-wide state — deliberately isolated from stock-specific info.
    """

    def __init__(self, shared_dim: int, num_regimes: int = 4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(shared_dim, shared_dim // 2),
            nn.ReLU(),
            nn.Linear(shared_dim // 2, num_regimes),
        )

    def forward(self, f_s: torch.Tensor) -> torch.Tensor:
        return self.head(f_s)   # (B, num_regimes)


# ─────────────────────────────────────────────
# 6. Full SPA-MSJF Model
# ─────────────────────────────────────────────

class SPAMSJF(nn.Module):
    """
    Full SPA-MSJF model for 5-stock multi-task financial forecasting.

    Args:
        num_stocks:            Number of stocks (K=5)
        input_dim:             Number of input features per timestep (e.g. 5 for OHLCV)
        private_hidden:        Hidden size for private LSTMs (default 32)
        shared_hidden:         Hidden size for shared LSTM (default 64)
        num_direction_classes: Classes for direction head (default 3: down/flat/up)
        num_regimes:           Classes for regime head (default 4)
        dropout:               Dropout rate (default 0.2)
    """

    def __init__(
        self,
        num_stocks: int = 5,
        input_dim: int = 5,
        private_hidden: int = 32,
        shared_hidden: int = 64,
        spa_dim: int = 32,
        num_direction_classes: int = 3,
        num_regimes: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_stocks = num_stocks

        # One private encoder per stock
        self.private_encoders = nn.ModuleList([
            PrivateEncoder(input_dim, private_hidden, dropout)
            for _ in range(num_stocks)
        ])

        # Single shared encoder across all stocks (sees raw time series)
        self.shared_encoder = SharedEncoder(num_stocks, input_dim, shared_hidden, dropout)

        # One SPA module per stock (projects to common spa_dim, then weighted sum)
        self.spa_modules = nn.ModuleList([
            SPA(shared_hidden, private_hidden, attn_dim=spa_dim)
            for _ in range(num_stocks)
        ])

        # Per-stock task heads operate on spa_dim (the SPA output dimension)
        self.task_heads = nn.ModuleList([
            StockTaskHeads(spa_dim, num_direction_classes)
            for _ in range(num_stocks)
        ])

        # Shared regime head (uses f_s only)
        self.regime_head = RegimeHead(shared_hidden, num_regimes)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: shape (B, K, T, input_dim)
               B = batch size, K = num_stocks, T = seq len, input_dim = features

        Returns:
            dict with keys:
              "stocks": list of K dicts, each with keys:
                  "return", "volatility", "sharpe" → (B,)
                  "direction"                       → (B, num_direction_classes)
              "regime": (B, num_regimes)
        """
        assert x.shape[1] == self.num_stocks, \
            f"Expected {self.num_stocks} stocks, got {x.shape[1]}"

        # 1. Private encoding: each stock independently
        private_feats = [
            self.private_encoders[k](x[:, k, :, :])   # (B, private_hidden)
            for k in range(self.num_stocks)
        ]

        # 2. Shared encoding: sees raw time series from ALL stocks (paper Eq. 5)
        f_s = self.shared_encoder(x)                   # (B, shared_hidden)

        # 3. SPA + task heads per stock
        stock_outputs = []
        for k in range(self.num_stocks):
            f_combined = self.spa_modules[k](f_s, private_feats[k])   # (B, combined_dim)
            preds = self.task_heads[k](f_combined)
            stock_outputs.append(preds)

        # 4. Regime (shared encoder only)
        regime_logits = self.regime_head(f_s)          # (B, num_regimes)

        return {
            "stocks": stock_outputs,
            "regime": regime_logits,
        }


# ─────────────────────────────────────────────
# 7. Joint Loss
# ─────────────────────────────────────────────

class JointLoss(nn.Module):
    """
    Weighted combination of per-task losses across all stocks.

    Loss = λ1·MSE(return) + λ2·MSE(vol) + λ3·Huber(sharpe)
         + λ4·CE(direction) + λ5·CE(regime)

    All per-stock losses are averaged across stocks.
    """

    def __init__(
        self,
        lambda_return: float = 1.0,
        lambda_vol: float = 0.5,
        lambda_sharpe: float = 0.3,
        lambda_direction: float = 1.0,
        lambda_regime: float = 0.5,
        huber_delta: float = 1.0,
    ):
        super().__init__()
        self.lambdas = {
            "return":    lambda_return,
            "vol":       lambda_vol,
            "sharpe":    lambda_sharpe,
            "direction": lambda_direction,
            "regime":    lambda_regime,
        }
        self.mse    = nn.MSELoss()
        self.huber  = nn.HuberLoss(delta=huber_delta)
        self.ce     = nn.CrossEntropyLoss()

    def forward(self, outputs: dict, targets: dict) -> tuple[torch.Tensor, dict]:
        """
        Args:
            outputs: model forward() output dict
            targets: dict with keys:
                "return":    (B, K)  float
                "volatility":(B, K)  float
                "sharpe":    (B, K)  float
                "direction": (B, K)  long  (class indices)
                "regime":    (B,)    long  (class indices)

        Returns:
            total_loss: scalar tensor
            loss_dict:  per-component losses for logging
        """
        K = len(outputs["stocks"])
        loss_return = loss_vol = loss_sharpe = loss_direction = 0.0

        for k in range(K):
            preds = outputs["stocks"][k]
            loss_return    += self.mse(preds["return"],     targets["return"][:, k])
            loss_vol       += self.mse(preds["volatility"], targets["volatility"][:, k])
            loss_sharpe    += self.huber(preds["sharpe"],   targets["sharpe"][:, k])
            loss_direction += self.ce(preds["direction"],   targets["direction"][:, k])

        # Average over stocks
        loss_return    /= K
        loss_vol       /= K
        loss_sharpe    /= K
        loss_direction /= K

        loss_regime = self.ce(outputs["regime"], targets["regime"])

        total = (
            self.lambdas["return"]    * loss_return    +
            self.lambdas["vol"]       * loss_vol       +
            self.lambdas["sharpe"]    * loss_sharpe    +
            self.lambdas["direction"] * loss_direction +
            self.lambdas["regime"]    * loss_regime
        )

        loss_dict = {
            "total":     total.item(),
            "return":    loss_return.item() if isinstance(loss_return, torch.Tensor) else loss_return,
            "vol":       loss_vol.item() if isinstance(loss_vol, torch.Tensor) else loss_vol,
            "sharpe":    loss_sharpe.item() if isinstance(loss_sharpe, torch.Tensor) else loss_sharpe,
            "direction": loss_direction.item() if isinstance(loss_direction, torch.Tensor) else loss_direction,
            "regime":    loss_regime.item(),
        }
        return total, loss_dict


# ─────────────────────────────────────────────
# 8. Dataset
# ─────────────────────────────────────────────

class StockDataset(Dataset):
    """
    Sliding-window dataset for multi-stock, multi-task forecasting.

    Args:
        data:        np.ndarray, shape (T_total, K, input_dim)
                     T_total = total time steps, K = num stocks
        targets:     dict of np.ndarrays:
                       "return":     (T_total, K)
                       "volatility": (T_total, K)
                       "sharpe":     (T_total, K)
                       "direction":  (T_total, K)  int
                       "regime":     (T_total,)     int
        seq_len:     number of lookback timesteps (default 30)
        horizon:     how many steps ahead to predict (default 1)
    """

    def __init__(
        self,
        data: np.ndarray,
        targets: dict,
        seq_len: int = 30,
        horizon: int = 1,
    ):
        self.data     = torch.tensor(data, dtype=torch.float32)
        self.seq_len  = seq_len
        self.horizon  = horizon

        self.targets = {
            "return":     torch.tensor(targets["return"],     dtype=torch.float32),
            "volatility": torch.tensor(targets["volatility"], dtype=torch.float32),
            "sharpe":     torch.tensor(targets["sharpe"],     dtype=torch.float32),
            "direction":  torch.tensor(targets["direction"],  dtype=torch.long),
            "regime":     torch.tensor(targets["regime"],     dtype=torch.long),
        }

        self.valid_idx = range(seq_len, len(data) - horizon + 1)

    def __len__(self) -> int:
        return len(self.valid_idx)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        t = self.valid_idx[idx]
        # x: (K, seq_len, input_dim)
        x = self.data[t - self.seq_len : t].permute(1, 0, 2)
        # targets at time t + horizon - 1
        t_idx = t + self.horizon - 1
        target = {k: v[t_idx] for k, v in self.targets.items()}
        return x, target


# ─────────────────────────────────────────────
# 9. Trainer (sliding window)
# ─────────────────────────────────────────────

class SlidingWindowTrainer:
    """
    Implements the sliding-window training protocol from the paper:
      - Train on 6 months of data
      - Test on the following 1 month
      - Slide forward by 1 month and repeat

    Args:
        model:       SPAMSJF instance
        criterion:   JointLoss instance
        device:      torch.device
        lr:          learning rate (default 1e-3)
        epochs:      training epochs per window (default 50)
        batch_size:  (default 32)
        patience:    early stopping patience (default 10)
    """

    def __init__(
        self,
        model: SPAMSJF,
        criterion: JointLoss,
        device: torch.device,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
        patience: int = 10,
    ):
        self.model      = model.to(device)
        self.criterion  = criterion
        self.device     = device
        self.epochs     = epochs
        self.batch_size = batch_size
        self.patience   = patience
        self.optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5,
        )

    def _run_epoch(self, loader: DataLoader, train: bool) -> dict:
        self.model.train(train)
        total_losses = {}
        with torch.set_grad_enabled(train):
            for x, targets in loader:
                x = x.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                outputs = self.model(x)
                loss, loss_dict = self.criterion(outputs, targets)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                for k, v in loss_dict.items():
                    total_losses[k] = total_losses.get(k, 0.0) + v
        return {k: v / len(loader) for k, v in total_losses.items()}

    def train_window(
        self,
        train_dataset: StockDataset,
        val_dataset: Optional[StockDataset] = None,
    ) -> list[dict]:
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=self.batch_size) if val_dataset else None

        # Re-init optimizer for each window to avoid stale momentum
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.optimizer.param_groups[0]["lr"])

        history = []
        best_val_loss = float("inf")
        patience_count = 0
        best_state = None

        for epoch in range(1, self.epochs + 1):
            train_losses = self._run_epoch(train_loader, train=True)
            log = {"epoch": epoch, **{f"train_{k}": v for k, v in train_losses.items()}}

            if val_loader:
                val_losses = self._run_epoch(val_loader, train=False)
                log.update({f"val_{k}": v for k, v in val_losses.items()})
                self.scheduler.step(val_losses["total"])

                if val_losses["total"] < best_val_loss:
                    best_val_loss = val_losses["total"]
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= self.patience:
                        print(f"  Early stopping at epoch {epoch}")
                        break

            history.append(log)
            if epoch % 10 == 0:
                print(f"  Epoch {epoch:3d} | train_total: {train_losses['total']:.4f}"
                      + (f" | val_total: {val_losses['total']:.4f}" if val_loader else ""))

        if best_state:
            self.model.load_state_dict(best_state)

        return history

    @torch.no_grad()
    def evaluate(self, test_dataset: StockDataset) -> dict:
        """
        Returns per-stock, per-task predictions and targets for the test window.

        Output structure:
            {
                "stocks": [  # list of K dicts
                    {"return": {"preds": Tensor, "targets": Tensor},
                     "volatility": ..., "sharpe": ..., "direction": ...},
                    ...
                ],
                "regime": {"preds": Tensor, "targets": Tensor},
            }
        """
        loader = DataLoader(test_dataset, batch_size=self.batch_size)
        K = self.model.num_stocks
        reg_tasks = ["return", "volatility", "sharpe"]

        # Per-stock accumulators
        stock_preds = [[[] for _ in reg_tasks + ["direction"]] for _ in range(K)]
        stock_targets = [[[] for _ in reg_tasks + ["direction"]] for _ in range(K)]
        regime_preds, regime_targets = [], []

        self.model.eval()
        for x, targets in loader:
            x = x.to(self.device)
            outputs = self.model(x)

            for k in range(K):
                preds_k = outputs["stocks"][k]
                for ti, task in enumerate(reg_tasks):
                    stock_preds[k][ti].append(preds_k[task].cpu())
                    stock_targets[k][ti].append(targets[task][:, k].cpu())
                # Direction
                stock_preds[k][3].append(preds_k["direction"].argmax(-1).cpu())
                stock_targets[k][3].append(targets["direction"][:, k].cpu())

            regime_preds.append(outputs["regime"].argmax(-1).cpu())
            regime_targets.append(targets["regime"].cpu())

        # Assemble results
        task_names = reg_tasks + ["direction"]
        stock_results = []
        for k in range(K):
            d = {}
            for ti, task in enumerate(task_names):
                d[task] = {
                    "preds":   torch.cat(stock_preds[k][ti]),
                    "targets": torch.cat(stock_targets[k][ti]),
                }
            stock_results.append(d)

        return {
            "stocks": stock_results,
            "regime": {
                "preds":   torch.cat(regime_preds),
                "targets": torch.cat(regime_targets),
            },
        }


# ─────────────────────────────────────────────
# 10. Example usage
# ─────────────────────────────────────────────

def make_dummy_data(T=500, K=5, input_dim=5, seed=42):
    """
    Generate synthetic OHLCV-like data with planted signals for smoke-testing.

    Planted structure:
      - Each stock follows a random walk with drift (realistic price dynamics)
      - Returns, volatility, Sharpe, and direction targets are computed FROM the
        generated prices, so they have real statistical relationships to the inputs
      - A shared "market factor" drives correlated moves across stocks
      - Regime is a slow-switching state (bull=0, bear=1, volatile=2, recovery=3)
        that affects drift and volatility, giving the model something learnable

    This ensures the model can learn real signals rather than fitting pure noise.
    """
    rng = np.random.default_rng(seed)

    # --- Regime: slow-switching market state ---
    regime = np.zeros(T, dtype=np.int64)
    current_regime = 0
    for t in range(1, T):
        if rng.random() < 0.02:  # 2% chance of regime switch per day
            current_regime = rng.integers(0, 4)
        regime[t] = current_regime

    # Regime-dependent drift and vol
    regime_drift = {0: 0.001, 1: -0.002, 2: 0.0, 3: 0.0015}
    regime_vol   = {0: 0.01,  1: 0.025,  2: 0.04, 3: 0.015}

    # --- Generate per-stock price series ---
    prices = np.zeros((T, K))
    prices[0] = rng.uniform(50, 300, K)  # Random starting prices

    # Shared market factor (drives cross-stock correlation)
    market_noise = rng.normal(0, 0.01, T)

    for t in range(1, T):
        drift = regime_drift[regime[t]]
        vol = regime_vol[regime[t]]
        # Each stock: drift + market factor + idiosyncratic noise
        stock_noise = rng.normal(0, vol, K)
        betas = np.array([1.0, 1.2, 0.8, 1.5, 0.6])[:K]  # stock sensitivities
        daily_return = drift + betas * market_noise[t] + stock_noise
        prices[t] = prices[t-1] * (1 + daily_return)

    # --- Build OHLCV features from prices ---
    data = np.zeros((T, K, input_dim), dtype=np.float32)
    for k in range(K):
        close = prices[:, k]
        noise_h = np.abs(rng.normal(0, 0.005, T)) * close
        noise_l = np.abs(rng.normal(0, 0.005, T)) * close
        open_p  = close * (1 + rng.normal(0, 0.002, T))
        high    = np.maximum(close, open_p) + noise_h
        low     = np.minimum(close, open_p) - noise_l
        volume  = rng.lognormal(mean=15, sigma=0.5, size=T)  # log-normal volume

        data[:, k, 0] = open_p
        data[:, k, 1] = high
        data[:, k, 2] = low
        data[:, k, 3] = close
        data[:, k, 4] = volume / 1e6  # scale volume down

    # --- Compute targets from generated prices ---
    returns = np.zeros((T, K), dtype=np.float32)
    returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]

    # Rolling volatility (20-day window)
    volatility = np.zeros((T, K), dtype=np.float32)
    for t in range(20, T):
        volatility[t] = np.std(returns[t-20:t], axis=0)

    # Rolling Sharpe ratio (60-day window, annualized)
    sharpe = np.zeros((T, K), dtype=np.float32)
    for t in range(60, T):
        window = returns[t-60:t]
        mu = np.mean(window, axis=0)
        sigma = np.std(window, axis=0) + 1e-8
        sharpe[t] = np.sqrt(252) * mu / sigma

    # Direction: 0=down (<-0.5%), 1=flat, 2=up (>+0.5%)
    direction = np.ones((T, K), dtype=np.int64)  # default flat
    direction[returns < -0.005] = 0
    direction[returns > 0.005] = 2

    targets = {
        "return":     returns,
        "volatility": volatility,
        "sharpe":     sharpe,
        "direction":  direction,
        "regime":     regime,
    }
    return data, targets


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    K, INPUT_DIM, SEQ_LEN = 5, 5, 30

    # ── Build model ──────────────────────────────────────
    model = SPAMSJF(
        num_stocks=K,
        input_dim=INPUT_DIM,
        private_hidden=32,
        shared_hidden=64,
        spa_dim=32,
        num_direction_classes=3,
        num_regimes=4,
        dropout=0.2,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # ── Build loss ───────────────────────────────────────
    criterion = JointLoss(
        lambda_return=1.0,
        lambda_vol=0.5,
        lambda_sharpe=0.3,
        lambda_direction=1.0,
        lambda_regime=0.5,
    )

    # ── Synthetic data with planted signals ──────────────
    data, targets = make_dummy_data(T=500, K=K, input_dim=INPUT_DIM)

    train_end, val_end = 350, 400
    train_ds = StockDataset(data[:train_end], {k: v[:train_end] for k, v in targets.items()}, SEQ_LEN)
    val_ds   = StockDataset(data[train_end:val_end], {k: v[train_end:val_end] for k, v in targets.items()}, SEQ_LEN)
    test_ds  = StockDataset(data[val_end:], {k: v[val_end:] for k, v in targets.items()}, SEQ_LEN)

    print(f"Dataset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # ── Train ────────────────────────────────────────────
    trainer = SlidingWindowTrainer(
        model=model,
        criterion=criterion,
        device=device,
        lr=1e-3,
        epochs=30,
        batch_size=32,
        patience=10,
    )

    print("\nTraining window 1...")
    history = trainer.train_window(train_ds, val_ds)

    # ── Evaluate ─────────────────────────────────────────
    print("\nEvaluating on test set...")
    results = trainer.evaluate(test_ds)

    # Print per-stock, per-task shapes
    print("\nResults structure:")
    for k, stock_res in enumerate(results["stocks"]):
        for task, vals in stock_res.items():
            print(f"  Stock {k} | {task:12s} | preds: {tuple(vals['preds'].shape)}, targets: {tuple(vals['targets'].shape)}")
    regime = results["regime"]
    print(f"  Regime       | preds: {tuple(regime['preds'].shape)}, targets: {tuple(regime['targets'].shape)}")

    # Quick metrics
    from sklearn.metrics import mean_squared_error, accuracy_score
    print("\n── Quick Metrics ──")
    for k, stock_res in enumerate(results["stocks"]):
        ret_mse = mean_squared_error(stock_res["return"]["targets"], stock_res["return"]["preds"])
        vol_mse = mean_squared_error(stock_res["volatility"]["targets"], stock_res["volatility"]["preds"])
        dir_acc = accuracy_score(stock_res["direction"]["targets"], stock_res["direction"]["preds"])
        print(f"  Stock {k} | return_MSE={ret_mse:.6f}  vol_MSE={vol_mse:.6f}  dir_acc={dir_acc:.3f}")

    regime_acc = accuracy_score(regime["targets"], regime["preds"])
    print(f"  Regime accuracy: {regime_acc:.3f}")
    print("\nDone. Model is ready for real data.")