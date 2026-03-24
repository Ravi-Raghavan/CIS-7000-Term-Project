"""
SPA-MSJF: Multi-series Jointly Forecasting with Shared-private Attention
Adapted for 5-stock multi-task financial forecasting. Refer to architecture.png for further details

Targets per stock:
  - Return          (regression, MSE)
  - Volatility      (regression, MSE)
  - Sharpe ratio    (regression, Huber)
  - Direction       (classification, CrossEntropy, 3 classes: down/flat/up)

Shared target (market-level):
  - Regime          (classification, CrossEntropy, N classes)

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
# 2. Private Decoder (one per stock)
# ─────────────────────────────────────────────

class PrivateDecoder(nn.Module):
    """
    Single-Stock Decoder.
    Input:  (B, T, input_dim)  — OHLCV features over T timesteps
    Output: (B, private_dim)    — last hidden state
    """

    def __init__(self, input_dim: int, private_dim: int = 32, dropout: float = 0.2, 
                        num_heads: int = 4, num_layers: int = 2, dim_feedforward: int = 128,
                        max_len: int = 500):
        super().__init__()

        # Project Input features to Private Dimension Space
        self.input_proj = nn.Linear(input_dim, private_dim)

        # Learned Position Embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, private_dim))

        # Transformer Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=private_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Transformer Decoder
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def _causal_mask(self, T, device):
        return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        x = self.input_proj(x)
        x = x + self.pos_embedding[:, :T, :]

        tgt_mask = self._causal_mask(T, x.device)

        x = self.transformer(tgt=x, memory=x, tgt_mask=tgt_mask)

        out = x[:, -1, :]
        return self.dropout(out)


# ─────────────────────────────────────────────
# 3. Shared Decoder (sees all stocks)
# ─────────────────────────────────────────────

class SharedDecoder(nn.Module):
    """
    Encodes the concatenation of all stocks' raw time series to capture
    cross-stock (market-level) temporal patterns (paper Eq. 5).

    Input:  (B, K, T, input_dim) — raw time series for all stocks
    Output: (B, shared_dim)

    The shared LSTM sees all stocks' features concatenated along the
    feature dimension at each timestep, so it models temporal dynamics
    across the full market, not just a single-step projection.
    """

    def __init__(self, num_stocks: int, input_dim: int, shared_dim: int = 64, dropout: float = 0.2,
                 num_heads: int = 4, num_layers: int = 2, dim_feedforward: int = 128,
                 max_len: int = 500):
        super().__init__()

        self.input_proj = nn.Linear(num_stocks * input_dim, shared_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, shared_dim))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=shared_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)

    def _causal_mask(self, T, device):
        return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, K, T, D = x.shape

        x = x.permute(0, 2, 1, 3).reshape(B, T, K * D)

        x = self.input_proj(x)
        x = x + self.pos_embedding[:, :T, :]

        tgt_mask = self._causal_mask(T, x.device)

        x = self.transformer(tgt=x, memory=x, tgt_mask=tgt_mask)

        out = x[:, -1, :]
        return self.dropout(out)


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
    Market regime classification using only the shared decoder output.
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

        # One private decoder per stock
        self.private_decoder = PrivateDecoder(input_dim, private_hidden, dropout)

        # Single shared decoder across all stocks (sees raw time series)
        self.shared_decoder = SharedDecoder(num_stocks, input_dim, shared_hidden, dropout)

        # One SPA module per stock (projects to common spa_dim, then weighted sum)
        self.spa_module = SPA(shared_hidden, private_hidden, attn_dim=spa_dim)

        # Stock task head operates on spa_dim (the SPA output dimension)
        self.task_head = StockTaskHeads(spa_dim, num_direction_classes)

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
            self.private_decoder(x[:, k, :, :])   # (B, private_hidden)
            for k in range(self.num_stocks)
        ]

        # 2. Shared encoding: sees raw time series from ALL stocks (paper Eq. 5)
        f_s = self.shared_decoder(x)                   # (B, shared_hidden)

        # 3. SPA + task heads per stock
        stock_outputs = []
        for k in range(self.num_stocks):
            f_combined = self.spa_module(f_s, private_feats[k])   # (B, combined_dim)
            preds = self.task_head(f_combined)
            stock_outputs.append(preds)

        # 4. Regime (shared decoder only)
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