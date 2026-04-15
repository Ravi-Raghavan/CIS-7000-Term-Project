"""
Stage 1 — Domain Adaptation of Kronos Tokenizer via LoRA
=========================================================
Fine-tunes the pretrained Kronos tokenizer on US high-cap tech stock data
(AAPL, GOOG, META, NVDA, TSLA) using parameter-efficient LoRA adapters.

Training objective: minimize tokenizer reconstruction loss on the target domain.
Only LoRA adapter weights update; all pretrained Kronos weights remain frozen.

Usage:
    python finetune_kronos_lora.py --data_dir "../Finance Data"
    python finetune_kronos_lora.py --data_dir "../Finance Data" --lora_rank 4 --epochs 30

Output:
    kronos_lora_adapted/   (peft adapter weights, ~few KB)
"""

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Kronos"))
from model.kronos import KronosTokenizer

from build_data_kronos import (
    load_and_align,
    build_kronos_raw_input,
    split_indices,
    STOCKS,
    SPLIT,
)


# ─────────────────────────────────────────────
# Dataset: sliding windows of raw OHLCVA data
# ─────────────────────────────────────────────

class TokenizerWindowDataset(Dataset):
    """
    Yields (window_size, 6) float32 windows from (K, T, 6) raw OHLCVA data.
    Used exclusively to fine-tune the tokenizer's reconstruction loss.
    """
    def __init__(self, kronos_raw: np.ndarray, window: int = 256, stride: int = 128):
        self.windows = []
        K, T, C = kronos_raw.shape
        for k in range(K):
            for start in range(0, T - window + 1, stride):
                self.windows.append(
                    kronos_raw[k, start : start + window, :].astype(np.float32)
                )
        print(f"Tokenizer fine-tune dataset: {len(self.windows)} windows "
              f"({K} stocks × ~{len(self.windows)//K} windows each, "
              f"window={window}, stride={stride})")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.tensor(self.windows[idx], dtype=torch.float32)


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────

def finetune_lora(
    data_dir:    str,
    lora_rank:   int   = 8,
    lora_alpha:  int   = 16,
    lora_dropout:float = 0.1,
    window:      int   = 256,
    stride:      int   = 128,
    batch_size:  int   = 16,
    lr:          float = 1e-4,
    weight_decay:float = 1e-5,
    epochs:      int   = 20,
    patience:    int   = 5,
    output_dir:  str   = "kronos_lora_adapted",
    device:      torch.device = None,
):
    if device is None:
        device = (torch.device("cuda") if torch.cuda.is_available()
                  else torch.device("mps") if torch.backends.mps.is_available()
                  else torch.device("cpu"))
    print(f"Device: {device}")

    # ── Load and split data ──────────────────────────────────────────────────
    frames      = load_and_align(data_dir)
    kronos_raw  = build_kronos_raw_input(frames)               # (K=5, T, 6)
    T           = kronos_raw.shape[1]
    train_end, val_end = split_indices(T, SPLIT)

    kronos_train = kronos_raw[:, :train_end, :]                # (K, T_train, 6)
    kronos_val   = kronos_raw[:, train_end:val_end, :]         # (K, T_val,   6)

    train_ds = TokenizerWindowDataset(kronos_train, window, stride)
    val_ds   = TokenizerWindowDataset(kronos_val,   window, stride)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    # ── Load tokenizer and apply LoRA ────────────────────────────────────────
    try:
        from peft import get_peft_model, LoraConfig
    except ImportError:
        raise ImportError("peft is required. Run: pip install peft>=0.10.0")

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        # q_proj and v_proj are the exact attribute names in MultiHeadAttentionWithRoPE
        # (confirmed in Kronos/model/module.py lines 322-324)
        target_modules=["q_proj", "v_proj"],
        lora_dropout=lora_dropout,
        bias="none",
    )
    tokenizer = get_peft_model(tokenizer, lora_config)
    tokenizer.print_trainable_parameters()
    tokenizer = tokenizer.to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, tokenizer.parameters()),
        lr=lr, weight_decay=weight_decay
    )

    # ── Training ─────────────────────────────────────────────────────────────
    best_val_loss   = float("inf")
    patience_count  = 0

    print(f"\n{'Epoch':>6}  {'Train loss':>12}  {'Val loss':>10}  {'':>5}")
    print("-" * 40)

    for epoch in range(1, epochs + 1):
        # Train
        tokenizer.train()
        train_loss_sum = 0.0
        for batch in train_dl:
            batch = batch.to(device)                           # (B, window, 6)
            optimizer.zero_grad()

            # forward() returns: (z_pre, z_recon), bsq_loss, quantized, z_indices
            (z_pre, z_recon), bsq_loss, _, _ = tokenizer(batch)

            # Reconstruction losses:
            #   z_recon: reconstructed from FULL codebook (s1+s2 bits)
            #   z_pre:   reconstructed from s1 bits only (coarse signal)
            recon_loss     = F.mse_loss(z_recon, batch)
            recon_loss_pre = F.mse_loss(z_pre,   batch)
            loss = recon_loss + 0.5 * recon_loss_pre + bsq_loss

            loss.backward()
            nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, tokenizer.parameters()),
                max_norm=1.0
            )
            optimizer.step()
            train_loss_sum += loss.item()

        train_loss = train_loss_sum / len(train_dl)

        # Validate
        tokenizer.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in val_dl:
                batch = batch.to(device)
                (z_pre, z_recon), bsq_loss, _, _ = tokenizer(batch)
                recon_loss     = F.mse_loss(z_recon, batch)
                recon_loss_pre = F.mse_loss(z_pre,   batch)
                val_loss_sum  += (recon_loss + 0.5 * recon_loss_pre + bsq_loss).item()

        val_loss = val_loss_sum / len(val_dl)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss  = val_loss
            patience_count = 0
            tokenizer.save_pretrained(output_dir)
        else:
            patience_count += 1

        print(f"{epoch:>6}  {train_loss:>12.5f}  {val_loss:>10.5f}  "
              f"{'* saved' if is_best else ''}")

        if patience_count >= patience:
            print(f"\nEarly stopping — no val improvement for {patience} epochs.")
            break

    print(f"\nBest val reconstruction loss: {best_val_loss:.5f}")
    print(f"LoRA adapter saved to: {output_dir}/")
    print("\nToken distribution check (frozen baseline vs LoRA-adapted):")
    print("  Run build_data_kronos.py with and without --lora_path to compare")
    print("  np.unique(s1_tokens).size  — more unique tokens = more adapted vocabulary")

    return best_val_loss


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 1: LoRA domain adaptation of Kronos tokenizer"
    )
    parser.add_argument("--data_dir",     default="../Finance Data",
                        help="Directory containing Historical_Data_*.csv files")
    parser.add_argument("--lora_rank",    type=int,   default=8,
                        help="LoRA rank r (4=conservative, 8=balanced, 16=aggressive)")
    parser.add_argument("--lora_alpha",   type=int,   default=16,
                        help="LoRA alpha (scaling = alpha/rank)")
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--window",       type=int,   default=256,
                        help="Sliding window size for tokenizer input (<=512)")
    parser.add_argument("--stride",       type=int,   default=128,
                        help="Stride for sliding window (default: 50%% overlap)")
    parser.add_argument("--batch_size",   type=int,   default=16)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--epochs",       type=int,   default=20)
    parser.add_argument("--patience",     type=int,   default=5)
    parser.add_argument("--output_dir",   default="kronos_lora_adapted",
                        help="Directory to save LoRA adapter weights")
    args = parser.parse_args()

    finetune_lora(
        data_dir     = args.data_dir,
        lora_rank    = args.lora_rank,
        lora_alpha   = args.lora_alpha,
        lora_dropout = args.lora_dropout,
        window       = args.window,
        stride       = args.stride,
        batch_size   = args.batch_size,
        lr           = args.lr,
        epochs       = args.epochs,
        patience     = args.patience,
        output_dir   = args.output_dir,
    )
