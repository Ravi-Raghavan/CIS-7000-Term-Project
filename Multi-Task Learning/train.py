"""
train.py
--------
Training entry point for SPA-MSJF.
 
Imports model/loss from spa_msjf.py and data pipeline from finance_data.py.
 
Usage:
    python train.py path/to/csvs/
 
    # or from a notebook:
    from train import train
    results = train(model, criterion, train_dl, val_dl, test_dl, device)
"""
 
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
from sklearn.metrics import accuracy_score
 
from spa_msjf import SPAMSJF, JointLoss
from build_data import build_dataloaders

# ─────────────────────────────────────────────
# Evaluation utilities
# ─────────────────────────────────────────────

STOCKS = ["AAPL", "GOOG", "META", "NVDA", "TSLA"]
 
@torch.no_grad()
def evaluate(model, dataloader: DataLoader, device: torch.device) -> dict:
    """
    Compute per-task metrics over a dataloader.
        Regression  (return, volatility, sharpe) → MSE per stock
        Classification (direction, regime)        → accuracy
    Returns: metrics[task][ticker_or_"market"]["MSE" or "accuracy"]
    """
    model.eval() # Put model in evaluation mode
 
    all_ret_pred  = [[] for _ in STOCKS]
    all_ret_true  = [[] for _ in STOCKS]
    all_vol_pred  = [[] for _ in STOCKS]
    all_vol_true  = [[] for _ in STOCKS]
    all_shp_pred  = [[] for _ in STOCKS]
    all_shp_true  = [[] for _ in STOCKS]
    all_dir_pred  = [[] for _ in STOCKS]
    all_dir_true  = [[] for _ in STOCKS]
    all_reg_pred  = []
    all_reg_true  = []
 
    for x, targets in dataloader:
        x       = x.to(device)
        outputs = model(x)
 
        for k in range(len(STOCKS)):
            preds = outputs["stocks"][k]
            all_ret_pred[k].append(preds["return"].cpu())
            all_ret_true[k].append(targets["return"][:, k])
            all_vol_pred[k].append(preds["volatility"].cpu())
            all_vol_true[k].append(targets["volatility"][:, k])
            all_shp_pred[k].append(preds["sharpe"].cpu())
            all_shp_true[k].append(targets["sharpe"][:, k])
            all_dir_pred[k].append(preds["direction"].argmax(-1).cpu())
            all_dir_true[k].append(targets["direction"][:, k])
 
        all_reg_pred.append(outputs["regime"].argmax(-1).cpu())
        all_reg_true.append(targets["regime"])
 
    def cat(lst): return torch.cat(lst).numpy()
 
    metrics = {"return": {}, "volatility": {}, "sharpe": {}, "direction": {}, "regime": {}}
 
    for k, ticker in enumerate(STOCKS):
        for task, preds_list, trues_list in [
            ("return",     all_ret_pred[k], all_ret_true[k]),
            ("volatility", all_vol_pred[k], all_vol_true[k]),
            ("sharpe",     all_shp_pred[k], all_shp_true[k]),
        ]:
            p = cat(preds_list)
            t = cat(trues_list)
            metrics[task][ticker] = {"MSE": float(np.mean((p - t) ** 2))}
 
        p = cat(all_dir_pred[k])
        t = cat(all_dir_true[k])
        metrics["direction"][ticker] = {"accuracy": float(accuracy_score(t, p))}
 
    p = cat(all_reg_pred)
    t = cat(all_reg_true)
    metrics["regime"]["market"] = {"accuracy": float(accuracy_score(t, p))}
 
    return metrics
 
 
def print_metrics(metrics: dict, split_name: str = ""):
    """Pretty-print the metrics dict returned by evaluate()."""
    header = f"── Metrics ({split_name}) " if split_name else "── Metrics "
    print(f"\n{header}{'─' * (50 - len(header))}")
 
    for task in ["return", "volatility", "sharpe"]:
        print(f"\n  {task.upper()}  (MSE)")
        print(f"  {'Stock':<8} {'MSE':>12}")
        for ticker, m in metrics[task].items():
            print(f"  {ticker:<8} {m['MSE']:>12.6f}")
 
    print(f"\n  DIRECTION  (accuracy)")
    print(f"  {'Stock':<8} {'Accuracy':>10}")
    for ticker, m in metrics["direction"].items():
        print(f"  {ticker:<8} {m['accuracy']:>10.4f}")
 
    print(f"\n  REGIME  (accuracy)")
    print(f"  Accuracy: {metrics['regime']['market']['accuracy']:.4f}")
    print()
 
# ─────────────────────────────────────────────
# Training function
# ─────────────────────────────────────────────
 
def train(
    model:      SPAMSJF,
    criterion:  JointLoss,
    train_dl:   DataLoader,
    val_dl:     DataLoader,
    test_dl:    DataLoader,
    device:     torch.device,
    epochs:     int   = 100,
    lr:         float = 1e-3,
    patience:   int   = 15,
    checkpoint: Optional[str] = "best_spa_msjf.pt",
) -> dict:
    """
    Full training loop with validation, early stopping, and final test eval.
 
    Args:
        model:       SPAMSJF instance (moved to device inside)
        criterion:   JointLoss instance
        train_dl:    training DataLoader
        val_dl:      validation DataLoader
        test_dl:     test DataLoader
        device:      torch.device
        epochs:      max training epochs (default 100)
        lr:          Adam learning rate (default 1e-3)
        patience:    early stopping patience in epochs (default 15)
        checkpoint:  path to save best model weights; None to skip saving
 
    Returns:
        dict with keys:
            "history"      : list of per-epoch loss dicts
            "best_epoch"   : epoch index with lowest val loss
            "val_metrics"  : full metrics on val set at best checkpoint
            "test_metrics" : full metrics on test set at best checkpoint
    """

    # Move model to device
    model = model.to(device)
 
    # weight_decay=1e-4 adds L2 regularisation — penalises large weights
    # and reduces memorisation of training-era patterns
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Halve LR after 7 epochs of no val improvement.
    # Patience=4 was tried but decayed LR too aggressively (8x drop by ep 23),
    # preventing escape from local minima — patience=7 gives more exploration room.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=7, factor=0.5
    )
 
    # Store best validation loss, state, and epoch
    best_val_loss  = float("inf")
    best_state     = None
    best_epoch     = 0
    patience_count = 0
    history        = []
 
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device}  |  Trainable params: {n_params:,}")
    print(f"Epochs: {epochs}  |  LR: {lr}  |  Early stop patience: {patience}")
    print(f"\n{'Epoch':>6}  {'Train loss':>12}  {'Val loss':>10}  {'LR':>10}  {'':>5}")
    print("─" * 52)
 
    # Iterate through Epochs
    for epoch in range(1, epochs + 1):
 
        # ── Train ──────────────────────────────────────────
        model.train()
        train_losses = {}
        for x, targets in train_dl:
            x       = x.to(device) # Get batch data to device
            targets = {k: v.to(device) for k, v in targets.items()}
 
            # Zero out optimizer
            optimizer.zero_grad()

            # Run Forward Pass
            outputs         = model(x)

            # Compute Loss and Update
            loss, loss_dict = criterion(outputs, targets)
            loss.backward()

            # Clip Gradient Norm to prevent Exploding Gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update Model
            optimizer.step()

            # Accumulate Loss values across batches
            for k, v in loss_dict.items():
                train_losses[k] = train_losses.get(k, 0.0) + v
 
        # Average out Loss values across batches
        train_losses = {k: v / len(train_dl) for k, v in train_losses.items()}
 
        # ── Validate ───────────────────────────────────────
        model.eval()
        val_losses = {}
        with torch.no_grad():
            for x, targets in val_dl:
                x       = x.to(device)
                targets = {k: v.to(device) for k, v in targets.items()}
                _, loss_dict = criterion(model(x), targets)
                for k, v in loss_dict.items():
                    val_losses[k] = val_losses.get(k, 0.0) + v
 
        val_losses = {k: v / len(val_dl) for k, v in val_losses.items()}
        scheduler.step(val_losses["total"])
 
        # ── Checkpoint ─────────────────────────────────────
        is_best = val_losses["total"] < best_val_loss
        if is_best:
            best_val_loss  = val_losses["total"]
            best_epoch     = epoch
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
            if checkpoint:
                torch.save(best_state, checkpoint)
        else:
            patience_count += 1
 
        # ── Log ────────────────────────────────────────────
        history.append({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_losses.items()},
            **{f"val_{k}":   v for k, v in val_losses.items()},
            "lr": optimizer.param_groups[0]["lr"],
        })
 
        print(
            f"{epoch:>6}  "
            f"{train_losses['total']:>12.5f}  "
            f"{val_losses['total']:>10.5f}  "
            f"{optimizer.param_groups[0]['lr']:>10.2e}  "
            f"{'*' if is_best else ''}"
        )
 
        if patience_count >= patience:
            print(f"\nEarly stopping — no val improvement for {patience} epochs.")
            break
 
    # ── Restore best weights ───────────────────────────────
    if best_state:
        model.load_state_dict(best_state)
        print(f"\nRestored best weights from epoch {best_epoch} "
              f"(val loss: {best_val_loss:.5f})")
 
    # ── Final evaluation ───────────────────────────────────
    print("\n── Val metrics ──────────────────────────────────────")
    val_metrics = evaluate(model, val_dl, device)
    print_metrics(val_metrics, split_name="val")
 
    print("── Test metrics ─────────────────────────────────────")
    test_metrics = evaluate(model, test_dl, device)
    print_metrics(test_metrics, split_name="test")
 
    return {
        "history":      history,
        "best_epoch":   best_epoch,
        "val_metrics":  val_metrics,
        "test_metrics": test_metrics,
    }
 
 
# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
 
if __name__ == "__main__":
    data_dir = "../Finance Data"

    def get_device() -> torch.device:
        """CUDA (NVIDIA) → MPS (Apple Silicon) → CPU."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = get_device()
 
    # ── Data ──────────────────────────────────────────────
    # seq_len=60 gives the LSTM 2 months of context per sample (up from 30)
    train_dl, val_dl, test_dl, meta = build_dataloaders(
        data_dir,
        seq_len=60,
        horizon=1,
        batch_size=32,
    )

    # ── Model ─────────────────────────────────────────────
    # Hyperparams from Phase-1 grid search (tune.py):
    #   private_hidden=32, shared_hidden=64, spa_dim=32 — compact avoids overfit
    #   dropout=0.1 — lower dropout preserved gradient signal on this dataset
    model = SPAMSJF(
        num_stocks=5,
        input_dim=5,              # 5 engineered features (see build_data.py)
        private_hidden=32,
        shared_hidden=64,
        spa_dim=32,
        num_direction_classes=3,
        num_regimes=2,            # bull / bear
        dropout=0.1,              # tuner found 0.1 > 0.3 on this dataset
    )

    # ── Loss ──────────────────────────────────────────────
    criterion = JointLoss(
        lambda_return=1.0,
        lambda_vol=0.5,
        lambda_sharpe=0.3,
        lambda_direction=1.0,
        lambda_regime=0.5,
    )

    # ── Train ─────────────────────────────────────────────
    results = train(
        model=model,
        criterion=criterion,
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        device=device,
        epochs=100,
        lr=5e-4,       # tuner: slower convergence improves generalisation
        patience=15,
        checkpoint="best_spa_msjf.pt",
    )