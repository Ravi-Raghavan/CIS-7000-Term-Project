import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
from sklearn.metrics import accuracy_score

from spa_msjf import SPAMSJF, JointLoss
from build_data_kronos import build_dataloaders

# ─────────────────────────────────────────────
# Evaluation utilities
# ─────────────────────────────────────────────

STOCKS = ["AAPL", "GOOG", "META", "NVDA", "TSLA"]
 
@torch.no_grad()
def evaluate(model, wrapper, dataloader: DataLoader, device: torch.device) -> dict:
    model.eval()
    wrapper.eval()
 
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
        s1, s2 = x
        s1, s2 = s1.to(device), s2.to(device)
        x_emb = wrapper(s1, s2)
        outputs = model(x_emb)
 
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
    header = f"-- Metrics ({split_name}) " if split_name else "-- Metrics "
    print(f"\n{header}{'-' * (50 - len(header))}")
 
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


def save_results_to_markdown(metrics: dict, epochs: int, lr: float, checkpoint: str, version="1.0"):
    import datetime
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(current_dir, "results", "test_results.md")
    
    lines = [
        f"## Test Run - {now}",
        f"**Version**: {version}",
        f"**Configuration**: Epochs={epochs}, LR={lr}, Checkpoint={checkpoint}",
        "",
        "### Test Metrics",
        ""
    ]

    for task in ["return", "volatility", "sharpe"]:
        lines.append(f"**{task.upper()} (MSE)**")
        lines.append("| Stock | MSE |")
        lines.append("|---|---|")
        for ticker, m in metrics[task].items():
            lines.append(f"| {ticker} | {m['MSE']:.6f} |")
        lines.append("")

    lines.append("**DIRECTION (Accuracy)**")
    lines.append("| Stock | Accuracy |")
    lines.append("|---|---|")
    for ticker, m in metrics["direction"].items():
        lines.append(f"| {ticker} | {m['accuracy']:.4f} |")
    lines.append("")

    lines.append(f"**REGIME (Accuracy)**")
    lines.append(f"Accuracy: {metrics['regime']['market']['accuracy']:.4f}")
    lines.append("\n---\n")

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "a") as f:
        f.write("\n".join(lines))
    print(f"Results saved to {filepath}")

 
# ─────────────────────────────────────────────
# Training function
# ─────────────────────────────────────────────
 
def train(
    model:      SPAMSJF,
    wrapper:    nn.Module,
    criterion:  JointLoss,
    train_dl:   DataLoader,
    val_dl:     DataLoader,
    test_dl:    DataLoader,
    device:     torch.device,
    epochs:     int   = 100,
    lr:         float = 1e-3,
    patience:   int   = 15,
    checkpoint: Optional[str] = "best_spa_msjf_kronos.pt",
) -> dict:

    model = model.to(device)
    wrapper = wrapper.to(device)
 
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(wrapper.parameters()),
        lr=lr, weight_decay=5e-4
    )

    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
 
    best_val_loss  = float("inf")
    best_state     = None
    best_wrapper_state = None
    best_epoch     = 0
    patience_count = 0
    history        = []
 
    n_params_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_params_wrapper = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
    print(f"Device: {device}  |  Trainable params: {n_params_model + n_params_wrapper:,} (Model: {n_params_model}, Wrapper: {n_params_wrapper})")
    print(f"Epochs: {epochs}  |  LR: {lr}  |  Early stop patience: {patience}")
    print(f"\n{'Epoch':>6}  {'Train loss':>12}  {'Val loss':>10}  {'LR':>10}  {'':>5}")
    print("-" * 52)
 
    for epoch in range(1, epochs + 1):
        model.train()
        wrapper.train()
        wrapper.tokenizer.eval()  # keep frozen tokenizer in eval mode
        
        train_losses = {}
        for x, targets in train_dl:
            s1, s2 = x
            s1, s2 = s1.to(device), s2.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
 
            optimizer.zero_grad()
            
            x_emb = wrapper(s1, s2)
            outputs = model(x_emb)

            loss, loss_dict = criterion(outputs, targets)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            nn.utils.clip_grad_norm_(wrapper.parameters(), max_norm=1.0)

            optimizer.step()

            for k, v in loss_dict.items():
                train_losses[k] = train_losses.get(k, 0.0) + v
 
        train_losses = {k: v / len(train_dl) for k, v in train_losses.items()}
 
        model.eval()
        wrapper.eval()
        val_losses = {}
        with torch.no_grad():
            for x, targets in val_dl:
                s1, s2 = x
                s1, s2 = s1.to(device), s2.to(device)
                targets = {k: v.to(device) for k, v in targets.items()}
                
                x_emb = wrapper(s1, s2)
                _, loss_dict = criterion(model(x_emb), targets)
                
                for k, v in loss_dict.items():
                    val_losses[k] = val_losses.get(k, 0.0) + v
 
        val_losses = {k: v / len(val_dl) for k, v in val_losses.items()}
        scheduler.step()
 
        is_best = val_losses["total"] < best_val_loss
        if is_best:
            best_val_loss  = val_losses["total"]
            best_epoch     = epoch
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_wrapper_state = {k: v.cpu().clone() for k, v in wrapper.state_dict().items()}
            patience_count = 0
            if checkpoint:
                torch.save({"model": best_state, "wrapper": best_wrapper_state}, checkpoint)
        else:
            patience_count += 1
 
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
            print(f"\nEarly stopping - no val improvement for {patience} epochs.")
            break
 
    if best_state and checkpoint:
        ckpt = torch.load(checkpoint)
        model.load_state_dict(ckpt["model"])
        wrapper.load_state_dict(ckpt["wrapper"])
        print(f"\nRestored best weights from epoch {best_epoch} "
              f"(val loss: {best_val_loss:.5f})")
 
    print("\n-- Val metrics --------------------------------------")
    val_metrics = evaluate(model, wrapper, val_dl, device)
    print_metrics(val_metrics, split_name="val")
 
    print("-- Test metrics -------------------------------------")
    test_metrics = evaluate(model, wrapper, test_dl, device)
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
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = get_device()
 
    train_dl, val_dl, test_dl, meta = build_dataloaders(
        data_dir,
        seq_len=60,
        horizon=1,
        batch_size=32,
    )

    wrapper = meta["wrapper"]
    input_dim = meta.get("input_dim", 320)
    print(f"Input dim: {input_dim} (Kronos encoded)")

    model = SPAMSJF(
        num_stocks=5,
        input_dim=input_dim,
        private_hidden=64,
        shared_hidden=128,
        spa_dim=64,
        num_direction_classes=3,
        num_regimes=2,
        dropout=0.25,
        num_heads=4,
        num_layers=2,
    )

    criterion = JointLoss(
        lambda_return=1.0,
        lambda_vol=0.5,
        lambda_sharpe=0.3,
        lambda_direction=1.0,
        lambda_regime=1.0,
    )

    results = train(
        model=model,
        wrapper=wrapper,
        criterion=criterion,
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        device=device,
        epochs=10,
        lr=3e-4,
        patience=15,
        checkpoint="best_spa_msjf_kronos.pt",
    )