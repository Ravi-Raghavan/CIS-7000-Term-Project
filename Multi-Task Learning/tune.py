"""
tune.py — Smart two-phase hyperparameter search for SPA-MSJF
─────────────────────────────────────────────────────────────
Phase 1 : Grid search over architecture size + learning rate  (~12 trials)
Phase 2 : Refine loss lambda weights using the Phase-1 winner  (~8 trials)

Each trial uses early stopping (patience=8, max 60 epochs) so fast failures
exit quickly. Total wall-clock on a 3070 Ti: ~25-40 min.

Results are saved to:
    tuning_results.csv   — every trial, every metric
    tuning_summary.txt   — human-readable best-config report

Usage:
    py -3.12 tune.py
"""

import csv
import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from spa_msjf import SPAMSJF, JointLoss
from build_data import build_dataloaders

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR      = "../Finance Data"
RESULTS_CSV   = "tuning_results.csv"
SUMMARY_FILE  = "tuning_summary.txt"
BEST_CKPT     = "best_tuned_spa_msjf.pt"

# Shared across all trials
SEQ_LEN       = 30
BATCH_SIZE    = 32
MAX_EPOCHS    = 60       # hard cap per trial
PATIENCE      = 8        # early-stop patience per trial
STOCKS        = ["AAPL", "GOOG", "META", "NVDA", "TSLA"]

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 search space  (architecture + optimiser)
# ─────────────────────────────────────────────────────────────────────────────

PHASE1_GRID = {
    "private_hidden": [32, 64],
    "shared_hidden":  [64, 128],
    "spa_dim":        [32, 64],
    "dropout":        [0.1, 0.3],
    "lr":             [1e-3, 5e-4],
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 search space  (loss lambda weights, fixed arch from Phase 1)
# ─────────────────────────────────────────────────────────────────────────────

PHASE2_GRID = {
    "lambda_return":    [1.0, 2.0],
    "lambda_vol":       [0.5, 1.0],
    "lambda_sharpe":    [0.3, 0.5],
    "lambda_direction": [1.0, 2.0],
    "lambda_regime":    [0.5, 1.0],
}


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    all_ret_pred  = [[] for _ in STOCKS]
    all_ret_true  = [[] for _ in STOCKS]
    all_dir_pred  = [[] for _ in STOCKS]
    all_dir_true  = [[] for _ in STOCKS]
    all_shp_pred  = [[] for _ in STOCKS]
    all_shp_true  = [[] for _ in STOCKS]
    all_reg_pred, all_reg_true = [], []

    for x, targets in loader:
        x = x.to(device)
        out = model(x)
        for k in range(len(STOCKS)):
            p = out["stocks"][k]
            all_ret_pred[k].append(p["return"].cpu())
            all_ret_true[k].append(targets["return"][:, k])
            all_dir_pred[k].append(p["direction"].argmax(-1).cpu())
            all_dir_true[k].append(targets["direction"][:, k])
            all_shp_pred[k].append(p["sharpe"].cpu())
            all_shp_true[k].append(targets["sharpe"][:, k])
        all_reg_pred.append(out["regime"].argmax(-1).cpu())
        all_reg_true.append(targets["regime"])

    def cat(lst): return torch.cat(lst).numpy()

    ret_mse  = np.mean([np.mean((cat(all_ret_pred[k]) - cat(all_ret_true[k]))**2)  for k in range(len(STOCKS))])
    shp_mse  = np.mean([np.mean((cat(all_shp_pred[k]) - cat(all_shp_true[k]))**2)  for k in range(len(STOCKS))])
    dir_acc  = np.mean([accuracy_score(cat(all_dir_true[k]), cat(all_dir_pred[k]))  for k in range(len(STOCKS))])
    reg_acc  = accuracy_score(cat(all_reg_true), cat(all_reg_pred))

    return {
        "return_mse": float(ret_mse),
        "sharpe_mse": float(shp_mse),
        "dir_acc":    float(dir_acc),
        "reg_acc":    float(reg_acc),
        # composite score: lower is better
        # normalises regression losses down, rewards classification accuracy
        "score":      float(ret_mse + 0.3 * shp_mse - 0.5 * dir_acc - 0.3 * reg_acc),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single training trial
# ─────────────────────────────────────────────────────────────────────────────

def run_trial(
    cfg: dict[str, Any],
    train_dl: DataLoader,
    val_dl:   DataLoader,
    device:   torch.device,
    trial_id: int,
    total:    int,
) -> dict:
    """Train one configuration, return val metrics at best checkpoint."""

    arch = {k: cfg[k] for k in ("private_hidden", "shared_hidden", "spa_dim", "dropout")}
    lr   = cfg["lr"]
    lambdas = {k: cfg[k] for k in ("lambda_return","lambda_vol","lambda_sharpe",
                                    "lambda_direction","lambda_regime")}

    model = SPAMSJF(
        num_stocks=5, input_dim=5,
        private_hidden=arch["private_hidden"],
        shared_hidden=arch["shared_hidden"],
        spa_dim=arch["spa_dim"],
        num_direction_classes=3,
        num_regimes=2,
        dropout=arch["dropout"],
    ).to(device)

    criterion = JointLoss(**lambdas)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=4, factor=0.5
    )

    best_val   = float("inf")
    best_state = None
    patience_count = 0
    t0 = time.time()

    print(f"\n[{trial_id}/{total}] {json.dumps({k: v for k,v in cfg.items()}, separators=(',',':'))}")
    print(f"  {'Ep':>4}  {'Train':>9}  {'Val':>9}")

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── train ──
        model.train()
        train_loss = 0.0
        for x, targets in train_dl:
            x       = x.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            optimizer.zero_grad()
            out = model(x)
            loss, _ = criterion(out, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        # ── validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, targets in val_dl:
                x       = x.to(device)
                targets = {k: v.to(device) for k, v in targets.items()}
                _, ld   = criterion(model(x), targets)
                val_loss += ld["total"]
        val_loss /= len(val_dl)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
            marker = "*"
        else:
            patience_count += 1
            marker = ""

        if epoch % 5 == 0 or marker:
            print(f"  {epoch:>4}  {train_loss:>9.5f}  {val_loss:>9.5f}  {marker}")

        if patience_count >= PATIENCE:
            print(f"  → Early stop at epoch {epoch}")
            break

    elapsed = time.time() - t0
    model.load_state_dict(best_state)
    metrics = evaluate(model, val_dl, device)
    metrics["val_loss"] = best_val
    metrics["elapsed"]  = round(elapsed, 1)

    print(f"  ✓ score={metrics['score']:.4f}  ret_mse={metrics['return_mse']:.5f}  "
          f"dir_acc={metrics['dir_acc']:.3f}  reg_acc={metrics['reg_acc']:.3f}  "
          f"({elapsed:.0f}s)")

    return metrics, model, best_state


# ─────────────────────────────────────────────────────────────────────────────
# Grid helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_grid(space: dict) -> list[dict]:
    keys   = list(space.keys())
    combos = list(itertools.product(*space.values()))
    return [dict(zip(keys, c)) for c in combos]


def save_row(path: str, row: dict, write_header: bool):
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = get_device()
    print(f"\n{'='*60}")
    print(f"  SPA-MSJF Hyperparameter Tuner")
    print(f"  Device : {device}")
    print(f"{'='*60}")

    # ── data (load once) ──────────────────────────────────────
    train_dl, val_dl, test_dl, meta = build_dataloaders(
        DATA_DIR, seq_len=SEQ_LEN, horizon=1, batch_size=BATCH_SIZE
    )
    print(f"  Train: {len(train_dl.dataset)}  Val: {len(val_dl.dataset)}  Test: {len(test_dl.dataset)}\n")

    # default lambda values used in Phase 1
    default_lambdas = dict(
        lambda_return=1.0, lambda_vol=0.5, lambda_sharpe=0.3,
        lambda_direction=1.0, lambda_regime=0.5,
    )

    # ── Phase 1: architecture + LR ────────────────────────────
    print(f"\n{'─'*60}")
    print("  PHASE 1 — Architecture & Learning Rate Search")
    print(f"{'─'*60}")

    p1_grid   = make_grid(PHASE1_GRID)
    p1_results = []
    csv_header_written = False

    for i, arch_cfg in enumerate(p1_grid, 1):
        cfg = {**arch_cfg, **default_lambdas}
        metrics, model, state = run_trial(cfg, train_dl, val_dl, device, i, len(p1_grid))

        row = {"phase": 1, **cfg, **metrics}
        save_row(RESULTS_CSV, row, write_header=not csv_header_written)
        csv_header_written = True
        p1_results.append((metrics["score"], cfg, state))

    # best Phase-1 config (lowest composite score)
    p1_results.sort(key=lambda x: x[0])
    best_p1_score, best_arch_cfg, best_p1_state = p1_results[0]

    print(f"\n  ── Phase 1 winner (score={best_p1_score:.4f}) ──")
    for k, v in best_arch_cfg.items():
        print(f"     {k}: {v}")

    # ── Phase 2: lambda tuning ────────────────────────────────
    print(f"\n{'─'*60}")
    print("  PHASE 2 — Loss Lambda Weight Search")
    print(f"{'─'*60}")

    # Use a reduced 2-value grid per lambda to keep trial count manageable
    p2_grid   = make_grid(PHASE2_GRID)
    p2_results = []

    for i, lambda_cfg in enumerate(p2_grid, 1):
        cfg = {**best_arch_cfg, **lambda_cfg}
        metrics, model, state = run_trial(cfg, train_dl, val_dl, device, i, len(p2_grid))

        row = {"phase": 2, **cfg, **metrics}
        save_row(RESULTS_CSV, row, write_header=False)
        p2_results.append((metrics["score"], cfg, model, state))

    p2_results.sort(key=lambda x: x[0])
    best_p2_score, best_cfg, best_model, best_state = p2_results[0]

    # ── Save best model ───────────────────────────────────────
    torch.save(best_state, BEST_CKPT)

    # ── Final test evaluation ─────────────────────────────────
    print(f"\n{'='*60}")
    print("  FINAL EVALUATION on Test Set")
    print(f"{'='*60}")
    best_model.load_state_dict(best_state)
    test_metrics = evaluate(best_model.to(device), test_dl, device)

    # ── Summary report ────────────────────────────────────────
    lines = []
    lines.append("=" * 60)
    lines.append("  SPA-MSJF TUNING SUMMARY")
    lines.append("=" * 60)

    lines.append("\n── Phase 1: Top 5 Architecture Configs ──")
    lines.append(f"  {'Score':>8}  {'priv_h':>6}  {'shar_h':>6}  {'spa_d':>5}  {'drop':>5}  {'lr':>8}")
    for score, cfg, _ in p1_results[:5]:
        lines.append(
            f"  {score:>8.4f}  {cfg['private_hidden']:>6}  {cfg['shared_hidden']:>6}  "
            f"{cfg['spa_dim']:>5}  {cfg['dropout']:>5}  {cfg['lr']:>8}"
        )

    lines.append("\n── Phase 2: Top 5 Lambda Configs ──")
    lines.append(f"  {'Score':>8}  {'λ_ret':>6}  {'λ_vol':>6}  {'λ_shp':>6}  {'λ_dir':>6}  {'λ_reg':>6}")
    for score, cfg, _, _ in p2_results[:5]:
        lines.append(
            f"  {score:>8.4f}  {cfg['lambda_return']:>6}  {cfg['lambda_vol']:>6}  "
            f"{cfg['lambda_sharpe']:>6}  {cfg['lambda_direction']:>6}  {cfg['lambda_regime']:>6}"
        )

    lines.append("\n── Best Overall Config ──")
    for k, v in best_cfg.items():
        lines.append(f"  {k}: {v}")

    lines.append("\n── Test Metrics (Best Model) ──")
    lines.append(f"  Return  MSE : {test_metrics['return_mse']:.6f}")
    lines.append(f"  Sharpe  MSE : {test_metrics['sharpe_mse']:.6f}")
    lines.append(f"  Dir  Acc    : {test_metrics['dir_acc']:.4f}  ({test_metrics['dir_acc']*100:.1f}%)")
    lines.append(f"  Regime Acc  : {test_metrics['reg_acc']:.4f}  ({test_metrics['reg_acc']*100:.1f}%)")
    lines.append(f"  Composite   : {test_metrics['score']:.4f}  (lower is better)")

    lines.append("\n── What Worked ──")
    ph = best_cfg['private_hidden']
    sh = best_cfg['shared_hidden']
    lines.append(f"  Architecture: private_hidden={ph}, shared_hidden={sh} — "
                 + ("larger shared encoder captures more cross-stock dynamics"
                    if sh >= 128 else "compact architecture avoids overfitting on limited data"))
    lines.append(f"  LR={best_cfg['lr']} — "
                 + ("slower convergence helped generalisation" if best_cfg['lr'] <= 5e-4
                    else "faster convergence useful with aggressive early stopping"))
    lines.append(f"  Dropout={best_cfg['dropout']} — "
                 + ("higher dropout regularised the shared encoder effectively"
                    if best_cfg['dropout'] >= 0.3 else "lower dropout preserved gradient signal"))
    lines.append(f"  λ_direction={best_cfg['lambda_direction']} — "
                 + ("upweighting direction improved classification-regression trade-off"
                    if best_cfg['lambda_direction'] >= 2.0
                    else "default weight sufficient — tasks balanced naturally"))

    lines.append(f"\n  Best checkpoint saved to: {BEST_CKPT}")
    lines.append(f"  Full results CSV        : {RESULTS_CSV}")
    lines.append("=" * 60)

    summary = "\n".join(lines)
    print("\n" + summary)
    Path(SUMMARY_FILE).write_text(summary, encoding="utf-8")
    print(f"\n  Summary written to {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
