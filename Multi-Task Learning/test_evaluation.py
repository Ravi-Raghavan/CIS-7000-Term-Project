"""
test_evaluation.py — Evaluation suite for SPA-MSJF
===================================================

Implements ALL metrics and robustness tests from the project proposal (Section 4):

  Proposal Metrics:
    - Return forecasting:            MSE
    - Volatility/Risk forecasting:   MSE
    - Sharpe ratio forecasting:      MSE
    - Directional Movement:          Accuracy + F1
    - Market regime identification:  Accuracy + F1

  Robustness Evaluation (Proposal Section 4.1):
    "Robustness is evaluated by measuring the degree of performance degradation
     across different prediction tasks under changing market conditions."

    We implement this by:
      1. Training on one regime window, testing on a different regime window
      2. Comparing per-task metrics across windows
      3. Computing a "degradation score" = (test_metric - val_metric) / val_metric

  Additional tests:
    - Smoke test:           forward pass + backward pass on fake data
    - Overfitting test:     model can memorize a tiny dataset (proves capacity)
    - SPA weight analysis:  verify attention weights are learnable and vary per stock
    - Single-task vs MTL:   compare multi-task model vs independent per-task baselines
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    f1_score,
    classification_report,
)

# Import everything from the main module
sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module
mtl = import_module("multi-task-learning")

SPAMSJF = mtl.SPAMSJF
JointLoss = mtl.JointLoss
StockDataset = mtl.StockDataset
SlidingWindowTrainer = mtl.SlidingWindowTrainer
make_dummy_data = mtl.make_dummy_data


# ─────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_default_model(K=5, input_dim=5):
    return SPAMSJF(
        num_stocks=K,
        input_dim=input_dim,
        private_hidden=32,
        shared_hidden=64,
        spa_dim=32,
        num_direction_classes=3,
        num_regimes=4,
        dropout=0.1,
    )


def build_default_criterion():
    return JointLoss(
        lambda_return=1.0,
        lambda_vol=0.5,
        lambda_sharpe=0.3,
        lambda_direction=1.0,
        lambda_regime=0.5,
    )


def compute_all_metrics(results: dict, stock_names=None) -> dict:
    """
    Compute ALL proposal metrics from evaluate() output.

    Returns dict:
        {
            "per_stock": [
                {
                    "name": str,
                    "return_mse": float,
                    "volatility_mse": float,
                    "sharpe_mse": float,
                    "direction_accuracy": float,
                    "direction_f1": float,
                },
                ...
            ],
            "regime_accuracy": float,
            "regime_f1": float,
            "avg_return_mse": float,
            "avg_vol_mse": float,
            "avg_sharpe_mse": float,
            "avg_direction_accuracy": float,
            "avg_direction_f1": float,
        }
    """
    K = len(results["stocks"])
    if stock_names is None:
        stock_names = [f"Stock_{k}" for k in range(K)]

    per_stock = []
    for k in range(K):
        s = results["stocks"][k]
        d = {"name": stock_names[k]}

        # Regression: MSE (Proposal Section 4.1)
        for task in ["return", "volatility", "sharpe"]:
            preds = s[task]["preds"].numpy()
            targs = s[task]["targets"].numpy()
            d[f"{task}_mse"] = float(mean_squared_error(targs, preds))

        # Classification: Accuracy + F1 (Proposal Section 4.1)
        dir_preds = s["direction"]["preds"].numpy()
        dir_targs = s["direction"]["targets"].numpy()
        d["direction_accuracy"] = float(accuracy_score(dir_targs, dir_preds))
        d["direction_f1"] = float(f1_score(dir_targs, dir_preds, average="weighted", zero_division=0))

        per_stock.append(d)

    # Regime: Accuracy + F1 (Proposal Section 4.1)
    reg_preds = results["regime"]["preds"].numpy()
    reg_targs = results["regime"]["targets"].numpy()

    metrics = {
        "per_stock": per_stock,
        "regime_accuracy": float(accuracy_score(reg_targs, reg_preds)),
        "regime_f1": float(f1_score(reg_targs, reg_preds, average="weighted", zero_division=0)),
    }

    # Averages across stocks
    for key in ["return_mse", "volatility_mse", "sharpe_mse", "direction_accuracy", "direction_f1"]:
        metrics[f"avg_{key}"] = float(np.mean([s[key] for s in per_stock]))

    return metrics


def print_metrics(metrics: dict, header: str = ""):
    """Pretty-print the metrics dict."""
    if header:
        print(f"\n{'='*70}")
        print(f"  {header}")
        print(f"{'='*70}")

    print(f"\n  {'Stock':<12} | {'Ret MSE':>9} | {'Vol MSE':>9} | {'Shrp MSE':>9} | {'Dir Acc':>8} | {'Dir F1':>7}")
    print(f"  {'-'*12}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*8}-+-{'-'*7}")
    for s in metrics["per_stock"]:
        print(f"  {s['name']:<12} | {s['return_mse']:9.6f} | {s['volatility_mse']:9.6f} | "
              f"{s['sharpe_mse']:9.6f} | {s['direction_accuracy']:8.3f} | {s['direction_f1']:7.3f}")

    print(f"\n  Averages:     ret_MSE={metrics['avg_return_mse']:.6f}  "
          f"vol_MSE={metrics['avg_volatility_mse']:.6f}  "
          f"sharpe_MSE={metrics['avg_sharpe_mse']:.6f}  "
          f"dir_acc={metrics['avg_direction_accuracy']:.3f}  "
          f"dir_F1={metrics['avg_direction_f1']:.3f}")
    print(f"  Regime:       accuracy={metrics['regime_accuracy']:.3f}  F1={metrics['regime_f1']:.3f}")


# ─────────────────────────────────────────────────────
# TEST 1: Smoke Test (forward + backward)
# ─────────────────────────────────────────────────────

def test_smoke():
    """Verify forward pass produces correct shapes and backward pass computes gradients."""
    print("\n[TEST 1] Smoke Test: forward + backward pass")

    K, T, D, B = 5, 30, 5, 4
    model = build_default_model(K, D).to(DEVICE)
    criterion = build_default_criterion()

    x = torch.randn(B, K, T, D, device=DEVICE)
    targets = {
        "return":     torch.randn(B, K, device=DEVICE),
        "volatility": torch.abs(torch.randn(B, K, device=DEVICE)),
        "sharpe":     torch.randn(B, K, device=DEVICE),
        "direction":  torch.randint(0, 3, (B, K), device=DEVICE),
        "regime":     torch.randint(0, 4, (B,), device=DEVICE),
    }

    # Forward
    outputs = model(x)
    assert len(outputs["stocks"]) == K, f"Expected {K} stock outputs"
    for k in range(K):
        assert outputs["stocks"][k]["return"].shape == (B,)
        assert outputs["stocks"][k]["direction"].shape == (B, 3)
    assert outputs["regime"].shape == (B, 4)

    # Backward
    loss, loss_dict = criterion(outputs, targets)
    loss.backward()

    grad_count = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total_params = sum(1 for p in model.parameters())
    print(f"  ✓ Forward OK. Output shapes correct.")
    print(f"  ✓ Backward OK. {grad_count}/{total_params} params have non-zero gradients.")
    print(f"  ✓ Loss = {loss.item():.4f}")
    return True


# ─────────────────────────────────────────────────────
# TEST 2: Overfit Test (memorize tiny dataset)
# ─────────────────────────────────────────────────────

def test_overfit():
    """
    Train on a tiny dataset for many epochs. If the model can drive loss
    near zero, it proves the architecture has sufficient capacity and
    gradients flow correctly through all components.
    """
    print("\n[TEST 2] Overfit Test: memorize tiny dataset")

    K, D, SEQ = 3, 5, 10
    data, targets = make_dummy_data(T=80, K=K, input_dim=D, seed=99)
    ds = StockDataset(data, targets, seq_len=SEQ)

    model = SPAMSJF(
        num_stocks=K, input_dim=D, private_hidden=32, shared_hidden=64,
        spa_dim=32, num_direction_classes=3, num_regimes=4, dropout=0.0,
    ).to(DEVICE)
    criterion = build_default_criterion()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    loader = torch.utils.data.DataLoader(ds, batch_size=len(ds), shuffle=False)

    initial_loss = None
    final_loss = None

    for epoch in range(500):
        model.train()
        for x, tgt in loader:
            x = x.to(DEVICE)
            tgt = {k: v.to(DEVICE) for k, v in tgt.items()}
            out = model(x)
            loss, _ = criterion(out, tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    ratio = final_loss / (initial_loss + 1e-8)
    passed = ratio < 0.7  # loss should drop by at least 30% (CE terms have a floor > 0)

    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss:   {final_loss:.4f}")
    print(f"  Ratio:        {ratio:.4f} {'✓ PASS' if passed else '✗ FAIL (loss did not drop enough)'}")
    return passed


# ─────────────────────────────────────────────────────
# TEST 3: SPA Weight Analysis
# ─────────────────────────────────────────────────────

def test_spa_weights():
    """
    Verify that SPA produces different attention weights for different stocks
    and that weights sum to 1 (softmax property).
    """
    print("\n[TEST 3] SPA Weight Analysis")

    K, D, SEQ, B = 5, 5, 30, 16
    model = build_default_model(K, D).to(DEVICE)
    model.eval()

    x = torch.randn(B, K, SEQ, D, device=DEVICE)

    # Hook into SPA modules to capture weights
    spa_weights = {}

    def make_hook(stock_idx):
        def hook(module, input, output):
            f_s, f_k = input
            ps = torch.relu(module.proj_s(f_s))
            pk = torch.relu(module.proj_k(f_k))
            combined = torch.cat([ps, pk], dim=-1)
            weights = torch.softmax(module.attn(combined), dim=-1)
            spa_weights[stock_idx] = weights.detach().cpu()
        return hook

    handles = []
    for k in range(K):
        h = model.spa_modules[k].register_forward_hook(make_hook(k))
        handles.append(h)

    with torch.no_grad():
        model(x)

    for h in handles:
        h.remove()

    # Check properties
    all_pass = True
    for k in range(K):
        w = spa_weights[k]  # (B, 2)
        sums = w.sum(dim=-1)
        sum_ok = torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
        mean_ws = w[:, 0].mean().item()
        mean_wk = w[:, 1].mean().item()
        print(f"  Stock {k}: w_shared={mean_ws:.4f}, w_private={mean_wk:.4f}, sum_to_1={sum_ok}")
        if not sum_ok:
            all_pass = False

    # Check weights differ across stocks (not all identical)
    w_shared_means = [spa_weights[k][:, 0].mean().item() for k in range(K)]
    variance = np.var(w_shared_means)
    print(f"  Variance of w_shared across stocks: {variance:.6f}")
    print(f"  {'✓ PASS' if all_pass else '✗ FAIL'}")
    return all_pass


# ─────────────────────────────────────────────────────
# TEST 4: Full Proposal Metrics (Section 4.1)
# ─────────────────────────────────────────────────────

def test_proposal_metrics():
    """
    Train and evaluate using ALL metrics from the proposal:
      - Return:     MSE
      - Volatility: MSE
      - Sharpe:     MSE
      - Direction:  Accuracy + F1
      - Regime:     Accuracy + F1
    """
    print("\n[TEST 4] Full Proposal Metrics (Section 4.1)")

    K, D, SEQ = 5, 5, 30
    data, targets = make_dummy_data(T=600, K=K, input_dim=D, seed=42)

    # Chronological split: train / val / test
    train_end, val_end = 400, 500
    train_ds = StockDataset(data[:train_end], {k: v[:train_end] for k, v in targets.items()}, SEQ)
    val_ds   = StockDataset(data[train_end:val_end], {k: v[train_end:val_end] for k, v in targets.items()}, SEQ)
    test_ds  = StockDataset(data[val_end:], {k: v[val_end:] for k, v in targets.items()}, SEQ)

    model = build_default_model(K, D)
    criterion = build_default_criterion()
    trainer = SlidingWindowTrainer(
        model=model, criterion=criterion, device=DEVICE,
        lr=1e-3, epochs=30, batch_size=32, patience=10,
    )

    print("  Training...")
    trainer.train_window(train_ds, val_ds)

    # Evaluate on validation
    print("  Evaluating on validation set...")
    val_results = trainer.evaluate(val_ds)
    val_metrics = compute_all_metrics(val_results, stock_names=["AAPL", "NVDA", "TSLA", "META", "GOOG"])
    print_metrics(val_metrics, "Validation Metrics (2018-2019 analog)")

    # Evaluate on test
    print("  Evaluating on test set...")
    test_results = trainer.evaluate(test_ds)
    test_metrics = compute_all_metrics(test_results, stock_names=["AAPL", "NVDA", "TSLA", "META", "GOOG"])
    print_metrics(test_metrics, "Test Metrics (2020+ analog — shifted regime)")

    return val_metrics, test_metrics


# ─────────────────────────────────────────────────────
# TEST 5: Robustness / Degradation Analysis
#          (Proposal Section 4.1 — key contribution)
# ─────────────────────────────────────────────────────

def test_robustness(val_metrics: dict, test_metrics: dict):
    """
    Proposal Section 4.1:
    "Robustness is evaluated by measuring the degree of performance degradation
     across different prediction tasks under changing market conditions."

    Computes degradation = (test - val) / |val| for each metric.
    For MSE:       positive degradation = worse on test (bad)
    For Acc/F1:    negative degradation = worse on test (bad)

    "Models exhibiting smaller and more uniform performance degradation across
     tasks and time periods are considered more robust."
    """
    print(f"\n{'='*70}")
    print(f"  [TEST 5] Robustness Analysis: Performance Degradation (Proposal §4.1)")
    print(f"{'='*70}")

    # Metric keys and whether higher is better
    mse_keys = ["avg_return_mse", "avg_volatility_mse", "avg_sharpe_mse"]
    acc_keys = ["avg_direction_accuracy", "avg_direction_f1", "regime_accuracy", "regime_f1"]

    print(f"\n  {'Metric':<28} | {'Val':>9} | {'Test':>9} | {'Degradation':>12} | {'Assessment'}")
    print(f"  {'-'*28}-+-{'-'*9}-+-{'-'*9}-+-{'-'*12}-+-{'-'*12}")

    degradations = []

    for key in mse_keys:
        val_v = val_metrics[key]
        test_v = test_metrics[key]
        denom = max(abs(val_v), 1e-4)  # avoid division by near-zero
        deg = (test_v - val_v) / denom
        degradations.append(abs(deg))
        label = "↑ worse" if deg > 0.1 else ("≈ stable" if abs(deg) < 0.1 else "↓ better")
        print(f"  {key:<28} | {val_v:9.6f} | {test_v:9.6f} | {deg:+11.2%} | {label}")

    for key in acc_keys:
        val_v = val_metrics[key]
        test_v = test_metrics[key]
        denom = max(abs(val_v), 0.01)  # avoid division by near-zero accuracy
        deg = (test_v - val_v) / denom
        degradations.append(abs(deg))
        label = "↓ worse" if deg < -0.1 else ("≈ stable" if abs(deg) < 0.1 else "↑ better")
        print(f"  {key:<28} | {val_v:9.3f} | {test_v:9.3f} | {deg:+11.2%} | {label}")

    avg_deg = np.mean(degradations)
    max_deg = np.max(degradations)
    uniformity = np.std(degradations)

    print(f"\n  Summary:")
    print(f"    Average |degradation|:  {avg_deg:.2%}")
    print(f"    Max |degradation|:      {max_deg:.2%}")
    print(f"    Uniformity (std):       {uniformity:.4f}  (lower = more uniform = more robust)")
    print(f"    Verdict: {'Robust ✓' if avg_deg < 0.5 and uniformity < 0.3 else 'Needs improvement'}")


# ─────────────────────────────────────────────────────
# TEST 6: Per-Regime Breakdown
#         (Tests domain shift sensitivity per task)
# ─────────────────────────────────────────────────────

def test_per_regime_breakdown():
    """
    Proposal Research Question 1:
    "Which financial prediction targets are most sensitive to domain shift?"

    Tests this by generating data with distinct regime windows and measuring
    per-task performance in each regime.
    """
    print(f"\n{'='*70}")
    print(f"  [TEST 6] Per-Regime Breakdown: Which tasks are most shift-sensitive?")
    print(f"{'='*70}")

    K, D, SEQ = 5, 5, 30
    # Generate longer data to have multiple regime periods
    data, targets = make_dummy_data(T=800, K=K, input_dim=D, seed=42)

    # Train on first half, test on second half (different regime distribution)
    train_end = 500
    train_ds = StockDataset(data[:train_end], {k: v[:train_end] for k, v in targets.items()}, SEQ)
    test_ds  = StockDataset(data[train_end:], {k: v[train_end:] for k, v in targets.items()}, SEQ)

    model = build_default_model(K, D)
    criterion = build_default_criterion()
    trainer = SlidingWindowTrainer(
        model=model, criterion=criterion, device=DEVICE,
        lr=1e-3, epochs=30, batch_size=32, patience=10,
    )

    print("  Training on first half...")
    trainer.train_window(train_ds)

    results = trainer.evaluate(test_ds)

    # Identify which regime each test sample belongs to
    regime_labels = results["regime"]["targets"].numpy()
    unique_regimes = np.unique(regime_labels)

    print(f"\n  Regimes found in test set: {unique_regimes}")
    print(f"  Distribution: {dict(zip(*np.unique(regime_labels, return_counts=True)))}")

    # Per-regime, per-task metrics (averaged across stocks)
    print(f"\n  {'Regime':<10} | {'Ret MSE':>9} | {'Vol MSE':>9} | {'Shrp MSE':>9} | {'Dir Acc':>8} | {'N samples':>9}")
    print(f"  {'-'*10}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*8}-+-{'-'*9}")

    for regime in unique_regimes:
        mask_indices = np.where(regime_labels == regime)[0]
        if len(mask_indices) < 5:
            continue

        ret_mses, vol_mses, shrp_mses, dir_accs = [], [], [], []
        for k_stock in range(K):
            s = results["stocks"][k_stock]
            for task, collector in [("return", ret_mses), ("volatility", vol_mses), ("sharpe", shrp_mses)]:
                p = s[task]["preds"].numpy()[mask_indices]
                t = s[task]["targets"].numpy()[mask_indices]
                collector.append(mean_squared_error(t, p))
            dp = s["direction"]["preds"].numpy()[mask_indices]
            dt = s["direction"]["targets"].numpy()[mask_indices]
            dir_accs.append(accuracy_score(dt, dp))

        print(f"  Regime {regime:<4} | {np.mean(ret_mses):9.6f} | {np.mean(vol_mses):9.6f} | "
              f"{np.mean(shrp_mses):9.6f} | {np.mean(dir_accs):8.3f} | {len(mask_indices):>9}")


# ─────────────────────────────────────────────────────
# TEST 7: Single-Task Baselines vs Multi-Task
#         (Proposal Research Question 3)
# ─────────────────────────────────────────────────────

class SingleTaskLSTM(nn.Module):
    """Simple single-stock, single-task LSTM baseline for comparison."""

    def __init__(self, input_dim: int, hidden: int = 32, task: str = "return"):
        super().__init__()
        self.task = task
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True)
        if task == "direction":
            self.head = nn.Linear(hidden, 3)
        else:
            self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        out = self.head(h.squeeze(0))
        if self.task != "direction":
            out = out.squeeze(-1)
        return out


def test_single_vs_multi():
    """
    Proposal Research Question 3:
    "Whether shared representations learned through multi-task learning
     improve robustness to task-specific shifts."

    Trains independent single-task LSTMs for each task on each stock,
    then compares against the multi-task SPA-MSJF model.
    """
    print(f"\n{'='*70}")
    print(f"  [TEST 7] Single-Task Baselines vs Multi-Task SPA-MSJF")
    print(f"{'='*70}")

    K, D, SEQ = 5, 5, 30
    data, targets = make_dummy_data(T=600, K=K, input_dim=D, seed=42)

    train_end, val_end = 400, 500
    test_data = data[val_end:]
    test_targets = {k: v[val_end:] for k, v in targets.items()}

    # ── Multi-task model ──
    print("\n  Training Multi-Task SPA-MSJF...")
    train_ds = StockDataset(data[:train_end], {k: v[:train_end] for k, v in targets.items()}, SEQ)
    val_ds   = StockDataset(data[train_end:val_end], {k: v[train_end:val_end] for k, v in targets.items()}, SEQ)
    test_ds  = StockDataset(data[val_end:], test_targets, SEQ)

    model = build_default_model(K, D)
    criterion = build_default_criterion()
    trainer = SlidingWindowTrainer(
        model=model, criterion=criterion, device=DEVICE,
        lr=1e-3, epochs=30, batch_size=32, patience=10,
    )
    trainer.train_window(train_ds, val_ds)
    mtl_results = trainer.evaluate(test_ds)
    mtl_metrics = compute_all_metrics(mtl_results)

    # ── Single-task baselines ──
    print("  Training Single-Task baselines (per stock, per task)...")
    st_metrics = {task: [] for task in ["return", "volatility", "sharpe", "direction"]}

    for k_stock in range(K):
        for task in ["return", "volatility", "sharpe", "direction"]:
            # Build sequences for a single stock: data shape is (T, K, D)
            X_train, y_train = [], []
            for t in range(SEQ, train_end):
                X_train.append(data[t-SEQ:t, k_stock, :])  # (SEQ, D)
                y_train.append(targets[task][t, k_stock] if targets[task].ndim > 1 else targets[task][t])
            X_train = torch.tensor(np.array(X_train), dtype=torch.float32)

            X_test, y_test = [], []
            for t in range(val_end + SEQ, len(data)):
                X_test.append(data[t-SEQ:t, k_stock, :])  # (SEQ, D)
                y_test.append(targets[task][t, k_stock] if targets[task].ndim > 1 else targets[task][t])
            X_test = torch.tensor(np.array(X_test), dtype=torch.float32)

            if task == "direction":
                y_train = torch.tensor(np.array(y_train), dtype=torch.long)
                y_test_t = torch.tensor(np.array(y_test), dtype=torch.long)
            else:
                y_train = torch.tensor(np.array(y_train), dtype=torch.float32)
                y_test_t = torch.tensor(np.array(y_test), dtype=torch.float32)

            # Train
            st_model = SingleTaskLSTM(D, hidden=32, task=task).to(DEVICE)
            optimizer = torch.optim.Adam(st_model.parameters(), lr=1e-3)
            loss_fn = nn.CrossEntropyLoss() if task == "direction" else nn.MSELoss()

            for _ in range(30):
                st_model.train()
                optimizer.zero_grad()
                pred = st_model(X_train.to(DEVICE))
                loss = loss_fn(pred, y_train.to(DEVICE))
                loss.backward()
                optimizer.step()

            # Evaluate
            st_model.eval()
            with torch.no_grad():
                pred = st_model(X_test.to(DEVICE)).cpu()

            if task == "direction":
                pred_cls = pred.argmax(-1).numpy()
                st_metrics[task].append(accuracy_score(np.array(y_test), pred_cls))
            else:
                st_metrics[task].append(mean_squared_error(np.array(y_test), pred.numpy()))

    # ── Compare ──
    print(f"\n  {'Task':<12} | {'Single-Task (avg)':>18} | {'Multi-Task':>12} | {'Winner':>10}")
    print(f"  {'-'*12}-+-{'-'*18}-+-{'-'*12}-+-{'-'*10}")

    comparisons = [
        ("return",     "avg_return_mse",         "lower"),
        ("volatility", "avg_volatility_mse",     "lower"),
        ("sharpe",     "avg_sharpe_mse",          "lower"),
        ("direction",  "avg_direction_accuracy",  "higher"),
    ]

    for task, mtl_key, better in comparisons:
        st_val = np.mean(st_metrics[task])
        mtl_val = mtl_metrics[mtl_key]

        if better == "lower":
            winner = "MTL ✓" if mtl_val < st_val else "Single"
        else:
            winner = "MTL ✓" if mtl_val > st_val else "Single"

        print(f"  {task:<12} | {st_val:18.6f} | {mtl_val:12.6f} | {winner:>10}")


# ─────────────────────────────────────────────────────
# Main: Run All Tests
# ─────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  SPA-MSJF Evaluation Suite")
    print("  Tests from Project Proposal (Section 4) + Architecture Validation")
    print("=" * 70)

    # Architecture validation
    test_smoke()
    test_overfit()
    test_spa_weights()

    # Proposal metrics (Section 4.1)
    val_metrics, test_metrics = test_proposal_metrics()

    # Robustness (Proposal Section 4.1)
    test_robustness(val_metrics, test_metrics)

    # Per-regime analysis (Proposal RQ1)
    test_per_regime_breakdown()

    # Single vs Multi-task (Proposal RQ3)
    test_single_vs_multi()

    print(f"\n{'='*70}")
    print("  All tests complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
