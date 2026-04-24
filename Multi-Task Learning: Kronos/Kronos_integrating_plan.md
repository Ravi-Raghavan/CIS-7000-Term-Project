# Plan: Integrate Kronos Tokenizer into Multi-Task Learning Pipeline

## Context

The existing SPA-MSJF model uses TS2Vec (self-supervised contrastive encoder) to transform
5 engineered financial features into 320-dim dense embeddings before feeding into the
Shared-Private Attention Transformer model. The goal is to build a new preprocessing module
in `preprocessing_MTL/` that **replaces TS2Vec** with the Kronos discrete tokenizer,
while keeping the downstream SPA-MSJF model, outputs, and evaluation metrics identical
for a clean research comparison.

**Why replace (not cooperate)?** Both serve the same role: temporal encoder → fixed-dim
representation fed to downstream decoders. Replacing gives a clean comparison; cooperating
would require model architecture changes that break metric comparability.

---

## Key Clarifications

### Which Kronos model?
There are 2 tokenizer sizes (not 4). The 4 "variants" are **predictor sizes** (mini/small/base/large).

| Tokenizer | Context | Used by | Our choice |
|-----------|---------|---------|------------|
| `Kronos-Tokenizer-2k` | 2048 | Kronos-mini | — |
| `Kronos-Tokenizer-base` | 512 | Kronos-small/base/large | **✓ Use this** |

**Use `Kronos-Tokenizer-base`** — 512-step context covers our 60-day windows; used by 3/4
predictors (better pretrained quality). For precomputing full T≈3700 series, process in
chunks of ≤512 steps and concatenate outputs.

Both tokenizers output: `s1_tokens` and `s2_tokens`, each `(B, T)` int64, vocab size = 1024.
Token format is identical across all variants (s1_bits=10, s2_bits=10).

**We do NOT use the Kronos predictor at all.** We only use the tokenizer to get discrete
token indices, then build our own `nn.Embedding` layers trained from scratch. This removes
the dependency on `HierarchicalEmbedding` from Kronos and simplifies implementation.

### Raw prices → log_return (two independent pipelines)
This is the critical architectural point — these are completely separate:

```
Raw OHLCV ──→ Kronos tokenizer ──→ 320D repr ──→ SPA-MSJF ──→  predicted log_return
    │                                                               (model output)
    └──→ compute_targets() ──────────────────────────────────────→ ground-truth log_return
                                                                    (training label)
```

The encoder INPUT is Kronos representations. The prediction TARGETS are computed
independently from closing prices in `compute_targets()` — **this function is unchanged**.
The model learns to map Kronos representations → log_return, supervised by ground-truth
targets. The encoder never needs to "output" log_return directly.

### Training compute: local vs Google Cloud
- **Local is sufficient** for the base experiment: dataset is tiny (~2800 windows),
  Kronos is frozen (precomputed once), trainable portion is similar size to original model
- **Save cloud credits for**: hyperparameter tuning (tune_kronos.py, 39+ trials) or
  fine-tuning the Kronos tokenizer on NASDAQ stocks

---

## Architecture

```
Raw OHLCV CSV
    ↓ build_kronos_raw_input()
(K=5, T≈3700, 6) float32  [log1p-scaled, clipped ±5]   ← raw prices + amount=O×V
    ↓ KronosTokenizer.encode(chunk, half=True) [FROZEN]  ← chunks of ≤512 steps
s1_tokens (K, T) int64  +  s2_tokens (K, T) int64       ← precomputed once
    ↓  stored in StockDataset, loaded per window
(B=32, K=5, T=60) s1  +  (B=32, K=5, T=60) s2  [long]
    ↓ KronosEncoderWrapper.forward() [TRAINABLE]
    nn.Embedding(1024, 256) for s1
    nn.Embedding(1024, 256) for s2
    nn.Linear(512, 320) fusion
(B=32, K=5, T=60, 320) float32                          ← IDENTICAL interface to TS2Vec
    ↓ SPA-MSJF model (ZERO CHANGES to spa_msjf.py)
return / vol / sharpe / direction / regime outputs
```

---

## Files to Create in `preprocessing_MTL/`

```
preprocessing_MTL/
├── Kronos_integrating_plan.md    ← this file
├── kronos_tokenizer_wrapper.py   ← NEW: frozen tokenizer + trainable embedding/adapter
├── build_data_kronos.py          ← ADAPTED from Multi-Task Learning/build_data.py
├── train_kronos.py               ← ADAPTED from Multi-Task Learning/train.py
├── requirements.txt              ← NEW: Kronos-specific deps
└── Kronos/                       ← git clone shiyu-coder/Kronos
    └── model/
        ├── kronos.py             ← KronosTokenizer class
        └── module.py             ← supporting modules
```

`spa_msjf.py` and `losses.py` are imported via `sys.path` from `../Multi-Task Learning/`
— zero copies, zero changes.

---

## Step-by-Step Implementation

### Step 1: Install Kronos

```bash
cd "C:\Users\Debby\Desktop\7000 project\CIS-7000-Term-Project\preprocessing_MTL"
git clone https://github.com/shiyu-coder/Kronos.git
pip install -r Kronos/requirements.txt
# installs: einops, huggingface_hub, safetensors, tqdm
```

### Step 2: `kronos_tokenizer_wrapper.py`

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Kronos"))
from model.kronos import KronosTokenizer

KRONOS_EMBED_DIM = 256    # per-token embedding dim
KRONOS_OUTPUT_DIM = 320   # must match TS2VEC_OUTPUT_DIM in build_data.py
VOCAB_SIZE = 1024         # 2^10 for both s1 and s2

class KronosEncoderWrapper(nn.Module):
    def __init__(self, device, kronos_model_name="NeoQuasar/Kronos-Tokenizer-base"):
        # Frozen tokenizer — loaded once
        self.tokenizer = KronosTokenizer.from_pretrained(kronos_model_name)
        self.tokenizer.eval()
        for p in self.tokenizer.parameters():
            p.requires_grad_(False)

        # Trainable: our own embedding lookup (NOT Kronos HierarchicalEmbedding)
        self.emb_s1  = nn.Embedding(VOCAB_SIZE, KRONOS_EMBED_DIM)
        self.emb_s2  = nn.Embedding(VOCAB_SIZE, KRONOS_EMBED_DIM)
        self.fusion  = nn.Linear(KRONOS_EMBED_DIM * 2, KRONOS_OUTPUT_DIM)

    def precompute_tokens(self, raw_ohlcva: np.ndarray, chunk_size=512):
        """
        Runs frozen tokenizer over full time series.
        Args:
            raw_ohlcva: (K, T, 6) float32 — log1p-scaled, clipped ±5
            chunk_size: max timesteps per forward pass (≤512 for Tokenizer-base)
        Returns:
            s1_tokens: (K, T) int64 numpy
            s2_tokens: (K, T) int64 numpy
        """
        # Process each stock; chunk along T dimension; concatenate results

    def forward(self, s1: Tensor, s2: Tensor) -> Tensor:
        """
        Trainable path: token indices → 320D embeddings.
        Args:
            s1: (B, K, T) long
            s2: (B, K, T) long
        Returns:
            (B, K, T, 320) float32
        """
        B, K, T = s1.shape
        e1 = self.emb_s1(s1.view(B*K, T))      # (B*K, T, 256)
        e2 = self.emb_s2(s2.view(B*K, T))      # (B*K, T, 256)
        out = self.fusion(torch.cat([e1, e2], dim=-1))  # (B*K, T, 320)
        return out.view(B, K, T, 320)
```

Trainable parameter count: 2×(1024×256) + (512×320 + 320) = ~688K params (small).

### Step 3: `build_data_kronos.py`

Base on `Multi-Task Learning/build_data.py`. Detailed changes:

**ADD** `build_kronos_raw_input(frames)`:
```python
def build_kronos_raw_input(frames):
    """Returns (K, T, 6): log1p-scaled [open, high, low, close, volume, amount]"""
    for each ticker in STOCKS:
        amount = open_price × volume   # proxy for trading turnover
        stack [o, h, l, c, v, amt] per timestep
    apply np.log1p() to all 6 columns
    return np.clip(data, -5.0, 5.0)   # matches Kronos's internal clip
```

**REPLACE** TS2Vec encoding block (build_data.py lines ~22-23, ~350-360):
```python
# OLD (removed):
_ts2vec_model = TS2Vec(input_dims=5, ...)
_ts2vec_model.load("trained_ts2vec.pth")
# ... encode loop producing (K, T, 320)

# NEW:
wrapper = KronosEncoderWrapper(device)
kronos_raw = build_kronos_raw_input(frames)        # (K, T, 6) raw prices
s1_tokens, s2_tokens = wrapper.precompute_tokens(kronos_raw)  # (K, T) int64 each
# Store in StockDataset instead of float32 encoded arrays
```

**MODIFY** `StockDataset`:
- `__init__`: `self.s1 = torch.tensor(s1_tokens, dtype=torch.long)` (K, T)
              `self.s2 = torch.tensor(s2_tokens, dtype=torch.long)` (K, T)
- `__getitem__`: return `(s1[:, t-seq_len:t], s2[:, t-seq_len:t])` + same targets dict
  - Each token window: `(K=5, T=60)` long

**KEEP IDENTICAL** (copy verbatim):
- `load_and_align()`, `compute_targets()`, `build_arrays()` (scalers + target normalization)
- `split_indices()`, target normalization statistics (ret_std, vol_std, sharpe stats)
- DataLoader construction (batch_size, shuffle, num_workers)

**ADD** `wrapper` to returned `meta` dict: `meta["wrapper"] = wrapper`

### Step 4: `train_kronos.py`

Base on `Multi-Task Learning/train.py`. Key changes:

**Add** to sys.path at top:
```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../Multi-Task Learning"))
from spa_msjf import SPAMSJF, JointLoss
```

**Modify** forward pass (both train loop and evaluate()):
```python
# Each batch: x is a tuple (s1, s2) instead of float tensor
s1, s2 = x                    # each (B, K=5, T=60) long
x_emb = wrapper(s1, s2)       # (B, K, T, 320) — trainable embedding + fusion
outputs = model(x_emb)        # SPA-MSJF: completely unchanged
```

**Modify** optimizer — include wrapper's trainable params:
```python
optimizer = AdamW(
    list(model.parameters()) + list(wrapper.parameters()),
    lr=lr, weight_decay=5e-4
)
# Note: wrapper.tokenizer params have requires_grad=False, AdamW ignores them
```

**Add** to start of each epoch:
```python
wrapper.tokenizer.eval()  # keep frozen tokenizer in eval mode even during training
```

**Modify** checkpoint save/load:
```python
# Save:
torch.save({"model": model.state_dict(), "wrapper": wrapper.state_dict()}, path)
# Load for evaluation:
ckpt = torch.load(path); model.load_state_dict(ckpt["model"])
```

**KEEP IDENTICAL** (copy verbatim):
- `evaluate()` function — same MSE/accuracy metrics, same composite score
- Scheduler (cosine with warmup), early stopping, gradient clipping
- All printing and logging

### Step 5: `requirements.txt`

```
einops>=0.7.0
huggingface_hub>=0.23.0
safetensors>=0.4.0
tqdm>=4.65.0
# Kronos: git clone https://github.com/shiyu-coder/Kronos.git into this directory
# PyTorch: install separately per CUDA version (requires torch>=2.0.0)
```

---

## Tensor Shapes at Each Stage

| Stage | Shape | Dtype | Notes |
|-------|-------|-------|-------|
| Raw CSV | `(T≈3700, 5)` | float64 | OHLCV per stock |
| After alignment | `(K=5, T, 5)` | float64 | common dates |
| Kronos raw input | `(K=5, T, 6)` | float32 | log1p + clip ±5; 6th = open×vol |
| Chunks for tokenizer | `(1, ≤512, 6)` | float32 | one stock, one chunk |
| s1/s2 tokens (precomputed) | `(K=5, T)` each | int64 | stored in StockDataset |
| Dataset item tokens | `(K=5, T=60)` each | long | one 60-day window |
| Batch tokens | `(B=32, K=5, T=60)` each | long | from DataLoader |
| After emb_s1 + emb_s2 | `(B*K=160, T=60, 256)` each | float32 | inside wrapper.forward |
| After fusion Linear | `(B=32, K=5, T=60, 320)` | float32 | **identical to TS2Vec output** |
| PrivateDecoder out | `(B=32, 64)` per stock | float32 | unchanged |
| SharedDecoder out | `(B=32, 128)` | float32 | unchanged |
| All task outputs | same as original | float32 | **IDENTICAL** |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Kronos expects 6 features, we have 5 OHLCV | Compute `amount = open × volume`; apply log1p |
| Kronos pretrained on Chinese A-shares | log1p + clip ±5 normalizes scale; zero-shot transfer is the experiment |
| T≈3700 > 512 context window | Process in chunks of ≤512 steps; concatenate token outputs |
| HuggingFace network access needed | Download once, cache locally; set `HF_HUB_OFFLINE=1` after |
| Kronos not pip-installable (no setup.py) | git clone + sys.path.insert pattern |
| StockDataset now returns (s1,s2) tuples | Isolated to `build_data_kronos.py`; original `build_data.py` untouched |
| Frozen tokenizer in train mode accidentally | Call `wrapper.tokenizer.eval()` at top of each epoch |

---

## Maintaining Comparability

- `compute_targets()`: copied verbatim — same log_return, vol, sharpe, direction, regime
- Target normalization (ret_std, vol_std, sharpe_mean/std): computed from same train split
- Date splits (80/10/10): identical chronological split on same aligned date intersection
- `evaluate()`: copied verbatim — same MSE, accuracy, composite score
- `spa_msjf.py` / `losses.py`: imported unchanged via sys.path — zero modifications
- `input_dim=320` passed to model: same value, different source

Metrics will differ in absolute value (that IS the research finding), but all metric definitions,
evaluation code, and targets are identical.

---

## What to Train Where

| Phase | Where to train | Why |
|-------|---------------|-----|
| Base experiment (frozen Kronos + adapter) | Local | Small dataset, trainable params similar to original |
| Hyperparameter tuning (tune_kronos.py) | Google Cloud | 39+ trials; parallelizable |
| Optional: fine-tune Kronos tokenizer | Google Cloud | Larger model, longer training |

---

## Implementation Sequence

1. `git clone https://github.com/shiyu-coder/Kronos.git preprocessing_MTL/Kronos`
2. `pip install -r preprocessing_MTL/Kronos/requirements.txt`
3. Write `kronos_tokenizer_wrapper.py` — `KronosEncoderWrapper` with `precompute_tokens()` + `forward()`
4. Write `build_data_kronos.py` — add `build_kronos_raw_input()`, modify dataset to store int64 tokens, keep all target logic identical
5. Write `train_kronos.py` — adapt forward pass, optimizer, and checkpointing
6. Write `requirements.txt`
7. Smoke test: `batch_size=2, epochs=1` — verify shapes end-to-end
8. Verify: `s1_tokens.min()=0, s1_tokens.max()≤1023`; adapter has `requires_grad=True`, tokenizer has `requires_grad=False`
9. Full training → `best_spa_msjf_kronos.pt`
10. Compare test metrics with `best_tuned_spa_msjf.pt` using same `evaluate()` function

---

## Critical Source Files

- `Multi-Task Learning/build_data.py` — base for `build_data_kronos.py`
- `Multi-Task Learning/train.py` — base for `train_kronos.py`
- `Multi-Task Learning/spa_msjf.py` — imported unchanged (sys.path)
- `Multi-Task Learning/losses.py` — imported unchanged (sys.path)
- `preprocessing_MTL/Kronos/model/kronos.py` — `KronosTokenizer` class
- `preprocessing_MTL/Kronos/model/module.py` — supporting modules (BSQuantizer, etc.)
