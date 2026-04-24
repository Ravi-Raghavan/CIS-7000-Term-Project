# Shared-Private Attention Multi-Stock Joint Forecasting (SPA-MSJF)

## Overview
This project implements a **Multi-Task Learning framework for Financial Time Series Forecasting** using a **Shared-Private architecture with Attention-Based Fusion (SPA)**.

The model performs joint multi-task prediction across several financial objectives. For each stock, it predicts:
- Returns (regression)  
- Volatility (regression)  
- Sharpe ratio (regression)  
- Direction (classification)  

In addition, it predicts the overall market regime (classification), which is shared across all stocks rather than being stock-specific. 

Overall, the key idea is to disentangle stock-specific signals (via a Private Encoder) from shared market-wide signals (via a Shared Encoder), and fuse them through a Shared-Private Attention (SPA) mechanism. For the stock-level tasks (returns, volatility, Sharpe ratio, and direction), the model uses a combination of private (idiosyncratic) and shared (market-driven) representations, with SPA dynamically weighting their contributions for each prediction. For the market regime classification, the model relies primarily on the shared encoder, since this task reflects global market conditions that are common across all stocks.

---

## Architecture

### Input
- Multiple stocks (K stocks)
- Time series input of shape **(B, K, T, D)**
- B: Batch Size 
- K: Number of Stocks in Portfolio 
- T: Length of Time Series
- D: Dimension of Input Features

---

### 1. Private Transformer (Stock-Specific)
- Processes each stock individually  
- Learns **stock-specific (idiosyncratic) features**  
- Output:  
  - $f_k$: private representation per stock  

---

### 2. Shared Transformer (Global)
- Processes all stocks jointly (Input is Reshaped to (B, T, K * D)) 
- Learns **shared/global market dynamics**  
- Runs in **parallel and independently** from private transformers  
- Output:  
  - $f_s$: shared representation  

---

### 3. Shared-Private Attention (SPA)

We fuse shared and private representations using a learned attention mechanism.

#### Step 1: Projection into a Common Space
Both shared and private features are projected into the same dimension:

$$
p_s = \mathrm{ReLU}(W_s f_s), \quad
p_k = \mathrm{ReLU}(W_k f_k)
$$

- $f_s$: shared representation  
- $f_k$: private representation  

---

#### Step 2: Attention Weight Computation

We concatenate the projected features and compute attention weights:

$$
z = [p_s \, \| \, p_k]
$$

$$
[w_s, w_k] = \mathrm{softmax}(W_a z)
$$

- $w_s, w_k \in (0,1)$, with $w_s + w_k = 1$

---

#### Step 3: Shared-Private Fusion

Final representation:

$$
f_{\text{combined}} = w_s \cdot p_s + w_k \cdot p_k
$$

---

#### Intuition

- $w_s$: importance of global (shared) features  
- $w_k$: importance of stock-specific (private) features  
- The model dynamically balances both per sample

---

### 4. Task Heads (Multi-Task Learning)

#### Regression Heads (use fused representation $f'$)
- Return prediction (MSE / Huber loss)  
- Volatility prediction (MSE / Huber loss)  
- Sharpe ratio prediction (MSE / Huber loss)  

#### Classification Heads
- Direction prediction (binary classification, CE loss) → uses $f'$
- Regime classification (multi-class, CE loss) → uses **shared representation $f_s$ only**

---

## Key Design Choices

- **Shared vs Private Decomposition**
  - Private transformer captures stock-level noise and signals  
  - Shared transformer captures market-wide structure  

- **Decoupled Regime Modeling**
  - Regime classification depends only on global features $f_s$
  - Avoids contamination from stock-specific noise  

- **Attention-Based Fusion (SPA)**
  - Learns how much to rely on shared vs private signals  
  - More flexible than simple concatenation or averaging  

---

## Repository Structure

### Architecture Diagram
- `architecture.png`

### Saved Model Weights (PyTorch Tensors) 
- `best_spa_msjf.pt`
- `best_tuned_spa_msjf.pt`

### TS2Vec PreProcessing
- `datautils.py` – Data preprocessing  
- `dilated_conv.py` – Temporal convolutions  
- `encoder.py` – Encoder backbone  
- `losses.py` – Contrastive loss  
- `ts2vec.py` – TS2Vec model  
- `ts2vec_experimentation.py` – Pretraining script (used to generate encodings of time series prior to SPA-MSJF)

### SPA-MSJF Files
- `spa_msjf.py` – Main architecture (shared/private + SPA + heads).
- `spa_msjf_lstm.py` – Older LSTM baseline (Original Version where Transformers were LSTM modules)
- `build_data.py` – Feature + multi-task label construction  
- `train.py` – Training loop  
- `tune.py` – Hyperparameter tuning 
- `test_evaluation.py` - Generate Evaluation Results for SPA-MSJF 

### Misc.
- `utils.py`: Extra Utility Functions
---

## Data Pipeline
1. Prepare multi-stock time series  
2. Construct:
   - Input sequences  
   - Multi-task targets:
     - Returns  
     - Volatility  
     - Sharpe  
     - Direction  
     - Regime  
3. Pretrain TS2Vec embeddings  
4. Train SPA-MSJF model  

---

## Training

### Train TS2Vec (Optional)
```bash
python ts2vec_experimentation.py
```

### Train SPA-MSJF
```bash
python train.py
```

### Hyperparameter Tuning
```bash
python tune.py
```

---

## Multi-Task Objective

The total loss is a weighted combination:

$$
\mathcal{L} =
\lambda_1 \mathcal{L}_{return} +
\lambda_2 \mathcal{L}_{volatility} +
\lambda_3 \mathcal{L}_{sharpe} +
\lambda_4 \mathcal{L}_{direction} +
\lambda_5 \mathcal{L}_{regime}
$$

- Regression: MSE / Huber  
- Classification: Cross-Entropy  

---

## Domain Shift Setup

- **Cross-stock generalization**  
  Train on subset of stocks, test on unseen stocks  

- **Temporal shift**  
  Train on past, evaluate on future  

- **Market regime shift**  
  Test across different volatility/market conditions  

---

## Key Features
- Multi-task forecasting across financial objectives  
- Shared-private transformer architecture  
- Attention-based feature fusion (SPA)  
- Explicit separation of global vs local signals  
- Robustness to domain shift  

---