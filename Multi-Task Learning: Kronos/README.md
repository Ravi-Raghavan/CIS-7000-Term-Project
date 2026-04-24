# Shared-Private Attention Multi-Stock Joint Forecasting (SPA-MSJF + Kronos)

## Overview
This project implements a **Multi-Task Learning framework for Financial Time Series Forecasting** using a **Shared-Private architecture with Attention-Based Fusion (SPA)**, where all time-series inputs are first processed using **Kronos embeddings**.

The model performs joint prediction across multiple financial objectives. For each stock, it predicts:
- Returns (regression)
- Volatility (regression)
- Sharpe ratio (regression)
- Direction (classification)

In addition, it predicts the **market regime** (classification), which is shared across all stocks.

The key idea is to:
- Use **Kronos embeddings as a strong temporal representation layer**
- Separate **stock-specific vs global market signals**
- Fuse them using **Shared-Private Attention (SPA)** for downstream forecasting

---

## Architecture

### Input
- K stocks over time
- Kronos-encoded time series features
- Shape: **(B, K, T, Dₖ)**
  - B: Batch size  
  - K: Number of stocks  
  - T: Time steps  
  - Dₖ: Kronos embedding dimension  

---

### 0. Kronos Preprocessing Layer

Raw financial time series are first transformed using **Kronos embeddings**.

Kronos captures:
- Multi-scale temporal dependencies across historical sequences  
- Long-range temporal structure in time series dynamics  
- Generalizable temporal patterns learned from large-scale pretraining  

Output:
- $x_k^{(kronos)} \in \mathbb{R}^{T \times D_k}$ per stock

These embeddings serve as input to both shared and private encoders.

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

### 1. Kronos Feature Backbone
- Replaces raw feature engineering
- Provides rich temporal embeddings
- Improves robustness under regime shifts

### 2. Shared-Private Decomposition
- Private encoder → stock-specific noise + alpha signals  
- Shared encoder → market structure + correlations  

### 3. SPA Fusion
- Learns dynamic weighting between global and local signals  
- More flexible than concatenation or averaging  

### 4. Regime Isolation
- Market regime prediction depends only on shared encoder  
- Prevents stock-level noise leakage  

---

## Data Pipeline

1. Raw financial time series  
2. Kronos embedding generation  
3. Construct multi-stock tensors  
4. Build multi-task labels:
   - Returns  
   - Volatility  
   - Sharpe ratio  
   - Direction  
   - Market regime  
5. Train SPA-MSJF model  

---