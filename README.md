# Multi-Task Tech Stock Price Prediction Under Domain Shift  
**CIS 7000: Deep Learning for Time Series**

## Overview
This project investigates **multi-task learning for financial time series prediction under domain shift**, focusing on large-cap technology stocks: **AAPL, GOOG, META, NVDA, TSLA**.

We study whether shared representations across:
- **stocks** (cross-sectional transfer)  
- **tasks** (Return, Volatility, Sharpe Ratio, Direction, Regime)  

can improve robustness under:
- **covariate shift** (changing input distributions over time)  
- **concept shift** (changing relationships, e.g., post-COVID dynamics)

Our core architecture builds on **Shared-Private Attention Multi-Stock Joint Forecasting (SPA-MSJF)**, combined with modern representation learning approaches.

---

## Tasks

### Regression Tasks
- Next-day Log Return  
- Volatility  
- Sharpe Ratio  

### Classification Tasks
- Direction (down / up)  
- Market regime (Bear / Bull)

---

## Dataset
- **Stocks**: AAPL, GOOG, META, NVDA, TSLA  
- **Features**:
  - OHLCV time series  
- **Splits**:
  - Chronological (train / validation / test)  

### Domain Shift Setup
- Cross-stock generalization (train on subset, test on unseen stock)
- Temporal shift (e.g., train on historical period, test on future period)

---

## Methodology

### 1. Multi-Task Architecture (SPA-MSJF)
- **Shared encoder**: Captures market-wide signals  
- **Private encoder**: Captures stock-specific behavior  
- **SPA module**: Learns weighted fusion of shared/private features  
- **Multi-head outputs**: One head per task  

Note: Refer to the README.md under the Multi-Task Learning folder for more implementation details!

---

### 2. Representation Learning

#### TS2Vec (Baseline)
- Self-supervised contrastive learning on time series  
- Produces **continuous embeddings**  
- Strong performance on downstream tasks  

#### Kronos Tokenizer (Alternative)
- Pretrained discrete tokenizer for financial time series  
- Converts sequences into:
  - **s1 tokens (coarse)**  
  - **s2 tokens (fine-grained)**  

- Requires:
  - Chunking (512 timesteps)  
  - Trainable embedding lookup tables (due to non-differentiable quantization)

---

### 3. Domain Adaptation (LoRA)
We adapt the pretrained Kronos tokenizer using **LoRA (Low-Rank Adaptation)**:
- Only **~0.1–1% parameters updated**
- Applied to attention layers  

**Training Objective:**
- Reconstruction loss (codebook + commitment loss)

**Pipeline:**
1. Stage 1: LoRA fine-tuning on training data  
2. Stage 2: Freeze encoder → train SPA-MSJF  

---

### 4. Training Improvements
Here is the list of improvements we made to the overall model architecture to get better results!
- Layer normalization  
- GELU activation  
- Label smoothing  
- Learning rate warmup + decay  
- Loss reweighting across tasks  

---

## Key Experiments

### 1. Baselines
- Naive (predict zero return)  
- ARIMA, GARCH  
- Linear/Ridge Regression  
- LSTM, GRU, Transformer  

**Finding:**  
No model beat the naive baseline for return prediction → strong evidence of **market efficiency + domain shift**

---

### 2. Multi-Task Learning
- Improved stability vs single-task models  
- Shared representations helped convergence  
- Limited gains on final predictive accuracy  

---

### 3. TS2Vec vs Kronos

| Property | TS2Vec | Kronos |
|----------|--------|--------|
| Representation | Continuous | Discrete (quantized) |
| Training | Contrastive | Tokenizer + embedding |
| Strength | Regression tasks | Transfer learning |
| Weakness | No explicit domain adaptation | Information bottleneck |

**Key Result:**
- TS2Vec significantly outperforms Kronos on:
  - Return prediction  
  - Sharpe ratio  
- Kronos suffers from **quantization loss**

---

### 4. LoRA Adaptation
- Improved **volatility prediction** on high-volatility stocks  
- Clean two-stage pipeline (no leakage)  
- Limited impact on classification tasks  

---

## Major Findings

### 1. Domain Shift is the Core Challenge
- Models trained on some stocks **fail to generalize** to others  
- Performance often drops to:
  - ~50% direction accuracy (random)  
  - naive-level RMSE  

---

### 2. Representation Matters More Than Model Size
- Increasing model capacity did **not** fix underfitting  
- Feature quality and training dynamics were more important  

---

### 3. Classification Collapse
- Direction and regime predictions converge to **majority class**
- Class weighting caused instability instead of improvement  
- Indicates **label imbalance + weak signal**

---

### 4. Discretization Bottleneck (Kronos)
- Mapping time series → tokens → embeddings:
  - loses fine-grained information  
- Explains large gap in regression performance vs TS2Vec  

---

### 5. Architecture ≠ Solution
- Hierarchical token routing (s1 vs s2) had **no measurable impact**  
- Suggests bottleneck is **data / labels / signal**, not architecture  

---

## Conclusions
- Naive baselines are extremely strong in finance  
- Data leakage can completely invalidate results  
- Multi-task learning improves stability, not necessarily accuracy  
- Better representations > larger models  
- Domain adaptation cannot fix weak signal problems  

--- 

## Project Structure
```text
├── Baseline: This folder contains the notebooks used to train the baseline models
├── Finance Data: Contains all the Stock Data (AAPL, GOOG, META, NVDA, TSLA)
├── Market Regime Modeling: Contains baseline models for Market Regime Prediction
└── Multi-Task Learning: Contains all the files used to train the SPA-MSJF Model
└── Multi-Task Learning: Kronos: Contains all the files used to train the SPA-MSJF Model (uses Kronos as TS2Vec for Pre-Processing)
└── Sharpe Ratio Modeling: Contains baseline models for Sharpe Ratio Prediction
└── Sentiment Analysis: Test Scripts to test methods to scrape financial news articles pertaining to a stock
└── Sentiment Modeling: Scripts to Fine-Tune BERT to classify sentiment on financial data. The sentiment approach was abandoned since scraping sentiment data proved to be very difficult
└── Scraped_data: Attempts to scrape from the Internet for Sentiment Analysis. 
```

Note: Please refer to the individual README files for Multi-Task Learning and Multi-Task Learning: Kronos for more implementation details! 