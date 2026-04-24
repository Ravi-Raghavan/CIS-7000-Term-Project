# Financial Sentiment Analysis with BERT (Fine-Tuning Pipeline)

This project fine-tunes a transformer-based language model (BERT) to perform **sentiment analysis on financial news text**. The model classifies news into **negative, neutral, or positive sentiment**

---

## Objective

The goal is to build a financial sentiment classifier using:

- Hugging Face Transformers
- Synthetic financial news data
- BERT-based sequence classification

---

## Model Overview

We fine-tune:

- `bert-base-uncased`

---

## Dataset

Since labeled financial news is limited, this project generates a **synthetic dataset** using structured templates.

### Companies included:
- Apple (AAPL)
- NVIDIA (NVDA)
- Tesla (TSLA)
- Meta (META)
- Google (GOOGL)

### Example samples:

**Positive**
- “Shares of NVIDIA surged 12% after reporting earnings above expectations.”

**Negative**
- “Tesla fell 9% after missing revenue estimates and lowering outlook.”

**Neutral**
- “Meta shares were little changed after mixed quarterly results.”

Each sample includes:
- text
- label
- ticker symbol

---

## Data Pipeline

### 1. Synthetic Data Generation
Financial news is generated using templated sentences with randomized:
- companies
- percentage moves
- sentiment tone

### 2. Tokenization
Text is tokenized using BERT tokenizer with:
- max length = 192
- truncation + padding

---

## Training Setup

Key hyperparameters:

- Seed: 42 (reproducibility)
- Batch size: 16
- Learning rate: 5e-4
- Epochs: 5
- Train/test split: 75/25

---

## Evaluation

The model is evaluated using:

- Accuracy
- Weighted F1 score

---

## Workflow Summary

### Step 1: Environment Setup
- Loads environment variables
- Sets reproducibility seeds

### Step 2: Dataset Creation
- Generates synthetic financial news
- Maps labels → IDs

### Step 3: Tokenization
- Converts text → token IDs
- Pads sequences for batching

### Step 4: Model Initialization
- Loads pretrained BERT
- Adds classification head

### Step 5: Fine-Tuning
- Trains model on synthetic dataset
- Evaluates periodically

### Step 6: Evaluation
- Computes accuracy and F1 score
- Prints validation metrics

### Step 7: Inference
Runs sample predictions on unseen financial headlines.

---

## Output

The trained model is saved to: fin-sentiment-scratch

Contents include:
- Fine-tuned model weights
- Tokenizer files
- Training checkpoints

---

## Example Inference

Input:

- “Apex Motors shares plunged after weak demand”
- “Nimbus Cloud reported results in line with expectations”
- “TechNova surged after earnings beat and strong guidance”

Output:

- Negative
- Neutral
- Positive

---