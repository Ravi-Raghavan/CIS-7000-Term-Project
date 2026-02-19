# Create a Financial Sentiment Model using HuggingFace

## Fetch Environment Variable Parameters
from dotenv import load_dotenv
from pathlib import Path
import os
project_root = Path(__file__).resolve().parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

## Misc. Imports
import random
from typing import Dict, List, Any
import numpy as np
import torch

## Import Hugging Face Tooling
os.environ["USE_TF"] = "0"
from datasets import Dataset
import evaluate
from transformers import (
    AutoTokenizer,
    BertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)

## Config
SEED = 42
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "fin-sentiment-scratch")
MODEL_NAME = os.environ.get("MODEL_NAME", "bert-base-uncased")
TOKENIZER_NAME = os.environ.get("TOKENIZER_NAME", "bert-base-uncased")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "192"))
TEST_SIZE = float(os.environ.get("TEST_SIZE", "0.25"))
EPOCHS = int(os.environ.get("EPOCHS", "8"))
LR = float(os.environ.get("LR", "5e-4"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))

## Establish Labeling Scheme
LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {lbl: i for i, lbl in enumerate(LABELS)}
ID2LABEL = {i: lbl for lbl, i in LABEL2ID.items()}

## Synthetic finance news dataset
def build_sample_financial_news(n_per_class: int = 80):
    """
    Synthetic finance news using tech stock portfolio: Apple, NVIDIA, Google, Meta, and Tesla.
    """

    pos_templates = [
        "Shares of {co} surged {pct}% after reporting earnings above expectations and raising guidance.",
        "{co} announced strong quarterly revenue growth, sending the stock up {pct}%.",
        "Analysts upgraded {co} citing strong demand in key segments; shares climbed {pct}%.",
        "{co} posted record profits this quarter, pushing the stock higher by {pct}%.",
    ]

    neg_templates = [
        "Shares of {co} fell {pct}% after missing revenue estimates and lowering outlook.",
        "{co} warned of slowing demand and margin pressure; shares dropped {pct}%.",
        "Regulatory scrutiny intensified for {co}, dragging the stock down {pct}%.",
        "{co} reported weaker-than-expected earnings; investors pushed shares down {pct}%.",
    ]

    neu_templates = [
        "{co} shares were little changed as the company reported mixed quarterly results.",
        "{co} reiterated prior guidance during its investor call; the stock was flat.",
        "{co} announced a new product line but provided no updated financial outlook.",
        "{co} completed a previously announced acquisition; shares remained stable.",
    ]

    companies = [
        ("Apple", "AAPL"),
        ("Nvidia", "NVDA"),
        ("Tesla", "TSLA"),
        ("Meta", "META"),
        ("Google", "GOOGL"),
    ]

    samples = []

    def generate_samples(templates, label):
        for _ in range(n_per_class):
            co, ticker = random.choice(companies)
            pct = random.randint(2, 15)
            text = random.choice(templates).format(co=co, pct=pct)

            samples.append({
                "text": text,
                "label": label,
                "ticker": ticker
            })

    generate_samples(pos_templates, "positive")
    generate_samples(neu_templates, "neutral")
    generate_samples(neg_templates, "negative")

    random.shuffle(samples)
    return samples


## Create Hugging Face Dataset
def make_dataset() -> Dataset:
    rows = build_sample_financial_news(n_per_class=90)  # ~270 examples
    for r in rows:
        r["label_id"] = LABEL2ID[r["label"]]
    return Dataset.from_list(rows)

## Tokenization of Text Corpus
def tokenize_batch(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )

## Establish Evaluation Metrics and Computation Function
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    out = {}
    out.update(accuracy.compute(predictions=preds, references=labels))
    out.update(f1.compute(predictions=preds, references=labels, average="weighted"))
    return out

## Main Function for execution 
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ## Create Tokenizer
    print(f"Tokenizer vocab source: {TOKENIZER_NAME}")
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)

    ## Create Dataset
    ds = make_dataset()
    ds = ds.train_test_split(test_size=TEST_SIZE, seed=SEED)

    ## Tokenize
    tokenized = ds.map(lambda x: tokenize_batch(x, tokenizer), batched=True)
    tokenized = tokenized.rename_column("label_id", "labels")
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    ## Instantiate Model
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels = len(LABELS))

    ## Training Arguments
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=1, # Just put 1 here for testing
        save_strategy="steps",      # save checkpoints every N steps
        save_steps=100,             # save every 100 steps
        eval_strategy="steps",      # evaluate every N steps
        eval_steps=100,             # evaluate every 100 steps
        logging_strategy="steps",
        logging_steps=100,          # log every 100 steps
        report_to="none",
        full_determinism=True
    )

    ## Instantiate Hugging Face Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        compute_metrics=compute_metrics,
    )

    print("\n=== Training from scratch (random weights) ===")
    trainer.train()

    print("\n=== Final evaluation ===")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\n=== Saving fine-tuned model ===")
    trainer.save_model(OUTPUT_DIR)

    # Inference Sample-Run
    demo_texts = [
        "Apex Motors shares plunged after the company cut guidance due to weak demand.",
        "Nimbus Cloud reported results in line with expectations; shares were flat.",
        "TechNova surged after beating earnings estimates and announcing a major partnership.",
    ]

    # Get the device the model is on (i.e. CPU / CUDA / MPS)
    device = trainer.args.device

    # Put model in eval mode
    model.eval()

    # Tokenize Demo Texts
    enc = tokenizer(
        demo_texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    # Move Inputs to same Device
    enc = {k: v.to(device) for k, v in enc.items()}

    # Generate Predictions
    with torch.no_grad():
        outputs = model(**enc)
        pred_ids = outputs.logits.argmax(dim=-1).cpu().tolist()

    print("\n=== Demo predictions ===")
    for text, pid in zip(demo_texts, pred_ids):
        print(f"- {ID2LABEL[int(pid)]}  |  {text}")

if __name__ == "__main__":
    main()