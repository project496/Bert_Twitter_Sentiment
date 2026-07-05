"""
train_bert.py
--------------------------------------------------------------
Assignment 03: BERT Training on Twitter Sentiment Dataset
Course: Natural Language Processing Lab

Trains a BERT (bert-base-uncased) sequence classification model
on the Twitter US Airline Sentiment dataset (dataset/twitter_sentiment.csv)
with 3 classes: negative, neutral, positive.

After running this script you will get:
    saved_bert_model/           -> trained model + tokenizer (used by app.py)
    results/loss_curve.png
    results/accuracy_curve.png
    results/confusion_matrix.png
    results/class_distribution.png
    results/metrics_report.txt

Exact model name, split, epochs and batch size (Precision standard):
    Model      : bert-base-uncased
    Train/Test : 80% / 20% (stratified)
    Epochs     : 3
    Batch size : 16
    Max length : 64 tokens
    Optimizer  : AdamW, lr = 2e-5
--------------------------------------------------------------
"""

import os
import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

# ------------------------------------------------------------------
# 0. Config
# ------------------------------------------------------------------
SEED = 42
DATA_PATH = "dataset/twitter_sentiment.csv"
TEXT_COL = "text"
LABEL_COL = "airline_sentiment"          # negative / neutral / positive
MODEL_NAME = "bert-base-uncased"
SAVE_DIR = "saved_bert_model"
RESULTS_DIR = "results"
MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ------------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------------
def clean_tweet(text: str) -> str:
    """Light cleaning suited for tweets: strip urls, @mentions, extra spaces."""
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df = df[[TEXT_COL, LABEL_COL]].dropna()
    df[TEXT_COL] = df[TEXT_COL].apply(clean_tweet)
    df = df[df[LABEL_COL].isin(LABEL2ID.keys())]
    df["label"] = df[LABEL_COL].map(LABEL2ID)
    return df.reset_index(drop=True)


# ------------------------------------------------------------------
# 2. Dataset class
# ------------------------------------------------------------------
class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ------------------------------------------------------------------
# 3. Train / eval loops
# ------------------------------------------------------------------
def run_epoch(model, loader, optimizer=None, scheduler=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.set_grad_enabled(is_train):
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        total_loss += loss.item() * batch["labels"].size(0)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch["labels"].detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_labels, all_preds


def main():
    # ---------------- Data ----------------
    print("Loading dataset...")
    df = load_data()
    print(f"Total usable rows: {len(df)}")
    print(df["label"].value_counts())

    # Class distribution plot
    plt.figure(figsize=(6, 4))
    sns.countplot(x=LABEL_COL, data=df, order=["negative", "neutral", "positive"])
    plt.title("Class Distribution of Twitter Sentiment Dataset")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/class_distribution.png")
    plt.close()

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df[TEXT_COL].tolist(),
        df["label"].tolist(),
        test_size=0.2,
        random_state=SEED,
        stratify=df["label"].tolist(),
    )

    # ---------------- Tokenizer / Model ----------------
    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL2ID)
    ).to(DEVICE)

    train_ds = TweetDataset(train_texts, train_labels, tokenizer)
    test_ds = TweetDataset(test_texts, test_labels, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # ---------------- Training ----------------
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, optimizer, scheduler)
        val_loss, val_acc, val_true, val_pred = run_epoch(model, test_loader)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_accs.append(tr_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    # ---------------- Final evaluation ----------------
    _, _, y_true, y_pred = run_epoch(model, test_loader)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )
    report = classification_report(
        y_true, y_pred, target_names=list(LABEL2ID.keys())
    )

    print("\n=== Final Evaluation ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(report)

    with open(f"{RESULTS_DIR}/metrics_report.txt", "w") as f:
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Train/Test split: 80/20 (stratified)\n")
        f.write(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, LR: {LR}\n\n")
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall   : {recall:.4f}\n")
        f.write(f"F1-score : {f1:.4f}\n\n")
        f.write(report)

    # ---------------- Graphs ----------------
    epochs_range = range(1, EPOCHS + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs_range, train_losses, label="Train Loss", marker="o")
    plt.plot(epochs_range, val_losses, label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/loss_curve.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs_range, train_accs, label="Train Accuracy", marker="o")
    plt.plot(epochs_range, val_accs, label="Validation Accuracy", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/accuracy_curve.png")
    plt.close()

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(LABEL2ID.keys()),
        yticklabels=list(LABEL2ID.keys()),
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png")
    plt.close()

    # ---------------- Save model + tokenizer ----------------
    print(f"Saving model and tokenizer to '{SAVE_DIR}/' ...")
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print("Done. Model ready to be loaded from the PyQt GUI (app.py).")


if __name__ == "__main__":
    main()
