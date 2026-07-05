# Assignment 03 — BERT Twitter Sentiment Analysis with PyQt GUI

Natural Language Processing Lab — Shifa Tameer-e-Millat University, Islamabad

## Dataset Source

`dataset/twitter_sentiment.csv` is the **Twitter US Airline Sentiment**
dataset (`Tweets.csv`), originally released on Kaggle / Crowdflower's Data
for Everyone library:
https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment

- Text column: `text`
- Label column: `airline_sentiment` (`negative`, `neutral`, `positive`)
- 14,640 tweets total.

## Project Structure

```
assignment_03_bert_twitter_sentiment/
|-- train_bert.py          # trains and saves the BERT model
|-- app.py                 # PyQt5 GUI application
|-- requirements.txt
|-- README.md
|-- dataset/
|   |-- twitter_sentiment.csv
|-- saved_bert_model/      # created after running train_bert.py
|-- results/               # created after running train_bert.py
|   |-- loss_curve.png
|   |-- accuracy_curve.png
|   |-- confusion_matrix.png
|   |-- class_distribution.png
|   |-- metrics_report.txt
|-- screenshots/
|   |-- gui_home.png
|   |-- prediction_result.png
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

A GPU is strongly recommended for training but not required (CPU also works,
just slower).

## Step 1 — Train the model

```bash
python train_bert.py
```

This will:
1. Load and lightly clean `dataset/twitter_sentiment.csv`.
2. Split into 80% train / 20% test (stratified).
3. Fine-tune `bert-base-uncased` for sentiment classification
   (negative / neutral / positive).
4. Evaluate with accuracy, precision, recall, F1-score, and confusion matrix.
5. Save all result graphs to `results/`.
6. Save the trained model + tokenizer to `saved_bert_model/`.

**Training configuration:** `bert-base-uncased`, 3 epochs, batch size 16,
max sequence length 64, learning rate 2e-5, AdamW optimizer.

## Step 2 — Run the GUI

```bash
python app.py
```

1. Click **Load Model** and select the `saved_bert_model/` folder.
2. Click **Load Dataset** and select `dataset/twitter_sentiment.csv`
   (or any CSV with a `text`/`tweet` column).
3. Click any tweet in the list to see its predicted sentiment.
4. Or type your own sentence in the manual box and click **Predict**.

## Results

See `results/metrics_report.txt` and the graphs in `results/` after training
for accuracy, precision, recall, F1-score, and the confusion matrix.

## Critical Thinking Notes (Paul's Standards)

- **Clarity:** Dataset has one text column (`text`) and one label column
  (`airline_sentiment`) with 3 classes.
- **Accuracy:** Metrics are reported directly from `results/metrics_report.txt`
  without cherry-picking.
- **Precision:** Model = `bert-base-uncased`, split = 80/20 stratified,
  epochs = 3, batch size = 16.
- **Relevance:** Loss/accuracy curves show learning progress; the confusion
  matrix shows per-class errors; the class distribution graph shows the
  dataset is imbalanced (majority negative).
- **Depth:** Because the dataset skews heavily negative, the model tends to
  predict negative more confidently than neutral/positive; short or
  sarcastic tweets are the most common error source.
- **Logic:** `train_bert.py` saves the model to `saved_bert_model/`, and
  `app.py` loads that exact folder via `from_pretrained()`, keeping
  train → save → load → predict connected end to end.
- **Fairness:** Class imbalance (mostly negative tweets), airline-specific
  slang, and sarcasm are potential sources of bias/error.
