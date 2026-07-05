"""
app.py
--------------------------------------------------------------
Assignment 03: PyQt GUI for BERT Twitter Sentiment Analysis
Course: Natural Language Processing Lab

Run with:
    python app.py

Workflow:
    1. Click "Load Model" -> choose the saved_bert_model/ folder
       created by train_bert.py.
    2. Click "Load Dataset" -> choose a Twitter sentiment CSV file
       (e.g. dataset/twitter_sentiment.csv).
    3. Click any tweet in the list -> predicted sentiment is shown.
    4. Or type a sentence in the manual box and click "Predict".
--------------------------------------------------------------
"""

import sys
import pandas as pd
import torch

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QListWidget,
    QLineEdit,
    QFileDialog,
    QMessageBox,
    QGroupBox,
)
from PyQt5.QtCore import Qt

from transformers import BertTokenizerFast, BertForSequenceClassification

ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}
TEXT_CANDIDATE_COLS = ["text", "tweet", "Tweet", "Text"]
LABEL_CANDIDATE_COLS = ["airline_sentiment", "sentiment", "label"]
MAX_LEN = 64


class SentimentApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BERT Twitter Sentiment Analysis")
        self.resize(800, 600)

        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.df = None
        self.text_col = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        main_layout = QVBoxLayout()

        # --- Top buttons ---
        top_box = QGroupBox("Setup")
        top_layout = QHBoxLayout()
        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.load_model)
        self.load_dataset_btn = QPushButton("Load Dataset")
        self.load_dataset_btn.clicked.connect(self.load_dataset)
        top_layout.addWidget(self.load_model_btn)
        top_layout.addWidget(self.load_dataset_btn)
        top_box.setLayout(top_layout)
        main_layout.addWidget(top_box)

        # --- Status label ---
        self.status_label = QLabel("Status: No model or dataset loaded yet.")
        self.status_label.setStyleSheet("color: #444; font-style: italic;")
        main_layout.addWidget(self.status_label)

        # --- Dataset view ---
        dataset_box = QGroupBox("Dataset (click a tweet to predict its sentiment)")
        dataset_layout = QVBoxLayout()
        self.tweet_list = QListWidget()
        self.tweet_list.itemClicked.connect(self.predict_selected_tweet)
        dataset_layout.addWidget(self.tweet_list)
        dataset_box.setLayout(dataset_layout)
        main_layout.addWidget(dataset_box, stretch=1)

        # --- Manual input ---
        manual_box = QGroupBox("Manual Sentence Prediction")
        manual_layout = QHBoxLayout()
        self.manual_input = QLineEdit()
        self.manual_input.setPlaceholderText("Type a sentence here...")
        self.predict_btn = QPushButton("Predict")
        self.predict_btn.clicked.connect(self.predict_manual_text)
        manual_layout.addWidget(self.manual_input)
        manual_layout.addWidget(self.predict_btn)
        manual_box.setLayout(manual_layout)
        main_layout.addWidget(manual_box)

        # --- Prediction result ---
        result_box = QGroupBox("Prediction Result")
        result_layout = QVBoxLayout()
        self.result_label = QLabel("Predicted Sentiment: -")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet(
            "font-size: 20px; font-weight: bold; padding: 12px;"
        )
        result_layout.addWidget(self.result_label)
        result_box.setLayout(result_layout)
        main_layout.addWidget(result_box)

        central.setLayout(main_layout)
        self.setCentralWidget(central)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def load_model(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select saved_bert_model folder"
        )
        if not folder:
            return
        try:
            self.tokenizer = BertTokenizerFast.from_pretrained(folder)
            self.model = BertForSequenceClassification.from_pretrained(folder)
            self.model.to(self.device)
            self.model.eval()
            self.status_label.setText(f"Status: Model loaded from '{folder}'.")
            QMessageBox.information(self, "Model Loaded", "BERT model loaded successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Model", str(e))

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------
    def load_dataset(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Twitter sentiment CSV", "", "CSV Files (*.csv)"
        )
        if not path:
            return
        try:
            df = pd.read_csv(path)
            text_col = next((c for c in TEXT_CANDIDATE_COLS if c in df.columns), None)
            if text_col is None:
                raise ValueError(
                    "No recognizable text column found "
                    f"(expected one of {TEXT_CANDIDATE_COLS})."
                )
            self.df = df.dropna(subset=[text_col]).reset_index(drop=True)
            self.text_col = text_col

            self.tweet_list.clear()
            for tweet in self.df[text_col].astype(str).tolist():
                self.tweet_list.addItem(tweet)

            self.status_label.setText(
                f"Status: Dataset loaded ({len(self.df)} tweets) from '{path}'."
            )
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Dataset", str(e))

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _predict(self, text: str) -> str:
        if self.model is None or self.tokenizer is None:
            QMessageBox.warning(self, "No Model", "Please load a trained model first.")
            return "-"
        if not text or not text.strip():
            return "-"

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            logits = self.model(**enc).logits
            pred_id = torch.argmax(logits, dim=1).item()

        return ID2LABEL.get(pred_id, str(pred_id))

    def predict_selected_tweet(self, item):
        text = item.text()
        label = self._predict(text)
        self.result_label.setText(f"Predicted Sentiment: {label}")

    def predict_manual_text(self):
        text = self.manual_input.text()
        label = self._predict(text)
        self.result_label.setText(f"Predicted Sentiment: {label}")


def main():
    app = QApplication(sys.argv)
    window = SentimentApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
