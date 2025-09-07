"""
train_model.py

Train a simple TF-IDF + LogisticRegression career classifier
from data/training_data.csv and save the pipeline to models/career_model.pkl.

CSV expected columns: text,label
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle

DATA_PATH = os.path.join("data", "training_data.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "career_model.pkl")

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # Basic cleaning
    df = df.dropna(subset=["text", "label"])
    return df["text"].astype(str).tolist(), df["label"].astype(str).tolist()

def train_and_save():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found. Create training CSV first.")

    texts, labels = load_data(DATA_PATH)
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)

    # Simple pipeline: TF-IDF -> Logistic Regression
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=15000)),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    print("Training model on", len(X_train), "samples...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    preds = pipeline.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"Validation accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_val, preds))

    # Create model directory and save
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_and_save()
