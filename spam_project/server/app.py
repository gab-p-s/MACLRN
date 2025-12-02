# server/app.py

import os
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

app = Flask(__name__)

BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
BASE_DATA_FILE = os.path.join(BASE_DATA_PATH, "SMSSpamCollection")
FEEDBACK_FILE = os.path.join(BASE_DATA_PATH, "feedback.csv")
MODEL_FILE = os.path.join(BASE_DATA_PATH, "spam_model.joblib")

model = None  # will be initialized in load_or_train_model()


def load_base_dataset():
    """
    Load the original SMS Spam Collection dataset.
    File format: tab-separated, columns: label, text
    """
    if not os.path.exists(BASE_DATA_FILE):
        raise FileNotFoundError(
            f"Base dataset not found at {BASE_DATA_FILE}. "
            f"Make sure you downloaded 'SMSSpamCollection' there."
        )
    df = pd.read_csv(
        BASE_DATA_FILE,
        sep="\t",
        header=None,
        names=["label", "text"],
        encoding="utf-8"
    )
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def load_feedback_dataset():
    """
    Load feedback data if it exists, otherwise return empty DataFrame.
    """
    if not os.path.exists(FEEDBACK_FILE):
        return pd.DataFrame(columns=["label", "text", "label_num"])
    df = pd.read_csv(FEEDBACK_FILE, encoding="utf-8")
    if "label_num" not in df.columns:
        df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def save_feedback_row(label: str, text: str):
    """
    Append a single labeled example to feedback.csv.
    """
    os.makedirs(BASE_DATA_PATH, exist_ok=True)
    row = pd.DataFrame(
        [{"label": label, "text": text, "label_num": 1 if label == "spam" else 0}]
    )
    if os.path.exists(FEEDBACK_FILE):
        row.to_csv(FEEDBACK_FILE, mode="a", header=False, index=False, encoding="utf-8")
    else:
        row.to_csv(FEEDBACK_FILE, mode="w", header=True, index=False, encoding="utf-8")


def train_new_model():
    """
    Train a TF-IDF + Logistic Regression classifier on base + feedback data.
    Returns the trained sklearn Pipeline.
    """
    base_df = load_base_dataset()
    feedback_df = load_feedback_dataset()

    if not feedback_df.empty:
        all_df = pd.concat([base_df, feedback_df], ignore_index=True)
    else:
        all_df = base_df

    X = all_df["text"].values
    y = all_df["label_num"].values

    # Very simple pipeline: TF-IDF → Logistic Regression
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    # Optional: simple train/test split to compute accuracy
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    val_acc = pipeline.score(X_val, y_val)
    print(f"[TRAIN] Validation accuracy: {val_acc:.3f}")

    # Save to disk so we can reuse later
    joblib.dump(pipeline, MODEL_FILE)
    return pipeline, float(val_acc)


def load_or_train_model():
    global model
    if os.path.exists(MODEL_FILE):
        print("[INFO] Loading existing model from disk...")
        model = joblib.load(MODEL_FILE)
        # You could optionally compute accuracy here by re-evaluating.
        return
    print("[INFO] No saved model found. Training a new one...")
    model, acc = train_new_model()


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict spam/ham for a given text.
    Request JSON: { "text": "some message" }
    Response JSON: { "prediction": "spam"/"ham", "probability": 0.87 }
    """
    global model
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"]
    proba_spam = model.predict_proba([text])[0][1]
    pred_label = "spam" if proba_spam >= 0.5 else "ham"

    return jsonify(
        {
            "prediction": pred_label,
            "probability_spam": float(proba_spam),
            "probability_ham": float(1.0 - proba_spam),
        }
    )


@app.route("/feedback", methods=["POST"])
def feedback():
    """
    Accept user feedback for a given text.
    Request JSON: { "text": "...", "true_label": "spam"/"ham" }
    Stores it in feedback.csv for future retraining.
    """
    data = request.get_json()
    if not data or "text" not in data or "true_label" not in data:
        return jsonify({"error": "Expected 'text' and 'true_label'"}), 400

    true_label = data["true_label"].lower()
    if true_label not in ("spam", "ham"):
        return jsonify({"error": "true_label must be 'spam' or 'ham'"}), 400

    save_feedback_row(true_label, data["text"])
    return jsonify({"status": "ok", "message": "Feedback saved"})


@app.route("/retrain", methods=["POST"])
def retrain():
    """
    Retrain the model using base dataset + feedback labels.
    Response JSON: { "status": "ok", "val_accuracy": 0.93 }
    """
    global model
    model, val_acc = train_new_model()
    return jsonify(
        {
            "status": "ok",
            "message": "Model retrained using base + feedback data",
            "val_accuracy": val_acc,
        }
    )


if __name__ == "__main__":
    # Initialize model at startup
    load_or_train_model()
    # Run Flask development server
    app.run(host="127.0.0.1", port=5000, debug=True)
