import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, hamming_loss, jaccard_score

from src.data.dataset_reader import load_split
from src.utils.file_io import save_json

def apply_thresholds(probs, thresholds, label_names):
    preds = np.zeros_like(probs, dtype=int)
    for i, label in enumerate(label_names):
        t = thresholds.get(label, 0.5)
        preds[:, i] = (probs[:, i] >= t).astype(int)
    return preds

def evaluate_lr_baseline():
    # Load test split
    X_test, y_test_labels, _ = load_split("test")

    # Load model + artifacts
    bundle = joblib.load("models/tfidf_lr_cuad.joblib")
    vectorizer = bundle["vectorizer"]
    model = bundle["model"]
    mlb = bundle["mlb"]

    # Load thresholds
    with open("artifacts/thresholds/thresholds_lr.json", "r") as f:
        thresholds = json.load(f)

    # Vectorize
    X_test_vec = vectorizer.transform(X_test)
    y_test = mlb.transform(y_test_labels)

    # Predict probabilities
    y_probs = model.predict_proba(X_test_vec)

    # Apply thresholds
    y_pred = apply_thresholds(y_probs, thresholds, mlb.classes_)

    # Metrics
    metrics = {
        "f1_micro": f1_score(y_test, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "hamming_loss": hamming_loss(y_test, y_pred),
        "jaccard_score": jaccard_score(y_test, y_pred, average="samples", zero_division=0)
    }

    # Per-label F1
    per_label_f1 = {
        label: f1_score(y_test[:, i], y_pred[:, i], zero_division=0)
        for i, label in enumerate(mlb.classes_)
    }

    # Save outputs
    Path("artifacts/eval").mkdir(parents=True, exist_ok=True)

    save_json(metrics, "artifacts/eval/baseline_lr_test_metrics.json")

    pd.DataFrame.from_dict(
        per_label_f1, orient="index", columns=["f1"]
    ).to_csv("artifacts/eval/baseline_lr_per_label_f1.csv")

    print("✅ LR baseline test evaluation complete")
    print(metrics)

if __name__ == "__main__":
    evaluate_lr_baseline()
