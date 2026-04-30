import json
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import f1_score

from src.data.dataset_reader import load_split
from src.utils.file_io import save_json

THRESHOLD_GRID = np.linspace(0.05, 0.95, 19)

def tune_thresholds_lr():
    X_val, y_val, _ = load_split("val")
    y_val = np.array(y_val)

    bundle = joblib.load("models/tfidf_lr_cuad.joblib")
    vectorizer = bundle["vectorizer"]
    model = bundle["model"]

    X_val_vec = vectorizer.transform(X_val)
    y_val_probs = model.predict_proba(X_val_vec)

    print("Prob stats:", y_val_probs.min(), y_val_probs.mean(), y_val_probs.max())

    thresholds = {}
    num_labels = y_val.shape[1]

    for i in range(num_labels):
        best_f1 = 0.0
        best_t = 0.5

        y_true = y_val[:, i]
        probs = y_val_probs[:, i]

        if y_true.sum() == 0:
            thresholds[str(i)] = 0.5
            continue

        for t in THRESHOLD_GRID:
            y_pred = (probs >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)

        thresholds[str(i)] = best_t

    save_json(thresholds, "artifacts/thresholds/thresholds_lr.json")
    print("✅ Threshold tuning complete (LR)")

if __name__ == "__main__":
    tune_thresholds_lr()
