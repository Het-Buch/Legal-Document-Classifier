import numpy as np
from sklearn.metrics import f1_score, hamming_loss, jaccard_score


def multilabel_metrics(y_true, y_pred):
    return {
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "jaccard_score": jaccard_score(
            y_true, y_pred, average="samples", zero_division=0
        )
    }


def compute_metrics(y_true, y_pred, label_set):
    """
    Returns:
      - overall metrics dict
      - per-label metrics dict (label -> F1)
    """

    overall = multilabel_metrics(y_true, y_pred)

    per_label = {}
    for i, label in enumerate(label_set):
        per_label[label] = f1_score(
            y_true[:, i],
            y_pred[:, i],
            zero_division=0
        )

    return overall, per_label
