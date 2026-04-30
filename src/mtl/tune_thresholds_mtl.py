import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from pathlib import Path

from src.mtl.model import MTLModel
from src.mtl.dataset import CUADChunkDataset
from src.utils.file_io import save_json

MODEL_PATH = "models/checkpoints/dap"
WEIGHTS_PATH = "models/checkpoints/mtl/mtl_bundle.pt"
BATCH_SIZE = 2

THRESHOLD_GRID = np.linspace(0.05, 0.95, 19)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tune_thresholds():
    dataset = CUADChunkDataset("val", MODEL_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    bundle = torch.load(WEIGHTS_PATH, map_location="cuda")
    model = MTLModel(MODEL_PATH, len(dataset.label_set))
    model.load_state_dict(bundle["state_dict"])
    model.cuda()
    model.eval()

    doc_preds = {}
    doc_labels = {}

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cpu().numpy()
            doc_ids = batch["doc_id"].cpu().numpy()

            _, doc_logits = model(input_ids, attention_mask)
            probs = sigmoid(doc_logits.cpu().numpy())

            for i in range(len(probs)):
                doc_id = int(doc_ids[i])
                doc_preds.setdefault(doc_id, []).append(probs[i])
                doc_labels.setdefault(doc_id, labels[i])

    y_true, y_prob = [], []
    for doc_id in sorted(doc_preds.keys()):
        y_true.append(doc_labels[doc_id])
        y_prob.append(np.mean(doc_preds[doc_id], axis=0))

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    thresholds = {}
    for i, label in enumerate(dataset.label_set):
        if y_true[:, i].sum() == 0:
            thresholds[label] = 0.5
            continue

        best_f1, best_t = 0.0, 0.5
        for t in THRESHOLD_GRID:
            preds = (y_prob[:, i] >= t).astype(int)
            f1 = f1_score(y_true[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)

        thresholds[label] = best_t

    Path("artifacts/thresholds").mkdir(parents=True, exist_ok=True)
    save_json(thresholds, "artifacts/thresholds/thresholds_mtl.json")

    print("✅ MTL threshold tuning complete")
    print("Saved to artifacts/thresholds/thresholds_mtl.json")


if __name__ == "__main__":
    tune_thresholds()
