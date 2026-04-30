import torch
import numpy as np
from torch.utils.data import DataLoader
from src.mtl.model import MTLModel
from src.mtl.dataset import CUADChunkDataset
from src.utils.metrics import compute_metrics
from src.utils.file_io import save_json, save_csv
from src.utils.file_io import load_json
MODEL_PATH = "models/checkpoints/dap"
WEIGHTS_PATH = "models/checkpoints/mtl/mtl_bundle.pt"
BATCH_SIZE = 2

# Temporary global threshold (Stage 7B will replace this)
DEFAULT_THRESHOLD = 0.5
LABEL_THRESHOLDS = load_json("artifacts/thresholds/thresholds_mtl.json")
bundle_thresholds = load_json("artifacts/thresholds/thresholds_mtl.json")

def evaluate(split):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CUADChunkDataset(split, MODEL_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    # 🔐 SAFETY CHECK (PUT IT HERE)
    bundle = torch.load(WEIGHTS_PATH, map_location=device)

    bundle_label_set = bundle["label_set"]
    # bundle_thresholds = bundle["thresholds"]
    assert set(bundle["label_set"]) == set(bundle_thresholds.keys()), \
        "Label set mismatch between checkpoint and thresholds"
    model = MTLModel(MODEL_PATH, len(bundle_label_set))
    model.load_state_dict(bundle["state_dict"])
    model.to(device)
    model.eval()

    doc_preds = {}
    doc_labels = {}

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()
            doc_ids = batch["doc_id"].cpu().numpy()

            _, doc_logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(doc_logits).cpu().numpy()

            for i in range(len(probs)):
                doc_id = int(doc_ids[i])
                doc_preds.setdefault(doc_id, []).append(probs[i])
                doc_labels.setdefault(doc_id, labels[i])

    y_true, y_pred = [], []

    for doc_id in sorted(doc_preds.keys()):
        avg_pred = np.mean(doc_preds[doc_id], axis=0)
        label_thresholds = np.array(
            [bundle_thresholds[lbl] for lbl in bundle_label_set]
        )
        y_pred.append((avg_pred >= label_thresholds).astype(int))
        y_true.append(doc_labels[doc_id])

    metrics, per_label = compute_metrics(
        np.array(y_true),
        np.array(y_pred),
        bundle_label_set
    )

    return metrics, per_label


if __name__ == "__main__":
    for split in ["val", "test"]:
        metrics, per_label = evaluate(split)

        save_json(
            metrics,
            f"artifacts/eval/mtl_{split}_metrics.json"
        )
        save_csv(
            per_label,
            f"artifacts/eval/mtl_{split}_per_label.csv"
        )

        print(f"✅ MTL {split.upper()} evaluation complete")
        print(metrics)
