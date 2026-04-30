import pandas as pd
from pathlib import Path
import numpy as np

CUAD_ROOT = Path("data/raw/CUAD_v1")

def load_cuad_dataset():
    csv_path = CUAD_ROOT / "master_clauses.csv"
    txt_dir = CUAD_ROOT / "full_contract_txt"

    df = pd.read_csv(csv_path)

    documents = []
    labels = []

    label_columns = [c for c in df.columns if c not in ["Filename", "Clause"]]

    def is_positive(val):
        if isinstance(val, str):
            return val.strip().lower() == "yes"
        if isinstance(val, (int, float)):
            return val == 1
        if isinstance(val, bool):
            return val
        return False

    for _, row in df.iterrows():
        fname = row["Filename"]

        if isinstance(fname, str) and fname.lower().endswith(".pdf"):
            fname = fname[:-4] + ".txt"

        txt_path = txt_dir / fname
        if not txt_path.exists():
            continue

        text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue

        documents.append(text)

        label_vector = [1 if is_positive(row[label]) else 0 for label in label_columns]
        labels.append(label_vector)

    # ✅ Sanity check (runs once)
    labels_np = np.array(labels)
    print("Total positives:", labels_np.sum())
    print("Docs with ≥1 label:", (labels_np.sum(axis=1) > 0).sum())

    return documents, labels, {"label_set": label_columns}
