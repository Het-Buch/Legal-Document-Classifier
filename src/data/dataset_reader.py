import json
from src.data.cuad_loader import load_cuad_dataset

SPLIT_PATH = "data/processed/splits.json"

def load_split(split="train"):
    # Load full dataset
    docs, labels, meta = load_cuad_dataset()

    # Load precomputed splits (DO NOT regenerate)
    with open(SPLIT_PATH, "r") as f:
        splits = json.load(f)

    idxs = splits[split]

    split_docs = [docs[i] for i in idxs]
    split_labels = [labels[i] for i in idxs]

    return split_docs, split_labels, meta