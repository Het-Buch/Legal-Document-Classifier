import json
import random
from pathlib import Path

def write_splits(seed=42, train_ratio=0.7, val_ratio=0.15):
    from src.data.cuad_loader import load_cuad_dataset
    docs, _, _ = load_cuad_dataset()

    random.seed(seed)
    idxs = list(range(len(docs)))
    random.shuffle(idxs)

    n = len(idxs)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    splits = {
        "train": idxs[:train_end],
        "val": idxs[train_end:val_end],
        "test": idxs[val_end:]
    }

    out = Path("data/processed/splits.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(splits, indent=2))
    return splits

if __name__ == "__main__":
    write_splits()
