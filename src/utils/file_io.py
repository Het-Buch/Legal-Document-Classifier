import json
import csv
import os


def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_json(obj, path):
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_csv(per_label_dict, path):
    """
    per_label_dict: Dict[label -> metric_value]
    """
    ensure_dir(path)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "f1"])
        for label, score in per_label_dict.items():
            writer.writerow([label, score])
            
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)