# tools/robustness_suite.py
"""
Robustness Suite

Implements:
- OCR noise simulation
- Paraphrase augmentation (lightweight back-translation-like heuristic)
- Random token deletion/insertion/swap
- Runner to evaluate a given predictor (UnifiedPredictor) across perturbations

Usage:
    python tools/robustness_suite.py --model baseline --baseline models/tfidf_lr_cuad.joblib --thresholds models/thresholds.json
"""
import argparse
import random
import numpy as np
from typing import List, Callable, Dict
from src.infer.predictor import UnifiedPredictor
from src.data.cuad_loader import load_cuad_dataset
from src.data.prepare_splits import write_splits
from src.utils.metrics import multilabel_evaluate
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("robustness")

def ocr_noise(text: str, p_replace: float = 0.02) -> str:
    """Simulate OCR character-level noise: random char substitution, deletion."""
    import string
    chars = list(text)
    for i in range(len(chars)):
        if random.random() < p_replace:
            op = random.choice(["sub", "del", "swap"])
            if op == "sub":
                chars[i] = random.choice(string.ascii_lowercase + " ")
            elif op == "del":
                chars[i] = ""
            elif op == "swap" and i + 1 < len(chars):
                chars[i], chars[i+1] = chars[i+1], chars[i]
    return "".join(chars)

def simple_paraphrase(text: str) -> str:
    """
    Lightweight paraphrasing: synonym substitution with a small internal map,
    sentence reordering if multi-sentence.
    (This is NOT a full paraphraser — it's a deterministic, reproducible augmenter.)
    """
    synonyms = {
        "terminate": "end",
        "termination": "end",
        "party": "counterparty",
        "agreement": "contract",
        "indemnify": "compensate",
        "warranty": "guarantee",
        "liable": "responsible"
    }
    # token-level substitution
    toks = text.split()
    out = [synonyms.get(t.lower().strip(".,;:"), t) for t in toks]
    out_text = " ".join(out)
    # sentence shuffle
    sents = [s.strip() for s in out_text.split(".") if s.strip()]
    if len(sents) > 1 and random.random() < 0.3:
        random.shuffle(sents)
    return ". ".join(sents) + (("." if out_text.strip().endswith(".") else ""))

def token_deletion(text: str, p_drop: float = 0.05) -> str:
    toks = text.split()
    out = [t for t in toks if random.random() > p_drop]
    if not out:
        return toks[:1]
    return " ".join(out)

def apply_perturbation(docs: List[str], perturb_fn: Callable[[str], str]) -> List[str]:
    return [perturb_fn(d) for d in docs]

def evaluate_on_perturbation(predictor: UnifiedPredictor, docs: List[str], labels_list: List[List[str]],
                             label_set: List[str], pert_name: str, pert_fn: Callable[[str], str]) -> Dict:
    # apply perturbation to docs
    docs_pert = apply_perturbation(docs, pert_fn)
    res = predictor.predict_documents(docs_pert, model_kind="auto")
    probs = res["probs"]
    preds = res["preds"]
    # binarize truth
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=label_set)
    Y = mlb.fit_transform(labels_list)
    metrics = multilabel_evaluate(Y, preds, y_prob=probs, label_set=label_set)
    logger.info("Perturbation=%s metrics: %s", pert_name, metrics)
    return {"perturbation": pert_name, "metrics": metrics}

def run_suite(predictor: UnifiedPredictor, which_split="test", out_dir="artifacts/robustness"):
    Path = __import__("pathlib").Path
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    docs, labels_list, meta = load_cuad_dataset()
    splits = write_splits()
    idxs = splits.get(which_split, splits["test"])
    docs_split = [docs[i] for i in idxs]
    labels_split = [labels_list[i] for i in idxs]
    label_set = meta["label_set"]

    perturbations = [
        ("ocr_noise", lambda t: ocr_noise(t, p_replace=0.03)),
        ("paraphrase", simple_paraphrase),
        ("token_deletion", lambda t: token_deletion(t, p_drop=0.07)),
    ]
    results = []
    # baseline evaluation on clean data
    clean_res = predictor.predict_documents(docs_split, model_kind="auto")
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=label_set)
    Y = mlb.fit_transform(labels_split)
    clean_metrics = multilabel_evaluate(Y, clean_res["preds"], y_prob=clean_res["probs"], label_set=label_set)
    results.append({"perturbation": "clean", "metrics": clean_metrics})
    logger.info("Clean metrics: %s", clean_metrics)

    for name, fn in perturbations:
        r = evaluate_on_perturbation(predictor, docs_split, labels_split, label_set, name, fn)
        results.append(r)

    out_path = Path(out_dir) / f"robustness_{predictor.__class__.__name__}.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Saved robustness suite results to %s", out_path)
    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, default="models/tfidf_lr_cuad.joblib")
    parser.add_argument("--thresholds", type=str, default="models/thresholds.json")
    parser.add_argument("--mtl", type=str, default=None)
    parser.add_argument("--out", type=str, default="artifacts/robustness/robustness_summary.json")
    args = parser.parse_args()
    predictor = UnifiedPredictor(baseline_path=args.baseline if Path(args.baseline).exists() else None,
                                 thresholds_path=args.thresholds if Path(args.thresholds).exists() else None,
                                 mtl_checkpoint=args.mtl if args.mtl and Path(args.mtl).exists() else None)
    run_suite(predictor, out_dir=str(Path(args.out).parent))
    print("Done.")
