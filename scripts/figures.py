# scripts/generate_paper_figures.py

from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Output directory
# -------------------------------------------------
FIG_DIR = Path("artifacts/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# Input files
# -------------------------------------------------
BASELINE_PATH = "artifacts/eval/baseline_svm_eval.json"
MTL_METRICS_PATH = "artifacts/eval/mtl_test_metrics.json"
PER_LABEL_PATH = "artifacts/eval/mtl_test_per_label.csv"
THRESHOLD_PATH = "artifacts/thresholds/thresholds_mtl.json"

# -------------------------------------------------
# Utility: safe F1 extraction
# -------------------------------------------------
def extract_f1(metrics):
    """
    Supports:
    - {"f1_micro": x, "f1_macro": y}
    - {"micro": {"f1": x}, "macro": {"f1": y}}
    """
    if "f1_micro" in metrics:
        return metrics["f1_micro"], metrics["f1_macro"]
    if "micro" in metrics and "macro" in metrics:
        return metrics["micro"]["f1"], metrics["macro"]["f1"]
    raise KeyError("Unsupported metric format")

# -------------------------------------------------
# Load data
# -------------------------------------------------
with open(BASELINE_PATH) as f:
    baseline = json.load(f)

with open(MTL_METRICS_PATH) as f:
    mtl = json.load(f)

per_label = pd.read_csv(PER_LABEL_PATH)
thresholds = json.load(open(THRESHOLD_PATH))

svm_micro, svm_macro = extract_f1(baseline)
mtl_micro, mtl_macro = extract_f1(mtl)

# =================================================
# FIG 1 — Overall performance comparison
# =================================================
labels = ["Micro F1", "Macro F1"]
svm_vals = [svm_micro, svm_macro]
mtl_vals = [mtl_micro, mtl_macro]

x = np.arange(len(labels))
w = 0.35

plt.figure()
plt.bar(x - w/2, svm_vals, w, label="SVM")
plt.bar(x + w/2, mtl_vals, w, label="MTL-DAP")
plt.xticks(x, labels)
plt.ylabel("F1 Score")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "fig1_overall_comparison.png", dpi=300)
plt.close()

# =================================================
# FIG 2 — Top-15 clause-wise F1 (MTL-DAP)
# =================================================
top = per_label.sort_values("f1", ascending=False).head(15)

plt.figure(figsize=(10, 5))
plt.barh(top["label"], top["f1"])
plt.xlabel("F1 Score")
plt.title("Top-15 Clause-wise F1 (MTL-DAP)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(FIG_DIR / "fig2_top15_clause_f1.png", dpi=300)
plt.close()

# =================================================
# FIG 3 — Micro vs Macro gap (imbalance robustness)
# =================================================
plt.figure()
plt.bar(
    ["SVM", "MTL-DAP"],
    [
        svm_micro - svm_macro,
        mtl_micro - mtl_macro
    ]
)
plt.ylabel("Micro–Macro F1 Gap")
plt.title("Robustness to Label Imbalance")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig3_micro_macro_gap.png", dpi=300)
plt.close()

# =================================================
# FIG 4 — Per-label F1 improvement (if available)
# =================================================
if "svm_f1" in per_label.columns:
    delta = per_label["f1"] - per_label["svm_f1"]

    plt.figure(figsize=(10, 6))
    plt.barh(per_label["label"], delta)
    plt.xlabel("Δ F1 (MTL − SVM)")
    plt.title("Per-label Performance Improvement")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_delta_f1.png", dpi=300)
    plt.close()

# =================================================
# FIG 5 — Threshold distribution
# =================================================
plt.figure()
plt.hist(list(thresholds.values()), bins=10)
plt.xlabel("Threshold Value")
plt.ylabel("Number of Labels")
plt.title("Optimized Per-label Threshold Distribution")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig5_threshold_distribution.png", dpi=300)
plt.close()

# =========================================================
# FIG 6 — Error Sensitivity (Hamming Loss)
# =========================================================
plt.figure()
plt.bar(
    ["SVM", "MTL-DAP"],
    [
        baseline.get("hamming_loss"),
        mtl.get("hamming_loss")
    ]
)
plt.ylabel("Hamming Loss")
plt.title("Error Sensitivity Comparison")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig7_hamming_loss.png", dpi=300)
plt.close()

# =========================================================
# FIG 7 — Label Overlap Quality (Jaccard Score)
# =========================================================
plt.figure()
plt.bar(
    ["SVM", "MTL-DAP"],
    [
        baseline.get("jaccard_score"),
        mtl.get("jaccard_score")
    ]
)
plt.ylabel("Jaccard Score")
plt.title("Label Overlap Quality Comparison")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig8_jaccard_score.png", dpi=300)
plt.close()

print("✅ All 7 paper figures saved to artifacts/figures/")