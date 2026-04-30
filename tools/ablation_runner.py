# tools/ablation_runner.py
"""
Ablation runner to execute the key ablations:
- doc-only (no chunk head)
- chunk-only (no doc head)
- no-DAP (use original backbone)
- DAP + MTL (normal)
It orchestrates training runs using configuration overrides and stores per-run metrics and metadata.

Assumptions:
- Uses src/mtl/train_mtl.py code but allows toggling flags for ablations.
"""
import argparse
import subprocess
import json
from pathlib import Path
import time
import logging

logger = logging.getLogger("ablation")
logging.basicConfig(level=logging.INFO)

ABLATION_CONFIGS = [
    {"name": "mtl_doc_only", "extra_args": ["--alpha", "0.0"], "desc": "Only doc-level loss (alpha=0 -> chunk loss weight 0)"},
    {"name": "mtl_chunk_only", "extra_args": ["--alpha", "1.0"], "desc": "Only chunk-level loss (alpha=1 -> doc loss weight 0)"},
    {"name": "mtl_no_dap", "extra_args": ["--no_pretrained"], "desc": "No pretrained backbone (train from scratch)"},
    {"name": "mtl_with_dap", "extra_args": [], "desc": "MTL starting from DAP checkpoint (pass --backbone_checkpoint)"},
]

def run_command(cmd: str):
    logger.info("Running: %s", cmd)
    start = time.time()
    res = subprocess.run(cmd, shell=True)
    end = time.time()
    return res.returncode, end - start

def main(output_dir="artifacts/ablations", dap_ckpt=None):
    Path = __import__("pathlib").Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = []
    base_cmd = "python src/mtl/train_mtl.py --epochs 1 --batch_size 2 --eval_batch_size 2 --use_cuda"  # example; user can modify
    for cfg in ABLATION_CONFIGS:
        name = cfg["name"]
        args = cfg["extra_args"].copy()
        cmd = base_cmd + " " + " ".join(args)
        if name == "mtl_with_dap" and dap_ckpt:
            cmd += f" --backbone_checkpoint {dap_ckpt}"
        # redirect logs
        logfile = Path(output_dir) / f"{name}.log"
        cmd_full = f"{cmd} 2>&1 | tee {logfile}"
        rc, elapsed = run_command(cmd_full)
        results.append({"name": name, "cmd": cmd, "rc": rc, "elapsed_s": elapsed, "log": str(logfile)})
    Path(output_dir).joinpath("summary.json").write_text(json.dumps(results, indent=2))
    logger.info("Ablation runs complete. Summary saved to %s", Path(output_dir)/"summary.json")
    return Path(output_dir)/"summary.json"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dap_ckpt", type=str, default=None)
    parser.add_argument("--out", type=str, default="artifacts/ablations")
    args = parser.parse_args()
    main(output_dir=args.out, dap_ckpt=args.dap_ckpt)
