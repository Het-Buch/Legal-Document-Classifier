# src/dap/dap_dataset.py
"""
DAP dataset utilities for Masked Language Modeling (MLM).

Provides:
- CorpusTextDataset: concatenates/streams text files or CSV columns to produce text examples.
- LineByLineTextDataset: splits text into lines (useful for short documents).
- DataCollator is used from transformers (DataCollatorForLanguageModeling).

Expectations:
- CUAD corpus lives under data/raw/CUAD_v1/
  - full_contract_txt/ (many .txt files)
  - master_clauses.csv
"""
from pathlib import Path
from typing import Iterator, List, Optional
import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def iterate_full_contract_txt(root: str = "data/raw/CUAD_v1") -> Iterator[str]:
    root_p = Path(root)
    txt_dir = root_p / "full_contract_txt"
    if not txt_dir.exists():
        logger.warning("No full_contract_txt/ directory at %s", txt_dir)
        return
    for p in sorted(txt_dir.glob("*.txt")):
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            text = p.read_text(encoding="latin-1")
        if text and text.strip():
            yield text

def iterate_master_clauses_text(root: str = "data/raw/CUAD_v1", csv_name: str = "master_clauses.csv",
                                text_col_candidates: Optional[List[str]] = None) -> Iterator[str]:
    root_p = Path(root)
    csv_path = root_p / csv_name
    if not csv_path.exists():
        logger.warning("master_clauses.csv not found at %s", csv_path)
        return
    try:
        df = pd.read_csv(csv_path, dtype=str).fillna("")
    except Exception as e:
        logger.warning("Failed to read CSV %s: %s", csv_path, e)
        return
    # candidate columns
    if text_col_candidates is None:
        text_col_candidates = ["text", "clause_text", "clause", "content", "clause_text_clean", "raw_text"]
    text_col = None
    for c in text_col_candidates:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        # fallback: concatenate all columns per row
        for _, row in df.iterrows():
            s = " ".join([str(v).strip() for v in row.values if str(v).strip()])
            if s:
                yield s
    else:
        for _, row in df.iterrows():
            s = str(row[text_col]).strip()
            if s:
                yield s

def build_corpus_iter(root: str = "data/raw/CUAD_v1", prefer_full_contracts: bool = True):
    """
    Yields text chunks suitable for tokenization.
    Order: if prefer_full_contracts True, yield full_contract_txt first, then master_clauses texts.
    """
    if prefer_full_contracts:
        for t in iterate_full_contract_txt(root):
            yield t
        for t in iterate_master_clauses_text(root):
            yield t
    else:
        for t in iterate_master_clauses_text(root):
            yield t
        for t in iterate_full_contract_txt(root):
            yield t

def demo_corpus(n_samples: int = 200):
    """
    Small synthetic corpus for demo/testing. Each sample is a short legal-ish sentence.
    """
    base = [
        "This Agreement is made between the parties.",
        "The party shall indemnify the other for losses arising out of breach.",
        "Confidentiality obligations survive termination.",
        "The license grants the right to use the software within the scope.",
        "Governing law shall be the laws of the State of New York.",
        "The indemnity clause includes attorneys' fees and costs.",
        "This Agreement may be terminated upon written notice."
    ]
    out = []
    for i in range(n_samples):
        out.append(" ".join(base[i % len(base)] for _ in range(1 + (i % 3))))
    return out
