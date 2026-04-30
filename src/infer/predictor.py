# src/infer/predictor.py
"""
Unified predictor for Baseline (TF-IDF OvR) and MTL (chunk+doc) models.

API:
    p = UnifiedPredictor(kind="baseline", baseline_path="models/tfidf_lr_cuad.joblib", thresholds="models/thresholds.json")
    docs = ["..."]
    doc_preds = p.predict_documents(docs)  # returns dict with label_set, probs, preds
    chunk_res = p.predict_chunks(docs)     # returns per-doc list of chunks with per-label scores

Notes:
- For baseline chunk-level scores, we vectorize each chunk separately and use estimator predict_proba/decision_function with sigmoid fallback.
- For MTL chunk-level scores, we use the model's chunk_logits directly.
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from joblib import load
import logging
from src.preprocessing.chunker import chunk_tokens

logger = logging.getLogger(__name__)

# Optional torch imports (only if MTL is used)
try:
    import torch
    TORCH = True
except Exception:
    TORCH = False

# helper for sigmoid
def _sigmoid(x):
    try:
        from scipy.special import expit
        return expit(x)
    except Exception:
        return 1.0 / (1.0 + np.exp(-x))

class UnifiedPredictor:
    def __init__(self,
                 baseline_path: Optional[str] = None,
                 thresholds_path: Optional[str] = None,
                 mtl_checkpoint: Optional[str] = None,
                 mtl_backbone: Optional[str] = None,
                 device: str = "cpu"):
        self.baseline_path = baseline_path
        self.thresholds_path = thresholds_path
        self.mtl_checkpoint = mtl_checkpoint
        self.mtl_backbone = mtl_backbone
        self.device = device
        self.baseline_obj = None
        self.vec = None
        self.clf = None
        self.mlb = None
        self.label_set = None
        self.thresholds = None
        # MTL model placeholders
        self.mtl_model = None
        self.mtl_tokenizer = None

        if baseline_path:
            self._load_baseline(baseline_path)
        if thresholds_path:
            self._load_thresholds(thresholds_path)
        if mtl_checkpoint:
            self._load_mtl(mtl_checkpoint, backbone_name=mtl_backbone)

    # -------------------------
    # Baseline loading + helpers
    # -------------------------
    def _load_baseline(self, path: str):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Baseline model not found at {path}")
        obj = load(path)
        self.baseline_obj = obj
        self.vec = obj.get("vectorizer", None)
        self.clf = obj.get("clf", None)
        self.mlb = obj.get("mlb", None)
        if self.mlb is not None:
            self.label_set = list(self.mlb.classes_)
        logger.info("Loaded baseline model from %s; labels=%s", path, self.label_set)

    def _baseline_predict_proba_for_texts(self, docs: List[str]) -> np.ndarray:
        """
        Return shape (N_docs, L)
        """
        if self.vec is None or self.clf is None:
            raise RuntimeError("Baseline vectorizer/clf not loaded")
        X = self.vec.transform(docs)
        # try predict_proba for each estimator, else decision_function mapped with sigmoid, else fallback to predict
        try:
            probs = np.vstack([est.predict_proba(X)[:,1] for est in self.clf.estimators_]).T
            return probs
        except Exception:
            try:
                scores = np.vstack([est.decision_function(X) for est in self.clf.estimators_]).T
                return _sigmoid(scores)
            except Exception:
                preds = self.clf.predict(X)
                return preds.astype(float)

    # compute baseline chunk-level scores by applying vectorizer/clf to each chunk text
    def baseline_chunk_scores(self, docs: List[str], chunker_fn) -> List[List[Dict[str, Any]]]:
        """
        chunker_fn: callable(text) -> list of chunk dicts with 'chunk_id' and 'tokens' (token list or text)
        Returns:
            per_doc_chunks: list (per doc) of list of {"chunk_id", "text", "probs": [L], "preds": [L]}
        """
        out = []
        for d in docs:
            chunks = chunker_fn(d)
            chunk_infos = []
            chunk_texts = [c["text"] for c in chunks]
            if len(chunk_texts) == 0:
                out.append([])
                continue
            probs = self._baseline_predict_proba_for_texts(chunk_texts)  # (n_chunks, L)
            for i, c in enumerate(chunks):
                p = probs[i].tolist()
                pred = [1 if p_j >= (self.thresholds[idx] if self.thresholds is not None else 0.5) else 0 for idx,p_j in enumerate(p)]
                chunk_infos.append({"chunk_id": c.get("chunk_id", i), "text": c["text"], "probs": p, "preds": pred})
            out.append(chunk_infos)
        return out

    # -------------------------
    # Thresholds
    # -------------------------
    def _load_thresholds(self, path: str):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Threshold file not found: {p}")
        th = json.loads(p.read_text(encoding="utf-8"))
        self.thresholds = [t.get("threshold", 0.5) for t in th.get("thresholds", [])]
        if self.label_set is None:
            self.label_set = th.get("label_set", [])
        logger.info("Loaded thresholds for %d labels", len(self.thresholds))

    # -------------------------
    # MTL loading + helpers
    # -------------------------
    def _load_mtl(self, checkpoint: str, backbone_name: Optional[str] = None):
        if not TORCH:
            raise RuntimeError("PyTorch required to load MTL model.")

        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"MTL checkpoint not found at {ckpt_path}")

        import torch

        bundle = torch.load(str(ckpt_path), map_location=self.device)

        # ✅ REQUIRED METADATA
        if "label_set" not in bundle:
            raise RuntimeError("MTL checkpoint missing label_set")

        self.label_set = bundle["label_set"]

        # ✅ ADD THIS (Answer labels only)
        self.answer_indices = [
            i for i, lbl in enumerate(self.label_set)
            if lbl.endswith("-Answer")
        ]

        raw_th = bundle.get("thresholds", None)

        # ---- parse thresholds safely ----
        if raw_th is None:
            self.thresholds = [0.5] * len(self.label_set)

        elif isinstance(raw_th, dict) and "thresholds" in raw_th:
            # JSON format: {"label_set": [...], "thresholds": [{label, threshold}, ...]}
            self.thresholds = [
                float(t["threshold"]) for t in raw_th["thresholds"]
            ]

        elif isinstance(raw_th, list):
            # already list of floats
            self.thresholds = [float(x) for x in raw_th]

        else:
            self.thresholds = [0.5] * len(self.label_set)

        # ✅ BUILD MODEL
        from src.mtl.model import MTLModel
        backbone = backbone_name or "nlpaueb/legal-bert-base-uncased"

        self.mtl_model = MTLModel(
            backbone,
            num_labels=len(self.label_set),
        )

        # ✅ LOAD WEIGHTS
        self.mtl_model.load_state_dict(bundle["state_dict"])
        self.mtl_model.to(self.device)
        self.mtl_model.eval()

        # ✅ TOKENIZER
        try:
            from transformers import AutoTokenizer
            self.mtl_tokenizer = AutoTokenizer.from_pretrained(backbone)
        except Exception:
            self.mtl_tokenizer = None

        logger.info(
            "Loaded MTL model: %d labels from %s",
            len(self.label_set), checkpoint
        )


    def mtl_predict_documents(self, docs: List[str], max_len: int = 512, stride: int = 128, batch_size: int = 8):
        """
        Returns: dict with keys:
            label_set, probs (N,L), preds (N,L), chunk_details (per doc list of chunk dicts with 'text','probs','preds')
        """
        if self.mtl_model is None:
            raise RuntimeError("MTL model not loaded")

        # Build dataset with tokenizer to chunk text into chunk texts
        # We'll reuse MTLChunkDocDataset but simpler: perform local chunking via tokenizer & chunker
        from src.preprocessing.tokenizer import get_tokenizer
        from src.preprocessing.chunker import chunk_tokens
        tokenizer = self.mtl_tokenizer if self.mtl_tokenizer is not None else get_tokenizer()
        # chunk every document into text chunks (strings)
        per_doc_chunk_texts = []
        for d in docs:
            # tokenize; if HF tokenizer present use tokenization to ids then reconstruct chunk texts using tokenizer.decode
            if hasattr(tokenizer, "encode") and hasattr(tokenizer, "decode"):
                enc = tokenizer(d, truncation=False, add_special_tokens=False)
                ids = enc["input_ids"]
                chunks_meta = chunk_tokens(ids, max_len=max_len, stride=stride)
                chunk_texts = [tokenizer.decode(ids[c["start"]:c["end"]], skip_special_tokens=True, clean_up_tokenization_spaces=True) for c in chunks_meta]
                # attach chunk_id
                per_doc_chunk_texts.append([{"chunk_id": c["chunk_id"], "text": chunk_texts[i]} for i, c in enumerate(chunks_meta)])
            else:
                # fallback: whitespace split and join tokens into chunk text
                toks = d.split()
                chunks_meta = chunk_tokens(toks, max_len=max_len, stride=stride)
                per_doc_chunk_texts.append([{"chunk_id": c["chunk_id"], "text": " ".join(c["tokens"])} for c in chunks_meta])

        # Flatten chunk texts and run batches through model to get chunk_logits
        all_chunk_texts = []
        doc_index_for_chunk = []
        for di, chunks in enumerate(per_doc_chunk_texts):
            for c in chunks:
                all_chunk_texts.append(c["text"])
                doc_index_for_chunk.append(di)
        if len(all_chunk_texts) == 0:
            # no text -> return zeros
            L = len(self.label_set)
            N = len(docs)
            probs = np.zeros((N, L)).tolist()
            preds = (np.zeros((N, L))).astype(int).tolist()
            return {"label_set": self.label_set, "probs": probs, "preds": preds, "chunks": per_doc_chunk_texts}

        # Create tensor batches using tokenizer (if available) else vectorize fallback with baseline vec if exists
        chunk_probs = []
        B = batch_size
        if self.mtl_tokenizer is not None:
            # use tokenizer to create input ids
            for i in range(0, len(all_chunk_texts), B):
                batch_texts = all_chunk_texts[i:i+B]
                enc = self.mtl_tokenizer(batch_texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
                input_ids = enc["input_ids"].to(self.device)
                attn = enc["attention_mask"].to(self.device)
                with torch.no_grad():
                    chunk_logits, doc_logits = self.mtl_model(
                        input_ids=input_ids,
                        attention_mask=attn
                    )
                    probs = torch.sigmoid(chunk_logits).cpu().numpy()
                for row in probs:
                    chunk_probs.append(row.tolist())
        else:
            # fallback: predict zeros
            chunk_probs = [[0.0]*len(self.label_set) for _ in all_chunk_texts]

        # assemble per-doc chunk dicts with preds using thresholds
        if self.thresholds is None:
            thresholds = np.full(len(self.label_set), 0.5, dtype=float)

        else:
            th = np.array(self.thresholds, dtype=float)

            # scalar → expand
            if th.ndim == 0:
                thresholds = np.full(len(self.label_set), float(th), dtype=float)

            # correct shape
            elif th.ndim == 1 and len(th) == len(self.label_set):
                thresholds = th

            # fallback safety
            else:
                thresholds = np.full(len(self.label_set), 0.5, dtype=float)

        per_doc_chunks_with_scores = [[] for _ in docs]
        for idx, (prob_row, doc_idx) in enumerate(zip(chunk_probs, doc_index_for_chunk)):
            pred_row = [1 if p >= thresholds[j] else 0 for j,p in enumerate(prob_row)]
            per_doc_chunks_with_scores[doc_idx].append({
                "chunk_id": idx,
                "text": all_chunk_texts[idx],
                "probs": prob_row,
                "preds": pred_row
            })

        # Aggregate chunk probs to document-level: mean pooling
        N = len(docs)
        L = len(self.label_set)
        doc_probs = np.zeros((N, L), dtype=float)
        for di in range(N):
            cps = per_doc_chunks_with_scores[di]
            if len(cps) == 0:
                continue
            arr = np.array([c["probs"] for c in cps])
            doc_probs[di] = arr.mean(axis=0)
        doc_preds = np.zeros_like(doc_probs, dtype=int)
        for i in range(N):
            if len(per_doc_chunks_with_scores[i]) == 0:
                continue
            chunk_arr = np.array([c["probs"] for c in per_doc_chunks_with_scores[i]])
            doc_preds[i] = (chunk_arr.max(axis=0) >= thresholds).astype(int)
            
        return {"label_set": self.label_set, "probs": doc_probs.tolist(), "preds": doc_preds, "chunks": per_doc_chunks_with_scores}

    # -------------------------
    # Generic document-level prediction (baseline or mtl based on availability)
    # -------------------------
    def predict_documents(self, docs: List[str], model_kind: str = "baseline", batch_size: int = 32) -> Dict[str, Any]:
        """
        model_kind: "baseline" or "mtl" or "auto"
        Returns dict with keys: label_set, probs (N,L), preds (N,L)
        """
        if model_kind == "auto":
            model_kind = "mtl" if self.mtl_model is not None else "baseline"

        if model_kind == "baseline":
            probs = self._baseline_predict_proba_for_texts(docs)
            th = np.array(self.thresholds) if self.thresholds is not None else np.array([0.5]*probs.shape[1])
            preds = (probs >= th).astype(int)
            return {"label_set": self.label_set, "probs": probs.tolist(), "preds": preds.tolist()}

        elif model_kind == "mtl":
            res = self.mtl_predict_documents(docs, batch_size=batch_size)
            return res
        else:
            raise ValueError("model_kind must be one of ['baseline','mtl','auto']")

