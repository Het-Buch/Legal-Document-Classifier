# deployment/streamlit_app.py

import sys
from pathlib import Path

# -------------------------------------------------
# Make project root visible for `src` imports
# -------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import streamlit as st
import numpy as np
from src.infer.predictor import UnifiedPredictor
from src.preprocessing.chunker import chunk_tokens
from transformers import AutoTokenizer

import re

LEGAL_KEYWORDS = {
    "agreement", "party", "parties", "shall", "may", "must",
    "license", "term", "termination", "governed", "law",
    "effective", "notice", "liability", "assign", "transfer"
}

LEGAL_MODALS = {"shall", "may", "must", "will"}

def is_likely_legal_text(text: str) -> bool:
    tokens = re.findall(r"\b\w+\b", text.lower())

    # 1️⃣ length check
    if len(tokens) < 8:
        return False

    # 2️⃣ keyword presence
    keyword_hits = sum(1 for t in tokens if t in LEGAL_KEYWORDS)
    if keyword_hits < 2:
        return False

    # 3️⃣ legal modal verbs
    if not any(t in LEGAL_MODALS for t in tokens):
        return False

    return True

def confidence_label(p: float) -> str:
    if p >= 0.85:
        return "Very High"
    elif p >= 0.70:
        return "High"
    elif p >= 0.50:
        return "Medium"
    else:
        return "Low"


def document_confidence(probs):
    if not probs:
        return "No confident clauses detected"

    max_p = max(probs)
    count_high = sum(p >= 0.7 for p in probs)

    if max_p >= 0.85 and count_high >= 2:
        return "Very High confidence document"
    elif max_p >= 0.70:
        return "High confidence document"
    elif max_p >= 0.50:
        return "Medium confidence document"
    else:
        return "Low confidence document"


# -------------------------------------------------
# Streamlit page config
# -------------------------------------------------
st.set_page_config(
    page_title="Legal Clause Classification",
    layout="wide"
)

st.title("Legal Clause Classification")
st.caption("Domain-Adaptive Pretrained LegalBERT with Multi-Task Learning")

# -------------------------------------------------
# Sidebar: model & artifact paths
# -------------------------------------------------
st.sidebar.header("Model & Artifacts")

baseline_path = st.sidebar.text_input(
    "Baseline model path (joblib)",
    value="models/tfidf_svm_cuad.joblib"
)

thresholds_path = st.sidebar.text_input(
    "Thresholds JSON",
    value="artifacts/thresholds/thresholds_mtl.json"
)

mtl_ckpt = st.sidebar.text_input(
    "MTL checkpoint (.pt)",
    value="models/checkpoints/mtl/mtl_bundle.pt"
)

mtl_backbone = st.sidebar.text_input(
    "MTL backbone path",
    value="nlpaueb/legal-bert-base-uncased"
)

model_kind = st.sidebar.text_input(
    "Model to use",
    value= "mtl"
)

load_button = st.sidebar.button("Load predictor")

# -------------------------------------------------
# Session state
# -------------------------------------------------
if "predictor" not in st.session_state:
    st.session_state.predictor = None

# -------------------------------------------------
# Load predictor
# -------------------------------------------------
if load_button:
    try:
        predictor = UnifiedPredictor(
            baseline_path if Path(baseline_path).exists() else None,
            thresholds_path if Path(thresholds_path).exists() else None,
            mtl_ckpt if Path(mtl_ckpt).exists() else None,
            mtl_backbone
        )
        st.session_state.predictor = predictor
        st.sidebar.success(
            "Predictor loaded successfully."
        )
    except Exception as e:
        st.sidebar.error(f"Failed to load predictor:\n{e}")

predictor = st.session_state.predictor

# -------------------------------------------------
# Input section
# -------------------------------------------------
st.header("Input Document")

doc_text = st.text_area(
    "Paste a legal clause or contract text",
    value="This Agreement may be terminated by either party upon thirty (30) days written notice.",
    height=250
)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("Predict"):
    if predictor is None:
        st.error("Please load a predictor from the sidebar first.")
        st.stop()

    if not is_likely_legal_text(doc_text):
        st.warning(
            "⚠️ The input does not appear to be a legal contract.\n\n"
            "Please provide a contractual clause or agreement text."
        )
        st.stop()

    with st.spinner("Running prediction..."):
        result = predictor.predict_documents(
            [doc_text],
            model_kind=model_kind
        )

        label_set = result["label_set"]
        probs = result["probs"][0]
        
        st.subheader("Detected Clauses")

        TOP_K = 5
        MIN_PROB = 0.30
        DEMO_THRESHOLD = 0.60

        rows = []

        for i, lbl in enumerate(label_set):
            if not lbl.endswith("-Answer"):
                continue

            prob = float(probs[i])

            if prob >= MIN_PROB:
                rows.append({
                    "Clause": lbl.replace("-Answer", ""),
                    "Probability": round(prob, 3),
                    "Confidence": confidence_label(prob)
                })

        # Sort by probability
        rows = sorted(rows, key=lambda x: x["Probability"], reverse=True)[:TOP_K]

        if rows:
            st.table(rows)

            doc_conf = document_confidence([r["Probability"] for r in rows])
            st.success(f"📄 Document-level confidence: **{doc_conf}**")
        else:
            st.info("No clauses detected above threshold.")

        # -------------------------------------------------
        # Chunk highlighting
        # -------------------------------------------------
        st.subheader("Chunk-level Highlighting")

        try:
            tokenizer = AutoTokenizer.from_pretrained(mtl_backbone)
            enc = tokenizer(
                doc_text,
                truncation=False,
                add_special_tokens=False
            )
            ids = enc["input_ids"]

            chunks_meta = chunk_tokens(
                ids,
                max_len=512,
                stride=128
            )

            chunk_texts = [
                tokenizer.decode(
                    ids[c["start"]:c["end"]],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                for c in chunks_meta
            ]
        except Exception:
            tokens = doc_text.split()
            chunks_meta = chunk_tokens(
                tokens,
                max_len=512,
                stride=128
            )
            chunk_texts = [
                " ".join(tokens[c["start"]:c["end"]])
                for c in chunks_meta
            ]

        # Get chunk scores
        if model_kind in ["auto", "baseline"]:
            try:
                chunk_infos = predictor.baseline_chunk_scores(
                    [doc_text],
                    lambda _: [
                        {"chunk_id": i, "text": chunk_texts[i]}
                        for i in range(len(chunk_texts))
                    ]
                )[0]
            except Exception:
                chunk_infos = []
        else:
            try:
                mtl_chunk_res = predictor.mtl_predict_documents(
                    [doc_text],
                    max_len=512,
                    stride=128
                )
                chunk_infos = mtl_chunk_res["chunks"][0]
            except Exception:
                chunk_infos = []

        CHUNK_THRESHOLD = 0.60   # strict for paper demo
        MIN_ALPHA = 0.08        # almost invisible
        MAX_ALPHA = 0.85

        for ci in chunk_infos:
            highlight_probs = []

            for i, lbl in enumerate(predictor.label_set):
                # ONLY Answer labels
                if not lbl.endswith("-Answer"):
                    continue

                # Document-level must be confident
                if probs[i] < DEMO_THRESHOLD:
                    continue

                # Chunk-level must be confident
                if ci.get("probs") and ci["probs"][i] >= CHUNK_THRESHOLD:
                    highlight_probs.append(ci["probs"][i])

            # ❌ No confident clause → NO highlight
            if not highlight_probs:
                alpha = 0.0
            else:
                max_p = max(highlight_probs)
                alpha = min(MAX_ALPHA, MIN_ALPHA + max_p * 0.85)

            # White background if no highlight
            if alpha == 0.0:
                color = "rgba(255,255,255,0.0)"
            else:
                color = f"rgba(255, 0, 0, {alpha})"

            st.markdown(
                f"""
                <div style="
                    background:{color};
                    padding:10px;
                    border-radius:6px;
                    margin-bottom:8px;
                ">
                {ci['text'][:500]}
                </div>
                """,
                unsafe_allow_html=True
            )

        st.success("Prediction completed.")
        # st.subheader(alpha)