<<<<<<< HEAD
# Legal-ML-Pipeline (CUAD)

End-to-end pipeline for **multi-label legal clause classification** on the
[CUAD v1](https://www.atticusprojectai.org/cuad) dataset.

The repo implements **Approach A**:

- **TF-IDF baselines** — One-vs-Rest Logistic Regression and Linear SVM
- **Domain-Adaptive Pretraining (DAP)** on LegalBERT
- **Multi-Task Learning (MTL)** with chunk-level + document-level heads
- **Long-document chunking** (`max_len=512`, `stride=128`)
- **Per-label threshold tuning**, robustness tests, and ablations
- **FastAPI** inference server + **Streamlit** dashboard
- **Docker / docker-compose** for reproducible deployment

---

## Repo layout

```
legal-ml-pipeline/
├── configs/                 # YAML configs (baseline, DAP, MTL, eval)
├── data/
│   └── raw/CUAD_v1/         # place CUAD here (gitignored)
├── src/
│   ├── data/                # loaders, splitters
│   ├── preprocessing/       # cleaning, chunking, tokenization
│   ├── baselines/           # TF-IDF + LR / SVM
│   ├── dap/                 # domain-adaptive pretraining
│   ├── mtl/                 # multi-task learning
│   ├── infer/               # inference helpers
│   ├── experiments/         # training / eval entrypoints
│   └── utils/
├── tools/                   # ablation runner, robustness suite, ONNX export
├── scripts/                 # shell entrypoints + figure generation
├── deployment/
│   ├── fastapi_server.py    # REST API
│   └── streamlit_app.py     # Streamlit demo
├── models/                  # saved sklearn models + transformer checkpoints
├── artifacts/               # eval outputs, thresholds, figures, ablations
├── tests/                   # pytest suite
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quickstart

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> `requirements.txt` pins the PyTorch CUDA 11.8 wheel index. If you need
> CPU-only or a different CUDA version, edit the `--index-url` line first.

### 2. Data

Download CUAD v1 and unpack it into:

```
data/raw/CUAD_v1/
├── full_contract_txt/
├── full_contract_pdf/
├── label_group_xlsx/
└── master_clauses.csv
```

The contract bodies are gitignored — only the master CSV / metadata are
tracked.

### 3. Train baselines

```bash
python -m src.baselines.train --config configs/baseline_config.yaml
```

Saved artifacts:

- `models/tfidf_lr_cuad.joblib`
- `models/tfidf_svm_cuad.joblib`

### 4. DAP + MTL

```bash
python -m src.dap.run_dap   --config configs/dap_config.yaml
python -m src.mtl.train_mtl --config configs/mtl_config.yaml
```

### 5. Evaluation

```bash
python -m src.experiments.evaluate --config configs/eval_config.yaml
```

Writes per-label thresholds and metrics to `artifacts/eval/` and
`artifacts/thresholds/`.

### 6. Robustness + ablations

```bash
python tools/robustness_suite.py
python tools/ablation_runner.py
```

Outputs land in `artifacts/robustness/` and `artifacts/ablations/`.

### 7. Figures

```bash
python scripts/figures.py
```

---

## Serving

### FastAPI

```bash
bash scripts/run_fastapi.sh
# or
uvicorn deployment.fastapi_server:app --host 0.0.0.0 --port 8000
```

### Streamlit dashboard

```bash
bash scripts/run_streamlit.sh
# or
streamlit run deployment/streamlit_app.py
```

### Docker

```bash
docker compose up --build
```

- API:        http://localhost:8000
- Streamlit:  http://localhost:8501

---

## Tests

```bash
pytest -q
```

---

## Results snapshot

See `SVM Results.txt` and `MTL Results.txt` in the repo root for the
latest macro-/micro-F1 numbers on the held-out split.

---

## Citation

If you use this pipeline, please cite the CUAD dataset:

> Hendrycks, D., Burns, C., Chen, A., Ball, S. *CUAD: An Expert-Annotated
> NLP Dataset for Legal Contract Review.* NeurIPS 2021 Datasets and
> Benchmarks Track.
=======
# Legal-Document-Classifier
>>>>>>> cb1c587b4e52e8d8c8bd44ac0f8a68f76fa2e3c2
