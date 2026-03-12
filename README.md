# MLOps — Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![MLflow](https://img.shields.io/badge/MLflow-2.10-orange) ![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

A production-grade MLOps pipeline for credit card fraud detection, built as part of a 6-week roadmap targeting ML Platform engineering roles at top-tier companies.

---

## Overview

This project builds an end-to-end ML pipeline to preprocess transaction data, train fraud detection models, track experiments with MLflow, and manage model versions with automated promotion logic.

The dataset is highly imbalanced (577:1), making fraud detection a realistic real-world ML challenge.

---

## Project Structure

```
MLops-credit-card-fraud/
├── main.py                     # Main entry point
├── journal.md                  # Development notes
├── requirements.txt
├── data/                       # Dataset (not included in repo)
├── mlruns/                     # MLflow tracking artifacts
└── src/
    ├── pipeline/
    │   ├── preprocess.py       # Chronological split, feature scaling, baseline export
    │   └── train.py            # XGBoost training + MLflow logging
    └── registry/
        └── promote.py          # Champion/challenger promotion logic
```

---

## Setup

### Prerequisites
- Python 3.10+
- pip
- Kaggle account (for dataset download)

### Installation

```bash
git clone https://github.com/rhrishi-99/MLops-credit-card-fraud.git
cd MLops-credit-card-fraud
pip install -r requirements.txt
```

---

## Dataset

This project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle (284,807 transactions, 492 fraud cases).

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcard.zip -d data/
```

> The dataset is excluded from this repository due to its size (150MB).

---

## Usage

### Run the pipeline

```bash
python main.py
```

### Launch MLflow UI

```bash
mlflow ui
```

Navigate to `http://localhost:5000` to view experiments, metrics, parameters, and registered model versions.

---

## Pipeline Design

### Preprocessing — `src/pipeline/preprocess.py`
Data is split **chronologically** — first 80% for training, last 20% simulates incoming production traffic. This avoids data leakage that random splits introduce. The `Amount` feature is scaled; V1–V28 are already PCA-transformed. Training distribution is saved as `train_baseline.parquet` for downstream drift detection.

### Training — `src/pipeline/train.py`
XGBoost with `scale_pos_weight=577` to handle class imbalance. Every run logs hyperparameters, evaluation metrics, and the model binary to MLflow automatically.

### Model Registry — `src/registry/promote.py`
New models register as `@challenger`. Promotion to `@champion` (production) only occurs if the new model's F1 strictly exceeds the current champion's — preventing accidental regression in production.

## Results (Week 1 Baseline)

| Metric | Value |
|--------|-------|
| F1 Score | 0.8553 |
| ROC-AUC | 0.9760 |
| Training samples | ~182,000 |
| Class imbalance ratio | 577:1 |

---


## Roadmap

- [x] Week 1 — Data pipeline, MLflow experiment tracking, champion/challenger model registry
- [ ] Week 2 — Apache Airflow DAG: `ingest → validate → train → evaluate → register_if_better`
- [ ] Week 3 — FastAPI serving, Dockerization, GCP Cloud Run deployment
- [ ] Week 4 — Evidently drift detection against training baseline
- [ ] Week 5 — LangGraph autonomous monitoring agent (ReAct pattern)
- [ ] Week 6 — Human-in-the-loop Slack approval workflow + audit trail

---

## Acknowledgements

- ULB Machine Learning Group (dataset creators)
- MLflow documentation
- Kaggle community

---

*Rhrishi R G · rgrhrishi@gmail.com · PES University, Bengaluru*