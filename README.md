# MLOps — Credit Card Fraud Detection

End-to-end MLOps pipeline built for Google/Apple ML Platform roles.
Covers data ingestion, experiment tracking, model registry, and automated promotion.

## Project Structure
```
fraud-mlops/
├── main.py
├── src/
│   ├── pipeline/
│   │   ├── preprocess.py     # data loading, chronological split, feature scaling
│   │   └── train.py          # XGBoost training + MLflow logging
│   └── registry/
│       └── promote.py        # champion/challenger promotion logic
└── data/                     # gitignored — see Dataset section
```

## Setup
```bash
pip install mlflow xgboost scikit-learn pandas numpy evidently langgraph fastapi uvicorn
```

## Dataset

Download from Kaggle and place in `data/`:
```bash
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcard.zip -d data/
```

## Run
```bash
python main.py
mlflow ui   # http://localhost:5000
```

## How It Works

1. Data is split **chronologically** — first 80% trains, last 20% simulates production
2. XGBoost trains with `scale_pos_weight=577` to handle the 577:1 class imbalance
3. Every run logs params, metrics, and model binary to MLflow automatically
4. New models register as `@challenger` — only promoted to `@champion` if F1 beats the current production model

## Results (Week 1)

| Metric | Value |
|--------|-------|
| F1 Score | 0.8553 |
| ROC-AUC | 0.9760 |

## Roadmap

- [x] Week 1 — Data pipeline + MLflow tracking + model registry
- [ ] Week 2 — Airflow DAG orchestration
- [ ] Week 3 — FastAPI + Docker + GCP Cloud Run
- [ ] Week 4 — Evidently drift detection
- [ ] Week 5 — LangGraph monitoring agent
- [ ] Week 6 — Human-in-the-loop + Slack approvals