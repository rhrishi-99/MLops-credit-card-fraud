Week 1 — MLOps Journal
March 11, 2026
Got the pipeline running. Data → preprocess → train → MLflow → model registry. The whole loop.

XGBoost with scale_pos_weight=577 for the class imbalance. First run: F1 0.855, AUC 0.976. Numbers are fine — the point this week was the plumbing, not the model.

MLflow logs everything automatically now. Every run, every param, every metric. Open localhost:5000 and it's all there. This is the thing people skip and regret later.

Also saved train_baseline.parquet for Evidently in Week 4. Easy thing to forget.
Next: Airflow DAG. Same logic, runs on a schedule.



# Week 1 — What We Did

## Goal
Build the foundation of an MLOps pipeline.

## Dataset
UCI Credit Card Fraud — 284k transactions, heavily imbalanced (577:1 legit to fraud ratio).

## Steps

**1. Preprocessing** — Loaded data, split chronologically (train/prod), scaled features, saved `train_baseline.parquet` for drift detection later.

**2. Training** — XGBoost with `scale_pos_weight=577` to handle imbalance. Results: F1 0.855, AUC 0.976.

**3. MLflow Tracking** — Every run automatically logs params, metrics, and the model binary. Viewable at `localhost:5000`.

**4. Model Registry** — New models register as `@challenger`. Only promoted to `@champion` (production) if F1 beats the current champion. Prevents shipping a worse model.

## Fixed Along the Way
- Removed deprecated `use_label_encoder` param from XGBoost
- Rewrote registry to use MLflow aliases instead of deprecated stages API

## Next
Week 2 — Airflow DAG to automate this entire pipeline on a schedule.