from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '/opt/airflow')

from src.pipeline.preprocess import load_and_preprocess
from src.pipeline.train import train_and_log
from src.registry.promote import register_and_promote

default_args = {
    "owner": "rhrishi",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="fraud_detection_training",
    default_args=default_args,
    schedule_interval="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "fraud"],
) as dag:

    def ingest(**context):
        X_train, X_val, y_train, y_val, prod_df, scaler = load_and_preprocess("data/creditcard.csv")
        context["ti"].xcom_push(key="data_shape", value=str(X_train.shape))

    def validate(**context):
        shape = context["ti"].xcom_pull(key="data_shape", task_ids="ingest_data")
        rows = int(shape.split(",")[0].strip("("))
        if rows < 1000:
            raise ValueError(f"Too few rows: {rows}")
        print(f"Data valid — {rows} training rows")

    def train(**context):
        X_train, X_val, y_train, y_val, _, _ = load_and_preprocess("data/creditcard.csv")
        run_id, _, metrics = train_and_log(X_train, X_val, y_train, y_val)
        context["ti"].xcom_push(key="run_id", value=run_id)
        context["ti"].xcom_push(key="f1_score", value=metrics["f1_score"])

    def evaluate(**context):
        f1 = context["ti"].xcom_pull(key="f1_score", task_ids="train_model")
        print(f"F1 Score: {f1:.4f}")
        if f1 < 0.70:
            raise ValueError(f"F1 too low to promote: {f1:.4f}")

    def register(**context):
        run_id = context["ti"].xcom_pull(key="run_id", task_ids="train_model")
        f1 = context["ti"].xcom_pull(key="f1_score", task_ids="train_model")
        register_and_promote(run_id, f1)

    ingest_task   = PythonOperator(task_id="ingest_data",         python_callable=ingest)
    validate_task = PythonOperator(task_id="validate_data",       python_callable=validate)
    train_task    = PythonOperator(task_id="train_model",         python_callable=train)
    evaluate_task = PythonOperator(task_id="evaluate_model",      python_callable=evaluate)
    register_task = PythonOperator(task_id="register_if_better",  python_callable=register)

    ingest_task >> validate_task >> train_task >> evaluate_task >> register_task