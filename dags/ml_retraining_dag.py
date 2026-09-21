"""Basic Airflow DAG for periodic model retraining."""

from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator


def retrain_model() -> None:
    """Placeholder for the model training pipeline."""
    print("Run data preparation, training, evaluation, and model registration.")


with DAG(
    dag_id="ml_retraining_dag",
    start_date=datetime(2024, 1, 1),
    schedule="@weekly",
    catchup=False,
) as dag:
    retrain = PythonOperator(
        task_id="retrain_model",
        python_callable=retrain_model,
    )
