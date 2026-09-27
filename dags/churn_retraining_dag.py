"""
DAG de reentrenamiento automático para el modelo de churn.

Flujo: start -> check_data_availability -> detect_drift -> (branch)
       -> si hay drift: retrain_model -> reload_api_model -> end
       -> si no hay drift: skip_retraining -> end
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

import sys
import os

sys.path.insert(0, '/app/src')

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'churn_retraining_pipeline',
    default_args=default_args,
    description='Detección de drift y reentrenamiento automático del modelo de churn',
    schedule_interval='@weekly',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['mlops', 'churn', 'retraining'],
)


def check_data_availability():
    data_path = '/app/data/Telco-Customer-Churn.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No se encontró: {data_path}")
    import pandas as pd
    df = pd.read_csv(data_path)
    print(f"✅ Datos disponibles: {len(df)} registros")
    return True


def detect_drift_task(**context):
    from drift_detection import detect_drift, save_drift_metrics_to_mlflow

    print("🔍 Ejecutando detección de drift...")
    should_retrain, drift_share, report_path, used_real_data = detect_drift()

    try:
        save_drift_metrics_to_mlflow(drift_share, should_retrain, used_real_data, report_path)
    except Exception as e:
        print(f"⚠️ Error guardando en MLflow: {e}")

    context['ti'].xcom_push(key='should_retrain', value=should_retrain)
    context['ti'].xcom_push(key='drift_share', value=drift_share)

    print(f"📊 Drift share: {drift_share:.2%}")
    print(f"🔄 Reentrenar: {should_retrain}")
    return should_retrain


def decide_retrain(**context):
    should_retrain = context['ti'].xcom_pull(key='should_retrain', task_ids='detect_drift')
    return 'retrain_model' if should_retrain else 'skip_retraining'


def retrain_model(**context):
    import subprocess
    print("🔄 Iniciando reentrenamiento...")
    result = subprocess.run(
        ['python', '/app/src/pipeline/train.py'],
        capture_output=True, text=True, cwd='/app',
        env={**os.environ, 'MLFLOW_TRACKING_URI': 'sqlite:////mlflow_data/mlflow.db'}
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise Exception(f"Error en reentrenamiento: {result.stderr}")
    print("✅ Reentrenamiento completado")
    return True


def reload_api_model(**context):
    """
    Le pide a la API que recargue el modelo desde disco, vía HTTP.
    No usa el socket de Docker (evita depender de docker-compose / docker
    CLI instalados dentro del contenedor de Airflow, que es una fuente
    común de fallos silenciosos).
    """
    import requests
    print("🔄 Pidiendo a la API que recargue el modelo...")
    try:
        response = requests.post('http://api:8000/reload-model', timeout=30)
        response.raise_for_status()
        print(f"✅ API recargó el modelo: {response.json()}")
    except Exception as e:
        raise Exception(f"No se pudo recargar el modelo en la API: {e}")
    return True


start = EmptyOperator(task_id='start', dag=dag)

check_data = PythonOperator(
    task_id='check_data_availability',
    python_callable=check_data_availability,
    dag=dag,
)

detect_drift_op = PythonOperator(
    task_id='detect_drift',
    python_callable=detect_drift_task,
    dag=dag,
)

branch = BranchPythonOperator(
    task_id='decide_retrain',
    python_callable=decide_retrain,
    dag=dag,
)

retrain = PythonOperator(
    task_id='retrain_model',
    python_callable=retrain_model,
    dag=dag,
)

reload_api = PythonOperator(
    task_id='reload_api_model',
    python_callable=reload_api_model,
    dag=dag,
)

skip = EmptyOperator(task_id='skip_retraining', dag=dag)
end = EmptyOperator(task_id='end', dag=dag)

start >> check_data >> detect_drift_op >> branch
branch >> retrain >> reload_api >> end
branch >> skip >> end
