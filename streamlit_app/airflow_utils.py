"""
Cliente liviano para la API REST de Apache Airflow (adaptado de la
implementación usada por el equipo, simplificado a un solo archivo).
"""
import os
from datetime import datetime
import requests
from requests.auth import HTTPBasicAuth

AIRFLOW_URL = os.getenv("AIRFLOW_URL", "http://airflow-webserver:8080")
AIRFLOW_USER = os.getenv("AIRFLOW_USER", "airflow")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "airflow")
DAG_ID = "churn_retraining_pipeline"

_auth = HTTPBasicAuth(AIRFLOW_USER, AIRFLOW_PASSWORD)


def check_airflow_connection() -> bool:
    try:
        r = requests.get(f"{AIRFLOW_URL}/health", auth=_auth, timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def trigger_dag(conf: dict = None):
    try:
        r = requests.post(
            f"{AIRFLOW_URL}/api/v1/dags/{DAG_ID}/dagRuns",
            json={"conf": conf or {}},
            auth=_auth,
            timeout=30,
        )
        if r.status_code in (200, 201):
            body = r.json()
            return True, body.get("dag_run_id", ""), "DAG disparado correctamente"
        return False, "", f"Error {r.status_code}: {r.text}"
    except Exception as e:
        return False, "", f"Error: {e}"


def get_dag_run(dag_run_id: str):
    try:
        r = requests.get(
            f"{AIRFLOW_URL}/api/v1/dags/{DAG_ID}/dagRuns/{dag_run_id}",
            auth=_auth, timeout=10,
        )
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def get_task_instances(dag_run_id: str):
    try:
        r = requests.get(
            f"{AIRFLOW_URL}/api/v1/dags/{DAG_ID}/dagRuns/{dag_run_id}/taskInstances",
            auth=_auth, timeout=10,
        )
        return r.json().get("task_instances", []) if r.status_code == 200 else []
    except Exception:
        return []


def get_recent_dag_runs(limit: int = 10):
    try:
        r = requests.get(
            f"{AIRFLOW_URL}/api/v1/dags/{DAG_ID}/dagRuns",
            params={"limit": limit, "order_by": "-start_date"},
            auth=_auth, timeout=10,
        )
        return r.json().get("dag_runs", []) if r.status_code == 200 else []
    except Exception:
        return []
