"""
Cliente liviano para la API REST de MLflow. No importa el paquete
`mlflow` (para no inflar la imagen de Streamlit); solo hace requests
HTTP directos a los endpoints estables de la REST API de MLflow.
"""
import os
import requests

MLFLOW_URL = os.getenv("MLFLOW_URL", "http://mlflow-ui:5000")


def get_experiment_id(experiment_name: str):
    try:
        r = requests.get(
            f"{MLFLOW_URL}/api/2.0/mlflow/experiments/get-by-name",
            params={"experiment_name": experiment_name},
            timeout=10,
        )
        if r.status_code == 200:
            return r.json()["experiment"]["experiment_id"]
        return None
    except Exception:
        return None


def search_runs(experiment_name: str, max_results: int = 50):
    """Devuelve una lista de runs (mas reciente primero) con sus métricas y parámetros."""
    exp_id = get_experiment_id(experiment_name)
    if exp_id is None:
        return []

    try:
        r = requests.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/runs/search",
            json={
                "experiment_ids": [exp_id],
                "max_results": max_results,
                "order_by": ["attribute.start_time DESC"],
            },
            timeout=10,
        )
        if r.status_code != 200:
            return []

        runs = r.json().get("runs", [])
        parsed = []
        for run in runs:
            info = run.get("info", {})
            data = run.get("data", {})
            metrics = {m["key"]: m["value"] for m in data.get("metrics", [])}
            params = {p["key"]: p["value"] for p in data.get("params", [])}
            parsed.append({
                "run_id": info.get("run_id"),
                "run_name": info.get("run_name", ""),
                "status": info.get("status"),
                "start_time": info.get("start_time"),
                "metrics": metrics,
                "params": params,
            })
        return parsed
    except Exception:
        return []


def get_registered_model(name: str):
    """Devuelve las versiones registradas de un modelo (incluyendo su stage)."""
    try:
        r = requests.get(
            f"{MLFLOW_URL}/api/2.0/mlflow/registered-models/get",
            params={"name": name},
            timeout=10,
        )
        if r.status_code == 200:
            return r.json().get("registered_model", {})
        return None
    except Exception:
        return None
