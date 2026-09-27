import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "running"


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_model_info():
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["num_features"] == 19


def test_predict_valid_payload():
    payload = {
        "tenure": 12,
        "MonthlyCharges": 70.5,
        "TotalCharges": 846.0,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "churn_prediction" in body
    assert body["churn_prediction"] in (0, 1)
    assert 0 <= body["churn_probability"] <= 1
    assert body["churn_label"] in ("Churn", "No Churn")


def test_predict_missing_required_field():
    # Falta MonthlyCharges y TotalCharges, que no tienen default
    r = client.post("/predict", json={"tenure": 12})
    assert r.status_code == 422


def test_stats_endpoint_after_prediction():
    payload = {"tenure": 5, "MonthlyCharges": 50.0, "TotalCharges": 250.0}
    client.post("/predict", json=payload)

    r = client.get("/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["total_predictions"] >= 1
