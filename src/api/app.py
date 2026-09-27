import os
import json
import sqlite3
from datetime import datetime, timezone

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Churn Prediction API")

MODEL_PATH = os.getenv('MODEL_PATH', 'models/model.joblib')
SCALER_PATH = os.getenv('SCALER_PATH', 'models/scaler.joblib')
ENCODER_PATH = os.getenv('ENCODER_PATH', 'models/label_encoders.joblib')

# ============================================================
# MONITOREO: SQLite para loggear cada predicción realizada
# ============================================================
MONITORING_DB_PATH = os.getenv('MONITORING_DB_PATH', 'monitoring/predictions.db')


def init_monitoring_db():
    db_dir = os.path.dirname(MONITORING_DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(MONITORING_DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            input_json TEXT NOT NULL,
            churn_prediction INTEGER NOT NULL,
            churn_probability REAL NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def log_prediction(input_data: dict, prediction: int, probability: float):
    try:
        conn = sqlite3.connect(MONITORING_DB_PATH)
        conn.execute(
            "INSERT INTO predictions (timestamp, input_json, churn_prediction, churn_probability) "
            "VALUES (?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                json.dumps(input_data),
                prediction,
                probability,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"No se pudo loggear la predicción: {e}")


init_monitoring_db()

# ============================================================
# COLUMNAS EXACTAS EN EL ORDEN EXACTO DEL ENTRENAMIENTO
# ============================================================
FEATURE_COLUMNS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

CATEGORICAL_COLUMNS = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

NUMERICAL_COLUMNS = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']


class CustomerData(BaseModel):
    # Numéricas
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    SeniorCitizen: int = 0

    # Categóricas con valores por defecto
    gender: str = "Male"
    Partner: str = "No"
    Dependents: str = "No"
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "DSL"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"


class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    churn_label: str


try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODER_PATH)
    logger.info(f"✅ Artefactos cargados. Columnas esperadas: {FEATURE_COLUMNS}")
except Exception as e:
    logger.error(f"❌ Error cargando artefactos: {e}")
    model = None


@app.get("/")
async def root():
    return {"message": "Churn Prediction API", "status": "running", "model_loaded": model is not None}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/model-info")
async def model_info():
    return {
        "model_type": "RandomForestClassifier",
        "is_loaded": model is not None,
        "expected_features": FEATURE_COLUMNS,
        "num_features": len(FEATURE_COLUMNS)
    }


@app.get("/stats")
async def stats():
    """Estadísticas de monitoreo sobre las predicciones realizadas hasta ahora."""
    if not os.path.exists(MONITORING_DB_PATH):
        return {"total_predictions": 0, "churn_rate": None, "avg_probability": None, "recent_predictions": []}

    conn = sqlite3.connect(MONITORING_DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*), AVG(churn_prediction), AVG(churn_probability) FROM predictions")
    total, churn_rate, avg_proba = cur.fetchone()

    cur.execute(
        "SELECT timestamp, churn_prediction, churn_probability FROM predictions ORDER BY id DESC LIMIT 10"
    )
    recent = [
        {"timestamp": r[0], "churn_prediction": r[1], "churn_probability": r[2]}
        for r in cur.fetchall()
    ]
    conn.close()

    return {
        "total_predictions": total or 0,
        "churn_rate": round(churn_rate, 4) if churn_rate is not None else None,
        "avg_probability": round(avg_proba, 4) if avg_proba is not None else None,
        "recent_predictions": recent,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(data: CustomerData):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    try:
        # Convertir a diccionario
        input_data = data.dict()
        logger.info(f"Datos recibidos: {input_data}")

        # Crear DataFrame con TODAS las columnas en el ORDEN EXACTO
        df_input = pd.DataFrame([input_data])

        # Asegurar que TODAS las columnas estén presentes
        for col in FEATURE_COLUMNS:
            if col not in df_input.columns:
                logger.warning(f"Columna faltante: {col}. Agregando valor por defecto.")
                df_input[col] = 0

        # Reordenar EXACTAMENTE como FEATURE_COLUMNS
        df_input = df_input[FEATURE_COLUMNS]

        logger.info(f"Columnas antes de codificar: {list(df_input.columns)}")
        logger.info(f"Total columnas: {len(df_input.columns)}")

        # Codificar variables categóricas
        for col in CATEGORICAL_COLUMNS:
            if col in df_input.columns and col in encoders:
                try:
                    df_input[col] = encoders[col].transform(df_input[col].astype(str))
                except ValueError:
                    df_input[col] = encoders[col].transform([encoders[col].classes_[0]])[0]

        # Escalar variables numéricas
        numerical_present = [c for c in NUMERICAL_COLUMNS if c in df_input.columns]
        if numerical_present:
            df_input[numerical_present] = scaler.transform(df_input[numerical_present])

        logger.info(f"Columnas finales: {list(df_input.columns)}")

        # Predecir
        proba = model.predict_proba(df_input)[0, 1]
        pred = int(proba > 0.5)

        # Loggear la predicción para monitoreo (no debe romper la respuesta si falla)
        log_prediction(input_data, pred, float(proba))

        return PredictionResponse(
            churn_prediction=pred,
            churn_probability=round(proba, 4),
            churn_label="Churn" if pred == 1 else "No Churn"
        )
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Error en la predicción: {str(e)}")
