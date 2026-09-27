"""
Módulo de detección de drift usando Evidently AI.

A diferencia de una simulación estática, este módulo usa como "datos de
producción" las predicciones reales que la API ya registra en
monitoring/predictions.db (ver src/api/app.py). Si todavía no hay
suficientes predicciones reales, cae de vuelta a una simulación (clara-
mente marcada como tal) solo para poder demostrar el flujo.
"""
import os
import json
import sqlite3
from datetime import datetime

import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric

# ============================================================
# RUTAS (configurables por variable de entorno, ver nota del
# error de permisos que ya nos pasó con /mlflow_data: todo lo
# que pueda correr fuera de Docker debe poder sobreescribirse)
# ============================================================
DATA_PATH = os.getenv('DATA_PATH', 'data/Telco-Customer-Churn.csv')
MONITORING_DB_PATH = os.getenv('MONITORING_DB_PATH', 'monitoring/predictions.db')
REPORTS_DIR = os.getenv('REPORTS_DIR', 'reports')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'sqlite:////mlflow_data/mlflow.db')

DRIFT_THRESHOLD = 0.3
MIN_PRODUCTION_SAMPLES = 30  # mínimo de predicciones reales para no usar simulación

FEATURE_COLUMNS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]


def load_reference_data():
    """Datos de referencia: el dataset de entrenamiento original."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"No se encontró el dataset de referencia en {DATA_PATH}. "
            f"Corre el entrenamiento primero (genera/usa data/Telco-Customer-Churn.csv)."
        )
    df = pd.read_csv(DATA_PATH)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)
    return df[FEATURE_COLUMNS]


def load_production_data_from_predictions():
    """
    Reconstruye un DataFrame de 'producción' a partir de las predicciones
    reales que la API fue logueando en monitoring/predictions.db.
    Retorna None si no hay suficientes registros todavía.
    """
    if not os.path.exists(MONITORING_DB_PATH):
        return None

    conn = sqlite3.connect(MONITORING_DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT input_json FROM predictions ORDER BY id DESC LIMIT 2000")
    rows = cur.fetchall()
    conn.close()

    if len(rows) < MIN_PRODUCTION_SAMPLES:
        return None

    records = [json.loads(r[0]) for r in rows]
    df = pd.DataFrame(records)

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        return None

    return df[FEATURE_COLUMNS]


def simulate_production_data(reference: pd.DataFrame) -> pd.DataFrame:
    """
    Fallback: simula un lote de producción con corrimiento intencional,
    SOLO para poder demostrar el flujo cuando aún no hay tráfico real
    suficiente (menos de MIN_PRODUCTION_SAMPLES predicciones reales).
    """
    print(f"⚠️  Menos de {MIN_PRODUCTION_SAMPLES} predicciones reales registradas. "
          f"Simulando datos de producción con drift para poder demostrar el flujo.")
    df_sim = reference.sample(n=min(500, len(reference)), random_state=42).copy()
    df_sim['MonthlyCharges'] = df_sim['MonthlyCharges'] * 1.25
    df_sim['tenure'] = (df_sim['tenure'] * 0.7).astype(int)
    mask = np.random.random(len(df_sim)) < 0.7
    df_sim.loc[mask, 'Contract'] = 'Month-to-month'
    return df_sim


def detect_drift():
    """Detecta drift entre los datos de entrenamiento y la producción real (o simulada)."""
    print("🔍 Iniciando detección de drift...")

    reference = load_reference_data()
    production = load_production_data_from_predictions()

    used_real_data = production is not None
    if not used_real_data:
        production = simulate_production_data(reference)

    print(f"📊 Referencia (entrenamiento): {len(reference)} registros")
    print(f"📊 Producción ({'real' if used_real_data else 'simulada'}): {len(production)} registros")

    report = Report(metrics=[DatasetDriftMetric()])
    report.run(reference_data=reference, current_data=production)

    result = report.as_dict()
    drift_result = result['metrics'][0]['result']

    drift_detected = drift_result.get('dataset_drift', False)
    drift_share = drift_result.get('drift_share', 0.0)

    print(f"📊 Drift detectado: {drift_detected}")
    print(f"📈 Porcentaje de features con drift: {drift_share * 100:.1f}%")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(REPORTS_DIR, f'drift_report_{timestamp}.html')
    report.save_html(report_path)
    print(f"📄 Reporte guardado en: {report_path}")

    should_retrain = drift_detected or drift_share > DRIFT_THRESHOLD

    if should_retrain:
        print("🚨 DRIFT SIGNIFICATIVO DETECTADO. Se recomienda reentrenar.")
    else:
        print("✅ Drift dentro de límites aceptables.")

    return should_retrain, drift_share, report_path, used_real_data


def save_drift_metrics_to_mlflow(drift_share, should_retrain, used_real_data, report_path):
    """Guarda métricas de drift en MLflow, en un experimento separado del de entrenamiento."""
    import mlflow

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("drift_monitoring")

        with mlflow.start_run(run_name=f"drift_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_metric("drift_share", drift_share)
            mlflow.log_metric("should_retrain", 1 if should_retrain else 0)
            mlflow.log_param("used_real_production_data", used_real_data)
            if os.path.exists(report_path):
                mlflow.log_artifact(report_path)
            print("📝 Métricas de drift registradas en MLflow (experimento 'drift_monitoring')")
    except Exception as e:
        print(f"⚠️ Error guardando en MLflow: {e}")


if __name__ == "__main__":
    should_retrain, drift_share, report_path, used_real_data = detect_drift()
    save_drift_metrics_to_mlflow(drift_share, should_retrain, used_real_data, report_path)

    print(f"\n📊 Resultado final:")
    print(f"   Drift share: {drift_share:.2%}")
    print(f"   Datos usados: {'reales (monitoring/predictions.db)' if used_real_data else 'simulados'}")
    print(f"   Reentrenar: {should_retrain}")

    exit(1 if should_retrain else 0)
