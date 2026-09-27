import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import joblib
import os
import warnings
import sqlite3
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN DE MLflow
# Respeta MLFLOW_TRACKING_URI si viene del entorno (Docker la fija a
# sqlite:////mlflow_data/mlflow.db); si no, usa esa misma ruta por defecto.
# En GitHub Actions (sin Docker) se sobreescribe a una ruta relativa,
# porque el runner no tiene permiso de escritura en la raíz "/".
# ============================================================
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'sqlite:////mlflow_data/mlflow.db')
DB_PATH = MLFLOW_TRACKING_URI.replace('sqlite:///', '', 1)
print(f"🔍 MLflow URI: {MLFLOW_TRACKING_URI}")

db_dir = os.path.dirname(DB_PATH)
if db_dir:
    os.makedirs(db_dir, exist_ok=True)

# Eliminar base de datos corrupta si existe
if os.path.exists(DB_PATH):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("SELECT 1 FROM sqlite_master LIMIT 1")
        conn.close()
    except Exception as e:
        print(f"⚠️ Base de datos corrupta: {e}. Eliminando...")
        os.remove(DB_PATH)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ============================================================
# COLUMNAS EXACTAS QUE USARÁ EL MODELO (SIN customerID)
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

MODEL_REGISTRY_NAME = "churn_model"

# Modelos candidatos a comparar. Agregar/quitar entradas aquí para experimentar.
CANDIDATE_MODELS = {
    "random_forest": (
        RandomForestClassifier,
        {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'random_state': 42}
    ),
    "logistic_regression": (
        LogisticRegression,
        {'max_iter': 1000, 'random_state': 42}
    ),
    "gradient_boosting": (
        GradientBoostingClassifier,
        {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'random_state': 42}
    ),
}


# ============================================================
# FUNCIONES DE DATOS (sin cambios respecto a la versión original)
# ============================================================
def load_and_preprocess_data():
    """Carga y preprocesa los datos"""
    print("📊 Cargando datos...")
    data_path = 'data/Telco-Customer-Churn.csv'

    if not os.path.exists(data_path):
        print("⚠️  Dataset no encontrado. Creando datos de ejemplo...")
        create_sample_data()

    df = pd.read_csv(data_path)
    print(f"✅ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")

    # Limpieza
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)

    # Separar features y target
    y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    # Seleccionar SOLO las columnas que el modelo usará (sin customerID)
    X = df[FEATURE_COLUMNS].copy()
    print(f"✅ Features seleccionadas: {len(X.columns)} columnas")
    print(f"   {list(X.columns)}")

    # Codificar variables categóricas
    le_dict = {}
    for col in CATEGORICAL_COLUMNS:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            le_dict[col] = le

    # Escalar variables numéricas
    scaler = StandardScaler()
    numerical_present = [c for c in NUMERICAL_COLUMNS if c in X.columns]
    X[numerical_present] = scaler.fit_transform(X[numerical_present])

    print(f"✅ Columnas finales del modelo: {list(X.columns)}")
    print(f"✅ Total columnas: {len(X.columns)}")

    return X, y, le_dict, scaler


def create_sample_data():
    """Crea datos de ejemplo"""
    np.random.seed(42)
    n_samples = 7043

    data = {
        'customerID': [f'CUST-{i:04d}' for i in range(n_samples)],
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'Partner': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples),
        'tenure': np.random.randint(0, 72, n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples, p=[0.4, 0.5, 0.1]),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.4, 0.2]),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.4, 0.4, 0.2]),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.4, 0.4, 0.2]),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.25, 0.2]),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n_samples),
        'MonthlyCharges': np.random.uniform(18, 120, n_samples),
        'TotalCharges': np.random.uniform(0, 9000, n_samples)
    }

    churn_prob = 0.5 * (np.array(data['tenure']) < 12) + \
                 0.3 * (np.array(data['Contract']) == 'Month-to-month') + \
                 0.2 * (np.array(data['InternetService']) == 'Fiber optic')
    churn_prob = np.clip(churn_prob / 1.0, 0, 1)
    churn = np.random.binomial(1, churn_prob)
    data['Churn'] = ['Yes' if c == 1 else 'No' for c in churn]

    df = pd.DataFrame(data)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/Telco-Customer-Churn.csv', index=False)
    print(f"✅ Datos de ejemplo creados: {len(df)} registros")


# ============================================================
# ENTRENAMIENTO Y COMPARACIÓN DE MODELOS
# ============================================================
def train_and_log_model(model_name, model_cls, params, X_train, X_test, y_train, y_test):
    """Entrena un modelo, calcula métricas y las registra como un run de MLflow."""
    with mlflow.start_run(run_name=model_name) as run:
        print(f"🧠 Entrenando {model_name}...")
        model = model_cls(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }

        mlflow.log_param("model_type", model_name)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        print(f"   {model_name} -> f1={metrics['f1_score']:.4f}  roc_auc={metrics['roc_auc']:.4f}")
        return run.info.run_id, model, metrics


def train_model():
    """Entrena y compara varios modelos, guarda el mejor y lo registra en el Model Registry."""
    print("🚀 Iniciando entrenamiento con comparación de modelos...")

    X, y, le_dict, scaler = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"📊 Datos: Train {len(X_train)}, Test {len(X_test)}")

    experiment_name = "churn_prediction"
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
            print(f"📝 Experimento creado: {experiment_name}")
        mlflow.set_experiment(experiment_name)

        results = {}
        for model_name, (model_cls, params) in CANDIDATE_MODELS.items():
            run_id, model, metrics = train_and_log_model(
                model_name, model_cls, params, X_train, X_test, y_train, y_test
            )
            results[model_name] = {"run_id": run_id, "model": model, "metrics": metrics}

        # Selecciona el mejor modelo por f1_score (cámbialo a 'roc_auc' si prefieres ese criterio)
        best_name = max(results, key=lambda name: results[name]["metrics"]["f1_score"])
        best = results[best_name]
        print(f"\n🏆 Mejor modelo: {best_name} (f1_score={best['metrics']['f1_score']:.4f})")

        # Guardar artefactos del mejor modelo para que la API los use directamente
        os.makedirs('models', exist_ok=True)
        joblib.dump(best["model"], 'models/model.joblib')
        joblib.dump(scaler, 'models/scaler.joblib')
        joblib.dump(le_dict, 'models/label_encoders.joblib')

        # Registrar el mejor modelo en el MLflow Model Registry
        try:
            client = MlflowClient()
            model_uri = f"runs:/{best['run_id']}/model"
            registered = mlflow.register_model(model_uri=model_uri, name=MODEL_REGISTRY_NAME)
            client.transition_model_version_stage(
                name=MODEL_REGISTRY_NAME,
                version=registered.version,
                stage="Production",
                archive_existing_versions=True
            )
            print(f"📦 Modelo registrado: {MODEL_REGISTRY_NAME} v{registered.version} (stage=Production)")
        except Exception as e:
            print(f"⚠️ No se pudo registrar el modelo en el registry: {e}")

        print("\n📊 Comparación de modelos:")
        for name, r in results.items():
            m = r["metrics"]
            marker = "🏆" if name == best_name else "  "
            print(f"{marker} {name:20s} f1={m['f1_score']:.4f}  roc_auc={m['roc_auc']:.4f}  acc={m['accuracy']:.4f}")

        print(f"\n✅ Modelo guardado. Columnas ({len(X.columns)}): {list(X.columns)}")

        return best["model"], best["metrics"]

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    print("=" * 60)
    train_model()
    print("=" * 60)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
