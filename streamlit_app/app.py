import os
from datetime import datetime

import requests
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

import mlflow_utils
import airflow_utils

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Predicción de Churn", page_icon="📉", layout="wide")
st.title("📉 Predicción de Deserción de Clientes (Churn)")
st.caption("Telco Customer Churn · FastAPI + MLflow + Airflow + Evidently AI")

tab_predict, tab_models, tab_drift, tab_status = st.tabs(
    ["🔮 Predicción", "📊 Comparación de Modelos", "🔍 Drift & Reentrenamiento", "⚙️ Estado del Sistema"]
)

# ============================================================
# PESTAÑA 1: PREDICCIÓN (tu formulario original, sin cambios)
# ============================================================
with tab_predict:
    with st.form("churn_form"):
        col1, col2 = st.columns(2)

        with col1:
            tenure = st.number_input("Meses como cliente (tenure)", min_value=0, max_value=100, value=12)
            monthly_charges = st.number_input("Cargo mensual (MonthlyCharges)", min_value=0.0, value=70.5, step=0.5)
            total_charges = st.number_input("Cargo total (TotalCharges)", min_value=0.0, value=846.0, step=1.0)
            senior_citizen = st.selectbox("¿Adulto mayor?", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
            gender = st.selectbox("Género", ["Male", "Female"])
            partner = st.selectbox("¿Tiene pareja?", ["Yes", "No"])
            dependents = st.selectbox("¿Tiene dependientes?", ["Yes", "No"])
            phone_service = st.selectbox("Servicio telefónico", ["Yes", "No"])
            multiple_lines = st.selectbox("Líneas múltiples", ["Yes", "No", "No phone service"])

        with col2:
            internet_service = st.selectbox("Servicio de internet", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Seguridad en línea", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Backup en línea", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Protección de dispositivo", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Soporte técnico", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            contract = st.selectbox("Contrato", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Facturación electrónica", ["Yes", "No"])
            payment_method = st.selectbox(
                "Método de pago",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
            )

        submitted = st.form_submit_button("Predecir")

    if submitted:
        payload = {
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "SeniorCitizen": senior_citizen,
            "gender": gender,
            "Partner": partner,
            "Dependents": dependents,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
        }

        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()

            proba = result["churn_probability"] * 100
            label = result["churn_label"]

            st.subheader(f"Resultado: {'🔴' if label == 'Churn' else '🟢'} {label}")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba,
                number={'suffix': "%"},
                title={'text': "Probabilidad de Churn"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred" if proba > 50 else "darkgreen"},
                    'steps': [
                        {'range': [0, 50], 'color': "#d4f4dd"},
                        {'range': [50, 100], 'color': "#fbd6d6"},
                    ],
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

        except requests.exceptions.RequestException as e:
            st.error(f"No se pudo conectar con la API ({API_URL}): {e}")

    st.divider()
    if st.button("📊 Ver estadísticas de predicciones (monitoreo)"):
        try:
            stats = requests.get(f"{API_URL}/stats", timeout=10).json()
            c1, c2 = st.columns(2)
            c1.metric("Total de predicciones", stats.get("total_predictions", 0))
            if stats.get("churn_rate") is not None:
                c2.metric("Tasa de churn observada", f"{stats['churn_rate'] * 100:.1f}%")
            st.json(stats)
        except requests.exceptions.RequestException as e:
            st.error(f"No se pudo obtener /stats: {e}")

# ============================================================
# PESTAÑA 2: COMPARACIÓN DE MODELOS (datos REALES desde MLflow,
# no hardcodeados — lee los runs que train.py fue registrando)
# ============================================================
with tab_models:
    st.header("📊 Comparación de modelos (MLflow)")
    st.caption("Lee en vivo los runs registrados por train.py en el experimento 'churn_prediction'.")

    if st.button("🔄 Actualizar desde MLflow", key="refresh_models"):
        st.rerun()

    runs = mlflow_utils.search_runs("churn_prediction", max_results=50)

    if not runs:
        st.warning(
            "No se encontraron runs todavía. Corre `./run.sh train` (o `docker-compose up train`) "
            "al menos una vez, o revisa que MLFLOW_URL apunte a http://mlflow-ui:5000."
        )
    else:
        rows = []
        for r in runs:
            m = r["metrics"]
            rows.append({
                "Fecha": datetime.fromtimestamp(r["start_time"] / 1000).strftime("%Y-%m-%d %H:%M") if r["start_time"] else "",
                "Modelo": r["params"].get("model_type", r["run_name"]),
                "Accuracy": float(m.get("accuracy", 0)),
                "Precision": float(m.get("precision", 0)),
                "Recall": float(m.get("recall", 0)),
                "F1-Score": float(m.get("f1_score", 0)),
                "ROC-AUC": float(m.get("roc_auc", 0)),
            })
        df = pd.DataFrame(rows)

        st.dataframe(
            df.style.format({
                "Accuracy": "{:.2%}", "Precision": "{:.2%}", "Recall": "{:.2%}",
                "F1-Score": "{:.2%}", "ROC-AUC": "{:.2%}",
            }),
            use_container_width=True,
        )

        # Comparación de la corrida más reciente (los últimos N runs, que
        # corresponden a los 3 modelos entrenados en la última llamada a train.py)
        st.subheader("🏆 Última comparación de modelos")
        latest_batch = df.head(3) if len(df) >= 3 else df
        fig = go.Figure()
        for metric in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
            fig.add_trace(go.Bar(name=metric, x=latest_batch["Modelo"], y=latest_batch[metric]))
        fig.update_layout(barmode="group", yaxis_range=[0, 1], height=420,
                           title="Métricas por modelo (última corrida de entrenamiento)")
        st.plotly_chart(fig, use_container_width=True)

        best_row = latest_batch.loc[latest_batch["F1-Score"].idxmax()]
        st.success(
            f"**Mejor modelo (por F1-Score):** {best_row['Modelo']} "
            f"— F1={best_row['F1-Score']:.2%}, ROC-AUC={best_row['ROC-AUC']:.2%}"
        )

    st.divider()
    st.subheader("📦 Model Registry")
    registered = mlflow_utils.get_registered_model("churn_model")
    if registered and registered.get("latest_versions"):
        for v in registered["latest_versions"]:
            st.markdown(f"- **Versión {v['version']}** — stage: `{v['current_stage']}`")
    else:
        st.info("Aún no hay versiones registradas de 'churn_model'.")

    st.markdown(f"🔗 Ver todo el detalle en [MLflow UI](http://localhost:5000)")

# ============================================================
# PESTAÑA 3: DRIFT & REENTRENAMIENTO (dispara el DAG de Airflow)
# ============================================================
with tab_drift:
    st.header("🔍 Monitoreo de Drift y Reentrenamiento")
    st.caption("Dispara el DAG de Airflow 'churn_retraining_pipeline' (también corre solo, cada semana).")

    if airflow_utils.check_airflow_connection():
        st.success("✅ Airflow está disponible")
    else:
        st.error("❌ Airflow no está disponible. Verifica que `docker-compose-airflow.yml` esté levantado.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Ejecutar análisis de drift ahora", use_container_width=True, type="primary"):
            with st.spinner("Disparando el DAG en Airflow..."):
                ok, run_id, msg = airflow_utils.trigger_dag({"triggered_by": "streamlit_ui"})
            if ok:
                st.success(msg)
                st.session_state["last_dag_run_id"] = run_id
                st.info(f"DAG Run ID: `{run_id}` — revisa el progreso abajo o en "
                        f"[Airflow UI](http://localhost:8080) (usuario/clave: airflow/airflow)")
            else:
                st.error(msg)

    with col2:
        if st.button("🧠 Forzar reentrenamiento", use_container_width=True):
            with st.spinner("Disparando reentrenamiento forzado..."):
                ok, run_id, msg = airflow_utils.trigger_dag({"triggered_by": "streamlit_ui", "force_retrain": True})
            if ok:
                st.success(msg)
                st.session_state["last_dag_run_id"] = run_id
            else:
                st.error(msg)

    if "last_dag_run_id" in st.session_state:
        st.subheader("📊 Última ejecución disparada desde aquí")
        run_id = st.session_state["last_dag_run_id"]
        if st.button("🔄 Actualizar estado"):
            status = airflow_utils.get_dag_run(run_id)
            if status:
                state = status.get("state")
                icon = {"success": "✅", "running": "⏳", "failed": "❌"}.get(state, "⚠️")
                st.markdown(f"{icon} **Estado:** `{state}`")
                tasks = airflow_utils.get_task_instances(run_id)
                for t in tasks:
                    t_icon = {"success": "✅", "running": "⏳", "failed": "❌", "skipped": "⏭️", "queued": "⏸️"}.get(t.get("state"), "❓")
                    st.markdown(f"{t_icon} `{t.get('task_id')}`: {t.get('state')}")

    st.divider()
    st.subheader("📜 Historial de ejecuciones")
    recent = airflow_utils.get_recent_dag_runs(limit=10)
    if recent:
        hist = pd.DataFrame([{
            "DAG Run": r.get("dag_run_id", "")[:35],
            "Estado": r.get("state"),
            "Inicio": (r.get("start_date") or "")[:19],
        } for r in recent])
        st.dataframe(hist, use_container_width=True)
    else:
        st.info("No hay ejecuciones registradas todavía (o Airflow no está disponible).")

    st.markdown("""
    ---
    **¿Cómo funciona el reentrenamiento automático?**
    Cada semana (`schedule_interval='@weekly'`), Airflow corre este mismo DAG solo:
    revisa que haya datos, mide el drift comparando el dataset de entrenamiento
    contra las predicciones reales que la API fue logueando, y si el drift supera
    el umbral (30%), reentrena los 3 modelos y le pide a la API que recargue el
    ganador — todo sin intervención manual.
    """)

# ============================================================
# PESTAÑA 4: ESTADO DEL SISTEMA
# ============================================================
with tab_status:
    st.header("⚙️ Estado del Sistema")

    services = [
        {"name": "API (FastAPI)", "url": f"{API_URL}/health", "port": 8000},
        {"name": "MLflow UI", "url": f"{mlflow_utils.MLFLOW_URL}/", "port": 5000},
        {"name": "Airflow", "url": f"{airflow_utils.AIRFLOW_URL}/health", "port": 8080},
    ]

    for svc in services:
        c1, c2, c3 = st.columns([3, 1, 2])
        c1.markdown(f"**{svc['name']}**")
        try:
            r = requests.get(svc["url"], timeout=5)
            c2.success("✅ Activo") if r.status_code == 200 else c2.warning(f"⚠️ {r.status_code}")
        except Exception:
            c2.error("❌ Inactivo")
        c3.markdown(f"[Abrir](http://localhost:{svc['port']})")

    st.divider()
    st.markdown("""
    | Servicio | Puerto | URL |
    |---|---|---|
    | API / Swagger | 8000 | http://localhost:8000/docs |
    | MLflow UI | 5000 | http://localhost:5000 |
    | Streamlit (esta app) | 8501 | http://localhost:8501 |
    | Airflow UI | 8080 | http://localhost:8080 (airflow/airflow) |
    """)
