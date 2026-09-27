import os
import requests
import streamlit as st
import plotly.graph_objects as go

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Predicción de Churn", page_icon="📉", layout="centered")
st.title("📉 Predicción de Deserción de Clientes (Churn)")
st.caption("Telco Customer Churn · Random Forest + FastAPI + MLflow")

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
        st.metric("Total de predicciones", stats.get("total_predictions", 0))
        if stats.get("churn_rate") is not None:
            st.metric("Tasa de churn observada", f"{stats['churn_rate'] * 100:.1f}%")
        st.json(stats)
    except requests.exceptions.RequestException as e:
        st.error(f"No se pudo obtener /stats: {e}")
