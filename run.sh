#!/bin/bash
# Script para manejar el flujo completo

echo "🚀 Iniciando proyecto MLOps con Docker"

case "$1" in
  train)
    echo "🏋️ Entrenando y comparando modelos (RandomForest, LogisticRegression, GradientBoosting)..."
    docker-compose up --build train
    ;;
  start)
    echo "▶️ Iniciando todos los servicios..."
    docker-compose up -d --build
    echo "✅ Servicios iniciados:"
    echo "  - API: http://localhost:8000"
    echo "  - Swagger UI: http://localhost:8000/docs"
    echo "  - MLflow UI: http://localhost:5000"
    echo "  - Interfaz Streamlit: http://localhost:8501"
    ;;
  stop)
    echo "⏹️ Deteniendo servicios..."
    docker-compose down
    ;;
  restart)
    echo "🔄 Reiniciando servicios..."
    docker-compose down
    docker-compose up -d --build
    ;;
  logs)
    docker-compose logs -f
    ;;
  test)
    echo "🧪 Probando la API..."
    curl -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"tenure": 12, "MonthlyCharges": 70.5, "TotalCharges": 846.0, "Contract": "Month-to-month", "InternetService": "Fiber optic", "OnlineSecurity": "No"}'
    ;;
  stats)
    echo "📊 Estadísticas de monitoreo (predicciones registradas)..."
    curl -s http://localhost:8000/stats | python3 -m json.tool 2>/dev/null || curl http://localhost:8000/stats
    ;;
  drift)
    echo "🔍 Corriendo detección de drift manualmente (sin Airflow)..."
    docker-compose run --rm train python src/drift_detection.py
    ;;
  airflow-start)
    echo "🌬️ Iniciando Airflow (esto puede tardar 1-2 minutos la primera vez)..."
    docker-compose -f docker-compose.yml -f docker-compose-airflow.yml up -d --build airflow-init
    docker-compose -f docker-compose.yml -f docker-compose-airflow.yml up -d --build
    echo "✅ Airflow UI: http://localhost:8080 (usuario: airflow / clave: airflow)"
    ;;
  airflow-stop)
    echo "⏹️ Deteniendo Airflow y el resto de servicios..."
    docker-compose -f docker-compose.yml -f docker-compose-airflow.yml down
    ;;
  *)
    echo "Uso: $0 {train|start|stop|restart|logs|test|stats|drift|airflow-start|airflow-stop}"
    echo "  train         - Entrena y compara modelos, registra el mejor en MLflow"
    echo "  start         - Inicia api, mlflow-ui y streamlit (sin Airflow)"
    echo "  stop          - Detiene esos servicios"
    echo "  restart       - Reinicia los servicios"
    echo "  logs          - Muestra los logs"
    echo "  test          - Prueba /predict con un ejemplo"
    echo "  stats         - Muestra /stats (monitoreo de predicciones)"
    echo "  drift         - Corre la detección de drift una sola vez, sin Airflow"
    echo "  airflow-start - Levanta TODO (incluyendo Airflow) para el reentrenamiento programado"
    echo "  airflow-stop  - Detiene todo, incluyendo Airflow"
    ;;
esac
