#!/bin/bash
# Script para manejar el flujo completo

echo "🚀 Iniciando proyecto MLOps con Docker"

case "$1" in
  train)
    echo "🏋️ Entrenando modelo..."
    docker-compose up train
    ;;
  start)
    echo "▶️ Iniciando todos los servicios..."
    docker-compose up -d
    echo "✅ Servicios iniciados:"
    echo "  - API: http://localhost:8000"
    echo "  - MLflow UI: http://localhost:5000"
    echo "  - Swagger UI: http://localhost:8000/docs"
    ;;
  stop)
    echo "⏹️ Deteniendo servicios..."
    docker-compose down
    ;;
  restart)
    echo "🔄 Reiniciando servicios..."
    docker-compose down
    docker-compose up -d
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
  *)
    echo "Uso: $0 {train|start|stop|restart|logs|test}"
    echo "  train   - Entrena el modelo"
    echo "  start   - Inicia todos los servicios"
    echo "  stop    - Detiene todos los servicios"
    echo "  restart - Reinicia los servicios"
    echo "  logs    - Muestra los logs"
    echo "  test    - Prueba la API con un ejemplo"
    ;;
esac
