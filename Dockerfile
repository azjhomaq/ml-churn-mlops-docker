# Dockerfile para la API FastAPI
FROM python:3.10-slim

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models/model.joblib \
    SCALER_PATH=/app/models/scaler.joblib \
    ENCODER_PATH=/app/models/label_encoders.joblib

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY src/ ./src/

# Exponer puerto
EXPOSE 8000

# Comando de inicio
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
