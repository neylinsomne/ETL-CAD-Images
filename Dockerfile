FROM python:3.12-slim

# opencv-python-headless necesita libglib en la imagen slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[service,infra]"
# Parámetros del filtro ajustados con scripts/tune.py (partición 30/70 balanceada)
COPY config ./config
COPY sql ./sql

ENV ENVELOPE_PARAMS=/app/config/envelope_params.json \
    DETECTOR=none

EXPOSE 8000
# La misma imagen corre la API (por defecto) o el worker (ver docker-compose.yml)
CMD ["uvicorn", "filtro_planos.service.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
