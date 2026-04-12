FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY services/price-prediction/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY services/price-prediction/app ./app
COPY ml_models /app/ml_models

EXPOSE 8002

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
