# Price Prediction Service

This directory contains the FastAPI-based ML prediction service for AlphaLens.

## Overview

The service loads local `.keras` models from `ml_models/`, fetches real-time stock data using `yfinance`, computes technical indicators, and returns ensemble predictions for the next 7 days.

Features:
- Ensemble predictions (BiLSTM + TCN-GRU models)
- 7-day forecast with weekend skipping
- Real-time data fetching
- Redis caching (5-minute TTL)
- WebSocket live updates
- Batch predictions for major companies
- Automatic background refresh every hour

## Install

```bash
cd services/price-prediction
python -m pip install -r requirements.txt
```

## Run locally

```bash
cd services/price-prediction
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
```

## Environment

Copy `.env.example` to `.env` and adjust values as needed.

## Endpoints

- `GET /` — health check
- `GET /models/status` — available symbols
- `POST /predict/{symbol}?days=7` — single-symbol prediction (next 7 days)
- `GET /predict/batch?symbols=AAPL,TSLA` — batch predictions
- `GET /predict/major` — predictions for major companies (20 symbols)
- `POST /refresh/major` — manually refresh cache for major companies
- `GET /ws/live/{symbol}` — live websocket predictions (updates every 60s)

## Models

Your model files are already unpacked in `ml_models/` and scalers in `ml_models/scalers/`.

Available symbols: AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA, ORCL, AMD, AVGO, BA, BAC, CAT, COST, CVX, GE, GS, HD, HON, JPM, KO, LOW, MA, MCD, MMM, MS, NKE, PG, SBUX, TGT, V, WMT, XOM

Example files:
- `ml_models/bilstm_AAPL.keras`
- `ml_models/tcn_gru_AAPL.keras`
- `ml_models/scalers/AAPL_scaler.pkl`

## Docker compose

You can start Redis and the prediction service locally with:

```bash
docker compose up --build
```

The service will be available at `http://localhost:8002`.

If you want the service to connect to a different Redis host, update `services/price-prediction/.env.example` or create a local `.env` file from it.
