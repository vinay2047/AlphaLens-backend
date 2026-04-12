"""
main.py – Unified FastAPI backend for AlphaLens.

Combines all three micro-services into a single deployable app:
    /api/sentiment/{ticker}     – News sentiment analysis (DistilRoBERTa)
    /api/predict/{symbol}       – Price prediction (BiLSTM + TCN-GRU)
    /api/inference              – Shadow Portfolio RL agent (PPO)

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 8000

Swagger UI:  http://localhost:8000/docs

Import strategy
---------------
Each service has its own `app/` sub-package with internal `from app.X` imports.
To avoid namespace collisions, we add each service's root to sys.path just
before importing that service's modules, then remove it after. The sentinel
`app` entry in sys.modules is swapped per service so that internal imports
within each service resolve to the correct `app/` package.
"""

from __future__ import annotations

import os
import sys
import importlib
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Path, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("alphalens")

# ---------------------------------------------------------------------------
# Project root path
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Service module loader
# ---------------------------------------------------------------------------
# Because each service uses bare `from app.X import Y` internally,
# we need to temporarily add each service directory to sys.path and
# rebind `app` in sys.modules, then stash the loaded modules under
# unique prefixed names.

_loaded = {}  # stash of loaded modules: "sentiment.headlines" → module


def _load_service_modules(service_dir: str, prefix: str, module_names: list[str]):
    """
    Load a service's app sub-modules, handling `from app.X` collisions.

    1. Add service_dir to sys.path[0]
    2. Purge any previous `app` / `app.*` from sys.modules
    3. Import each module — internal `from app.X` resolves to this service
    4. Stash under `_loaded[prefix.name]`
    5. Clean up sys.path
    """
    # Save and clear any existing 'app' module entries
    saved_app_modules = {}
    for key in list(sys.modules.keys()):
        if key == "app" or key.startswith("app."):
            saved_app_modules[key] = sys.modules.pop(key)

    # Temporarily put this service's directory first on sys.path
    sys.path.insert(0, service_dir)

    try:
        for name in module_names:
            mod = importlib.import_module(f"app.{name}")
            _loaded[f"{prefix}.{name}"] = mod
    finally:
        # Remove this service's directory from sys.path
        sys.path.remove(service_dir)

        # Stash these app modules under prefixed names and remove bare `app`
        for key in list(sys.modules.keys()):
            if key == "app" or key.startswith("app."):
                _loaded[f"{prefix}.{key}"] = sys.modules.pop(key)

        # Restore any previously-saved modules
        sys.modules.update(saved_app_modules)


# -- Shadow portfolio --
try:
    _shadow_portfolio_dir = os.path.join(_ROOT, "services", "shadow_portfolio")
    if _shadow_portfolio_dir not in sys.path:
        sys.path.insert(0, _shadow_portfolio_dir)
    from app.inference import load_model as shadow_load_model
    from app.inference import run_inference
    from app.schemas import InferenceRequest, InferenceResponse
    _shadow_ok = True
except Exception as e:
    logger.error("Shadow portfolio import failed: %s", e, exc_info=True)
    _shadow_ok = False

# -- Sentiment --
try:
    _load_service_modules(
        os.path.join(_ROOT, "services", "sentiment"),
        "sentiment",
        ["pipeline_loader", "analyzer", "headlines", "schemas"],
    )
    _sentiment_ok = True
except Exception as e:
    logger.error("Sentiment import failed: %s", e, exc_info=True)
    _sentiment_ok = False

# -- Price prediction --
try:
    _load_service_modules(
        os.path.join(_ROOT, "services", "price-prediction"),
        "prediction",
        ["config", "scaler_utils", "data_fetcher", "predictor"],
    )
    _prediction_ok = True
except Exception as e:
    logger.error("Price prediction import failed: %s", e, exc_info=True)
    _prediction_ok = False

# Pull out the modules we need
if _sentiment_ok:
    _get_headlines = _loaded["sentiment.headlines"].get_headlines
    _analyse_headlines = _loaded["sentiment.analyzer"].analyse_headlines
    _SentimentResponse = _loaded["sentiment.schemas"].SentimentResponse
    _get_sentiment_pipeline = _loaded["sentiment.pipeline_loader"].get_sentiment_pipeline

if _prediction_ok:
    _predict_symbol = _loaded["prediction.predictor"].predict_symbol
    _get_available_symbols = _loaded["prediction.predictor"].get_available_symbols
    _load_all_models = _loaded["prediction.predictor"].load_all_models
    _MAJOR_COMPANIES = _loaded["prediction.config"].MAJOR_COMPANIES


# ---------------------------------------------------------------------------
# Redis (optional — graceful fallback if unavailable)
# ---------------------------------------------------------------------------
import json
from redis import Redis
from redis.exceptions import RedisError

REDIS_HOST = os.environ.get("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB = int(os.environ.get("REDIS_DB", "0"))
REDIS_TTL = int(os.environ.get("REDIS_TTL_SECONDS", "300"))

try:
    redis_client = Redis(
        host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
        decode_responses=True, socket_connect_timeout=2,
    )
    redis_client.ping()
    _redis_ok = True
    logger.info("Redis connected at %s:%s", REDIS_HOST, REDIS_PORT)
except Exception:
    redis_client = None
    _redis_ok = False
    logger.warning("Redis unavailable — prediction caching disabled.")


def _cache_set(symbol: str, payload: dict) -> None:
    if not _redis_ok:
        return
    try:
        redis_client.set(f"prediction:{symbol}", json.dumps(payload), ex=REDIS_TTL)
    except RedisError as exc:
        logger.warning("Redis write failed: %s", exc)


def _cache_get(symbol: str) -> dict | None:
    if not _redis_ok:
        return None
    try:
        raw = redis_client.get(f"prediction:{symbol}")
        if not raw:
            return None
        payload = json.loads(raw)
        payload["cached"] = True
        return payload
    except RedisError:
        return None


# ---------------------------------------------------------------------------
# Lifespan — warm up all models at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Sentiment pipeline
    if _sentiment_ok:
        try:
            logger.info("Loading sentiment pipeline ...")
            _get_sentiment_pipeline()
            logger.info("Sentiment pipeline ready.")
        except Exception as e:
            logger.warning("Sentiment model failed to load: %s", e)

    # 2. Price prediction models
    if _prediction_ok:
        try:
            logger.info("Loading price prediction models ...")
            loaded = _load_all_models()
            logger.info("Price prediction models ready: %s", loaded)
        except Exception as e:
            logger.warning("Price prediction models failed to load: %s", e)

    # 3. Shadow portfolio PPO
    if _shadow_ok:
        try:
            logger.info("Loading shadow portfolio PPO model ...")
            shadow_load_model()
            logger.info("Shadow portfolio model ready.")
        except Exception as e:
            logger.warning("Shadow portfolio model failed to load: %s", e)

    yield
    logger.info("Shutting down AlphaLens backend.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AlphaLens API",
    description=(
        "Unified backend for AlphaLens — combines sentiment analysis, "
        "price prediction, and shadow portfolio RL inference into a "
        "single deployable service."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://localhost:3000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== HEALTH ==============================================================

@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "ok",
        "service": "alphalens-backend",
        "endpoints": {
            "sentiment": "/api/sentiment/{ticker}",
            "predict": "/api/predict/{symbol}",
            "predict_batch": "/api/predict/batch?symbols=...",
            "models_status": "/api/predict/models/status",
            "inference": "/api/inference",
        },
    }


# ===== SENTIMENT ===========================================================

@app.get(
    "/api/sentiment/{ticker}",
    response_model=_SentimentResponse,
    tags=["Sentiment"],
    summary="Analyse sentiment for a stock ticker",
)
async def get_sentiment(
    ticker: str = Path(
        ..., min_length=1, max_length=10,
        description="Stock ticker symbol, e.g. AAPL, TSLA, MSFT.",
        examples=["AAPL"],
    ),
):
    ticker = ticker.upper()
    logger.info("Sentiment request: %s", ticker)

    headlines = _get_headlines(ticker, count=5)
    if not headlines:
        return _SentimentResponse(
            ticker=ticker, consensus="Neutral",
            score_summary={"positive": 0, "negative": 0, "neutral": 0},
            analysed_headlines=[],
        )

    analysis = _analyse_headlines(headlines)
    return _SentimentResponse(
        ticker=ticker,
        consensus=analysis["consensus"],
        score_summary=analysis["score_summary"],
        analysed_headlines=analysis["analysed_headlines"],
    )


# ===== PRICE PREDICTION ====================================================

@app.get("/api/predict/models/status", tags=["Price Prediction"])
async def models_status():
    return {
        "available_symbols": _get_available_symbols(),
        "total": len(_get_available_symbols()),
    }


@app.get("/api/predict/batch", tags=["Price Prediction"])
async def predict_batch(symbols: str, days: int = 7) -> dict:
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    results = {}
    for sym in symbol_list:
        try:
            cached = _cache_get(sym)
            if cached:
                results[sym] = cached
                continue
            data = await asyncio.to_thread(_predict_symbol, sym, days)
            _cache_set(sym, data)
            results[sym] = data
        except Exception as exc:
            results[sym] = {"error": str(exc)}
    return results


@app.post("/api/predict/{symbol}", tags=["Price Prediction"])
async def predict(symbol: str, days: int = 7):
    symbol = symbol.upper()
    cached = _cache_get(symbol)
    if cached:
        return cached
    try:
        result = await asyncio.to_thread(_predict_symbol, symbol, days)
        _cache_set(symbol, result)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Prediction error for %s: %s", symbol, exc)
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/api/predict/refresh/major", tags=["Price Prediction"])
async def refresh_major_companies(days: int = 7) -> dict:
    results = {}
    for sym in _MAJOR_COMPANIES:
        try:
            data = await asyncio.to_thread(_predict_symbol, sym, days)
            _cache_set(sym, data)
            results[sym] = "refreshed"
        except Exception as exc:
            results[sym] = {"error": str(exc)}
    return results


# WebSocket for live predictions
class _ConnectionManager:
    def __init__(self):
        self.active: dict[str, list[WebSocket]] = {}

    async def connect(self, ws: WebSocket, symbol: str):
        await ws.accept()
        self.active.setdefault(symbol, []).append(ws)

    def disconnect(self, ws: WebSocket, symbol: str):
        if symbol in self.active:
            self.active[symbol].remove(ws)


_ws_manager = _ConnectionManager()


@app.websocket("/ws/live/{symbol}")
async def websocket_live(websocket: WebSocket, symbol: str):
    await _ws_manager.connect(websocket, symbol.upper())
    try:
        while True:
            result = await asyncio.to_thread(_predict_symbol, symbol.upper())
            await websocket.send_json(result)
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        _ws_manager.disconnect(websocket, symbol.upper())


# ===== SHADOW PORTFOLIO ====================================================

@app.post(
    "/api/inference",
    response_model=InferenceResponse,
    tags=["Shadow Portfolio"],
    summary="Run RL agent inference",
)
async def post_inference(request: InferenceRequest):
    """Run the Shadow Portfolio RL agent on the given ticker and date range."""
    try:
        result = run_inference(
            ticker=request.ticker,
            start_date=request.start_date,
            end_date=request.end_date,
        )
        return InferenceResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model not available: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
