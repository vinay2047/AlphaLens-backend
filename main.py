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
Service modules are loaded LAZILY — NOT at module import time. This ensures
uvicorn can bind the port within Render's timeout window. After binding, a
background warmup task loads all services sequentially (one at a time to
avoid memory spikes).
"""

from __future__ import annotations

import os
import sys
import importlib
import asyncio
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Path, HTTPException, WebSocket, WebSocketDisconnect, Query, Body
from pydantic import BaseModel
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
# Lazy service loader
# ---------------------------------------------------------------------------
# All heavy ML imports are deferred to first use. Each service has a lock to
# prevent double-loading under concurrent requests, and a status flag to
# avoid retrying after a permanent failure.

_loaded = {}          # stash of loaded modules: "sentiment.headlines" → module
_service_status = {}  # "sentiment" → True/False/None (None = not yet tried)
_service_locks = {
    "shadow": threading.Lock(),
    "sentiment": threading.Lock(),
    "prediction": threading.Lock(),
}


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


# ---------------------------------------------------------------------------
# Lazy loaders — called on first request, NOT at import time
# ---------------------------------------------------------------------------

def _ensure_shadow():
    """Lazily load the shadow portfolio service."""
    status = _service_status.get("shadow")
    if status is not None:
        return status  # already loaded (True) or failed (False)

    with _service_locks["shadow"]:
        # Double-check after acquiring lock
        if _service_status.get("shadow") is not None:
            return _service_status["shadow"]

        try:
            logger.info("Loading shadow portfolio service...")
            shadow_dir = os.path.join(_ROOT, "services", "shadow_portfolio")
            if shadow_dir not in sys.path:
                sys.path.insert(0, shadow_dir)
            from app.inference import load_model as shadow_load_model
            from app.inference import run_inference
            from app.schemas import InferenceRequest, InferenceResponse

            _loaded["shadow.load_model"] = shadow_load_model
            _loaded["shadow.run_inference"] = run_inference
            _loaded["shadow.InferenceRequest"] = InferenceRequest
            _loaded["shadow.InferenceResponse"] = InferenceResponse

            _service_status["shadow"] = True
            logger.info("Shadow portfolio service loaded.")
            return True
        except Exception as e:
            logger.error("Shadow portfolio import failed: %s", e, exc_info=True)
            _service_status["shadow"] = False
            return False


def _ensure_sentiment():
    """Lazily load the sentiment analysis service."""
    status = _service_status.get("sentiment")
    if status is not None:
        return status

    with _service_locks["sentiment"]:
        if _service_status.get("sentiment") is not None:
            return _service_status["sentiment"]

        try:
            logger.info("Loading sentiment service...")
            _load_service_modules(
                os.path.join(_ROOT, "services", "sentiment"),
                "sentiment",
                ["pipeline_loader", "analyzer", "headlines", "schemas"],
            )
            _service_status["sentiment"] = True
            logger.info("Sentiment service loaded.")
            return True
        except Exception as e:
            logger.error("Sentiment import failed: %s", e, exc_info=True)
            _service_status["sentiment"] = False
            return False


def _ensure_prediction():
    """Lazily load the price prediction service."""
    status = _service_status.get("prediction")
    if status is not None:
        return status

    with _service_locks["prediction"]:
        if _service_status.get("prediction") is not None:
            return _service_status["prediction"]

        try:
            logger.info("Loading price prediction service...")
            _load_service_modules(
                os.path.join(_ROOT, "services", "price-prediction"),
                "prediction",
                ["config", "scaler_utils", "data_fetcher", "predictor"],
            )
            _service_status["prediction"] = True
            logger.info("Price prediction service loaded.")
            return True
        except Exception as e:
            logger.error("Price prediction import failed: %s", e, exc_info=True)
            _service_status["prediction"] = False
            return False


# ---------------------------------------------------------------------------
# Redis (optional — graceful fallback if unavailable)
# ---------------------------------------------------------------------------
import json

REDIS_HOST = os.environ.get("REDIS_HOST", "")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB = int(os.environ.get("REDIS_DB", "0"))
REDIS_TTL = int(os.environ.get("REDIS_TTL_SECONDS", "300"))

_redis_ok = False
redis_client = None

if REDIS_HOST:
    try:
        from redis import Redis
        from redis.exceptions import RedisError
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
else:
    logger.info("REDIS_HOST not set — prediction caching disabled.")


def _cache_set(symbol: str, payload: dict) -> None:
    if not _redis_ok:
        return
    try:
        redis_client.set(f"prediction:{symbol}", json.dumps(payload), ex=REDIS_TTL)
    except Exception as exc:
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
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Background warmup — loads services sequentially AFTER port binds
# ---------------------------------------------------------------------------

async def _background_warmup():
    """
    Load all services in background after uvicorn binds the port.

    Services are loaded ONE AT A TIME (sequentially) to avoid compounding
    memory spikes. gc.collect() runs between loads to reclaim temporary
    allocations before the next heavy import.

    Order: shadow FIRST (imports torch — needs the most RAM while memory
    is cleanest), then sentiment (lightweight API call), then prediction.
    """
    import gc

    await asyncio.sleep(2)  # let uvicorn fully bind and answer health checks
    logger.info("Background warmup: starting sequential service loading...")

    loop = asyncio.get_event_loop()

    # Shadow first — torch import needs ~200MB, do it while RAM is cleanest
    for name, loader in [
        ("shadow", _ensure_shadow),
        ("sentiment", _ensure_sentiment),
        ("prediction", _ensure_prediction),
    ]:
        try:
            ok = await loop.run_in_executor(None, loader)
            logger.info("Background warmup: %s → %s", name, "loaded" if ok else "FAILED")
        except Exception as e:
            logger.error("Background warmup: %s failed: %s", name, e)

        # Free temporary memory before loading the next service
        gc.collect()
        await asyncio.sleep(1)

    logger.info("Background warmup complete.")

async def _keep_alive_task():
    """Ping the server's public URL every 14 minutes to prevent Render from suspending it."""
    import httpx
    url = "https://alpha-lens-3464.onrender.com/"
    while True:
        await asyncio.sleep(14 * 60)  # 14 minutes
        try:
            async with httpx.AsyncClient() as client:
                await client.get(url, timeout=10.0)
            logger.info("Keep-alive ping sent to %s", url)
        except Exception as e:
            logger.warning("Keep-alive ping failed: %s", e)



# ---------------------------------------------------------------------------
# Lifespan — bind port first, then warm up in background
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AlphaLens backend starting — warmup in background.")
    warmup_task = asyncio.create_task(_background_warmup())
    keep_alive = asyncio.create_task(_keep_alive_task())
    yield
    warmup_task.cancel()
    keep_alive.cancel()
    try:
        await warmup_task
    except asyncio.CancelledError:
        pass
    try:
        await keep_alive
    except asyncio.CancelledError:
        pass
    logger.info("Shutting down AlphaLens backend.")


# ---------------------------------------------------------------------------
# App — this must execute FAST so uvicorn can bind the port immediately
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



if os.getenv("ENV") == "production":
    origins = [
        "https://alpha-lens-trading-partner-pv55.vercel.app",
    ]
else:
    origins = [
        "http://localhost:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
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
        "services_loaded": {
            "sentiment": _service_status.get("sentiment", "not_loaded"),
            "prediction": _service_status.get("prediction", "not_loaded"),
            "shadow": _service_status.get("shadow", "not_loaded"),
        },
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
    if not _ensure_sentiment():
        raise HTTPException(status_code=503, detail="Sentiment service unavailable.")

    _get_headlines = _loaded["sentiment.headlines"].get_headlines
    _analyse_headlines = _loaded["sentiment.analyzer"].analyse_headlines
    _SentimentResponse = _loaded["sentiment.schemas"].SentimentResponse

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
    if not _ensure_prediction():
        raise HTTPException(status_code=503, detail="Prediction service unavailable.")
    _get_available_symbols = _loaded["prediction.predictor"].get_available_symbols
    return {
        "available_symbols": _get_available_symbols(),
        "total": len(_get_available_symbols()),
    }


@app.get("/api/predict/batch", tags=["Price Prediction"])
async def predict_batch(symbols: str, days: int = 7) -> dict:
    if not _ensure_prediction():
        raise HTTPException(status_code=503, detail="Prediction service unavailable.")
    _predict_symbol = _loaded["prediction.predictor"].predict_symbol

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


class PredictPriceRequest(BaseModel):
    ticker: str
    open: list[float]
    high: list[float]
    low: list[float]
    close: list[float]
    volume: list[float]
    days: int = 7

@app.post("/api/predict", tags=["Price Prediction"])
async def predict(req: PredictPriceRequest):
    if not _ensure_prediction():
        raise HTTPException(status_code=503, detail="Prediction service unavailable.")
    _predict_symbol = _loaded["prediction.predictor"].predict_symbol

    symbol = req.ticker.upper()
    days = req.days
    cached = _cache_get(symbol)
    if cached:
        return cached
    try:
        features = {
            "open": req.open,
            "high": req.high,
            "low": req.low,
            "close": req.close,
            "volume": req.volume
        }
        result = await asyncio.to_thread(_predict_symbol, symbol, days, features)
        _cache_set(symbol, result)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Prediction error for %s: %s", symbol, exc)
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/api/predict/refresh/major", tags=["Price Prediction"])
async def refresh_major_companies(days: int = 7) -> dict:
    if not _ensure_prediction():
        raise HTTPException(status_code=503, detail="Prediction service unavailable.")
    _predict_symbol = _loaded["prediction.predictor"].predict_symbol
    _MAJOR_COMPANIES = _loaded["prediction.config"].MAJOR_COMPANIES

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
    if not _ensure_prediction():
        await websocket.close(code=1013, reason="Prediction service unavailable")
        return
    _predict_symbol = _loaded["prediction.predictor"].predict_symbol

    await _ws_manager.connect(websocket, symbol.upper())
    try:
        while True:
            result = await asyncio.to_thread(_predict_symbol, symbol.upper())
            await websocket.send_json(result)
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        _ws_manager.disconnect(websocket, symbol.upper())


# ===== SHADOW PORTFOLIO ====================================================

class ShadowInferenceRequest(BaseModel):
    ticker: str = "SPY"
    start_date: str = "2024-01-01"
    end_date: str = "2024-06-30"

@app.post(
    "/api/inference",
    tags=["Shadow Portfolio"],
    summary="Run RL agent inference",
)
async def post_inference(request: ShadowInferenceRequest):
    """Run the Shadow Portfolio RL agent on the given ticker and date range."""
    if not _ensure_shadow():
        raise HTTPException(status_code=503, detail="Shadow portfolio service unavailable.")

    InferenceRequest = _loaded["shadow.InferenceRequest"]
    InferenceResponse = _loaded["shadow.InferenceResponse"]
    run_inference = _loaded["shadow.run_inference"]

    try:
        req = InferenceRequest(**request.model_dump())
        result = run_inference(
            ticker=req.ticker,
            start_date=req.start_date,
            end_date=req.end_date,
        )
        return InferenceResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model not available: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")