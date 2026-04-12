"""
main.py – FastAPI entry-point for the Shadow Portfolio inference micro-service.

Endpoints:
    GET  /                                          → Health check
    POST /api/inference                             → Inference via JSON body
    GET  /api/inference/{ticker}                    → Inference via path + query params

Run locally:
    cd services/shadow_portfolio
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8001

Swagger UI :  http://localhost:8001/docs
ReDoc       :  http://localhost:8001/redoc
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Path, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.inference import load_model, run_inference
from app.schemas import InferenceRequest, InferenceResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan – pre-load the PPO model at startup so first request is fast
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained PPO model + VecNormalize stats into memory."""
    logger.info("🚀 Loading Shadow Portfolio PPO model …")
    try:
        load_model()
        logger.info("✅ PPO model loaded and ready for inference.")
    except FileNotFoundError as e:
        logger.warning("⚠️  Model not found at startup: %s", e)
        logger.warning("   Inference endpoints will fail until train.py is run.")
    yield
    logger.info("🛑 Shutting down Shadow Portfolio inference service.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AlphaLens – Shadow Portfolio Inference API",
    description=(
        "Runs the pre-trained PPO reinforcement learning agent on any ticker "
        "for a given date range. The agent was trained on SPY (2010–2021) to "
        "learn optimal allocation between the asset and cash, using 16 quant "
        "features and a Differential Sharpe Ratio reward.\n\n"
        "**Inference pipeline:** Fetch OHLCV → compute features → PPO forward "
        "pass per day → collect allocations + portfolio values → compute "
        "Sharpe, max drawdown, total return."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the Next.js frontend (localhost:3000) to call this service.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://localhost:3000",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
async def health_check():
    """Simple health / readiness probe."""
    return {"status": "ok", "service": "shadow-portfolio-inference"}


@app.post(
    "/api/inference",
    response_model=InferenceResponse,
    tags=["Inference"],
    summary="Run RL agent inference (JSON body)",
    description=(
        "Accepts a ticker, start_date, and end_date in the request body. "
        "Downloads market data, computes features, and runs the trained PPO "
        "agent day-by-day to produce allocation recommendations and "
        "performance metrics."
    ),
)
async def post_inference(request: InferenceRequest):
    """**POST /api/inference**

    Run the Shadow Portfolio RL agent on the given ticker and date range.

    The agent steps through each trading day:
    1. Observes 17-dim state (16 quant features + current allocation)
    2. PPO policy forward pass → allocation ∈ [0, 1]
    3. Portfolio return = w·r_asset + (1-w)·r_rf − transaction costs
    4. Repeat until end of date range

    Returns day-by-day allocations, portfolio values, and aggregated metrics
    (Sharpe, max drawdown, total return) for both the agent and a buy-and-hold
    baseline.
    """
    try:
        result = run_inference(
            ticker=request.ticker,
            start_date=request.start_date,
            end_date=request.end_date,
        )
        return InferenceResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model not available: {e}. Run train.py first.",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")


@app.get(
    "/api/inference/{ticker}",
    response_model=InferenceResponse,
    tags=["Inference"],
    summary="Run RL agent inference (path + query params)",
    description=(
        "Convenience GET endpoint. Specify the ticker in the path and dates "
        "as query parameters. Equivalent to the POST endpoint."
    ),
)
async def get_inference(
    ticker: str = Path(
        ...,
        min_length=1,
        max_length=10,
        description="Stock ticker symbol (e.g. SPY, QQQ, AAPL).",
        examples=["SPY"],
    ),
    start_date: str = Query(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Start date for inference (YYYY-MM-DD).",
        examples=["2024-01-01"],
    ),
    end_date: str = Query(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="End date for inference (YYYY-MM-DD).",
        examples=["2024-06-30"],
    ),
):
    """**GET /api/inference/{ticker}?start_date=...&end_date=...**

    Same as the POST endpoint but using path + query parameters.
    """
    try:
        result = run_inference(
            ticker=ticker.upper(),
            start_date=start_date,
            end_date=end_date,
        )
        return InferenceResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model not available: {e}. Run train.py first.",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
