"""
main.py – FastAPI entry-point for the News Sentiment Analysis micro-service.

Run locally:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Swagger UI :  http://localhost:8000/docs
ReDoc       :  http://localhost:8000/redoc
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Path
from fastapi.middleware.cors import CORSMiddleware

from services.sentiment.app.headlines import get_headlines
from services.sentiment.app.analyzer import analyse_headlines
from services.sentiment.app.pipeline_loader import get_sentiment_pipeline
from services.sentiment.app.schemas import SentimentResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan – warm-up the model at startup so the first request is fast
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the sentiment model into memory before accepting traffic."""
    logger.info("🚀 Warming up the sentiment pipeline …")
    get_sentiment_pipeline()          # downloads + caches on first call
    logger.info("✅ Sentiment pipeline warmed up and ready.")
    yield                             # application runs here
    logger.info("🛑 Shutting down sentiment service.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AlphaLens – News Sentiment API",
    description=(
        "Analyses recent financial headlines for a given stock ticker using "
        "the DistilRoBERTa model fine-tuned on financial news sentiment "
        "(*mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis*). "
        "Returns per-headline sentiment labels with confidence scores and an "
        "overall Bullish / Bearish / Neutral consensus."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the Next.js frontend (localhost:3000) to call this service.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
async def health_check():
    """Simple health / readiness probe."""
    return {"status": "ok", "service": "sentiment-analysis"}


@app.get(
    "/api/sentiment/{ticker}",
    response_model=SentimentResponse,
    tags=["Sentiment"],
    summary="Analyse sentiment for a stock ticker",
    description=(
        "Fetches 5 recent financial headlines for the given ticker, runs them "
        "through a DistilRoBERTa sentiment model, and returns individual "
        "labels with an aggregated Bullish / Bearish / Neutral consensus."
    ),
)
async def get_sentiment(
    ticker: str = Path(
        ...,
        min_length=1,
        max_length=10,
        description="Stock ticker symbol, e.g. AAPL, TSLA, MSFT.",
        examples=["AAPL"],
    ),
):
    """**GET /api/sentiment/{ticker}**

    1. Fetches up to 5 recent relevant financial headlines for *ticker*.
    2. Runs them through the cached sentiment pipeline (CPU).
    3. Aggregates scores → consensus.
    4. Returns a clean JSON response.
    """
    ticker = ticker.upper()
    logger.info("Sentiment request for ticker: %s", ticker)

    # Step 1 – Fetch headlines
    headlines = get_headlines(ticker, count=5)
    
    if not headlines:
        # Avoid crashing analyzer if no headlines
        return SentimentResponse(
            ticker=ticker,
            consensus="Neutral",
            score_summary={"positive": 0, "negative": 0, "neutral": 0},
            analysed_headlines=[],
        )

    # Step 2 + 3 – Analyse & aggregate
    analysis = analyse_headlines(headlines)

    # Step 4 – Build response
    return SentimentResponse(
        ticker=ticker,
        consensus=analysis["consensus"],
        score_summary=analysis["score_summary"],
        analysed_headlines=analysis["analysed_headlines"],
    )
