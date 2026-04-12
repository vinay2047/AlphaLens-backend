"""
schemas.py – Pydantic response models for the sentiment API.

Using explicit schemas keeps the OpenAPI docs clean and gives callers
strong contracts to code against.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class HeadlineResult(BaseModel):
    """Single headline with its model-assigned sentiment."""

    headline: str = Field(..., description="The financial news headline text.")
    source: str = Field(..., description="News outlet that published the headline.")
    sentiment_label: str = Field(
        ...,
        description="Normalised sentiment: 'positive', 'negative', or 'neutral'.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence score for the assigned label.",
    )


class ScoreSummary(BaseModel):
    """Counts of each sentiment class across the analysed headlines."""

    positive: int = 0
    negative: int = 0
    neutral: int = 0


class SentimentResponse(BaseModel):
    """Top-level response returned by ``GET /api/sentiment/{ticker}``."""

    ticker: str = Field(..., description="Stock ticker symbol (uppercased).")
    consensus: str = Field(
        ...,
        description="Overall market sentiment: 'Bullish', 'Bearish', or 'Neutral'.",
    )
    score_summary: ScoreSummary
    analysed_headlines: list[HeadlineResult]

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "ticker": "AAPL",
                "consensus": "Bullish",
                "score_summary": {"positive": 3, "negative": 1, "neutral": 1},
                "analysed_headlines": [
                    {
                        "headline": "AAPL beats Q3 earnings estimates, stock surges 8%",
                        "source": "Reuters",
                        "sentiment_label": "positive",
                        "confidence": 0.9732,
                    }
                ],
            }
        ]
    }}
