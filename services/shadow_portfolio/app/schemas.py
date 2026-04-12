"""
schemas.py – Pydantic models for the Shadow Portfolio inference API.

Defines request/response contracts for the inference endpoints.
Using explicit schemas keeps the OpenAPI docs clean and gives callers
strong contracts to code against.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class InferenceRequest(BaseModel):
    """Body for ``POST /api/inference``."""

    ticker: str = Field(
        default="SPY",
        min_length=1,
        max_length=10,
        description="Stock ticker symbol (e.g. SPY, QQQ, AAPL).",
    )
    start_date: str = Field(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Start date for inference window (YYYY-MM-DD).",
    )
    end_date: str = Field(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="End date for inference window (YYYY-MM-DD).",
    )

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "ticker": "SPY",
                "start_date": "2024-01-01",
                "end_date": "2024-06-30",
            }
        ]
    }}


# ---------------------------------------------------------------------------
# Response building blocks
# ---------------------------------------------------------------------------

class DayResult(BaseModel):
    """Per-day inference output."""

    date: str = Field(..., description="Trading date (YYYY-MM-DD).")
    allocation: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Agent's recommended allocation to the asset (0 = 100%% cash, 1 = 100%% invested).",
    )
    agent_portfolio_value: float = Field(
        ...,
        description="Cumulative portfolio value (agent), starting at 1.0.",
    )
    baseline_portfolio_value: float = Field(
        ...,
        description="Cumulative portfolio value (buy-and-hold baseline), starting at 1.0.",
    )
    daily_return: float = Field(
        ...,
        description="Agent's portfolio log-return for this day.",
    )


class InferenceMetrics(BaseModel):
    """Aggregated performance metrics for a run."""

    total_return: float = Field(
        ..., description="Total return as a decimal (e.g. 0.12 = +12%%)."
    )
    sharpe_ratio: float = Field(
        ..., description="Annualised Sharpe ratio (daily returns × √252)."
    )
    max_drawdown: float = Field(
        ..., description="Maximum peak-to-trough drawdown as a decimal."
    )
    final_portfolio_value: float = Field(
        ..., description="Portfolio value at the end of the window (started at 1.0)."
    )


class InferenceResponse(BaseModel):
    """Top-level response for the inference endpoints."""

    ticker: str
    start_date: str
    end_date: str
    trading_days: int = Field(
        ..., description="Number of trading days in the inference window."
    )

    agent_metrics: InferenceMetrics
    baseline_metrics: InferenceMetrics

    daily_results: list[DayResult] = Field(
        ..., description="Day-by-day allocation and portfolio value trace."
    )

    model_info: dict = Field(
        default_factory=dict,
        description="Metadata about the model used (training window, architecture, etc.).",
    )

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "ticker": "SPY",
                "start_date": "2024-01-01",
                "end_date": "2024-03-31",
                "trading_days": 62,
                "agent_metrics": {
                    "total_return": 0.082,
                    "sharpe_ratio": 1.45,
                    "max_drawdown": 0.043,
                    "final_portfolio_value": 1.082,
                },
                "baseline_metrics": {
                    "total_return": 0.128,
                    "sharpe_ratio": 1.12,
                    "max_drawdown": 0.085,
                    "final_portfolio_value": 1.128,
                },
                "daily_results": [
                    {
                        "date": "2024-01-02",
                        "allocation": 0.72,
                        "agent_portfolio_value": 1.003,
                        "baseline_portfolio_value": 1.004,
                        "daily_return": 0.003,
                    }
                ],
                "model_info": {
                    "algorithm": "PPO",
                    "trained_on": "SPY 2010-2021",
                    "features": 17,
                },
            }
        ]
    }}
