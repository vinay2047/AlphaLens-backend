"""
pipeline_loader.py – Sentiment analysis via the HuggingFace Inference API.

Calls the serverless HF API instead of loading torch + transformers locally.
This eliminates ~500MB of RAM usage, making it viable on Render's free tier.

The returned object is a callable with the same interface as a local
transformers pipeline: pipe(texts) → [{"label": ..., "score": ...}, ...]
"""
from __future__ import annotations

import os
import logging
import time

import httpx

logger = logging.getLogger(__name__)

_MODEL_ID = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
_API_URL = f"https://api-inference.huggingface.co/models/{_MODEL_ID}"
_HF_TOKEN = os.environ.get("HF_TOKEN", "")

_HEADERS = {"Content-Type": "application/json"}
if _HF_TOKEN:
    _HEADERS["Authorization"] = f"Bearer {_HF_TOKEN}"


class _HFInferencePipeline:
    """Callable wrapper that mimics the transformers pipeline interface.

    Usage matches the local pipeline exactly::

        pipe = get_sentiment_pipeline()
        results = pipe(["headline 1", "headline 2"])
        # → [{"label": "positive", "score": 0.92}, {"label": "negative", "score": 0.85}]
    """

    def __call__(self, texts: list[str]) -> list[dict]:
        """Send texts to HF Inference API, return top label per text."""
        payload = {"inputs": texts}

        for attempt in range(3):
            try:
                response = httpx.post(
                    _API_URL,
                    headers=_HEADERS,
                    json=payload,
                    timeout=60.0,
                )

                # Model cold-start: HF returns 503 while loading
                if response.status_code == 503:
                    body = response.json()
                    wait = min(body.get("estimated_time", 20), 30)
                    logger.info(
                        "HF model loading (attempt %d/3), waiting %.0fs...",
                        attempt + 1, wait,
                    )
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                results = response.json()

                # API returns [[{label, score}, ...], ...] per text (all labels).
                # Local pipeline returns [{label, score}] per text (top label only).
                # → take the first (highest-scoring) entry from each inner list.
                return [labels[0] for labels in results]

            except httpx.HTTPStatusError as e:
                logger.error("HF API error (attempt %d/3): %s", attempt + 1, e)
                if attempt < 2:
                    time.sleep(2)
                else:
                    raise
            except httpx.TimeoutException:
                logger.warning("HF API timeout (attempt %d/3)", attempt + 1)
                if attempt < 2:
                    time.sleep(2)
                else:
                    raise

        raise RuntimeError("HF Inference API failed after 3 attempts")


_pipeline = _HFInferencePipeline()


def get_sentiment_pipeline():
    """Return the HF Inference API wrapper (drop-in for transformers pipeline)."""
    return _pipeline