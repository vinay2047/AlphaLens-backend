"""
pipeline_loader.py – Singleton loader for the HuggingFace sentiment pipeline.
Loads **mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis**
lazily on first inference call (not at import/startup), so Render can bind
the port before the model download begins.

CPU-only optimisations applied:
    • ``device=-1``          → forces CPU execution
    • ``torch_dtype=float32``→ full-precision (no GPU half-precision)
    • ``truncation=True``    → guards against inputs > 512 tokens
    • ``batch_size=8``       → process headlines in one forward pass
"""
from __future__ import annotations

import logging
import threading
from transformers import pipeline as hf_pipeline

logger = logging.getLogger(__name__)

_MODEL_ID = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"

_pipeline = None
_lock = threading.Lock()          # prevents double-loading under concurrent requests


def get_sentiment_pipeline():
    """Return a cached HuggingFace ``sentiment-analysis`` pipeline.

    The model is downloaded and initialised **on the first call only**.
    Subsequent calls return the already-loaded pipeline immediately.
    A threading lock ensures only one thread loads the model even when
    multiple requests arrive simultaneously before loading completes.
    """
    global _pipeline

    if _pipeline is not None:       # fast path — no lock needed
        return _pipeline

    with _lock:                     # slow path — only one thread loads
        if _pipeline is None:       # double-checked locking
            logger.info("Loading sentiment model: %s …", _MODEL_ID)
            _pipeline = hf_pipeline(
                task="sentiment-analysis",
                model=_MODEL_ID,
                device=-1,          # CPU only
                truncation=True,
                batch_size=8,
            )
            logger.info("Sentiment pipeline ready (CPU).")

    return _pipeline