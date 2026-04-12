"""
pipeline_loader.py – Singleton loader for the HuggingFace sentiment pipeline.

Loads **mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis**
once at import / first call and reuses it for every request, avoiding the
latency of re-downloading or re-initialising on every inference.

CPU-only optimisations applied:
    • ``device=-1``          → forces CPU execution
    • ``torch_dtype=float32``→ full-precision (no GPU half-precision)
    • ``truncation=True``    → guards against inputs > 512 tokens
    • ``batch_size=8``       → process headlines in one forward pass
"""

from __future__ import annotations

import logging
from functools import lru_cache
from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

_MODEL_ID = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"


@lru_cache(maxsize=1)
def get_sentiment_pipeline():
    """Return a cached HuggingFace ``sentiment-analysis`` pipeline.

    Passing the model ID directly to the pipeline factory is more robust
    as it handles tokenizer and model loading logic internally.
    """
    logger.info("Loading sentiment model: %s …", _MODEL_ID)

    pipe = hf_pipeline(
        task="sentiment-analysis",
        model=_MODEL_ID,
        device=-1,            # CPU only
        truncation=True,
        batch_size=8,
    )

    logger.info("Sentiment pipeline ready (CPU).")
    return pipe
