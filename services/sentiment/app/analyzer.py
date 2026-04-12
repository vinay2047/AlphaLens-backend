"""
analyzer.py – Core sentiment analysis logic.

Responsibilities:
    1. Run headlines through the HuggingFace pipeline.
    2. Normalise model labels → canonical { positive, negative, neutral }.
    3. Aggregate individual scores into an overall consensus metric.
"""

from __future__ import annotations

from services.sentiment.app.pipeline_loader import get_sentiment_pipeline



_LABEL_MAP: dict[str, str] = {
    "positive": "positive",
    "negative": "negative",
    "neutral":  "neutral",
}


def _normalise_label(raw_label: str) -> str:
    """Map the raw model label to a canonical form."""
    return _LABEL_MAP.get(raw_label.lower(), "neutral")



def analyse_headlines(headlines: list[dict]) -> dict:
    """Run sentiment analysis on a list of headline dicts.

    Parameters
    ----------
    headlines : list[dict]
        Each dict must contain at least a ``"headline"`` key.

    Returns
    -------
    dict
        ``analysed_headlines`` – list of per-headline results
        ``consensus``         – overall Bullish / Bearish / Neutral verdict
        ``score_summary``     – count of each sentiment class
    """
    pipe = get_sentiment_pipeline()

    texts = [h["headline"] for h in headlines]
    results = pipe(texts)  # batch inference in one forward pass

    analysed: list[dict] = []
    counts = {"positive": 0, "negative": 0, "neutral": 0}

    for headline_meta, result in zip(headlines, results):
        label = _normalise_label(result["label"])
        counts[label] += 1

        analysed.append({
            "headline": headline_meta["headline"],
            "source":   headline_meta.get("source", "unknown"),
            "sentiment_label": label,
            "confidence": round(result["score"], 4),
        })

    consensus = _compute_consensus(counts)

    return {
        "consensus": consensus,
        "score_summary": counts,
        "analysed_headlines": analysed,
    }


def _compute_consensus(counts: dict[str, int]) -> str:
    """Derive an overall market-sentiment consensus.

    Rules
    -----
    • If positive > negative  → **Bullish**
    • If negative > positive  → **Bearish**
    • Otherwise               → **Neutral**
    """
    if counts["positive"] > counts["negative"]:
        return "Bullish"
    elif counts["negative"] > counts["positive"]:
        return "Bearish"
    return "Neutral"
