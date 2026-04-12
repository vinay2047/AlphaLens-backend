"""
headlines.py – Headline provider for financial news using Finnhub.
"""

from __future__ import annotations

import os
import re
import logging
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import finnhub

logger = logging.getLogger(__name__)

# Load environment variables (e.g. from .env file)
load_dotenv()

_FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
if not _FINNHUB_API_KEY:
    logger.warning("FINNHUB_API_KEY is not set in the environment.")

# Initialize Finnhub client
finnhub_client = finnhub.Client(api_key=_FINNHUB_API_KEY) if _FINNHUB_API_KEY else None

def _clean_company_name(profile_name: str) -> str:
    """Removes common corporate suffixes to get the core brand name."""
    if not profile_name:
        return ""
    # Strip common suffixes that mess up matching
    suffixes = [r'\binc\.?', r'\bcorp\.?', r'\bltd\.?', r'\bllc\b', r'\bcompany\b', r'\bholdings\b']
    clean_name = profile_name.lower()
    for suffix in suffixes:
        clean_name = re.sub(suffix, '', clean_name)
    
    # Remove special characters and extra whitespace
    clean_name = re.sub(r'[^a-z0-9\s]', '', clean_name)
    return clean_name.strip()

def get_headlines(ticker: str, count: int = 5, days_back: int = 7) -> list[dict]:
    """Return the most *relevant* financial headlines for *ticker* using scoring."""
    if not finnhub_client:
        logger.error("Finnhub client is not initialized.")
        return []

    ticker = ticker.upper()
    try:
        # 1. Smarter Company Name Extraction
        try:
            profile = finnhub_client.company_profile2(symbol=ticker)
            raw_name = profile.get("name", "") if profile else ""
            company_kw = _clean_company_name(raw_name)
        except Exception:
            company_kw = ""

        # 2. Shorter, tighter window
        end_date = datetime.now(tz=timezone.utc)
        start_date = end_date - timedelta(days=days_back)
        
        _to = end_date.strftime("%Y-%m-%d")
        _from = start_date.strftime("%Y-%m-%d")
        
        news_items = finnhub_client.company_news(ticker, _from=_from, to=_to)
        
        if not news_items:
            return []

        # Regex patterns for exact word boundary matching
        ticker_pattern = re.compile(rf'\b{re.escape(ticker)}\b', re.IGNORECASE)
        company_pattern = re.compile(rf'\b{re.escape(company_kw)}\b', re.IGNORECASE) if company_kw else None

        scored_articles = []
        seen_headlines = set()

        # 3. Score and Deduplicate
        for item in news_items:
            headline = item.get("headline", "No Title").strip()
            summary = item.get("summary", "").strip()
            
            # Deduplication check
            if headline.lower() in seen_headlines:
                continue
            seen_headlines.add(headline.lower())

            score = 0
            
            # High Priority: Ticker or Name in the actual Headline
            if ticker_pattern.search(headline):
                score += 5
            if company_pattern and company_pattern.search(headline):
                score += 4
                
            # Medium Priority: Ticker or Name in Summary
            if ticker_pattern.search(summary):
                score += 2
            if company_pattern and company_pattern.search(summary):
                score += 1

            # Only keep articles that score above a zero threshold
            if score > 0:
                pub_time = item.get("datetime")
                dt = datetime.fromtimestamp(pub_time, tz=timezone.utc) if pub_time else datetime.now(tz=timezone.utc)
                
                scored_articles.append({
                    "headline": headline,
                    "source": item.get("source", "Unknown"),
                    "timestamp": dt.isoformat(),
                    "score": score,  # Storing score for sorting
                    "datetime_obj": dt
                })

        # 4. Sort by Relevance (Score) first, then Recency (Timestamp)
        scored_articles.sort(key=lambda x: (x["score"], x["datetime_obj"]), reverse=True)

        # 5. Clean up the output to match original expected schema
        results = []
        for article in scored_articles[:count]:
            results.append({
                "headline": article["headline"],
                "source": article["source"],
                "timestamp": article["timestamp"]
            })
            
        return results

    except finnhub.FinnhubAPIException as e:
        logger.error(f"Finnhub API Error for {ticker}: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching headlines for {ticker}: {e}")
        return []
