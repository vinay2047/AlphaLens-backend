"""
headlines.py – Headline provider for financial news using Finnhub.
"""

from __future__ import annotations

import os
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

def get_headlines(ticker: str, count: int = 5) -> list[dict]:
    """Return up to *count* financial headlines for *ticker* using Finnhub.

    Each headline dict contains:
        - ``headline``  : the generated text
        - ``source``    : publisher
        - ``timestamp`` : ISO-8601 UTC string
    """
    if not finnhub_client:
        logger.error("Cannot fetch headlines. Finnhub client is not initialized (missing API key).")
        return []

    ticker = ticker.upper()
    try:
        # Extract a keyword from the company name to filter out macro noise
        try:
            profile = finnhub_client.company_profile2(symbol=ticker)
            short_name = profile.get("name", "") if profile else ""
            # e.g. "Apple Inc." -> "apple"
            company_kw = short_name.split()[0].lower() if short_name else ticker.lower()
        except Exception:
            company_kw = ticker.lower()

        # Finnhub company_news requires YYYY-MM-DD
        end_date = datetime.now(tz=timezone.utc)
        start_date = end_date - timedelta(days=7)
        
        _to = end_date.strftime("%Y-%m-%d")
        _from = start_date.strftime("%Y-%m-%d")
        
        news_items = finnhub_client.company_news(ticker, _from=_from, to=_to)
        
        if not news_items:
            return []
            
        results = []
        for item in news_items:
            headline = item.get("headline", "No Title")
            summary = item.get("summary", "")
            
            # RELEVANCE FILTER:
            # Skip article if neither the ticker nor the company name appears in it
            search_text = (headline + " " + summary).lower()
            if ticker.lower() not in search_text and company_kw not in search_text:
                continue

            pub_time = item.get("datetime")
            if pub_time:
                dt = datetime.fromtimestamp(pub_time, tz=timezone.utc)
                timestamp = dt.isoformat()
            else:
                timestamp = datetime.now(tz=timezone.utc).isoformat()
                
            results.append({
                "headline": headline,
                "source": item.get("source", "Unknown Publisher"),
                "timestamp": timestamp,
            })
            
            if len(results) >= count:
                break
                
        return results
        
    except finnhub.FinnhubAPIException as e:
        # This handles rate limits (HTTP 429) out of the box nicely
        logger.error(f"Finnhub API Error for {ticker}: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching headlines for {ticker}: {e}")
        return []
