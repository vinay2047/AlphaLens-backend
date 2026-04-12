"""
Data acquisition and walk-forward splitting for the Shadow Portfolio.

Multi-Regime Training & Stress-Test Evaluation
-----------------------------------------------
The data pipeline is configured for a hard date-based split that ensures
the agent trains on multiple market regimes and is tested on a bear market:

    |-------------- Train (2010–2021) --------------|--- Test (2022) ---|
    Bull runs, 2018 chop, COVID crash, recovery      2022 bear market

This is stricter than a percentage-based split because:
  1. The training window includes diverse regimes (bull, bear, crash, recovery)
  2. The test window is a known bear market — the ultimate stress test
  3. No ambiguity about what data the agent has seen vs. not seen
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


def _create_session():
    """
    Create a requests-compatible session that bypasses Yahoo's bot detection.

    Tries curl_cffi first (TLS fingerprint impersonation — most reliable on
    cloud servers), falls back to a standard requests.Session with browser
    headers if curl_cffi is unavailable.
    """
    try:
        from curl_cffi.requests import Session
        session = Session(impersonate="chrome")
        logger.info("Using curl_cffi session (TLS impersonation enabled)")
        return session
    except ImportError:
        import requests
        session = requests.Session()
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        })
        logger.info("curl_cffi not available, using requests session with browser UA")
        return session


def _try_download(ticker: str, start: str, end: str, session) -> pd.DataFrame:
    """Attempt download via yf.download()."""
    df = yf.download(
        ticker, start=start, end=end,
        auto_adjust=True, progress=False,
        session=session,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return df


def _try_ticker_history(ticker: str, start: str, end: str, session) -> pd.DataFrame:
    """Fallback: use yf.Ticker().history() which hits a different Yahoo endpoint."""
    t = yf.Ticker(ticker, session=session)
    df = t.history(start=start, end=end, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
    df = df[cols].dropna()
    return df


def fetch_data(ticker: str = "SPY", start: str = "2010-01-01",
               end: str = "2022-12-31", max_retries: int = 3) -> pd.DataFrame:
    """
    Download OHLCV data via yfinance with retry + fallback logic.

    Strategy (per attempt):
      1. yf.download() with curl_cffi TLS impersonation session
      2. yf.Ticker().history() fallback (different Yahoo API endpoint)

    Args:
        ticker: Yahoo Finance symbol (default: SPY as market proxy).
        start:  Start date string (YYYY-MM-DD).
        end:    End date string (YYYY-MM-DD).
        max_retries: Number of download attempts before giving up.

    Returns:
        DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume'].
    """
    session = _create_session()
    logger.info("Downloading %s from %s to %s...", ticker, start, end)

    for attempt in range(1, max_retries + 1):
        # --- Strategy 1: yf.download() ---
        try:
            df = _try_download(ticker, start, end, session)
            if len(df) > 0:
                logger.info("Downloaded %d days via yf.download() (attempt %d).", len(df), attempt)
                return df
        except Exception as e:
            logger.warning("yf.download() failed for %s (attempt %d): %s", ticker, attempt, e)

        # --- Strategy 2: yf.Ticker().history() fallback ---
        try:
            df = _try_ticker_history(ticker, start, end, session)
            if len(df) > 0:
                logger.info("Downloaded %d days via Ticker.history() (attempt %d).", len(df), attempt)
                return df
        except Exception as e:
            logger.warning("Ticker.history() failed for %s (attempt %d): %s", ticker, attempt, e)

        if attempt < max_retries:
            wait = 2 * attempt
            logger.info("Waiting %ds before retry...", wait)
            time.sleep(wait)

    logger.error("All %d download attempts failed for %s.", max_retries, ticker)
    return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])


def get_aligned_dates(df: pd.DataFrame, feature_length: int) -> pd.DatetimeIndex:
    """
    Recover the date index aligned with the feature matrix.

    build_feature_matrix() trims warmup rows from the start and returns
    numpy arrays without dates. This function recovers the corresponding
    dates by taking the last `feature_length` dates from the original
    DataFrame.

    Args:
        df: Original OHLCV DataFrame with DatetimeIndex.
        feature_length: Number of rows in the feature matrix (T).

    Returns:
        DatetimeIndex of length T, aligned with the feature matrix.
    """
    return df.index[-feature_length:]


def date_based_split(features: np.ndarray, returns: np.ndarray,
                     prices: np.ndarray, dates: pd.DatetimeIndex,
                     train_end: str = "2021-12-31",
                     test_start: str = "2022-01-01") -> dict:
    """
    Hard date-based chronological split into Train / Test.

    Unlike percentage-based splits, this ensures the exact market regimes
    in each partition are known and intentional:
      - Train: 2010 through 2021 (bulls, 2018 chop, COVID crash + recovery)
      - Test:  2022 (bear market — the stress test)

    Args:
        features: (T, 16) feature matrix.
        returns:  (T,) log return series.
        prices:   (T,) close price series.
        dates:    DatetimeIndex aligned with the feature matrix.
        train_end:  Last date (inclusive) for training data.
        test_start: First date (inclusive) for test data.

    Returns:
        Dictionary with 'train' and 'test' keys, each containing
        (features, returns, prices) tuples.
    """
    train_end_dt = pd.Timestamp(train_end)
    test_start_dt = pd.Timestamp(test_start)

    train_mask = dates <= train_end_dt
    test_mask = dates >= test_start_dt

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    splits = {
        'train': (
            features[train_idx],
            returns[train_idx],
            prices[train_idx],
        ),
        'test': (
            features[test_idx],
            returns[test_idx],
            prices[test_idx],
        ),
    }

    print(f"  Date-based split:")
    print(f"    Train: {dates[train_idx[0]].date()} to "
          f"{dates[train_idx[-1]].date()}  ({len(train_idx)} days)")
    print(f"    Test:  {dates[test_idx[0]].date()} to "
          f"{dates[test_idx[-1]].date()}  ({len(test_idx)} days)")
    return splits


# ---- Legacy function kept for backward compatibility ----

def walk_forward_split(features: np.ndarray, returns: np.ndarray,
                       prices: np.ndarray, train_pct: float = 0.6,
                       val_pct: float = 0.2) -> dict:
    """
    Chronological walk-forward split into Train / Validate / Test.

    This is the ONLY acceptable split strategy for time-series RL.
    Random splits would allow the agent to see future data patterns
    during training — a critical form of data leakage.

    Args:
        features: (T, 16) feature matrix.
        returns:  (T,) log return series.
        prices:   (T,) close price series.
        train_pct: Fraction of data for training (default 60%).
        val_pct:   Fraction of data for validation (default 20%).
                   Remaining = test set.

    Returns:
        Dictionary with 'train', 'val', 'test' keys, each containing
        (features, returns, prices) tuples.
    """
    n = len(features)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    splits = {
        'train': (
            features[:train_end],
            returns[:train_end],
            prices[:train_end],
        ),
        'val': (
            features[train_end:val_end],
            returns[train_end:val_end],
            prices[train_end:val_end],
        ),
        'test': (
            features[val_end:],
            returns[val_end:],
            prices[val_end:],
        ),
    }

    print(f"  Walk-forward split: "
          f"Train={train_end}, Val={val_end - train_end}, "
          f"Test={n - val_end}")
    return splits
