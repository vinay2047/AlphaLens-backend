"""
Feature engineering for the Shadow Portfolio environment.

All features are computed using data available at time t (no look-ahead).
The build_feature_matrix() function returns a (T, 16) array that the
environment augments with the current allocation to form a 17-dim observation.

Math Notes:
-----------
- Log returns: ln(P_t / P_{t-1}) — additive over time, symmetric for gains/losses.
- Fractional differentiation: d ≈ 0.4 balances stationarity (needed for RL
  generalization) with memory (preserving long-range price dependency).
  See: López de Prado, "Advances in Financial Machine Learning" (2018), Ch. 5.
- Rolling volatility: realized vol proxy via rolling std of log returns.
- ATR regime: True Range captures gaps; regime flag triggers when ATR > 1.5×
  its rolling median, indicating a volatility regime shift.
- Relative Strength: 20-day cumulative log return (momentum proxy). In a
  multi-asset setting, this would compare against a separate benchmark.
- SMA crossover: (SMA_fast − SMA_slow) / price normalizes across price levels;
  positive = bullish trend, negative = bearish.
"""

import numpy as np
import pandas as pd


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """
    Continuously compounded return: ln(P_t / P_{t-1}).

    Log returns are preferred over simple returns because they are:
    1. Additive over time (easy to sum for multi-period returns)
    2. Approximately symmetric for small changes
    3. More suitable for statistical modeling (closer to normal distribution)
    """
    return np.log(prices / prices.shift(1))


def fractional_diff(prices: pd.Series, d: float = 0.4,
                    thresh: float = 1e-5) -> pd.Series:
    """
    Fixed-width window fractional differentiation (López de Prado, 2018).

    Standard integer differencing (d=1) achieves stationarity but destroys
    memory. Fractional differencing with 0 < d < 1 finds the minimum d
    that makes the series stationary while preserving maximum memory.

    The binomial expansion weights are:
        w_0 = 1
        w_k = -w_{k-1} * (d - k + 1) / k

    These weights decay geometrically and are truncated when |w_k| < thresh,
    giving a finite impulse response filter.

    Args:
        prices: Raw price series.
        d: Fractional differencing order. d=0.4 is typically sufficient
           for stationarity while retaining ~60% of price memory.
        thresh: Weight truncation threshold for finite window width.

    Returns:
        Fractionally differenced series (NaN for warmup period).
    """
    # ---- Compute filter weights ----
    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < thresh:
            break
        weights.append(w)
        k += 1

    weights = np.array(weights[::-1])  # Reverse for convolution order
    width = len(weights)

    # ---- Apply filter via dot product ----
    result = pd.Series(index=prices.index, dtype=np.float64)
    for i in range(width - 1, len(prices)):
        result.iloc[i] = np.dot(weights, prices.iloc[i - width + 1: i + 1].values)

    return result


def rolling_volatility(returns: pd.Series, window: int) -> pd.Series:
    """
    Rolling standard deviation of log returns.

    A simple estimator for realized volatility. The window parameter
    controls the lookback: shorter windows (10d) capture recent vol,
    longer windows (20d) smooth out noise.
    """
    return returns.rolling(window).std()


def atr_regime(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = 14) -> pd.Series:
    """
    ATR-based volatility regime indicator.

    True Range = max(H-L, |H-C_{t-1}|, |L-C_{t-1}|) captures intraday
    range AND gap risk. ATR is the rolling mean of True Range.

    Regime classification: if ATR > 1.5× its rolling median over a
    longer lookback (5× period), we flag a high-volatility regime.
    The 1.5× threshold is a common risk management heuristic.

    Returns:
        Binary series: 1.0 = high-vol regime, 0.0 = normal regime.
    """
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = true_range.rolling(period).mean()
    # Longer lookback for regime baseline to avoid whipsawing
    median_atr = atr.rolling(period * 5).median()

    regime = (atr > 1.5 * median_atr).astype(float)
    return regime


def relative_strength(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    20-day cumulative log return as a momentum / relative strength proxy.

    In a multi-asset portfolio, this would be:
        RS = cumulative_return(asset) / cumulative_return(benchmark)
    Since SPY IS the market benchmark here, we use its own rolling
    cumulative return, which captures the momentum factor.

    Interpretation: positive = uptrend, negative = downtrend.
    """
    return returns.rolling(window).sum()


def sma_crossover(prices: pd.Series, fast: int = 10,
                  slow: int = 30) -> pd.Series:
    """
    Normalized SMA crossover signal: (SMA_fast − SMA_slow) / price.

    Dividing by price normalizes the spread across different price levels
    (a $1 spread means different things for a $10 vs $500 stock).

    Signal > 0 → short-term trend above long-term (bullish crossover).
    Signal < 0 → short-term trend below long-term (bearish crossover).
    """
    sma_fast = prices.rolling(fast).mean()
    sma_slow = prices.rolling(slow).mean()
    return (sma_fast - sma_slow) / prices


def build_feature_matrix(df: pd.DataFrame) -> tuple:
    """
    Assemble all features into an aligned matrix.

    The feature matrix has 16 columns (the env appends current allocation
    as the 17th dimension):

        [0:10]  Last 10 log returns (lag 0 = most recent)
        [10]    Fractionally differenced price (d=0.4)
        [11]    10-day rolling volatility
        [12]    20-day rolling volatility
        [13]    ATR volatility regime flag (binary)
        [14]    Relative strength / momentum (20-day cumulative return)
        [15]    SMA 10/30 crossover signal (normalized)

    Args:
        df: OHLCV DataFrame from yfinance.

    Returns:
        features:  np.ndarray of shape (T, 16)
        returns:   np.ndarray of shape (T,)   — log returns for reward
        prices:    np.ndarray of shape (T,)   — close prices for baseline
    """
    close = df['Close']
    log_ret = compute_log_returns(close)

    # ---- 10 lagged returns: captures short-term return distribution ----
    ret_lags = pd.DataFrame({
        f'ret_lag_{i}': log_ret.shift(i) for i in range(10)
    })

    # ---- Long-memory stationarity transform ----
    frac = fractional_diff(close, d=0.4)

    # ---- Volatility features ----
    vol_10 = rolling_volatility(log_ret, 10)
    vol_20 = rolling_volatility(log_ret, 20)
    regime = atr_regime(df['High'], df['Low'], close)

    # ---- Momentum features ----
    rel_str = relative_strength(log_ret, 20)
    sma_cross = sma_crossover(close, 10, 30)

    # ---- Combine into single DataFrame ----
    features = pd.concat([
        ret_lags,                           # 10 columns
        frac.rename('frac_diff'),           # 1
        vol_10.rename('vol_10'),            # 1
        vol_20.rename('vol_20'),            # 1
        regime.rename('atr_regime'),        # 1
        rel_str.rename('rel_strength'),     # 1
        sma_cross.rename('sma_cross'),      # 1
    ], axis=1)  # Total: 16 columns

    # ---- Align: drop warmup rows where any feature is NaN ----
    valid_mask = features.notna().all(axis=1) & log_ret.notna()
    first_valid = valid_mask.idxmax()

    features = features.loc[first_valid:]
    log_ret_aligned = log_ret.loc[first_valid:]
    prices_aligned = close.loc[first_valid:]

    # Fill any straggler NaNs (shouldn't occur after proper warmup)
    features = features.fillna(0.0)

    return (
        features.values.astype(np.float32),
        log_ret_aligned.values.astype(np.float32),
        prices_aligned.values.astype(np.float64),
    )
