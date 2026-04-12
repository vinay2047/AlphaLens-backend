"""
inference.py – Core inference engine for the Shadow Portfolio RL agent.

This module handles the complete inference pipeline:
    1. Fetch OHLCV data via yfinance for the requested ticker/date range
    2. Compute the 16-dimensional quant feature matrix
    3. Load the pre-trained PPO model and frozen VecNormalize statistics
    4. Step through the environment day-by-day, collecting allocations
    5. Compute performance metrics for both agent and buy-and-hold baseline

How Inference Works (detailed)
------------------------------
At each trading day t, the inference loop:

    a) Constructs a 17-dim observation:
         [0:10]  Last 10 log returns (captures recent return distribution)
         [10]    Fractionally differenced price (stationary + memory)
         [11-12] Rolling volatility at 10d and 20d (risk signal)
         [13]    ATR regime flag (1 = high-vol regime)
         [14]    20-day momentum (trend signal)
         [15]    SMA 10/30 crossover (trend confirmation)
         [16]    Current allocation (cost-awareness)

    b) VecNormalize scales the observation using FROZEN training statistics
       (mean/std computed during training on 2010-2021 SPY data). This ensures
       the neural network receives inputs in the same scale it was trained on.
       training=False prevents test data from contaminating the statistics.

    c) PPO's MLP policy (2 hidden layers of 64 units each) performs a forward
       pass through the neural network. The actor head outputs a Gaussian
       distribution over allocations, and we take the MEAN (deterministic=True)
       as the allocation decision. This avoids exploration noise during inference.

    d) The allocation w ∈ [0, 1] determines capital split:
         - w × capital → invested in the asset (earns market return)
         - (1-w) × capital → held in cash (earns risk-free rate: 5%/252 daily)
         - Transaction costs are deducted for allocation changes > 0.5%

    e) Portfolio value updates multiplicatively:
         V_{t+1} = V_t × exp(w · r_asset + (1-w) · r_rf − costs)

Note: The model was trained on SPY only. Running on other tickers is possible
but the features (especially the ATR regime thresholds and momentum windows)
are tuned for SPY's market microstructure. Results on other tickers should
be treated as experimental.
"""

import os
import sys
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ---- Resolve imports to sibling modules (data.py, features.py, env.py) ----
_SHADOW_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SHADOW_DIR not in sys.path:
    sys.path.insert(0, _SHADOW_DIR)

from data import fetch_data, get_aligned_dates          # noqa: E402
from features import build_feature_matrix               # noqa: E402
from env import ShadowPortfolioEnv                       # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model cache (loaded once at startup, reused across requests)
# ---------------------------------------------------------------------------

_cached_model: PPO | None = None
_cached_vecnorm_path: str | None = None


def get_model_paths() -> tuple[str, str]:
    """Return (model_path, vecnorm_path) for the saved PPO agent."""
    save_dir = os.path.join(_SHADOW_DIR, "saved_models")
    model_path = os.path.join(save_dir, "ppo_shadow_portfolio")
    vecnorm_path = os.path.join(save_dir, "vec_normalize.pkl")
    return model_path, vecnorm_path


def load_model() -> tuple[PPO, str]:
    """
    Load the pre-trained PPO model (cached after first call).

    Returns:
        (model, vecnorm_path) — the PPO instance and path to VecNormalize pkl.

    Raises:
        FileNotFoundError: If saved model files are missing.
    """
    global _cached_model, _cached_vecnorm_path

    if _cached_model is not None:
        return _cached_model, _cached_vecnorm_path

    model_path, vecnorm_path = get_model_paths()

    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(
            f"Trained model not found at {model_path}.zip — "
            f"run train.py first to create the model."
        )
    if not os.path.exists(vecnorm_path):
        raise FileNotFoundError(
            f"VecNormalize stats not found at {vecnorm_path} — "
            f"run train.py first."
        )

    logger.info("Loading PPO model from %s", model_path)
    import torch
    original_torch_load = torch.load

    def safe_torch_load(*args, **kwargs):
        # Force mmap=False to avoid Render's memory-mapping restrictions
        # which cause the PytorchStreamReader miniz read failures.
        kwargs["mmap"] = False
        kwargs["map_location"] = torch.device('cpu')
        return original_torch_load(*args, **kwargs)

    try:
        torch.load = safe_torch_load
        _cached_model = PPO.load(model_path, device="cpu")
    finally:
        torch.load = original_torch_load
    _cached_vecnorm_path = vecnorm_path
    logger.info("Model loaded successfully.")

    return _cached_model, _cached_vecnorm_path


def _compute_metrics(values: np.ndarray) -> dict:
    """
    Compute standard portfolio performance metrics.

    Args:
        values: Array of cumulative portfolio values (starting near 1.0).

    Returns:
        Dict with total_return, sharpe_ratio, max_drawdown, final_portfolio_value.
    """
    daily_returns = np.diff(np.log(values + 1e-10))
    total_return = float((values[-1] / values[0]) - 1.0)

    sharpe_ratio = float(
        np.mean(daily_returns) / (np.std(daily_returns) + 1e-10) * np.sqrt(252)
    )

    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / (peak + 1e-10)
    max_dd = float(np.max(drawdown))

    return {
        "total_return": round(total_return, 6),
        "sharpe_ratio": round(sharpe_ratio, 4),
        "max_drawdown": round(max_dd, 6),
        "final_portfolio_value": round(float(values[-1]), 6),
    }


def run_inference(
    ticker: str = "SPY",
    start_date: str = "2024-01-01",
    end_date: str = "2024-06-30",
) -> dict:
    """
    Run the trained PPO agent on a ticker for a given date range.

    Pipeline:
        1. Fetch OHLCV data (with warmup buffer for feature computation)
        2. Build the 16-dim feature matrix
        3. Extract the requested date window
        4. Load model + VecNormalize (frozen stats)
        5. Step through environment day-by-day
        6. Compute metrics for agent and buy-and-hold baseline

    Args:
        ticker:     Yahoo Finance ticker symbol.
        start_date: Inference window start (YYYY-MM-DD).
        end_date:   Inference window end (YYYY-MM-DD).

    Returns:
        Dictionary with ticker, dates, trading_days, agent_metrics,
        baseline_metrics, daily_results, and model_info.

    Raises:
        ValueError: If no trading data is found in the requested range.
        FileNotFoundError: If the trained model is missing.
    """
    logger.info("Inference request: %s from %s to %s", ticker, start_date, end_date)

    # ---- 1. Fetch data with warmup buffer --------------------------------
    # Features like SMA(30), rolling vol(20), and fractional diff need
    # historical warmup data. We fetch ~200 extra trading days before
    # the requested start_date to ensure all features are valid.
    warmup_start = (
        datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=400)
    ).strftime("%Y-%m-%d")

    logger.info("Fetching data from %s to %s (includes warmup)", warmup_start, end_date)
    df = fetch_data(ticker, start=warmup_start, end=end_date)

    if len(df) < 50:
        raise ValueError(
            f"Insufficient data for {ticker} between {warmup_start} and {end_date}. "
            f"Got only {len(df)} trading days (need at least 50 for feature warmup)."
        )

    # ---- 2. Build features ------------------------------------------------
    features, returns, prices = build_feature_matrix(df)
    logger.info("Feature matrix: %s", features.shape)

    # ---- 3. Extract the requested date window -----------------------------
    dates = get_aligned_dates(df, len(features))
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    mask = (dates >= start_ts) & (dates <= end_ts)
    idx = np.where(mask)[0]

    if len(idx) == 0:
        raise ValueError(
            f"No trading data found for {ticker} between {start_date} and {end_date}. "
            f"Check that the date range contains valid trading days."
        )

    window_features = features[idx]
    window_returns = returns[idx]
    window_dates = dates[idx]

    logger.info(
        "Inference window: %s to %s (%d trading days)",
        window_dates[0].date(), window_dates[-1].date(), len(idx),
    )

    # ---- 4. Load model + VecNormalize ------------------------------------
    model, vecnorm_path = load_model()

    # Create environment with FROZEN VecNormalize statistics.
    # training=False    → prevents test data from updating running mean/std
    # norm_reward=False → we want raw rewards for metric computation
    inf_env = DummyVecEnv([
        lambda f=window_features, r=window_returns:
            ShadowPortfolioEnv(features=f, returns=r)
    ])
    inf_env = VecNormalize.load(vecnorm_path, inf_env)
    inf_env.training = False
    inf_env.norm_reward = False

    # ---- 5. Inference loop -----------------------------------------------
    obs = inf_env.reset()
    agent_values = [1.0]
    allocations = []
    daily_returns_list = []

    done = False
    while not done:
        # PPO forward pass: deterministic=True takes the mean of the
        # Gaussian policy (no exploration noise during inference)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, info = inf_env.step(action)
        done = dones[0]

        agent_values.append(info[0]["portfolio_value"])
        allocations.append(info[0]["allocation"])
        daily_returns_list.append(info[0]["portfolio_return"])

    agent_values = np.array(agent_values)
    allocations = np.array(allocations)

    # ---- 6. Buy-and-hold baseline ----------------------------------------
    baseline_returns = window_returns[1:]  # first return is used as t=0
    baseline_values = [1.0]
    for r in baseline_returns:
        baseline_values.append(baseline_values[-1] * np.exp(r))
    baseline_values = np.array(baseline_values)

    # Align lengths (agent may terminate 1 step early)
    min_len = min(len(agent_values), len(baseline_values))
    agent_values = agent_values[:min_len]
    baseline_values = baseline_values[:min_len]

    # ---- 7. Compute metrics ----------------------------------------------
    agent_metrics = _compute_metrics(agent_values)
    baseline_metrics = _compute_metrics(baseline_values)

    # ---- 8. Build daily results ------------------------------------------
    # The environment steps from day 0 to day T-1, producing T-1 actions.
    # Daily results correspond to the days AFTER the first observation.
    result_dates = window_dates[1: 1 + len(allocations)]

    daily_results = []
    for i in range(min(len(allocations), len(result_dates))):
        daily_results.append({
            "date": str(result_dates[i].date()),
            "allocation": round(float(allocations[i]), 4),
            "agent_portfolio_value": round(float(agent_values[i + 1]), 6),
            "baseline_portfolio_value": round(
                float(baseline_values[i + 1]) if i + 1 < len(baseline_values) else float(baseline_values[-1]),
                6,
            ),
            "daily_return": round(float(daily_returns_list[i]), 6),
        })

    # ---- 9. Assemble response --------------------------------------------
    return {
        "ticker": ticker.upper(),
        "start_date": start_date,
        "end_date": end_date,
        "trading_days": len(daily_results),
        "agent_metrics": agent_metrics,
        "baseline_metrics": baseline_metrics,
        "daily_results": daily_results,
        "model_info": {
            "algorithm": "PPO (Proximal Policy Optimization)",
            "policy": "MlpPolicy (64×64 hidden layers)",
            "trained_on": "SPY 2010-01-01 to 2021-12-31",
            "training_timesteps": 500_000,
            "observation_dim": 17,
            "features": [
                "log_returns_lag_0..9 (10)",
                "fractional_diff_d0.4 (1)",
                "rolling_vol_10d (1)",
                "rolling_vol_20d (1)",
                "atr_regime_flag (1)",
                "relative_strength_20d (1)",
                "sma_crossover_10_30 (1)",
                "current_allocation (1)",
            ],
            "reward": "Differential Sharpe Ratio + scaled return",
            "transaction_costs": "10 bps fee + 0.5 bps slippage per rebalance",
            "note": (
                "Model was trained on SPY only. Inference on other tickers "
                "is experimental — features are tuned for SPY market dynamics."
            ),
        },
    }
