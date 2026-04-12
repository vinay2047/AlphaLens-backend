"""
ShadowPortfolioEnv — Custom Gymnasium environment for RL-based allocation.

The agent's sole decision is the allocation ratio between a basket of assets
(represented as a single price series) and cash. This environment implements
professional-grade reward shaping and cost modeling.

Architecture
------------
    Observation (17-dim):  Quant features + current position
    Action (1-dim [0,1]):  Asset allocation fraction
    Reward:                Blended DSR + scaled return (balanced risk/return)

Look-Ahead Bias Prevention
--------------------------
At step t, the agent observes features computed from data up to time t.
It then decides an allocation. The reward is computed from the return
realized between time t and t+1 (i.e., returns[t+1]).
The agent NEVER sees future price movements before making its decision.

    Timeline:
    ─────────────────────────────────────────────────────►
    t-1       t (observe + decide)     t+1 (reward)

Bug Fixes Applied
-----------------
1. Observation space = 17-dim (16 features + 1 allocation), not 22.
2. DSR denominator uses ε=1e-8 to prevent NaN; reward clipped to [-10, 10].
3. Cash portion earns daily risk-free rate (5% / 252 ≈ 0.0198% per day).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ShadowPortfolioEnv(gym.Env):
    """
    Shadow Portfolio allocation environment.

    Parameters
    ----------
    features : np.ndarray, shape (T, 16)
        Pre-computed feature matrix from build_feature_matrix().
    returns : np.ndarray, shape (T,)
        Log return series aligned with features.
    fee_rate : float
        Proportional transaction fee per unit of allocation change (default 10 bps).
    slippage_bps : float
        Fixed slippage cost per unit of allocation change (default 0.5 bps).
    risk_free_annual : float
        Annual risk-free rate applied daily to the cash portion (default 5%).
    dsr_eta : float
        Exponential moving average decay for DSR running averages (default 0.05).
        Higher η makes the Sharpe estimate more responsive to recent returns,
        preventing the EMA from being stuck near zero during early steps.
    rebalance_threshold : float
        Minimum |Δallocation| to trigger a trade (default 0.5%).
    return_weight : float
        Scaling factor for the portfolio-return component of the blended reward.
        Daily returns (~0.001) are multiplied by this to be comparable in
        magnitude to DSR values (~0.1–0.5). Default 100.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        fee_rate: float = 0.001,          # 10 bps per unit change
        slippage_bps: float = 0.5e-4,     # 0.5 bps slippage
        risk_free_annual: float = 0.05,   # 5% annual risk-free rate
        dsr_eta: float = 0.05,            # DSR EMA decay (faster adaptation)
        rebalance_threshold: float = 0.005,  # 0.5% threshold (smoother alloc)
        return_weight: float = 100.0,     # Scale factor for return component
    ):
        super().__init__()

        assert features.shape[0] == returns.shape[0], \
            f"Feature/return length mismatch: {features.shape[0]} vs {returns.shape[0]}"
        assert features.shape[1] == 16, \
            f"Expected 16 feature columns, got {features.shape[1]}"

        self.features = features
        self.returns = returns
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps
        self.dsr_eta = dsr_eta
        self.rebalance_threshold = rebalance_threshold
        self.return_weight = return_weight

        # ---- Fix #3: Daily risk-free rate for cash returns ----
        # rf_daily = (1 + rf_annual)^(1/252) - 1 ≈ rf_annual / 252
        # Using the simple approximation for small rates
        self.daily_rf = risk_free_annual / 252.0

        # ---- Fix #1: Observation space is strictly 17-dimensional ----
        # 16 market features + 1 current allocation = 17
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
        )

        # Continuous action: fraction of capital in assets [0, 1]
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # ---- Episode state (initialized properly in reset()) ----
        self._step_idx = 0
        self._max_steps = len(self.returns) - 1  # Need returns[t+1]
        self._allocation = 0.0
        self._portfolio_value = 1.0

        # Differential Sharpe Ratio running averages (Moody & Saffell, 1998)
        self._A = 0.0  # EMA of portfolio returns
        self._B = 0.0  # EMA of squared portfolio returns

    def _get_obs(self) -> np.ndarray:
        """
        Build the 17-dim observation vector.

        The 16 market features come from the pre-computed feature matrix at
        the current time step. The 17th dimension is the agent's current
        allocation, which is critical for the agent to understand its own
        position and the cost implications of rebalancing.

        No look-ahead: features[t] uses only data available at time t.
        """
        market_features = self.features[self._step_idx]  # shape (16,)
        obs = np.append(market_features, self._allocation).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        """Reset environment to the start of the episode."""
        super().reset(seed=seed)

        self._step_idx = 0
        self._allocation = 0.0
        self._portfolio_value = 1.0

        # Reset DSR running averages
        self._A = 0.0
        self._B = 0.0

        return self._get_obs(), {}

    def step(self, action):
        """
        Execute one step of the environment.

        Flow:
        1. Parse the new allocation from the action.
        2. Apply rebalancing threshold — skip small trades.
        3. Compute transaction costs for executed trades.
        4. Compute portfolio return (asset + risk-free on cash − costs).
        5. Update DSR running averages and compute reward.
        6. Advance the time index.

        Returns:
            obs, reward, terminated, truncated, info
        """
        new_alloc = float(np.clip(action[0], 0.0, 1.0))
        old_alloc = self._allocation

        # ---- Rebalancing threshold: ignore tiny allocation changes ----
        # This prevents unnecessary churn and the associated costs.
        # Only trades with |Δalloc| ≥ 1% are executed.
        delta_alloc = abs(new_alloc - old_alloc)

        if delta_alloc < self.rebalance_threshold:
            # No trade: keep existing allocation, zero cost
            new_alloc = old_alloc
            tx_cost = 0.0
        else:
            # Transaction cost model:
            #   Cost = |Δalloc| × fee_rate  (proportional fee)
            #        + |Δalloc| × slippage  (execution slippage)
            tx_cost = delta_alloc * (self.fee_rate + self.slippage_bps)

        self._allocation = new_alloc

        # ---- Portfolio return for this step ----
        # Asset return: log return from time t to t+1
        # This is the ONLY place future data enters — and it's the realized
        # return AFTER the agent has already committed to its allocation.
        asset_return = self.returns[self._step_idx + 1]

        # Fix #3: Cash earns risk-free rate, not zero.
        #   portfolio_return = w · r_asset + (1-w) · r_rf − costs
        # This incentivizes the agent to stay invested only when the
        # expected excess return over risk-free compensates for the risk.
        portfolio_return = (
            self._allocation * asset_return
            + (1.0 - self._allocation) * self.daily_rf
            - tx_cost
        )

        # Update portfolio value (multiplicative: V_t = V_{t-1} × exp(r_t))
        self._portfolio_value *= np.exp(portfolio_return)

        # ---- Differential Sharpe Ratio (DSR) reward ----
        #
        # From Moody & Saffell (1998), "Learning to Trade via Direct RL":
        #   A_t = A_{t-1} + η · (R_t − A_{t-1})     [EMA of returns]
        #   B_t = B_{t-1} + η · (R_t² − B_{t-1})    [EMA of squared returns]
        #
        #   DSR_t = (B_{t-1} · ΔA − 0.5 · A_{t-1} · ΔB) / (B_{t-1} − A_{t-1}²)^{3/2}
        #
        # where ΔA = A_t − A_{t-1}, ΔB = B_t − B_{t-1}.
        #
        # The denominator (B − A²) is the variance of returns. This directly
        # optimizes the Sharpe ratio by rewarding return improvements that are
        # large relative to risk, and penalizing return improvements that come
        # with excessive variance increase.

        A_prev = self._A
        B_prev = self._B

        # Update EMAs
        self._A = A_prev + self.dsr_eta * (portfolio_return - A_prev)
        self._B = B_prev + self.dsr_eta * (portfolio_return ** 2 - B_prev)

        delta_A = self._A - A_prev
        delta_B = self._B - B_prev

        # Variance estimate from previous EMAs
        variance = B_prev - A_prev ** 2

        # Fix #2: Add epsilon to prevent NaN when variance ≈ 0.
        # This happens when the agent holds 100% cash (constant returns)
        # or during early episode steps when EMAs haven't diverged.
        eps = 1e-8
        denom = (variance + eps) ** 1.5

        dsr_reward = (B_prev * delta_A - 0.5 * A_prev * delta_B) / denom

        # ---- Blended reward: DSR + scaled portfolio return ----
        #
        # Pure DSR biases heavily toward cash (zero variance = zero risk
        # penalty), causing the agent to under-invest. By adding a scaled
        # log-return component, the agent gets a direct signal for capturing
        # market gains. The DSR component still penalizes uncompensated risk,
        # creating a balanced risk/return tradeoff.
        #
        # return_weight scales daily returns (~0.001) to be comparable
        # in magnitude to DSR values (~0.1–0.5). VecNormalize further
        # stabilizes the combined signal.
        return_component = self.return_weight * portfolio_return
        reward = dsr_reward + return_component

        # Fix #2: Clip reward to [-10, 10] to prevent extreme gradient
        # updates that can destabilize PPO's trust region optimization.
        reward = float(np.clip(reward, -10.0, 10.0))

        # ---- Advance step ----
        self._step_idx += 1
        terminated = self._step_idx >= self._max_steps
        truncated = False

        obs = self._get_obs() if not terminated else np.zeros(17, dtype=np.float32)

        info = {
            "portfolio_value": self._portfolio_value,
            "allocation": self._allocation,
            "portfolio_return": portfolio_return,
            "tx_cost": tx_cost,
        }

        return obs, reward, terminated, truncated, info
