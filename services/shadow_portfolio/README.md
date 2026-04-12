# Shadow Portfolio — RL-Based Asset Allocation Agent

A reinforcement learning agent that learns optimal allocation between a basket of assets (SPY) and cash, using professional-grade quant features and risk-adjusted reward shaping.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    PPO Agent (MlpPolicy)                │
│            Decides: allocation ∈ [0, 1]                 │
└──────────────────────┬──────────────────────────────────┘
                       │ action
                       ▼
┌─────────────────────────────────────────────────────────┐
│              ShadowPortfolioEnv (Gymnasium)             │
│                                                         │
│  Observation (17-dim):                                  │
│    [0:10]  Log returns (10-day lookback)                │
│    [10]    Fractionally differenced price               │
│    [11-12] Rolling volatility (10d, 20d)                │
│    [13]    ATR volatility regime flag                   │
│    [14]    Relative strength (20d momentum)             │
│    [15]    SMA crossover (10/30)                        │
│    [16]    Current allocation (agent's position)        │
│                                                         │
│  Reward: Differential Sharpe Ratio - Transaction Costs  │
│  Cash: Earns daily risk-free rate (5%/252)              │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### Local
```bash
cd services/shadow_portfolio
pip install -r requirements.txt

# Train the agent (500,000 timesteps)
python train.py

# Evaluate against buy-and-hold
python evaluate.py
```

### Google Colab
```python
# Upload the shadow_portfolio/ directory, then:
%cd /content/shadow_portfolio
%run run_colab.py
```

## File Structure

| File | Description |
|------|-------------|
| `features.py` | Quant feature engineering (log returns, frac diff, vol, momentum) |
| `data.py` | yfinance data acquisition + walk-forward split |
| `env.py` | Custom Gymnasium environment (17-dim obs, DSR reward) |
| `train.py` | PPO training pipeline with VecNormalize |
| `evaluate.py` | Test evaluation + performance comparison plots |


## Key Design Decisions

### Observation Space (17 dimensions)
All features avoid look-ahead bias — computed from data available at decision time only.

- **Log Returns (10d)**: `ln(P_t/P_{t-1})` — additive, symmetric, near-normal
- **Fractional Differentiation (d=0.4)**: Balances stationarity with memory preservation (López de Prado, 2018)
- **Rolling Volatility**: Realized vol proxy at two timescales (10d, 20d)
- **ATR Regime**: Binary flag when volatility > 1.5× median — signals regime shifts
- **Relative Strength**: 20-day cumulative return — captures momentum factor
- **SMA Crossover**: Normalized (SMA₁₀ − SMA₃₀)/price — trend signal
- **Current Allocation**: Agent's own position — critical for cost-aware decisions

### Reward: Differential Sharpe Ratio
Instead of optimizing raw returns (which ignores risk), we use the step-by-step change in the Sharpe ratio (Moody & Saffell, 1998):

```
DSR_t = (B_{t-1} · ΔA_t − 0.5 · A_{t-1} · ΔB_t) / (B_{t-1} − A_{t-1}²)^{3/2}
```

This directly penalizes uncompensated risk: a return increase paired with a larger variance increase yields a *negative* reward.

### Transaction Cost Model
- **Proportional fee**: `|Δallocation| × 10 bps`
- **Slippage**: `|Δallocation| × 0.5 bps`
- **Rebalancing threshold**: Trades < 1% allocation change are ignored (no cost, no execution)

### Walk-Forward Validation
Chronological 60/20/20 split — no random shuffling, no future leakage:
```
|------- Train (60%) -------|--- Val (20%) ---|--- Test (20%) ---|
```

## Applied Bug Fixes

1. **Dimension Mismatch**: Observation space = 17 (16 features + 1 allocation)
2. **DSR NaN Trap**: ε=10⁻⁸ in variance denominator + reward clipped to [-10, 10]
3. **Risk-Free Rate**: Cash earns 5%/252 daily, not zero
4. **Timesteps**: 500,000 (up from 50,000) for proper convergence
5. **VecNormalize Leakage**: `training=False, norm_reward=False` during evaluation
