"""
Training pipeline for the Shadow Portfolio PPO agent.

Pipeline:
    1. Fetch historical data (SPY via yfinance)
    2. Compute quant features (log returns, frac diff, vol, momentum)
    3. Walk-forward split (60% train / 20% val / 20% test)
    4. Wrap environment in DummyVecEnv → VecNormalize
    5. Train PPO for 500,000 timesteps (Fix #4)
    6. Validate on held-out validation set
    7. Save model + VecNormalize statistics

VecNormalize
------------
Wrapping the env in VecNormalize automatically scales observations to
zero mean / unit variance and normalizes rewards. This is critical because:
  - Our 17 observation features span vastly different scales
    (log returns ≈ 0.01, frac diff prices ≈ 100s)
  - Without normalization, the neural network would be dominated by
    large-scale features, ignoring informative small-scale ones
  - Reward normalization stabilizes PPO's value function training

PPO Hyperparameters
-------------------
  - ent_coef=0.01: Entropy bonus encourages the agent to explore different
    allocation levels rather than collapsing to a deterministic policy early.
  - n_steps=2048: Rollout length balances bias (short) vs variance (long).
  - batch_size=64: Mini-batch size for PPO's surrogate objective updates.
"""

import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Resolve imports when running as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import fetch_data, walk_forward_split
from features import build_feature_matrix
from env import ShadowPortfolioEnv


def make_env(features, returns):
    """Factory function for DummyVecEnv (requires a callable)."""
    def _init():
        return ShadowPortfolioEnv(features=features, returns=returns)
    return _init


def train():
    """Full training pipeline: data → features → train → validate → save."""

    # ---- 1. Fetch data ----
    print("=" * 60)
    print("STEP 1: Fetching data")
    print("=" * 60)
    df = fetch_data("SPY", start="2010-01-01", end="2024-12-31")

    # ---- 2. Build features ----
    print("\n" + "=" * 60)
    print("STEP 2: Computing quant features")
    print("=" * 60)
    features, returns, prices = build_feature_matrix(df)
    print(f"  Feature matrix: {features.shape}  (T × 16)")
    print(f"  Returns:        {returns.shape}")
    print(f"  Prices:         {prices.shape}")

    # ---- 3. Walk-forward split ----
    print("\n" + "=" * 60)
    print("STEP 3: Walk-forward chronological split")
    print("=" * 60)
    splits = walk_forward_split(features, returns, prices)
    train_feat, train_ret, _ = splits['train']
    val_feat, val_ret, _ = splits['val']

    # ---- 4. Create vectorized + normalized environment ----
    print("\n" + "=" * 60)
    print("STEP 4: Building training environment")
    print("=" * 60)
    train_env = DummyVecEnv([make_env(train_feat, train_ret)])
    train_env = VecNormalize(
        train_env,
        norm_obs=True,       # Scale observations to N(0,1)
        norm_reward=True,    # Scale rewards for stable value function
        clip_obs=10.0,       # Clip normalized obs to [-10, 10]
    )
    print(f"  Observation space: {train_env.observation_space}")
    print(f"  Action space:      {train_env.action_space}")

    # ---- 5. Initialize PPO agent ----
    print("\n" + "=" * 60)
    print("STEP 5: Initializing PPO agent")
    print("=" * 60)
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,         # Rollout buffer size
        batch_size=64,        # Mini-batch size for SGD updates
        n_epochs=10,          # PPO epochs per rollout
        gamma=0.99,           # Discount factor
        ent_coef=0.01,        # Entropy bonus for exploration
        verbose=1,
        seed=42,
    )
    print("  Policy architecture: MlpPolicy (64x64 hidden layers)")
    print("  Entropy coefficient: 0.01 (exploration bonus)")

    # ---- 6. Train (Fix #4: 500k timesteps for convergence) ----
    print("\n" + "=" * 60)
    print("STEP 6: Training (500,000 timesteps)")
    print("=" * 60)
    model.learn(total_timesteps=500_000)

    # ---- 7. Save model + VecNormalize stats ----
    try:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "saved_models")
    except NameError:
        save_dir = os.path.join(os.getcwd(), "saved_models")

    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "ppo_shadow_portfolio")
    vecnorm_path = os.path.join(save_dir, "vec_normalize.pkl")

    model.save(model_path)
    train_env.save(vecnorm_path)
    print(f"\n  Model saved:       {model_path}")
    print(f"  VecNormalize saved: {vecnorm_path}")

    # ---- 8. Quick validation run ----
    print("\n" + "=" * 60)
    print("STEP 7: Validation run")
    print("=" * 60)
    val_env = DummyVecEnv([make_env(val_feat, val_ret)])
    val_env = VecNormalize.load(vecnorm_path, val_env)
    # Lock normalization stats during validation (same logic as Fix #5)
    val_env.training = False
    val_env.norm_reward = False

    obs = val_env.reset()
    total_reward = 0.0
    steps = 0
    val_pv = 1.0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = val_env.step(action)
        total_reward += reward[0]
        steps += 1
        val_pv = info[0].get("portfolio_value", val_pv)
        if done[0]:
            break

    print(f"  Validation steps:           {steps}")
    print(f"  Validation total reward:    {total_reward:.4f}")
    print(f"  Validation portfolio value: {val_pv:.4f}")
    print(f"  Validation return:          {(val_pv - 1) * 100:+.2f}%")

    print("\n" + "=" * 60)
    print("Training complete. Run evaluate.py for full test results.")
    print("=" * 60)

    return model, train_env


if __name__ == "__main__":
    train()
