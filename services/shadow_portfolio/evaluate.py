"""
Evaluation script — Shadow Bot vs Buy & Hold baseline.

Compares the trained RL agent's performance on the held-out test set
against a simple 100%-invested buy-and-hold strategy.

Metrics reported:
    - Total return (%)
    - Annualized Sharpe ratio
    - Maximum drawdown (%)
    - Final portfolio value (per $10,000 invested)

Plots generated:
    1. Cumulative portfolio value (agent vs baseline)
    2. Agent allocation over time
    3. Drawdown comparison

Bug Fix #5: VecNormalize Data Leakage Prevention
-------------------------------------------------
When loading VecNormalize for test evaluation, we MUST lock the running
statistics by setting:
    env.training = False    → Freeze running mean/variance
    env.norm_reward = False → Don't normalize rewards during eval

If training=True (the default), test observations would update the
running statistics, causing:
    1. Data leakage: test distribution leaks into normalization params
    2. Non-reproducibility: results change based on evaluation order
    3. Distribution shift: early vs late test observations are normalized
       with different statistics
"""

import os
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Headless backend (works in Colab and servers)
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Resolve imports when running as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import fetch_data, walk_forward_split
from features import build_feature_matrix
from env import ShadowPortfolioEnv


def compute_metrics(values: np.ndarray, label: str) -> dict:
    """
    Compute and print standard performance metrics.

    Sharpe ratio is annualized: Sharpe = mean(daily_ret) / std(daily_ret) × √252
    Max drawdown: largest peak-to-trough decline as a fraction of peak.
    """
    daily_returns = np.diff(np.log(values + 1e-10))
    total_return = (values[-1] / values[0]) - 1.0

    sharpe = (
        np.mean(daily_returns) / (np.std(daily_returns) + 1e-10) * np.sqrt(252)
    )

    # Max drawdown
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / (peak + 1e-10)
    max_dd = np.max(drawdown)

    print(f"\n{'=' * 45}")
    print(f"  {label}")
    print(f"{'=' * 45}")
    print(f"  Total Return:     {total_return * 100:+.2f}%")
    print(f"  Sharpe Ratio:     {sharpe:.3f}")
    print(f"  Max Drawdown:     {max_dd * 100:.2f}%")
    print(f"  Final Value:      ${values[-1] * 10000:,.2f}  (per $10k)")

    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "final_value": values[-1],
    }


def evaluate():
    """Run the trained agent on the test set and compare with buy-and-hold."""

    # ---- 1. Rebuild test data (same pipeline as training) ----
    print("=" * 60)
    print("EVALUATION: Shadow Bot vs Buy & Hold")
    print("=" * 60)

    print("\nRebuilding data pipeline...")
    df = fetch_data("SPY", start="2010-01-01", end="2024-12-31")
    features, returns, prices = build_feature_matrix(df)
    splits = walk_forward_split(features, returns, prices)
    test_feat, test_ret, test_prices = splits['test']
    print(f"  Test set: {len(test_feat)} trading days")

    # ---- 2. Load model and VecNormalize ----
    try:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "saved_models")
    except NameError:
        save_dir = os.path.join(os.getcwd(), "saved_models")

    vecnorm_path = os.path.join(save_dir, "vec_normalize.pkl")
    model_path = os.path.join(save_dir, "ppo_shadow_portfolio")

    print(f"\n  Loading model from:       {model_path}")
    print(f"  Loading VecNormalize from: {vecnorm_path}")

    # Create test environment
    test_env = DummyVecEnv([
        lambda: ShadowPortfolioEnv(features=test_feat, returns=test_ret)
    ])

    # ---- Fix #5: Lock VecNormalize to prevent data leakage ----
    # training=False: Freeze running mean/var (don't update with test data)
    # norm_reward=False: Return raw rewards for interpretable evaluation
    test_env = VecNormalize.load(vecnorm_path, test_env)
    test_env.training = False       # CRITICAL: freeze running statistics
    test_env.norm_reward = False    # CRITICAL: raw rewards for evaluation

    model = PPO.load(model_path)
    print("  Model and VecNormalize loaded successfully.")

    # ---- 3. Run agent on test set ----
    print("\nRunning agent on test set...")
    obs = test_env.reset()
    agent_values = [1.0]
    allocations = []

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, info = test_env.step(action)
        done = dones[0]
        agent_values.append(info[0]["portfolio_value"])
        allocations.append(info[0]["allocation"])

    agent_values = np.array(agent_values)
    allocations = np.array(allocations)

    # ---- 4. Buy-and-hold baseline ----
    # 100% invested in SPY from day 1. Accumulate returns multiplicatively.
    baseline_returns = test_ret[1:]  # Skip index 0 (need forward return)
    baseline_values = [1.0]
    for r in baseline_returns:
        baseline_values.append(baseline_values[-1] * np.exp(r))
    baseline_values = np.array(baseline_values)

    # Align lengths (agent may terminate 1 step early due to _max_steps)
    min_len = min(len(agent_values), len(baseline_values))
    agent_values = agent_values[:min_len]
    baseline_values = baseline_values[:min_len]

    # ---- 5. Performance metrics ----
    agent_metrics = compute_metrics(agent_values, "Shadow Bot (RL Agent)")
    baseline_metrics = compute_metrics(baseline_values, "Buy & Hold (100% SPY)")

    # ---- 6. Generate plots ----
    print("\nGenerating evaluation plots...")

    fig, axes = plt.subplots(
        3, 1, figsize=(14, 10),
        gridspec_kw={'height_ratios': [3, 1, 1]}
    )

    # --- Plot 1: Cumulative returns ---
    ax1 = axes[0]
    ax1.plot(agent_values, label='Shadow Bot', color='#2196F3',
             linewidth=1.5)
    ax1.plot(baseline_values, label='Buy & Hold (SPY)',
             color='#FF5722', linewidth=1.5, alpha=0.8)
    ax1.set_title('Shadow Portfolio vs Buy & Hold — Test Period',
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value (normalized to $1)')
    ax1.legend(fontsize=12, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Allocation over time ---
    ax2 = axes[1]
    ax2.fill_between(range(len(allocations)), allocations,
                     alpha=0.6, color='#4CAF50', label='Asset Allocation')
    ax2.set_ylabel('Allocation')
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title('Agent Asset Allocation Over Time', fontsize=12)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Drawdown comparison ---
    ax3 = axes[2]
    agent_peak = np.maximum.accumulate(agent_values)
    agent_dd = (agent_peak - agent_values) / (agent_peak + 1e-10)
    baseline_peak = np.maximum.accumulate(baseline_values)
    baseline_dd = (baseline_peak - baseline_values) / (baseline_peak + 1e-10)

    ax3.fill_between(range(len(agent_dd)), -agent_dd, alpha=0.5,
                     color='#2196F3', label='Shadow Bot DD')
    ax3.fill_between(range(len(baseline_dd)), -baseline_dd, alpha=0.5,
                     color='#FF5722', label='Buy & Hold DD')
    ax3.set_ylabel('Drawdown')
    ax3.set_xlabel('Trading Days')
    ax3.set_title('Drawdown Comparison', fontsize=12)
    ax3.legend(fontsize=10, loc='lower left')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    try:
        plot_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        plot_dir = os.getcwd()

    plot_path = os.path.join(plot_dir, "evaluation_results.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved: {plot_path}")

    # Try to show (works in Colab / Jupyter, silently fails in headless)
    try:
        plt.show()
    except Exception:
        pass

    return agent_metrics, baseline_metrics


if __name__ == "__main__":
    evaluate()
