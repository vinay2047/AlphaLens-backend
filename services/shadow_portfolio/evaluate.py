"""
Dual-Regime Evaluation — Shadow Bot vs Buy & Hold.

Tests the trained agent on TWO out-of-sample periods to demonstrate
its value across different market regimes:

    1. BEAR MARKET (2022): The 2022 drawdown — proves capital preservation.
    2. BULL MARKET (2023): The 2023 recovery rally — proves the agent
       doesn't just hide in cash, it participates in upside too.

Model was trained on 2010-2021 data ONLY. Both test periods are
strictly out-of-sample (the agent has never seen this data).

Bug Fix #5: VecNormalize Data Leakage Prevention
-------------------------------------------------
When loading VecNormalize for test evaluation, we lock the running
statistics by setting training=False and norm_reward=False.
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
from data import fetch_data, get_aligned_dates
from features import build_feature_matrix
from env import ShadowPortfolioEnv


def compute_metrics(values: np.ndarray, label: str, silent: bool = False) -> dict:
    """
    Compute standard performance metrics.

    Sharpe ratio is annualized: Sharpe = mean(daily_ret) / std(daily_ret) * sqrt(252)
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

    if not silent:
        print(f"\n{'=' * 50}")
        print(f"  {label}")
        print(f"{'=' * 50}")
        print(f"  [>] SHARPE RATIO:  {sharpe:.3f}")
        print(f"  [*] MAX DRAWDOWN:  {max_dd * 100:.2f}%")
        print(f"  Total Return:     {total_return * 100:+.2f}%")
        print(f"  Final Value:      ${values[-1] * 10000:,.2f}  (per $10k)")

    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "final_value": values[-1],
    }


def run_single_regime(model, vecnorm_path, features, returns, regime_label):
    """
    Run the agent on a single regime's data and return metrics + traces.

    Args:
        model: Loaded PPO model.
        vecnorm_path: Path to VecNormalize pickle.
        features: (T, 16) feature matrix for this regime.
        returns: (T,) log returns for this regime.
        regime_label: String label for printing (e.g., "2022 Bear Market").

    Returns:
        agent_metrics, baseline_metrics, agent_values, baseline_values, allocations
    """
    print(f"\n  Running on {regime_label} ({len(features)} trading days)...")

    # Create environment with locked VecNormalize
    test_env = DummyVecEnv([
        lambda f=features, r=returns: ShadowPortfolioEnv(features=f, returns=r)
    ])
    test_env = VecNormalize.load(vecnorm_path, test_env)
    test_env.training = False       # Freeze running statistics
    test_env.norm_reward = False    # Raw rewards for evaluation

    # Run agent
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

    # Buy-and-hold baseline
    baseline_returns = returns[1:]
    baseline_values = [1.0]
    for r in baseline_returns:
        baseline_values.append(baseline_values[-1] * np.exp(r))
    baseline_values = np.array(baseline_values)

    # Align lengths
    min_len = min(len(agent_values), len(baseline_values))
    agent_values = agent_values[:min_len]
    baseline_values = baseline_values[:min_len]

    # Compute metrics
    agent_metrics = compute_metrics(
        agent_values, f"Shadow Bot -- {regime_label}")
    baseline_metrics = compute_metrics(
        baseline_values, f"Buy & Hold -- {regime_label}")

    return agent_metrics, baseline_metrics, agent_values, baseline_values, allocations


def extract_regime(df, features, returns, prices, start_date, end_date):
    """
    Extract a date-based slice from the feature matrix.

    Args:
        df: Original OHLCV DataFrame with DatetimeIndex.
        features, returns, prices: Arrays from build_feature_matrix().
        start_date, end_date: Date strings for the regime window.

    Returns:
        regime_features, regime_returns, regime_prices
    """
    import pandas as pd
    dates = get_aligned_dates(df, len(features))
    mask = (dates >= pd.Timestamp(start_date)) & (dates <= pd.Timestamp(end_date))
    idx = np.where(mask)[0]

    if len(idx) == 0:
        raise ValueError(f"No data found for {start_date} to {end_date}")

    print(f"    {dates[idx[0]].date()} to {dates[idx[-1]].date()} ({len(idx)} days)")
    return features[idx], returns[idx], prices[idx]


def evaluate():
    """
    Dual-regime evaluation: Bear Market (2022) + Bull Market (2023).

    Both periods are out-of-sample — the model was trained on 2010-2021 only.
    """
    print("=" * 60)
    print("  DUAL-REGIME EVALUATION")
    print("  Model trained: 2010-2021 | Test: 2022 (Bear) + 2023 (Bull)")
    print("=" * 60)

    # ---- 1. Fetch full data range (2010 through 2023 for both test windows) ----
    print("\nFetching data...")
    df = fetch_data("SPY", start="2010-01-01", end="2023-12-31")
    features, returns, prices = build_feature_matrix(df)
    print(f"  Feature matrix: {features.shape}")

    # ---- 2. Extract regime slices ----
    print("\nExtracting regime windows:")
    print("  Bear Market (2022):")
    bear_feat, bear_ret, bear_prices = extract_regime(
        df, features, returns, prices, "2022-01-01", "2022-12-31")

    print("  Bull Market (2023):")
    bull_feat, bull_ret, bull_prices = extract_regime(
        df, features, returns, prices, "2023-01-01", "2023-12-31")

    # ---- 3. Load model ----
    try:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "saved_models")
    except NameError:
        save_dir = os.path.join(os.getcwd(), "saved_models")

    vecnorm_path = os.path.join(save_dir, "vec_normalize.pkl")
    model_path = os.path.join(save_dir, "ppo_shadow_portfolio")

    print(f"\n  Loading model from: {model_path}")
    model = PPO.load(model_path)
    print("  Model loaded successfully.")

    # ---- 4. Run both regimes ----
    print("\n" + "=" * 60)
    print("  REGIME 1: 2022 BEAR MARKET (Stress Test)")
    print("=" * 60)
    bear_agent, bear_baseline, bear_av, bear_bv, bear_alloc = run_single_regime(
        model, vecnorm_path, bear_feat, bear_ret, "2022 Bear Market")

    print("\n" + "=" * 60)
    print("  REGIME 2: 2023 BULL MARKET (Upside Participation)")
    print("=" * 60)
    bull_agent, bull_baseline, bull_av, bull_bv, bull_alloc = run_single_regime(
        model, vecnorm_path, bull_feat, bull_ret, "2023 Bull Market")

    # ---- 5. Print combined summary ----
    print("\n\n" + "=" * 60)
    print("  COMBINED RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n  {'Metric':<20} {'Bear 2022':>14} {'Bull 2023':>14}")
    print(f"  {'-'*20} {'-'*14} {'-'*14}")
    print(f"  {'Shadow Return':<20} "
          f"{bear_agent['total_return']*100:>+13.2f}% "
          f"{bull_agent['total_return']*100:>+13.2f}%")
    print(f"  {'B&H Return':<20} "
          f"{bear_baseline['total_return']*100:>+13.2f}% "
          f"{bull_baseline['total_return']*100:>+13.2f}%")
    print(f"  {'Shadow Sharpe':<20} "
          f"{bear_agent['sharpe']:>14.3f} "
          f"{bull_agent['sharpe']:>14.3f}")
    print(f"  {'B&H Sharpe':<20} "
          f"{bear_baseline['sharpe']:>14.3f} "
          f"{bull_baseline['sharpe']:>14.3f}")
    print(f"  {'Shadow Max DD':<20} "
          f"{bear_agent['max_dd']*100:>13.2f}% "
          f"{bull_agent['max_dd']*100:>13.2f}%")
    print(f"  {'B&H Max DD':<20} "
          f"{bear_baseline['max_dd']*100:>13.2f}% "
          f"{bull_baseline['max_dd']*100:>13.2f}%")

    # ---- 6. Generate dual-regime plot ----
    print("\nGenerating dual-regime evaluation plots...")

    fig, axes = plt.subplots(3, 2, figsize=(18, 12))

    # === LEFT COLUMN: 2022 Bear Market ===
    # Cumulative returns
    axes[0, 0].plot(bear_av, label='Shadow Bot', color='#2196F3', linewidth=1.5)
    axes[0, 0].plot(bear_bv, label='Buy & Hold', color='#FF5722',
                    linewidth=1.5, alpha=0.8)
    axes[0, 0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].set_title(
        f'2022 Bear Market (Out-of-Sample)\n'
        f'Shadow: {bear_agent["total_return"]*100:+.1f}% | '
        f'B&H: {bear_baseline["total_return"]*100:+.1f}%',
        fontsize=13, fontweight='bold')
    axes[0, 0].set_ylabel('Portfolio Value')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # Allocation
    axes[1, 0].fill_between(range(len(bear_alloc)), bear_alloc,
                            alpha=0.6, color='#4CAF50')
    axes[1, 0].set_ylabel('Allocation')
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].set_title('Agent Allocation (2022)', fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)

    # Drawdown
    bear_peak = np.maximum.accumulate(bear_av)
    bear_dd = (bear_peak - bear_av) / (bear_peak + 1e-10)
    bear_bpeak = np.maximum.accumulate(bear_bv)
    bear_bdd = (bear_bpeak - bear_bv) / (bear_bpeak + 1e-10)
    axes[2, 0].fill_between(range(len(bear_dd)), -bear_dd,
                            alpha=0.5, color='#2196F3', label='Shadow Bot')
    axes[2, 0].fill_between(range(len(bear_bdd)), -bear_bdd,
                            alpha=0.5, color='#FF5722', label='Buy & Hold')
    axes[2, 0].set_ylabel('Drawdown')
    axes[2, 0].set_xlabel('Trading Days')
    axes[2, 0].set_title(
        f'Drawdown: Shadow {bear_agent["max_dd"]*100:.1f}% vs '
        f'B&H {bear_baseline["max_dd"]*100:.1f}%', fontsize=11)
    axes[2, 0].legend(fontsize=9)
    axes[2, 0].grid(True, alpha=0.3)

    # === RIGHT COLUMN: 2023 Bull Market ===
    # Cumulative returns
    axes[0, 1].plot(bull_av, label='Shadow Bot', color='#2196F3', linewidth=1.5)
    axes[0, 1].plot(bull_bv, label='Buy & Hold', color='#FF5722',
                    linewidth=1.5, alpha=0.8)
    axes[0, 1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].set_title(
        f'2023 Bull Market (Out-of-Sample)\n'
        f'Shadow: {bull_agent["total_return"]*100:+.1f}% | '
        f'B&H: {bull_baseline["total_return"]*100:+.1f}%',
        fontsize=13, fontweight='bold')
    axes[0, 1].set_ylabel('Portfolio Value')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # Allocation
    axes[1, 1].fill_between(range(len(bull_alloc)), bull_alloc,
                            alpha=0.6, color='#4CAF50')
    axes[1, 1].set_ylabel('Allocation')
    axes[1, 1].set_ylim(-0.05, 1.05)
    axes[1, 1].set_title('Agent Allocation (2023)', fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)

    # Drawdown
    bull_peak = np.maximum.accumulate(bull_av)
    bull_dd = (bull_peak - bull_av) / (bull_peak + 1e-10)
    bull_bpeak = np.maximum.accumulate(bull_bv)
    bull_bdd = (bull_bpeak - bull_bv) / (bull_bpeak + 1e-10)
    axes[2, 1].fill_between(range(len(bull_dd)), -bull_dd,
                            alpha=0.5, color='#2196F3', label='Shadow Bot')
    axes[2, 1].fill_between(range(len(bull_bdd)), -bull_bdd,
                            alpha=0.5, color='#FF5722', label='Buy & Hold')
    axes[2, 1].set_ylabel('Drawdown')
    axes[2, 1].set_xlabel('Trading Days')
    axes[2, 1].set_title(
        f'Drawdown: Shadow {bull_agent["max_dd"]*100:.1f}% vs '
        f'B&H {bull_baseline["max_dd"]*100:.1f}%', fontsize=11)
    axes[2, 1].legend(fontsize=9)
    axes[2, 1].grid(True, alpha=0.3)

    plt.suptitle('Shadow Portfolio: Dual-Regime Out-of-Sample Evaluation\n'
                 '(Model trained on 2010-2021 only)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    try:
        plot_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        plot_dir = os.getcwd()

    plot_path = os.path.join(plot_dir, "evaluation_results.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved: {plot_path}")

    try:
        plt.show()
    except Exception:
        pass

    return {
        "bear": {"agent": bear_agent, "baseline": bear_baseline},
        "bull": {"agent": bull_agent, "baseline": bull_baseline},
    }


if __name__ == "__main__":
    evaluate()
