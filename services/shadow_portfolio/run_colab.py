"""
Shadow Portfolio RL Agent — Google Colab Entry Point

Run this script in a Colab notebook cell or as a standalone Python script.
It will install dependencies, train the PPO agent, and run evaluation.

Usage in Colab:
    1. Upload the entire shadow_portfolio/ directory to Colab
    2. Run: %cd /content/shadow_portfolio
    3. Run: %run run_colab.py

Usage locally:
    cd services/shadow_portfolio
    pip install -r requirements.txt
    python run_colab.py
"""

import subprocess
import sys
import os

# ---- Install dependencies (safe to re-run in Colab) ----
print("Installing dependencies...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "gymnasium>=0.29.0",
    "stable-baselines3>=2.1.0",
    "yfinance>=0.2.31",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "matplotlib>=3.7.0",
])
print("Dependencies installed.\n")

# ---- Ensure imports resolve from this directory ----
script_dir = os.path.dirname(os.path.abspath(__file__)) \
    if '__file__' in dir() else os.getcwd()
sys.path.insert(0, script_dir)

# ---- Run training pipeline ----
from train import train
print("\n" + "=" * 60)
print("  SHADOW PORTFOLIO — TRAINING PHASE")
print("=" * 60 + "\n")

model, train_env = train()

# ---- Run evaluation ----
from evaluate import evaluate
print("\n\n" + "=" * 60)
print("  SHADOW PORTFOLIO — EVALUATION PHASE")
print("=" * 60 + "\n")

agent_metrics, baseline_metrics = evaluate()

# ---- Summary ----
print("\n\n" + "=" * 60)
print("  FINAL COMPARISON")
print("=" * 60)
print(f"\n  {'Metric':<20} {'Shadow Bot':>12} {'Buy & Hold':>12}")
print(f"  {'-'*20} {'-'*12} {'-'*12}")
print(f"  {'Total Return':<20} "
      f"{agent_metrics['total_return']*100:>+11.2f}% "
      f"{baseline_metrics['total_return']*100:>+11.2f}%")
print(f"  {'Sharpe Ratio':<20} "
      f"{agent_metrics['sharpe']:>12.3f} "
      f"{baseline_metrics['sharpe']:>12.3f}")
print(f"  {'Max Drawdown':<20} "
      f"{agent_metrics['max_dd']*100:>11.2f}% "
      f"{baseline_metrics['max_dd']*100:>11.2f}%")
print(f"\n  See evaluation_results.png for detailed plots.")
print("=" * 60)
