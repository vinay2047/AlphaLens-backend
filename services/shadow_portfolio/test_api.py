"""Quick test script for the inference API."""
import urllib.request
import json

BASE = "http://localhost:8001"

# 1. Health check
print("=== Health Check ===")
r = urllib.request.urlopen(f"{BASE}/")
print(json.loads(r.read()))

# 2. POST inference
print("\n=== POST /api/inference ===")
body = json.dumps({
    "ticker": "SPY",
    "start_date": "2024-01-01",
    "end_date": "2024-03-31",
}).encode()

req = urllib.request.Request(
    f"{BASE}/api/inference",
    data=body,
    headers={"Content-Type": "application/json"},
)
r = urllib.request.urlopen(req)
data = json.loads(r.read())

print(f"Ticker: {data['ticker']}")
print(f"Trading days: {data['trading_days']}")
print(f"Agent metrics: {json.dumps(data['agent_metrics'], indent=2)}")
print(f"Baseline metrics: {json.dumps(data['baseline_metrics'], indent=2)}")
print(f"First 3 daily results:")
for d in data["daily_results"][:3]:
    print(f"  {d['date']}: alloc={d['allocation']:.2f}, "
          f"agent_pv={d['agent_portfolio_value']:.4f}, "
          f"bh_pv={d['baseline_portfolio_value']:.4f}")
print(f"Model: {data['model_info']['algorithm']}")

# 3. GET inference
print("\n=== GET /api/inference/SPY ===")
r = urllib.request.urlopen(
    f"{BASE}/api/inference/SPY?start_date=2024-04-01&end_date=2024-04-30"
)
data2 = json.loads(r.read())
print(f"Ticker: {data2['ticker']}, Trading days: {data2['trading_days']}")
print(f"Agent Sharpe: {data2['agent_metrics']['sharpe_ratio']}")
print(f"Baseline Sharpe: {data2['baseline_metrics']['sharpe_ratio']}")

print("\n✅ All API tests passed!")
