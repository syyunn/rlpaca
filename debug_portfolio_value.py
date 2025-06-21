#!/usr/bin/env python3
"""Debug why portfolio value goes flat"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from src.rl.realistic_offline_env import RealisticOfflineEnv
import matplotlib.pyplot as plt

# Load the 50k model
model = SAC.load('/Users/suyeolyun/gits/rlpaca/quick_model_50000.zip')
print("Loaded 50k model")

# Load data
bars_df = pd.read_parquet('/Users/suyeolyun/gits/rlpaca/data/alpaca_bars/NVDA_2025-06-20_bars.parquet')
ticks_df = pd.read_parquet('/Users/suyeolyun/gits/rlpaca/data/1min_ticks/NVDA_ticks_20250620_1min.parquet')

# Convert ticks
ticks = []
for _, tick in ticks_df.iterrows():
    ticks.append({
        'timestamp': tick.name,
        'bid': tick['bid'],
        'ask': tick['ask'],
        'bid_size': tick.get('bid_size', 100),
        'ask_size': tick.get('ask_size', 100),
        'price': (tick['bid'] + tick['ask']) / 2
    })

# Create environment
env = RealisticOfflineEnv(
    historical_ticks=ticks,
    historical_bars=bars_df,
    max_position=None,
    transaction_cost=0.0
)

# Run episode with detailed tracking
obs = env.reset()
done = False
step = 0

# Detailed tracking
debug_data = []

print("\nRunning episode with detailed tracking...")
print("Step | Time     | Action | Position | Cash    | Price   | Portfolio | Trades")
print("-" * 85)

while not done and step < 100:  # Limit to first 100 steps for debugging
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    
    # Get current market price from bars
    current_time = info['time']
    # Find the closest bar
    bar_times = bars_df.index
    closest_bar_idx = np.argmin(np.abs(bar_times - current_time))
    market_price = bars_df.iloc[closest_bar_idx]['close']
    
    # Calculate what portfolio value SHOULD be
    expected_portfolio = info['capital'] + info['position'] * market_price
    
    debug_data.append({
        'step': step,
        'time': info['time'],
        'action': action[0],
        'position': info['position'],
        'cash': info['capital'],
        'market_price': market_price,
        'portfolio_value': info['portfolio_value'],
        'expected_portfolio': expected_portfolio,
        'execution_price': info.get('execution_price', np.nan),
        'trades': len(env.trades_executed)
    })
    
    if step % 10 == 0:
        print(f"{step:4d} | {info['time'].strftime('%H:%M')} | "
              f"{action[0]:6.3f} | {info['position']:8.2f} | "
              f"${info['capital']:7.0f} | ${market_price:6.2f} | "
              f"${info['portfolio_value']:9.0f} | {len(env.trades_executed):6d}")
    
    step += 1

# Analyze the data
df = pd.DataFrame(debug_data)

print(f"\nAnalyzing {len(df)} steps...")

# Find where portfolio value stops changing
portfolio_diffs = df['portfolio_value'].diff().abs()
flat_start = None
for i in range(10, len(portfolio_diffs)):
    if all(portfolio_diffs[i:i+10] < 0.01):  # 10 consecutive steps with no change
        flat_start = i
        break

if flat_start:
    print(f"\nPortfolio value goes flat at step {flat_start}")
    print(f"Time: {df.iloc[flat_start]['time']}")
    print(f"Position: {df.iloc[flat_start]['position']:.2f} shares")
    print(f"Cash: ${df.iloc[flat_start]['cash']:.2f}")
    print(f"Market price: ${df.iloc[flat_start]['market_price']:.2f}")
    
    # Check discrepancy
    print("\nChecking portfolio calculation:")
    for i in range(max(0, flat_start-5), min(len(df), flat_start+5)):
        row = df.iloc[i]
        calc_portfolio = row['cash'] + row['position'] * row['market_price']
        discrepancy = row['portfolio_value'] - calc_portfolio
        print(f"Step {i}: Reported=${row['portfolio_value']:.2f}, "
              f"Calculated=${calc_portfolio:.2f}, "
              f"Discrepancy=${discrepancy:.2f}")

# Plot to visualize
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

steps = df['step']

# Market price
ax = axes[0]
ax.plot(steps, df['market_price'], 'b-', label='Market Price')
ax.set_ylabel('Market Price ($)')
ax.legend()
ax.grid(True, alpha=0.3)

# Portfolio values
ax = axes[1]
ax.plot(steps, df['portfolio_value'], 'g-', label='Reported Portfolio', linewidth=2)
ax.plot(steps, df['expected_portfolio'], 'r--', label='Expected Portfolio', alpha=0.7)
ax.set_ylabel('Portfolio Value ($)')
ax.legend()
ax.grid(True, alpha=0.3)

# Position and cash
ax = axes[2]
ax2 = ax.twinx()
ax.plot(steps, df['position'], 'purple', label='Position (shares)')
ax2.plot(steps, df['cash'], 'orange', label='Cash ($)')
ax.set_xlabel('Step')
ax.set_ylabel('Position (shares)', color='purple')
ax2.set_ylabel('Cash ($)', color='orange')
ax.grid(True, alpha=0.3)

plt.suptitle('Portfolio Value Debug Analysis')
plt.tight_layout()
plt.savefig('/Users/suyeolyun/gits/rlpaca/portfolio_debug.png')
print(f"\nDebug plot saved as portfolio_debug.png")