#!/usr/bin/env python3
"""Visualize what the RL model learned"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from src.rl.realistic_offline_env import RealisticOfflineEnv

# Load model
model_path = '/Users/suyeolyun/gits/rlpaca/quick_model_10000.zip'
model = SAC.load(model_path)
print(f"Loaded model: {model_path}")

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

# Test model behavior across different states
print("\nTesting model behavior across different market conditions...")

# 1. Test at different position levels
position_levels = np.linspace(0, 1, 11)  # 0%, 10%, 20%, ..., 100% invested
actions_by_position = []

for pos_frac in position_levels:
    env.reset()
    # Set position
    env.position = pos_frac * env.capital / 145.0  # Approx NVDA price
    env.capital = (1 - pos_frac) * env.initial_capital
    
    obs = env._get_state()
    action, _ = model.predict(obs, deterministic=True)
    actions_by_position.append(action[0])

# 2. Run full episode and track behavior
env.reset()
done = False
actions = []
positions = []
prices = []
times = []
portfolio_values = []

while not done:
    obs = env._get_state()
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    
    actions.append(action[0])
    positions.append(info['position'])
    prices.append(info.get('current_price', 0))
    times.append(info['time'])
    portfolio_values.append(info['portfolio_value'])

# Create visualizations
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle('RL Model Behavior Analysis (10k steps trained)', fontsize=16)

# 1. Actions by position level
ax = axes[0, 0]
ax.plot(position_levels * 100, actions_by_position, 'bo-', linewidth=2, markersize=8)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Position Level (% of capital invested)')
ax.set_ylabel('Action (position delta)')
ax.set_title('Model Actions vs Position Level')
ax.grid(True, alpha=0.3)
ax.set_ylim(-1.1, 1.1)

# 2. Action distribution histogram
ax = axes[0, 1]
ax.hist(actions, bins=50, edgecolor='black', alpha=0.7)
ax.set_xlabel('Action Value')
ax.set_ylabel('Frequency')
ax.set_title(f'Action Distribution (mean={np.mean(actions):.3f})')
ax.grid(True, alpha=0.3)

# 3. Price and actions over time
ax = axes[1, 0]
ax2 = ax.twinx()
time_hours = [(t.hour + t.minute/60) for t in times]
ax.plot(time_hours, prices, 'b-', label='Price', alpha=0.7)
ax2.scatter(time_hours, actions, c='red', alpha=0.5, s=10, label='Actions')
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Price ($)', color='blue')
ax2.set_ylabel('Action', color='red')
ax.set_title('Price and Model Actions Over Time')
ax.tick_params(axis='y', labelcolor='blue')
ax2.tick_params(axis='y', labelcolor='red')
ax.grid(True, alpha=0.3)

# 4. Position over time
ax = axes[1, 1]
ax.plot(time_hours, positions, 'g-', linewidth=2)
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Position (shares)')
ax.set_title('Position Over Time')
ax.grid(True, alpha=0.3)

# 5. Portfolio value over time
ax = axes[2, 0]
ax.plot(time_hours, portfolio_values, 'purple', linewidth=2)
ax.axhline(y=env.initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial capital')
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Portfolio Value ($)')
ax.set_title('Portfolio Value Over Time')
ax.grid(True, alpha=0.3)
ax.legend()

# 6. Model behavior summary
ax = axes[2, 1]
ax.axis('off')
summary_text = f"""Model Behavior Summary:
    
Total Actions: {len(actions)}
Unique Actions: {len(np.unique(actions))}
Mean Action: {np.mean(actions):.3f}
Std Action: {np.std(actions):.3f}

Buy Attempts: {(np.array(actions) > 0.1).sum()} ({(np.array(actions) > 0.1).sum()/len(actions)*100:.1f}%)
Sell Attempts: {(np.array(actions) < -0.1).sum()} ({(np.array(actions) < -0.1).sum()/len(actions)*100:.1f}%)
Hold Actions: {(np.abs(np.array(actions)) <= 0.1).sum()} ({(np.abs(np.array(actions)) <= 0.1).sum()/len(actions)*100:.1f}%)

Final Return: {(portfolio_values[-1] - env.initial_capital) / env.initial_capital * 100:.2f}%
Trades Executed: {len(env.trades_executed)}

Diagnosis: {"Model learned to avoid losses by not trading" if len(env.trades_executed) == 0 else "Model is actively trading"}
"""
ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, 
        fontsize=11, verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('/Users/suyeolyun/gits/rlpaca/model_behavior_10k.png', dpi=150)
print(f"\nVisualization saved as model_behavior_10k.png")
plt.show()