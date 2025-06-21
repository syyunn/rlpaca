#!/usr/bin/env python3
"""Compare behavior of all trained models"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from src.rl.realistic_offline_env import RealisticOfflineEnv

# Models to compare
models = {
    '10k steps': '/Users/suyeolyun/gits/rlpaca/quick_model_10000.zip',
    '50k steps': '/Users/suyeolyun/gits/rlpaca/quick_model_50000.zip',
    '100k steps': '/Users/suyeolyun/gits/rlpaca/quick_model_100000.zip'
}

# Load data once
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

# Test each model
results = {}

print("Model Comparison Analysis")
print("=" * 80)

for name, path in models.items():
    print(f"\n{name}:")
    model = SAC.load(path)
    
    # Create fresh environment
    env = RealisticOfflineEnv(
        historical_ticks=ticks,
        historical_bars=bars_df,
        max_position=None,
        transaction_cost=0.0
    )
    
    # Run episode
    obs = env.reset()
    done = False
    actions = []
    positions = []
    portfolio_values = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        actions.append(action[0])
        positions.append(info['position'])
        portfolio_values.append(info['portfolio_value'])
    
    # Calculate metrics
    actions = np.array(actions)
    final_return = (portfolio_values[-1] - env.initial_capital) / env.initial_capital * 100
    
    results[name] = {
        'final_return': final_return,
        'trades': len(env.trades_executed),
        'mean_action': np.mean(actions),
        'std_action': np.std(actions),
        'unique_actions': len(np.unique(actions)),
        'max_position': np.max(positions),
        'actions': actions,
        'positions': positions,
        'portfolio_values': portfolio_values
    }
    
    # Print summary
    print(f"  Return: {final_return:+.2f}%")
    print(f"  Trades: {len(env.trades_executed)}")
    print(f"  Mean action: {np.mean(actions):.3f}")
    print(f"  Action range: [{np.min(actions):.3f}, {np.max(actions):.3f}]")
    print(f"  Unique actions: {len(np.unique(actions))}")
    print(f"  Max position: {np.max(positions):.2f} shares")

# Create visualization
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Model Evolution: 10k vs 50k vs 100k Steps', fontsize=16)

model_names = list(models.keys())
colors = ['red', 'green', 'blue']

# 1. Action distributions
for i, (name, color) in enumerate(zip(model_names, colors)):
    ax = axes[0, i]
    actions = results[name]['actions']
    ax.hist(actions, bins=50, alpha=0.7, color=color, edgecolor='black')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f'{name} Actions')
    ax.set_xlabel('Action Value')
    ax.set_ylabel('Frequency')
    ax.set_xlim(-1.1, 1.1)

# 2. Position over time
for i, (name, color) in enumerate(zip(model_names, colors)):
    ax = axes[1, i]
    positions = results[name]['positions']
    steps = range(len(positions))
    ax.plot(steps[::50], positions[::50], color=color, linewidth=2)  # Sample every 50 steps
    ax.set_title(f'{name} Position')
    ax.set_xlabel('Step')
    ax.set_ylabel('Position (shares)')
    ax.set_ylim(-50, 750)

# 3. Portfolio value over time
for i, (name, color) in enumerate(zip(model_names, colors)):
    ax = axes[2, i]
    values = results[name]['portfolio_values']
    steps = range(len(values))
    ax.plot(steps[::50], values[::50], color=color, linewidth=2)  # Sample every 50 steps
    ax.axhline(y=100000, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f'{name} Portfolio')
    ax.set_xlabel('Step')
    ax.set_ylabel('Portfolio Value ($)')
    ax.set_ylim(98500, 101000)

plt.tight_layout()
plt.savefig('/Users/suyeolyun/gits/rlpaca/model_evolution_comparison.png', dpi=150)
print(f"\n\nVisualization saved as model_evolution_comparison.png")

# Training progression analysis
print("\n" + "="*80)
print("TRAINING PROGRESSION ANALYSIS")
print("="*80)
print("\n10k → 50k → 100k progression:")
print(f"  Action evolution: {results['10k steps']['mean_action']:.3f} → "
      f"{results['50k steps']['mean_action']:.3f} → "
      f"{results['100k steps']['mean_action']:.3f}")
print(f"  Return evolution: {results['10k steps']['final_return']:+.2f}% → "
      f"{results['50k steps']['final_return']:+.2f}% → "
      f"{results['100k steps']['final_return']:+.2f}%")

# Market comparison
market_return = (bars_df['close'].iloc[-1] - bars_df['close'].iloc[0]) / bars_df['close'].iloc[0] * 100
print(f"\nMarket return: {market_return:+.2f}%")
print("Alpha (vs market):")
for name in model_names:
    alpha = results[name]['final_return'] - market_return
    print(f"  {name}: {alpha:+.2f}%")