#!/usr/bin/env python3
"""Evaluate the 10k model with better error handling"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from src.rl.realistic_offline_env import RealisticOfflineEnv

# Load model
model = SAC.load('/Users/suyeolyun/gits/rlpaca/quick_model_10000.zip')
print("Loaded 10k quick model")

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

# Run full episode
obs = env.reset()
done = False
total_reward = 0
actions = []
positions = []
prices = []
times = []

print("\nRunning trading simulation...")
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    
    actions.append(action[0])
    positions.append(info['position'])
    prices.append(info.get('execution_price', info.get('current_price', 0)))
    times.append(info['time'])

# Analysis
actions = np.array(actions)
positions = np.array(positions)

print("\n" + "="*60)
print("TRADING SUMMARY")
print("="*60)

# Portfolio performance
initial_capital = env.initial_capital
final_value = info['portfolio_value']
returns = (final_value - initial_capital) / initial_capital * 100

print(f"\nPortfolio Performance:")
print(f"Initial capital: ${initial_capital:,.2f}")
print(f"Final value:     ${final_value:,.2f}")
print(f"Total return:    {returns:.2f}%")
print(f"Total reward:    {total_reward:.2f}")

# Trading activity
print(f"\nTrading Activity:")
print(f"Total steps:     {len(actions)}")
print(f"Trades executed: {len(env.trades_executed)}")

# Action analysis
print(f"\nAction Analysis:")
print(f"Mean action:     {np.mean(actions):.3f}")
print(f"Std action:      {np.std(actions):.3f}")
print(f"Min action:      {np.min(actions):.3f}")
print(f"Max action:      {np.max(actions):.3f}")

buy_attempts = (actions > 0.1).sum()
sell_attempts = (actions < -0.1).sum()
hold_actions = (np.abs(actions) <= 0.1).sum()

print(f"\nAction Distribution:")
print(f"Buy attempts:    {buy_attempts} ({buy_attempts/len(actions)*100:.1f}%)")
print(f"Sell attempts:   {sell_attempts} ({sell_attempts/len(actions)*100:.1f}%)")
print(f"Hold actions:    {hold_actions} ({hold_actions/len(actions)*100:.1f}%)")

# Position analysis
print(f"\nPosition Analysis:")
print(f"Max position:    {np.max(positions):.2f} shares")
print(f"Mean position:   {np.mean(positions):.2f} shares")
print(f"Final position:  {positions[-1]:.2f} shares")

# Check if model learned anything useful
if len(env.trades_executed) == 0:
    print("\n⚠️  WARNING: Model never executed any trades!")
    print("   The model is stuck trying to sell when it has no position.")
    print("   This suggests it needs more training to learn proper trading.")
else:
    print(f"\nExecuted Trades:")
    for i, trade in enumerate(env.trades_executed[:5]):  # Show first 5
        print(f"  {i+1}. {trade['side'].upper()} {trade['shares']:.2f} shares @ ${trade['price']:.2f}")
    if len(env.trades_executed) > 5:
        print(f"  ... and {len(env.trades_executed)-5} more trades")

print("\n" + "="*60)