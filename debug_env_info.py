#!/usr/bin/env python3
"""Debug what info the environment provides"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.rl.realistic_offline_env import RealisticOfflineEnv

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

# Reset and take one step
obs = env.reset()
action = [-1.0, 0.0]  # Try to sell
obs, reward, done, info = env.step(action)

print("Info keys provided by environment:")
for key, value in info.items():
    print(f"  {key}: {value} (type: {type(value).__name__})")

# Check what tick data looks like
print(f"\nFirst tick data:")
print(f"  Timestamp: {ticks[0]['timestamp']}")
print(f"  Bid: {ticks[0]['bid']}")
print(f"  Ask: {ticks[0]['ask']}")
print(f"  Price: {ticks[0]['price']}")

# Check minute bars
print(f"\nFirst minute bar:")
print(bars_df.iloc[0])