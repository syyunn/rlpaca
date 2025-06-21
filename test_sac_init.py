#!/usr/bin/env python3
"""Test SAC initialization and first steps"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pandas as pd
from stable_baselines3 import SAC
from src.rl.realistic_offline_env import RealisticOfflineEnv
from src.config.trading_config import default_config

print("Setting up environment...")
# Quick data load
date_str = '20250620'
tick_file = f'/Users/suyeolyun/gits/rlpaca/data/1min_ticks/NVDA_ticks_{date_str}_1min.parquet'
bars_file = f'/Users/suyeolyun/gits/rlpaca/data/alpaca_bars/NVDA_2025-06-20_bars.parquet'

ticks_df = pd.read_parquet(tick_file)
bars_df = pd.read_parquet(bars_file)

ticks = []
for _, row in ticks_df.iterrows():
    ticks.append({
        'timestamp': row['timestamp'],
        'bid': row['bid'],
        'ask': row['ask'],
        'bid_size': row['bid_size'],
        'ask_size': row['ask_size']
    })

env = RealisticOfflineEnv(
    historical_ticks=ticks,
    historical_bars=bars_df,
    decision_interval_seconds=default_config.DECISION_INTERVAL_SECONDS,
    initial_capital=default_config.INITIAL_CAPITAL,
    max_position=None,
    transaction_cost=default_config.TRANSACTION_COST
)

print("\nCreating SAC model...")
start = time.time()
model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=100000,
    learning_starts=1000,  # This might be the issue!
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    policy_kwargs=dict(net_arch=[512, 512, 256]),
    verbose=1  # More verbose to see what's happening
)
print(f"SAC model created in {time.time() - start:.2f}s")

print("\nTesting first 100 steps...")
start = time.time()

# Manually collect some steps
from stable_baselines3.common.callbacks import BaseCallback

class DebugCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.n_calls = 0
        
    def _on_step(self):
        self.n_calls += 1
        if self.n_calls % 10 == 0:
            print(f"Callback: {self.n_calls} steps")
        return True

try:
    model.learn(total_timesteps=100, callback=DebugCallback())
    print(f"100 steps completed in {time.time() - start:.2f}s")
except Exception as e:
    print(f"Error during learning: {e}")
    import traceback
    traceback.print_exc()