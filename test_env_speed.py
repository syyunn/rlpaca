#!/usr/bin/env python3
"""Test environment speed and identify bottlenecks"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pandas as pd
from src.rl.realistic_offline_env import RealisticOfflineEnv
from src.config.trading_config import default_config

print("Loading data...")
start = time.time()

# Load data
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

print(f"Data loaded in {time.time() - start:.2f}s")
print(f"Ticks: {len(ticks)}, Bars: {len(bars_df)}")

# Create environment
print("\nCreating environment...")
start = time.time()
env = RealisticOfflineEnv(
    historical_ticks=ticks,
    historical_bars=bars_df,
    decision_interval_seconds=default_config.DECISION_INTERVAL_SECONDS,
    initial_capital=default_config.INITIAL_CAPITAL,
    max_position=None,
    transaction_cost=default_config.TRANSACTION_COST
)
print(f"Environment created in {time.time() - start:.2f}s")

# Test reset
print("\nTesting reset...")
start = time.time()
obs = env.reset()
print(f"Reset in {time.time() - start:.2f}s")
print(f"Observation shape: {obs.shape}")

# Test steps
print("\nTesting step speed...")
step_times = []
for i in range(100):
    start = time.time()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    step_time = time.time() - start
    step_times.append(step_time)
    
    if i % 20 == 0:
        print(f"Step {i}: {step_time*1000:.1f}ms")
    
    if done:
        print(f"Episode ended at step {i}")
        break

import numpy as np
print(f"\nStep timing statistics:")
print(f"Mean: {np.mean(step_times)*1000:.1f}ms")
print(f"Max: {np.max(step_times)*1000:.1f}ms")
print(f"Min: {np.min(step_times)*1000:.1f}ms")

# Test full episode
print("\nTesting full episode speed...")
start = time.time()
obs = env.reset()
done = False
steps = 0
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    steps += 1

elapsed = time.time() - start
print(f"Full episode: {steps} steps in {elapsed:.1f}s ({steps/elapsed:.1f} steps/s)")

# Memory check
import psutil
process = psutil.Process()
print(f"\nMemory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")