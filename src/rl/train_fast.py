#!/usr/bin/env python3
"""Fast training script with minimal logging"""

import sys
import os
import argparse
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from stable_baselines3 import SAC
from realistic_offline_env import RealisticOfflineEnv
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config.trading_config import default_config

def load_training_data(date='2025-06-20'):
    """Load tick and bar data for training"""
    # Get absolute path to data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    date_str = date.replace('-', '')
    tick_file = os.path.join(project_root, 'data/1min_ticks', f'NVDA_ticks_{date_str}_1min.parquet')
    df = pd.read_parquet(tick_file)
    
    ticks = []
    for _, row in df.iterrows():
        ticks.append({
            'timestamp': row['timestamp'],
            'bid': row['bid'],
            'ask': row['ask'],
            'bid_size': row['bid_size'],
            'ask_size': row['ask_size']
        })
    
    # Load minute bars
    bars_file = os.path.join(project_root, 'data/alpaca_bars', f'NVDA_{date}_bars.parquet')
    minute_bars = pd.read_parquet(bars_file) if os.path.exists(bars_file) else pd.DataFrame()
    
    return ticks, minute_bars

def train_model(timesteps=1000):
    """Train model with minimal logging"""
    print(f"\n🚀 FAST TRAINING: {timesteps:,} timesteps")
    print("="*50)
    
    # Load data
    print("Loading data...")
    ticks, minute_bars = load_training_data()
    
    # Create environment
    env = RealisticOfflineEnv(
        historical_ticks=ticks,
        historical_bars=minute_bars,
        decision_interval_seconds=default_config.DECISION_INTERVAL_SECONDS,
        initial_capital=default_config.INITIAL_CAPITAL,
        max_position=None,
        transaction_cost=default_config.TRANSACTION_COST
    )
    
    # Create model
    print("Creating SAC model...")
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=dict(net_arch=[512, 512, 256]),
        verbose=0  # No verbose output
    )
    
    # Train
    print(f"Training for {timesteps:,} steps...")
    start_time = time.time()
    
    # Progress callback
    steps_done = 0
    def progress_callback(locals, globals):
        nonlocal steps_done
        steps_done += 1
        if steps_done % 1000 == 0:
            elapsed = time.time() - start_time
            speed = steps_done / elapsed
            eta = (timesteps - steps_done) / speed
            print(f"Progress: {steps_done:,}/{timesteps:,} ({steps_done/timesteps*100:.1f}%) - "
                  f"Speed: {speed:.1f} steps/s - ETA: {eta:.1f}s")
        return True
    
    model.learn(total_timesteps=timesteps, callback=progress_callback)
    
    # Save
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_name = f"fast_model_{timesteps}.zip"
    model_path = os.path.join(project_root, model_name)
    model.save(model_path)
    
    elapsed = time.time() - start_time
    print(f"\n✅ Training complete in {elapsed:.1f} seconds")
    print(f"Speed: {timesteps/elapsed:.1f} timesteps/second")
    print(f"Model saved as: {model_path}")
    
    # Quick test
    print("\n🧪 Quick behavior test...")
    obs = env.reset()
    
    # Test a few steps
    for i in range(5):
        action, _ = model.predict(obs, deterministic=True)
        print(f"Step {i}: action=[{action[0]:.3f}, {action[1]:.1f}]")
        obs, reward, done, info = env.step(action)
        if done:
            break
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=1000)
    args = parser.parse_args()
    
    train_model(args.timesteps)