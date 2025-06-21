#!/usr/bin/env python3
"""Quick training with optimized settings for faster results"""

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
    
    bars_file = os.path.join(project_root, 'data/alpaca_bars', f'NVDA_{date}_bars.parquet')
    minute_bars = pd.read_parquet(bars_file) if os.path.exists(bars_file) else pd.DataFrame()
    
    return ticks, minute_bars

def train_model(timesteps=10000):
    """Train model with optimized settings"""
    print(f"\n🚀 QUICK TRAINING: {timesteps:,} timesteps")
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
    
    # Create model with OPTIMIZED settings for faster training
    print("Creating optimized SAC model...")
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=10000,  # Smaller buffer
        learning_starts=100,  # Start learning much earlier!
        batch_size=64,  # Smaller batch for faster updates
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=dict(net_arch=[256, 256]),  # Smaller network!
        verbose=1
    )
    
    # Train with progress
    print(f"Training for {timesteps:,} steps...")
    start_time = time.time()
    
    from stable_baselines3.common.callbacks import BaseCallback
    
    class FastProgressCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.start_time = time.time()
            
        def _on_step(self):
            if self.n_calls % 500 == 0:
                elapsed = time.time() - self.start_time
                speed = self.n_calls / elapsed
                eta = (timesteps - self.n_calls) / speed
                print(f"Progress: {self.n_calls:,}/{timesteps:,} ({self.n_calls/timesteps*100:.1f}%) - "
                      f"Speed: {speed:.1f} steps/s - ETA: {eta/60:.1f} min")
            return True
    
    model.learn(total_timesteps=timesteps, callback=FastProgressCallback())
    
    # Save
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_name = f"quick_model_{timesteps}.zip"
    model_path = os.path.join(project_root, model_name)
    model.save(model_path)
    
    elapsed = time.time() - start_time
    print(f"\n✅ Training complete in {elapsed/60:.1f} minutes")
    print(f"Speed: {timesteps/elapsed:.1f} steps/second")
    print(f"Model saved as: {model_path}")
    
    # Quick test
    print("\n🧪 Testing model behavior...")
    obs = env.reset()
    
    # Test different portfolio states
    test_states = [
        (0.0, "No position"),
        (0.5, "Half invested"),
        (1.0, "Fully invested")
    ]
    
    for frac, desc in test_states:
        # Set position
        env.position = frac * env.capital / 145.0  # Approx NVDA price
        env.capital = (1 - frac) * env.initial_capital
        
        obs = env._get_state()
        action, _ = model.predict(obs, deterministic=True)
        print(f"{desc}: action=[{action[0]:6.3f}, {action[1]:6.1f}]")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=10000)
    args = parser.parse_args()
    
    train_model(args.timesteps)