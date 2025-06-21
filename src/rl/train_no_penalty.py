#!/usr/bin/env python3
"""
Train SAC model WITHOUT artificial constraint penalties
Let the model learn naturally that invalid actions yield no reward
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from src.rl.realistic_offline_env import RealisticOfflineEnv
import time

class DetailedProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.start_time = None
        self.last_print_time = None
        
    def _on_training_start(self):
        self.start_time = time.time()
        self.last_print_time = self.start_time
        
    def _on_step(self):
        current_time = time.time()
        if current_time - self.last_print_time >= 10:  # Print every 10 seconds
            elapsed = current_time - self.start_time
            progress = self.num_timesteps / self.total_timesteps
            speed = self.num_timesteps / elapsed
            eta = (self.total_timesteps - self.num_timesteps) / speed / 60
            
            print(f"Progress: {self.num_timesteps:,}/{self.total_timesteps:,} ({progress*100:.1f}%) - "
                  f"Speed: {speed:.1f} steps/s - ETA: {eta:.1f} min")
            self.last_print_time = current_time
        return True

def train_no_penalty(timesteps=50000):
    print("\n🚀 TRAINING WITHOUT CONSTRAINT PENALTIES")
    print("=" * 50)
    print("Let the model learn naturally that invalid actions yield no reward")
    print("=" * 50)
    
    # Load data
    print("\nLoading data...")
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
        initial_capital=100000,
        max_position=None,
        transaction_cost=0.0
    )
    
    # Create model with same hyperparameters as quick training
    print("\nCreating SAC model...")
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=50000,
        learning_starts=100,  # Start learning early
        batch_size=64,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=dict(
            net_arch=[256, 256]  # Smaller network for faster training
        ),
        verbose=0,
        tensorboard_log=f"logs/no_penalty/"
    )
    
    # Train
    print(f"\nTraining for {timesteps:,} steps...")
    model.learn(
        total_timesteps=timesteps,
        callback=DetailedProgressCallback(timesteps),
        tb_log_name=f"sac_{timesteps}"
    )
    
    # Save model
    model_path = f'/Users/suyeolyun/gits/rlpaca/no_penalty_model_{timesteps}.zip'
    model.save(model_path)
    print(f"\n✅ Model saved as: {model_path}")
    
    # Test the model
    print("\n🧪 Testing model behavior...")
    obs = env.reset()
    
    # Test different scenarios
    scenarios = [
        ("Start of day", obs),
    ]
    
    # Simulate some steps to get different states
    for i in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = env.step(action)
        if i == 49:
            scenarios.append((f"After 50 steps (pos={info['position']:.1f})", obs))
        if done:
            break
    
    print("\nModel behavior in different states:")
    for scenario, test_obs in scenarios:
        action, _ = model.predict(test_obs, deterministic=True)
        print(f"{scenario}: action=[{action[0]:6.3f}, {action[1]:5.1f}]")
    
    return model_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=50000,
                        help='Number of timesteps to train')
    args = parser.parse_args()
    
    train_no_penalty(args.timesteps)