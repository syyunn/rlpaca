#!/usr/bin/env python3
"""Evaluate a trained SAC model on test data"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import SAC
import argparse
import pandas as pd
from src.rl.realistic_offline_env import RealisticOfflineEnv
from scripts.download_data import download_nvda_with_bars
from datetime import datetime, timedelta

def evaluate_model(model_path: str, test_date: str = None):
    """Evaluate model performance on a test day"""
    
    # Load model
    model = SAC.load(model_path)
    print(f"Loaded model from {model_path}")
    
    # Get test data (use recent day if not specified)
    if test_date is None:
        test_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
    
    print(f"Downloading test data for {test_date}...")
    bars_df = download_nvda_with_bars(test_date, test_date)
    
    if bars_df.empty:
        print(f"No data available for {test_date}")
        return
    
    # Create dummy ticks (for now)
    ticks = []
    for _, bar in bars_df.iterrows():
        # Create synthetic ticks from bars
        for i in range(10):  # 10 ticks per minute bar
            tick = {
                'timestamp': bar.name + timedelta(seconds=i*6),
                'bid': bar['close'] - 0.01,
                'ask': bar['close'] + 0.01,
                'bid_size': 100,
                'ask_size': 100,
                'price': bar['close']
            }
            ticks.append(tick)
    
    # Create environment
    env = RealisticOfflineEnv(
        historical_ticks=ticks,
        historical_bars=bars_df,
        max_position=None,
        transaction_cost=0.0
    )
    
    # Run episode
    obs = env.reset()
    done = False
    total_reward = 0
    actions_taken = []
    
    print("\nRunning trading simulation...")
    print("Time         | Price  | Position | Action | Cash     | Value")
    print("-" * 70)
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        actions_taken.append(action[0])
        
        # Print every 30 steps (every 2.5 minutes)
        if len(actions_taken) % 30 == 0:
            print(f"{info['time'].strftime('%H:%M:%S')} | "
                  f"${info.get('execution_price', 0):.2f} | "
                  f"{info['position']:7.2f} | "
                  f"{action[0]:6.3f} | "
                  f"${info['capital']:8.2f} | "
                  f"${info['portfolio_value']:8.2f}")
    
    # Final results
    final_value = info['portfolio_value']
    returns = (final_value - env.initial_capital) / env.initial_capital * 100
    
    print("\n" + "="*70)
    print(f"FINAL RESULTS:")
    print(f"Initial capital: ${env.initial_capital:,.2f}")
    print(f"Final value:     ${final_value:,.2f}")
    print(f"Total return:    {returns:.2f}%")
    print(f"Total reward:    {total_reward:.2f}")
    print(f"Trades executed: {len(env.trades_executed)}")
    
    # Action statistics
    actions_array = np.array(actions_taken)
    print(f"\nAction statistics:")
    print(f"Mean action:  {np.mean(actions_array):.3f}")
    print(f"Std action:   {np.std(actions_array):.3f}")
    print(f"% Buy (>0.1): {(actions_array > 0.1).sum() / len(actions_array) * 100:.1f}%")
    print(f"% Sell (<-0.1): {(actions_array < -0.1).sum() / len(actions_array) * 100:.1f}%")
    print(f"% Hold:       {(np.abs(actions_array) <= 0.1).sum() / len(actions_array) * 100:.1f}%")
    
    return returns, total_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to the saved model")
    parser.add_argument("--date", help="Test date (YYYY-MM-DD)")
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.date)