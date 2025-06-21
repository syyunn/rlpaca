#!/usr/bin/env python3
"""Evaluate a trained SAC model on existing data"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from stable_baselines3 import SAC
import argparse
from src.rl.realistic_offline_env import RealisticOfflineEnv

def evaluate_model(model_path: str):
    """Evaluate model performance on existing data"""
    
    # Load model
    model = SAC.load(model_path)
    print(f"Loaded model from {model_path}")
    
    # Load existing data
    bars_df = pd.read_parquet('/Users/suyeolyun/gits/rlpaca/data/alpaca_bars/NVDA_2025-06-20_bars.parquet')
    ticks_df = pd.read_parquet('/Users/suyeolyun/gits/rlpaca/data/1min_ticks/NVDA_ticks_20250620_1min.parquet')
    
    print(f"Loaded data for 2025-06-20")
    print(f"Bars: {len(bars_df)} minute bars")
    print(f"Ticks: {len(ticks_df)} aggregated ticks")
    
    # Convert ticks to the format expected by the environment
    ticks = []
    for _, tick in ticks_df.iterrows():
        tick_dict = {
            'timestamp': tick.name,
            'bid': tick['bid'],
            'ask': tick['ask'],
            'bid_size': tick.get('bid_size', 100),
            'ask_size': tick.get('ask_size', 100),
            'price': (tick['bid'] + tick['ask']) / 2
        }
        ticks.append(tick_dict)
    
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
    
    step_count = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        actions_taken.append(action[0])
        step_count += 1
        
        # Print every 30 steps (every 2.5 minutes)
        if step_count % 30 == 0 or done:
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
    
    # Trade analysis
    if env.trades_executed:
        print(f"\nTrade analysis:")
        buy_trades = [t for t in env.trades_executed if t['side'] == 'buy']
        sell_trades = [t for t in env.trades_executed if t['side'] == 'sell']
        print(f"Buy trades:  {len(buy_trades)}")
        print(f"Sell trades: {len(sell_trades)}")
        
        if buy_trades:
            avg_buy_price = np.mean([t['price'] for t in buy_trades])
            print(f"Avg buy price: ${avg_buy_price:.2f}")
        if sell_trades:
            avg_sell_price = np.mean([t['price'] for t in sell_trades])
            print(f"Avg sell price: ${avg_sell_price:.2f}")
    
    return returns, total_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to the saved model")
    args = parser.parse_args()
    
    evaluate_model(args.model_path)