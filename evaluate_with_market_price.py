#!/usr/bin/env python3
"""Enhanced evaluation using actual market prices"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from src.rl.realistic_offline_env import RealisticOfflineEnv
import argparse

def evaluate_model_with_prices(model_path):
    """Evaluate model with proper market prices"""
    
    # Load model
    model = SAC.load(model_path)
    print(f"Evaluating: {model_path}")
    
    # Load data
    bars_df = pd.read_parquet('/Users/suyeolyun/gits/rlpaca/data/alpaca_bars/NVDA_2025-06-20_bars.parquet')
    ticks_df = pd.read_parquet('/Users/suyeolyun/gits/rlpaca/data/1min_ticks/NVDA_ticks_20250620_1min.parquet')
    
    # Get market prices from bars
    market_prices = bars_df['close'].values
    market_times = bars_df.index
    
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
    
    # Run episode
    obs = env.reset()
    done = False
    
    # Tracking
    actions = []
    positions = []
    portfolio_values = []
    rewards = []
    step_count = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        actions.append(action[0])
        positions.append(info['position'])
        portfolio_values.append(info['portfolio_value'])
        rewards.append(reward)
        step_count += 1
    
    # Analysis
    actions = np.array(actions)
    positions = np.array(positions)
    portfolio_values = np.array(portfolio_values)
    
    initial_capital = env.initial_capital
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_capital) / initial_capital * 100
    
    # Market performance
    market_return = (market_prices[-1] - market_prices[0]) / market_prices[0] * 100
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'Model Evaluation: {os.path.basename(model_path)}', fontsize=16)
    
    # 1. Market Price Movement
    ax = axes[0, 0]
    hours = [(t.hour + t.minute/60) for t in market_times]
    ax.plot(hours, market_prices, 'b-', linewidth=2)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('NVDA Price ($)')
    ax.set_title(f'Market Price (Return: {market_return:+.2f}%)')
    ax.grid(True, alpha=0.3)
    
    # 2. Portfolio Value
    ax = axes[0, 1]
    # Interpolate portfolio values to match market times
    portfolio_interp = np.interp(range(len(market_times)), 
                                np.linspace(0, len(market_times)-1, len(portfolio_values)),
                                portfolio_values)
    ax.plot(hours, portfolio_interp, 'g-', linewidth=2)
    ax.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Portfolio Value ($)')
    ax.set_title(f'Portfolio Performance (Return: {total_return:+.2f}%)')
    ax.grid(True, alpha=0.3)
    
    # 3. Position over time
    ax = axes[1, 0]
    position_interp = np.interp(range(len(market_times)), 
                               np.linspace(0, len(market_times)-1, len(positions)),
                               positions)
    ax.plot(hours, position_interp, 'purple', linewidth=2)
    ax.fill_between(hours, 0, position_interp, alpha=0.3, color='purple')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Position (shares)')
    ax.set_title('Position Size Over Time')
    ax.grid(True, alpha=0.3)
    
    # 4. Actions histogram
    ax = axes[1, 1]
    ax.hist(actions, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Action Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Action Distribution')
    ax.grid(True, alpha=0.3)
    
    # 5. Cumulative rewards
    ax = axes[2, 0]
    cumulative_rewards = np.cumsum(rewards)
    reward_interp = np.interp(range(len(market_times)), 
                             np.linspace(0, len(market_times)-1, len(cumulative_rewards)),
                             cumulative_rewards)
    ax.plot(hours, reward_interp, 'orange', linewidth=2)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title(f'Total Reward: {np.sum(rewards):.2f}')
    ax.grid(True, alpha=0.3)
    
    # 6. Summary stats
    ax = axes[2, 1]
    ax.axis('off')
    
    summary_text = f"""Performance Summary:
    
Portfolio Return: {total_return:+.2f}%
Market Return: {market_return:+.2f}%
Alpha: {(total_return - market_return):+.2f}%

Total Trades: {len(env.trades_executed)}
Buy Actions: {(actions > 0.1).sum()} ({(actions > 0.1).sum()/len(actions)*100:.1f}%)
Sell Actions: {(actions < -0.1).sum()} ({(actions < -0.1).sum()/len(actions)*100:.1f}%)

Max Position: {np.max(positions):.2f} shares
Avg Position: {np.mean(positions):.2f} shares
Final Position: {positions[-1]:.2f} shares

Total Reward: {np.sum(rewards):.2f}
"""
    
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, 
            fontsize=12, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    output_path = model_path.replace('.zip', '_market_analysis.png')
    plt.savefig(output_path, dpi=150)
    print(f"\nAnalysis saved as: {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Portfolio Return: {total_return:+.2f}%")
    print(f"Market Return:    {market_return:+.2f}%")
    print(f"Alpha:           {(total_return - market_return):+.2f}%")
    print(f"Trades Executed:  {len(env.trades_executed)}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to model")
    args = parser.parse_args()
    
    evaluate_model_with_prices(args.model_path)