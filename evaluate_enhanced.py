#!/usr/bin/env python3
"""Enhanced evaluation with detailed performance metrics"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from src.rl.realistic_offline_env import RealisticOfflineEnv
import argparse

def evaluate_model_enhanced(model_path):
    """Evaluate model with comprehensive metrics"""
    
    # Load model
    model = SAC.load(model_path)
    print(f"Loaded model: {model_path}")
    
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
    
    # Run episode and collect detailed data
    obs = env.reset()
    done = False
    
    # Tracking arrays
    actions = []
    positions = []
    prices = []
    times = []
    portfolio_values = []
    rewards = []
    cash_amounts = []
    
    print("\nRunning trading simulation...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        actions.append(action[0])
        positions.append(info['position'])
        prices.append(info.get('execution_price', 0))  # Use execution_price
        times.append(info['time'])
        portfolio_values.append(info['portfolio_value'])
        rewards.append(reward)
        cash_amounts.append(info['capital'])
    
    # Convert to numpy arrays
    actions = np.array(actions)
    positions = np.array(positions)
    prices = np.array(prices)
    portfolio_values = np.array(portfolio_values)
    rewards = np.array(rewards)
    
    # Calculate metrics
    initial_capital = env.initial_capital
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_capital) / initial_capital * 100
    
    # Maximum drawdown
    running_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - running_max) / running_max * 100
    max_drawdown = np.min(drawdowns)
    
    # Daily returns (approximate)
    returns_pct = np.diff(portfolio_values) / portfolio_values[:-1] * 100
    
    # Sharpe ratio (annualized, assuming minute data)
    if len(returns_pct) > 0 and np.std(returns_pct) > 0:
        sharpe = np.mean(returns_pct) / np.std(returns_pct) * np.sqrt(252 * 390)  # Annualized
    else:
        sharpe = 0.0
    
    # Price performance
    price_return = (prices[-1] - prices[0]) / prices[0] * 100
    
    # Win rate of trades
    profitable_trades = 0
    total_trades = len(env.trades_executed)
    
    if total_trades > 0:
        buy_prices = {}
        for trade in env.trades_executed:
            if trade['side'] == 'buy':
                buy_prices[trade['timestamp']] = trade['price']
            elif trade['side'] == 'sell' and buy_prices:
                # Find matching buy
                buy_price = list(buy_prices.values())[0]  # Simple FIFO
                if trade['price'] > buy_price:
                    profitable_trades += 1
    
    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Print comprehensive results
    print("\n" + "="*80)
    print("COMPREHENSIVE TRADING EVALUATION")
    print("="*80)
    
    print(f"\n📊 PORTFOLIO PERFORMANCE")
    print(f"Initial Capital:     ${initial_capital:,.2f}")
    print(f"Final Value:         ${final_value:,.2f}")
    print(f"Total Return:        {total_return:+.2f}%")
    print(f"Max Drawdown:        {max_drawdown:.2f}%")
    print(f"Sharpe Ratio:        {sharpe:.2f}")
    
    print(f"\n📈 ASSET PERFORMANCE")
    print(f"NVDA Start Price:    ${prices[0]:.2f}")
    print(f"NVDA End Price:      ${prices[-1]:.2f}")
    print(f"NVDA Return:         {price_return:+.2f}%")
    print(f"Alpha vs Buy&Hold:   {(total_return - price_return):+.2f}%")
    
    print(f"\n🎯 TRADING STATISTICS")
    print(f"Total Trades:        {total_trades}")
    print(f"Win Rate:            {win_rate:.1f}%")
    print(f"Avg Position:        {np.mean(positions):.2f} shares")
    print(f"Max Position:        {np.max(positions):.2f} shares")
    
    print(f"\n🤖 MODEL BEHAVIOR")
    print(f"Total Reward:        {np.sum(rewards):.2f}")
    print(f"Actions >0.1:        {(actions > 0.1).sum()} ({(actions > 0.1).sum()/len(actions)*100:.1f}%)")
    print(f"Actions <-0.1:       {(actions < -0.1).sum()} ({(actions < -0.1).sum()/len(actions)*100:.1f}%)")
    print(f"Hold Actions:        {(np.abs(actions) <= 0.1).sum()} ({(np.abs(actions) <= 0.1).sum()/len(actions)*100:.1f}%)")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Model Performance Analysis: {os.path.basename(model_path)}', fontsize=16)
    
    # 1. Portfolio value and NVDA price
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    
    time_hours = [(t.hour + t.minute/60) for t in times]
    ax1.plot(time_hours, portfolio_values, 'g-', linewidth=2, label='Portfolio Value')
    ax1.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5)
    ax1_twin.plot(time_hours, prices, 'b-', alpha=0.5, label='NVDA Price')
    
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Portfolio Value ($)', color='g')
    ax1_twin.set_ylabel('NVDA Price ($)', color='b')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Portfolio Value vs NVDA Price')
    
    # 2. Position over time
    ax2 = axes[0, 1]
    ax2.plot(time_hours, positions, 'purple', linewidth=2)
    ax2.fill_between(time_hours, 0, positions, alpha=0.3, color='purple')
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Position (shares)')
    ax2.set_title('Position Size Over Time')
    ax2.grid(True, alpha=0.3)
    
    # 3. Drawdown
    ax3 = axes[1, 0]
    ax3.fill_between(time_hours[1:], 0, drawdowns[1:], color='red', alpha=0.5)
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Drawdown (%)')
    ax3.set_title(f'Drawdown (Max: {max_drawdown:.2f}%)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Actions histogram
    ax4 = axes[1, 1]
    ax4.hist(actions, bins=50, edgecolor='black', alpha=0.7)
    ax4.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Action Value')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Action Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_name = model_path.replace('.zip', '_evaluation.png')
    plt.savefig(output_name, dpi=150)
    print(f"\n📊 Visualization saved as: {output_name}")
    
    return {
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'win_rate': win_rate,
        'trades': total_trades,
        'alpha': total_return - price_return
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to the saved model")
    args = parser.parse_args()
    
    results = evaluate_model_enhanced(args.model_path)