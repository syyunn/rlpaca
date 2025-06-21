#!/usr/bin/env python3
"""
Overfitting test with MUCH BIGGER network
Same 1-hour window but with a network that can actually handle 5,185 inputs
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
import time

# Reuse the HourlyEnv from train_overfit_1hour
from train_overfit_1hour import HourlyEnv

def train_big_network():
    print("\n🔬 BIG NETWORK OVERFITTING TEST")
    print("=" * 50)
    print("Input: 5,185 dimensions")
    print("Network: [2048, 1024, 512, 256] - Much bigger!")
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
    env = HourlyEnv(
        historical_ticks=ticks,
        historical_bars=bars_df,
        initial_capital=100000,
        max_position=None,
        transaction_cost=0.0
    )
    
    # Create model with MUCH BIGGER network
    print("\n🧠 Creating BIG model...")
    print("Parameters: ~13M (vs ~2.7M before)")
    
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=1e-4,  # Lower LR for bigger network
        buffer_size=50000,
        learning_starts=100,
        batch_size=512,  # Bigger batch for stability
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=4,
        policy_kwargs=dict(
            net_arch=[2048, 1024, 512, 256]  # MUCH bigger!
        ),
        verbose=0
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Train on same hour
    print(f"\n🏃 Training on same 1-hour window for 50 episodes...")
    print("(Using bigger network should show results faster)")
    
    episode_returns = []
    start_time = time.time()
    best_return = -100
    
    for episode in range(50):
        obs = env.reset()
        done = False
        episode_return = 0
        actions_taken = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            actions_taken.append(action[0])
            obs, reward, done, info = env.step(action)
            episode_return += reward
            
        episode_returns.append(episode_return)
        
        # Check if actions are non-zero
        unique_actions = len(np.unique(np.round(actions_taken, 3)))
        action_std = np.std(actions_taken)
        
        # Aggressive training
        if model.num_timesteps > model.learning_starts:
            model.train(gradient_steps=200)  # Even more training!
        
        if episode % 5 == 0:
            elapsed = time.time() - start_time
            avg_return = np.mean(episode_returns[-5:]) if len(episode_returns) >= 5 else episode_return
            print(f"Episode {episode}: Return={episode_return:.3f}%, "
                  f"Avg5={avg_return:.3f}%, "
                  f"Actions={unique_actions}, Std={action_std:.3f}, "
                  f"Time={elapsed:.1f}s")
            
            if episode_return > best_return:
                best_return = episode_return
                
    # Final evaluation
    print("\n📈 FINAL EVALUATION:")
    obs = env.reset()
    done = False
    final_actions = []
    trades = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        final_actions.append(action[0])
        obs, reward, done, info = env.step(action)
        if len(env.trades_executed) > trades:
            trades = len(env.trades_executed)
    
    final_return = (info['portfolio_value'] - env.initial_capital) / env.initial_capital * 100
    
    print(f"Final return: {final_return:.3f}%")
    print(f"Best return during training: {best_return:.3f}%")
    print(f"Trades executed: {trades}")
    print(f"Action diversity: {len(np.unique(np.round(final_actions, 3)))}")
    
    if final_return > 0.01 or best_return > 0.01:
        print("\n✅ SUCCESS! Big network shows signs of learning!")
        model.save('/Users/suyeolyun/gits/rlpaca/big_network_overfit_model.zip')
    else:
        print("\n❌ Even with 13M parameters, still not learning")
        print("Problem might be:")
        print("- Need recurrent architecture (LSTM/Transformer)")
        print("- State representation too noisy")
        print("- Reward too sparse")

if __name__ == "__main__":
    train_big_network()