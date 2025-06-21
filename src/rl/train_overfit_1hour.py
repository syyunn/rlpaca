#!/usr/bin/env python3
"""
Overfitting test: Same complex state (5,185 dims) but only 1 hour of trading
Keep everything the same as production, just shorter episode
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

class HourlyEnv(RealisticOfflineEnv):
    """Same as RealisticOfflineEnv but episodes end after 1 hour"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hour_steps = 720  # 1 hour = 3600 sec / 5 sec intervals
        
    def _is_end_of_day(self):
        """End after 1 hour instead of full day"""
        # Count steps from market open
        steps_since_open = (self.current_time - self.current_time.replace(
            hour=self.market_open.hour, 
            minute=self.market_open.minute,
            second=0,
            microsecond=0
        )).total_seconds() / 5
        
        return steps_since_open >= self.hour_steps

def train_overfit_1hour():
    print("\n🔬 OVERFITTING TEST: 1-hour window")
    print("=" * 50)
    print("Same 5,185-dim state, same environment, just shorter episodes")
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
    
    # Create environment - same as production but 1-hour episodes
    env = HourlyEnv(
        historical_ticks=ticks,
        historical_bars=bars_df,
        initial_capital=100000,
        max_position=None,
        transaction_cost=0.0
    )
    
    # Test one episode to see the window
    print("\n📊 Testing 1-hour window:")
    obs = env.reset()
    start_time = env.current_time
    done = False
    steps = 0
    while not done and steps < 800:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        steps += 1
    end_time = env.current_time
    print(f"Episode length: {steps} steps")
    print(f"Start: {start_time.strftime('%H:%M:%S')}")
    print(f"End: {end_time.strftime('%H:%M:%S')}")
    
    # Create model with larger network and aggressive training
    print("\n🧠 Creating model with [512, 512, 256] network...")
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=50000,
        learning_starts=100,
        batch_size=256,
        tau=0.01,  # Faster target updates
        gamma=0.99,
        train_freq=1,
        gradient_steps=4,  # More gradient steps
        policy_kwargs=dict(
            net_arch=[512, 512, 256]  # Larger network
        ),
        verbose=0,
        tensorboard_log=f"logs/overfit_1hour/"
    )
    
    # Train on the same hour repeatedly
    print(f"\n🏃 Training on same 1-hour window for 100 episodes...")
    
    episode_returns = []
    start_time = time.time()
    
    for episode in range(100):
        obs = env.reset()
        done = False
        episode_return = 0
        step = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)
            episode_return += reward
            step += 1
            
        episode_returns.append(episode_return)
        
        # Aggressive training after each episode
        if model.num_timesteps > model.learning_starts:
            model.train(gradient_steps=100)  # Many gradient steps!
        
        if episode % 10 == 0:
            elapsed = time.time() - start_time
            avg_return = np.mean(episode_returns[-10:]) if len(episode_returns) >= 10 else episode_return
            print(f"Episode {episode}: Return={episode_return:.3f}%, "
                  f"Avg10={avg_return:.3f}%, Time={elapsed:.1f}s")
    
    # Save model
    model_path = '/Users/suyeolyun/gits/rlpaca/overfit_1hour_model.zip'
    model.save(model_path)
    print(f"\n✅ Model saved as: {model_path}")
    
    # Final evaluation
    print("\n📈 FINAL EVALUATION (deterministic):")
    obs = env.reset()
    done = False
    actions = []
    positions = []
    portfolio_values = []
    trades = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        actions.append(action[0])
        positions.append(info['position'])
        portfolio_values.append(info['portfolio_value'])
        if len(env.trades_executed) > trades:
            trades = len(env.trades_executed)
    
    final_return = (portfolio_values[-1] - env.initial_capital) / env.initial_capital * 100
    unique_actions = len(np.unique(np.round(actions, 3)))
    
    print(f"Final return: {final_return:.3f}%")
    print(f"Trades executed: {trades}")
    print(f"Unique actions: {unique_actions}")
    print(f"Action range: [{np.min(actions):.3f}, {np.max(actions):.3f}]")
    print(f"Max position: {np.max(positions):.2f} shares")
    
    # Check if it learned anything
    if final_return > 0.1 and unique_actions > 5:
        print("\n✅ SUCCESS! Model shows signs of learning on 1-hour window!")
    else:
        print("\n❌ Still not learning even on 1-hour window")
        print("Need either: bigger network, more episodes, or simpler state space")

if __name__ == "__main__":
    train_overfit_1hour()