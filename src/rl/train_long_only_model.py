#!/usr/bin/env python3
"""
Train SAC model with long-only constraint
The model will learn that it can only sell if it has a position
"""

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from realistic_offline_env import RealisticOfflineEnv
import json
import structlog

logger = structlog.get_logger()

class VerboseTrainingCallback(BaseCallback):
    """Custom callback for logging training progress"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.constraint_violations = []
        
    def _on_step(self) -> bool:
        # Log info from the environment
        if 'infos' in self.locals:
            for info in self.locals['infos']:
                if info and 'constraint_violations' in info:
                    violations = info['constraint_violations']
                    if violations:
                        self.constraint_violations.extend(violations)
                        
        return True
    
    def _on_rollout_end(self) -> None:
        # Log constraint violations
        if self.constraint_violations:
            violation_counts = {}
            for v in self.constraint_violations:
                violation_counts[v] = violation_counts.get(v, 0) + 1
            logger.info("Constraint violations in rollout", violations=violation_counts)
            self.constraint_violations = []

def load_training_data(date='2025-06-20'):
    """Load tick and bar data for training"""
    date_str = date.replace('-', '')
    tick_file = f'data/1min_ticks/NVDA_ticks_{date_str}_1min.parquet'
    df = pd.read_parquet(tick_file)
    
    # Convert to list of dicts for the environment
    ticks = []
    for _, row in df.iterrows():
        ticks.append({
            'timestamp': row['timestamp'],
            'bid': row['bid'],
            'ask': row['ask'],
            'bid_size': row['bid_size'],
            'ask_size': row['ask_size']
        })
    
    # Load REAL minute bars directly from Alpaca
    bars_file = f'data/alpaca_bars/NVDA_{date}_bars.parquet'
    if not os.path.exists(bars_file):
        # Fallback to computing from trades if bars not downloaded
        trades_file = 'data/ticks_fixed/NVDA_trades_with_timestamps.parquet'
        trades_df = pd.read_parquet(trades_file)
        
        if 'timestamp' in trades_df.columns:
            trades_df.set_index('timestamp', inplace=True)
        
        trades_df = trades_df.between_time('09:30', '16:00')
        
        minute_bars = trades_df.resample('1min').agg({
            'price': ['first', 'max', 'min', 'last', 'count'],
            'size': 'sum'
        })
        
        minute_bars.columns = ['open', 'high', 'low', 'close', 'trade_count', 'volume']
        minute_bars = minute_bars.fillna(method='ffill')
        minute_bars['vwap'] = minute_bars['close']
    else:
        # Use Alpaca's official minute bars
        minute_bars = pd.read_parquet(bars_file)
        # Ensure we have all required columns
        if 'vwap' not in minute_bars.columns:
            minute_bars['vwap'] = minute_bars['close']
        if 'trade_count' not in minute_bars.columns:
            minute_bars['trade_count'] = minute_bars['volume'] / 100  # Estimate
    
    return ticks, minute_bars

def train_long_only_model(timesteps=1000, date='2025-06-20', model_name='long_only_sac_model'):
    """Train model with long-only constraint"""
    
    logger.info(f"Loading {date} tick and bar data...")
    ticks, minute_bars = load_training_data(date)
    logger.info(f"Loaded {len(ticks)} ticks")
    
    # Create environment with long-only constraint
    env = RealisticOfflineEnv(
        historical_ticks=ticks,
        historical_bars=minute_bars,
        decision_interval_seconds=5,
        initial_capital=100000,
        max_position=1000,  # Max 1000 shares long
        transaction_cost=0.001
    )
    
    logger.info("Environment created - LONG-ONLY mode enforced")
    
    # Create SAC model
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
        policy_kwargs=dict(
            net_arch=[512, 512, 256]  # Deeper network
        ),
        verbose=1,
        tensorboard_log="./tensorboard_logs/long_only_sac/"
    )
    
    logger.info("SAC model created")
    
    # Training parameters
    total_timesteps = timesteps
    
    # Create callback
    callback = VerboseTrainingCallback()
    
    logger.info(f"Starting training for {total_timesteps} timesteps...")
    logger.info("Model will learn:")
    logger.info("  - Can only buy (no short selling)")
    logger.info("  - Can only sell if has position")
    logger.info("  - Position limits (max 1000 shares)")
    
    start_time = datetime.now()
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    end_time = datetime.now()
    training_duration = (end_time - start_time).total_seconds()
    
    logger.info(f"Training completed in {training_duration:.1f} seconds")
    
    # Test the trained model
    logger.info("\nTesting trained model...")
    obs = env.reset()
    total_reward = 0
    trades = []
    positions = []
    
    for i in range(100):  # Test for 100 steps
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        positions.append(info['position'])
        if len(info['constraint_violations']) > 0:
            logger.warning(f"Step {i}: Constraint violations", violations=info['constraint_violations'])
            
        if done:
            logger.info("Episode done", total_reward=total_reward)
            break
    
    # Check if model learned long-only behavior
    min_position = min(positions)
    max_position = max(positions)
    logger.info(f"Position range during test: [{min_position:.2f}, {max_position:.2f}]")
    
    if min_position < 0:
        logger.warning("Model still trying to go short!")
    else:
        logger.info("✅ Model successfully learned long-only constraint!")
    
    # Save the model
    model_filename = f"{model_name}.zip"
    model.save(model_filename)
    logger.info(f"Model saved as {model_filename}")
    
    # Save training summary
    summary = {
        "training_duration_seconds": training_duration,
        "total_timesteps": total_timesteps,
        "final_test_reward": float(total_reward),
        "min_position_test": float(min_position),
        "max_position_test": float(max_position),
        "long_only_success": min_position >= 0,
        "training_date": datetime.now().isoformat()
    }
    
    with open("long_only_training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train long-only SAC model')
    parser.add_argument('--timesteps', type=int, default=1000,
                       help='Number of training timesteps (default: 1000 for quick test, use 50000 for real training)')
    parser.add_argument('--date', type=str, default='2025-06-20',
                       help='Training date (default: 2025-06-20)')
    parser.add_argument('--model-name', type=str, default='long_only_sac_model',
                       help='Model name to save (default: long_only_sac_model)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test mode - only 100 timesteps')
    
    args = parser.parse_args()
    
    # Override timesteps if quick-test
    if args.quick_test:
        args.timesteps = 100
        logger.info("🚀 QUICK TEST MODE - 100 timesteps only")
    
    logger.info("="*60)
    logger.info("TRAINING LONG-ONLY SAC MODEL")
    logger.info("="*60)
    logger.info(f"Timesteps: {args.timesteps}")
    logger.info(f"Date: {args.date}")
    logger.info(f"Model name: {args.model_name}")
    
    model = train_long_only_model(
        timesteps=args.timesteps,
        date=args.date,
        model_name=args.model_name
    )
    
    logger.info("\n✅ Training complete!")
    logger.info(f"Model saved as: {args.model_name}.zip")
    logger.info("Ready for deployment with long-only constraint")