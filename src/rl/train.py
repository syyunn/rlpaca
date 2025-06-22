#!/usr/bin/env python3
"""
Professional RL Training Script for NVDA Intraday Trading

Usage:
    python train.py --config configs/default.json --symbol NVDA --date 2025-06-20
    python train.py --config configs/quick_test.json --symbol NVDA --date 2025-06-20
    python train.py --config configs/production.json --symbol NVDA --date 2025-06-20
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    EvalCallback, 
    CheckpointCallback, 
    CallbackList,
    BaseCallback
)
from stable_baselines3.common.monitor import Monitor

from src.rl.realistic_offline_env import RealisticOfflineEnv
import structlog

logger = structlog.get_logger()


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    logger.info(f"Loaded configuration from {config_path}")
    return config


def load_market_data(symbol: str, date: str):
    """Load market data for training"""
    data_dir = Path(__file__).parent.parent.parent / "data"
    bars_file = data_dir / "alpaca_bars" / f"{symbol}_{date}_bars.parquet"
    ticks_file = data_dir / "1min_ticks" / f"{symbol}_ticks_{date.replace('-', '')}_1min.parquet"
    
    if not bars_file.exists() or not ticks_file.exists():
        raise FileNotFoundError(
            f"Data files not found for {symbol} on {date}\n"
            f"Expected: {bars_file} and {ticks_file}\n"
            f"Please run data download script first."
        )
    
    # Load data
    bars_df = pd.read_parquet(bars_file)
    ticks_df = pd.read_parquet(ticks_file)
    
    # Convert ticks to expected format
    ticks = []
    for timestamp, tick in ticks_df.iterrows():
        ticks.append({
            'timestamp': timestamp,
            'bid': tick['bid'],
            'ask': tick['ask'],
            'bid_size': tick.get('bid_size', 100),
            'ask_size': tick.get('ask_size', 100),
            'price': (tick['bid'] + tick['ask']) / 2
        })
    
    logger.info(f"Loaded {len(bars_df)} minute bars and {len(ticks)} ticks")
    return bars_df, ticks


class ProgressCallback(BaseCallback):
    """Custom callback for logging training progress"""
    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Calculate progress
            progress_pct = (self.num_timesteps / self.locals.get('total_timesteps', 1)) * 100
            
            # Try to get current episode info from the environment
            infos = self.locals.get('infos', [])
            current_info = infos[0] if infos else {}
            
            # Calculate PnL from portfolio value
            portfolio_value = current_info.get('portfolio_value', 100000)
            pnl = portfolio_value - 100000  # Assuming 100k initial capital
            position = current_info.get('position', 0)
            
            # Get current action to debug (this is the exact action at this step)
            actions = self.locals.get('actions', None)
            current_action = float(actions[0][0]) if actions is not None and len(actions) > 0 else 0
            
            # Try to get loss info if available
            loss_info = ""
            if hasattr(self.model, 'logger') and self.model.logger:
                # Get recent loss values from logger
                if hasattr(self.model.logger, 'name_to_value'):
                    actor_loss = self.model.logger.name_to_value.get('train/actor_loss', None)
                    critic_loss = self.model.logger.name_to_value.get('train/critic_loss', None)
                    if actor_loss is not None:
                        loss_info = f" | Actor Loss: {actor_loss:.4f}"
                    if critic_loss is not None:
                        loss_info += f" | Critic Loss: {critic_loss:.4f}"
            
            # Get latest completed episode info
            if len(self.model.ep_info_buffer) > 0:
                ep_info = self.model.ep_info_buffer[-1]
                logger.info(
                    f"Step: {self.num_timesteps:,}/{self.locals.get('total_timesteps', 0):,} ({progress_pct:.1f}%) | "
                    f"Episodes: {len(self.model.ep_info_buffer)} | "
                    f"Last Return: {ep_info.get('r', 0):.2f} | "
                    f"PnL: ${pnl:.2f} | Pos: {position:.0f} | Action: {current_action:.3f}{loss_info}"
                )
            else:
                # Show progress even without completed episodes
                logger.info(
                    f"Step: {self.num_timesteps:,}/{self.locals.get('total_timesteps', 0):,} ({progress_pct:.1f}%) | "
                    f"First episode in progress... | "
                    f"PnL: ${pnl:.2f} | Pos: {position:.0f} | Action: {current_action:.3f}{loss_info}"
                )
        return True


def create_environment(bars_df, ticks, env_config):
    """Create training environment"""
    env = RealisticOfflineEnv(
        historical_ticks=ticks,
        historical_bars=bars_df,
        initial_capital=env_config['initial_capital'],
        max_position=None,
        transaction_cost=env_config['transaction_cost'],
        decision_interval_seconds=env_config['decision_interval_seconds']
    )
    return Monitor(env)


def train_model(env, eval_env, config, model_name):
    """Train the SAC model"""
    
    model_config = config['model']
    training_config = config['training']
    logging_config = config['logging']
    
    # Create directories
    os.makedirs(f"{logging_config['model_dir']}/{model_name}", exist_ok=True)
    os.makedirs(f"{logging_config['checkpoint_dir']}/{model_name}", exist_ok=True)
    os.makedirs(f"{logging_config['tensorboard_log']}/{model_name}", exist_ok=True)
    
    # Create callbacks
    callbacks = []
    
    # Evaluation callback
    if eval_env is not None:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{logging_config['model_dir']}/{model_name}/",
            log_path=f"{logging_config['tensorboard_log']}/{model_name}/",
            eval_freq=training_config['eval_freq'],
            n_eval_episodes=training_config['n_eval_episodes'],
            deterministic=True,
            render=False,
            verbose=1
        )
        callbacks.append(eval_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=training_config['save_freq'],
        save_path=f"{logging_config['checkpoint_dir']}/{model_name}/",
        name_prefix="sac_checkpoint",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Progress callback - log every 100 steps for more frequent updates
    progress_callback = ProgressCallback(check_freq=100)
    callbacks.append(progress_callback)
    
    # Create model
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=model_config['learning_rate'],
        buffer_size=model_config['buffer_size'],
        learning_starts=model_config['learning_starts'],
        batch_size=model_config['batch_size'],
        tau=model_config['tau'],
        gamma=model_config['gamma'],
        ent_coef=model_config['ent_coef'],
        target_update_interval=model_config['target_update_interval'],
        gradient_steps=model_config['gradient_steps'],
        policy_kwargs=dict(
            net_arch=model_config['network_arch'],
            log_std_init=model_config['log_std_init']
        ),
        tensorboard_log=f"{logging_config['tensorboard_log']}/{model_name}/",
        verbose=training_config['verbose']
    )
    
    # Log model info
    total_params = sum(p.numel() for p in model.policy.parameters())
    logger.info(f"Created SAC model with {total_params:,} parameters")
    logger.info(f"Network architecture: {model_config['network_arch']}")
    
    # Train
    logger.info(f"Starting training for {training_config['total_timesteps']:,} timesteps...")
    model.learn(
        total_timesteps=training_config['total_timesteps'],
        callback=CallbackList(callbacks),
        tb_log_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Save final model
    final_path = f"{logging_config['model_dir']}/{model_name}/final_model.zip"
    model.save(final_path)
    logger.info(f"Training complete! Model saved to {final_path}")
    
    return model


def evaluate_model(model, env, n_episodes=5):
    """Evaluate the trained model"""
    returns = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_return = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            episode_return += reward
        
        # Get initial capital from the wrapped environment
        initial_capital = env.env.initial_capital if hasattr(env.env, 'initial_capital') else 100000
        final_return = (info['portfolio_value'] - initial_capital) / initial_capital * 100
        returns.append(final_return)
        logger.info(f"Episode {episode + 1}: {final_return:.2f}% return")
    
    avg_return = np.mean(returns)
    std_return = np.std(returns)
    logger.info(f"Average return: {avg_return:.2f}% ± {std_return:.2f}%")
    
    return avg_return, std_return


def main():
    parser = argparse.ArgumentParser(description="Train RL trading model")
    parser.add_argument("--config", default="configs/default.json", 
                        help="Path to configuration JSON file")
    parser.add_argument("--symbol", default="NVDA", help="Stock symbol")
    parser.add_argument("--date", default="2025-06-20", help="Training date (YYYY-MM-DD)")
    parser.add_argument("--name", help="Model name (default: {symbol}_{date})")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set model name
    if args.name:
        model_name = args.name
    else:
        config_name = Path(args.config).stem
        model_name = f"{args.symbol}_{config_name}_{args.date.replace('-', '')}"
    
    logger.info(f"Training model: {model_name}")
    
    # Load data
    bars_df, ticks = load_market_data(args.symbol, args.date)
    
    # Create environments
    train_env = create_environment(bars_df, ticks, config['environment'])
    eval_env = create_environment(bars_df, ticks, config['environment'])
    
    # Train model
    model = train_model(train_env, eval_env, config, model_name)
    
    # Final evaluation
    logger.info("Running final evaluation...")
    avg_return, std_return = evaluate_model(model, eval_env)
    
    # Save training summary
    summary = {
        "model_name": model_name,
        "config_file": args.config,
        "symbol": args.symbol,
        "date": args.date,
        "network_arch": config['model']['network_arch'],
        "total_timesteps": config['training']['total_timesteps'],
        "avg_return": float(avg_return),
        "std_return": float(std_return),
        "timestamp": datetime.now().isoformat()
    }
    
    summary_path = f"{config['logging']['model_dir']}/{model_name}/training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Training complete! Summary saved to {summary_path}")


if __name__ == "__main__":
    main()