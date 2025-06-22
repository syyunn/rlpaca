"""
Training Configuration for RL Trading System

This module defines all hyperparameters and network architectures
for training SAC models on NVDA intraday trading.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class NetworkConfig:
    """Neural network architecture configuration"""
    # Network sizes tested and validated
    SMALL: List[int] = None  # Will be set in __post_init__
    MEDIUM: List[int] = None
    LARGE: List[int] = None
    XLARGE: List[int] = None
    
    def __post_init__(self):
        # Tested architectures with 5,185-dim input
        self.SMALL = [256, 256]  # ~2.7M params - baseline
        self.MEDIUM = [512, 512, 256]  # ~5.5M params - good for testing
        self.LARGE = [1024, 512, 256]  # ~11M params - proven to work
        self.XLARGE = [2048, 1024, 512, 256]  # ~22M params - for complex patterns


@dataclass 
class SACConfig:
    """SAC algorithm hyperparameters"""
    # Learning parameters
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 50_000  # Reduced from 1M to fit in memory
    learning_starts: int = 1000
    
    # SAC specific
    tau: float = 0.005  # Soft update coefficient
    gamma: float = 0.99  # Discount factor
    ent_coef: str = 'auto'  # Automatic entropy tuning
    target_update_interval: int = 1
    gradient_steps: int = 1  # Updates per env step
    
    # Exploration
    log_std_init: float = -1  # Initial exploration (-3 is too low)
    
    # Training stability
    use_sde: bool = False  # State Dependent Exploration
    sde_sample_freq: int = -1
    

@dataclass
class TrainingConfig:
    """Complete training configuration"""
    # Model architecture
    network_arch: List[int] = None  # Set using NetworkConfig
    
    # Training duration
    total_timesteps: int = 1_000_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 5
    
    # Logging
    tensorboard_log: str = "./logs/"
    verbose: int = 1
    
    # Checkpointing
    save_freq: int = 50_000
    checkpoint_dir: str = "./checkpoints/"
    
    # Environment
    transaction_cost: float = 0.0  # Alpaca is commission-free
    initial_capital: float = 100_000
    
    def __post_init__(self):
        if self.network_arch is None:
            # Default to LARGE architecture that worked
            self.network_arch = NetworkConfig().LARGE


# Pre-configured settings for different use cases
class TrainingPresets:
    """Pre-configured training settings"""
    
    @staticmethod
    def quick_test():
        """Quick testing configuration"""
        config = TrainingConfig()
        config.total_timesteps = 50_000
        config.network_arch = NetworkConfig().MEDIUM
        config.eval_freq = 5_000
        return config, SACConfig(learning_starts=100, gradient_steps=4)
    
    @staticmethod
    def standard():
        """Standard training configuration"""
        config = TrainingConfig()
        config.total_timesteps = 1_000_000
        config.network_arch = NetworkConfig().LARGE
        return config, SACConfig()
    
    @staticmethod
    def aggressive():
        """Aggressive learning for overfitting tests"""
        config = TrainingConfig()
        config.total_timesteps = 200_000
        config.network_arch = NetworkConfig().LARGE
        sac_config = SACConfig(
            learning_rate=1e-3,  # Higher LR
            batch_size=512,  # Larger batches
            gradient_steps=10,  # More updates
            tau=0.01,  # Faster target updates
            learning_starts=100  # Start learning quickly
        )
        return config, sac_config
    
    @staticmethod
    def production():
        """Production deployment configuration"""
        config = TrainingConfig()
        config.total_timesteps = 5_000_000
        config.network_arch = NetworkConfig().XLARGE
        config.eval_freq = 50_000
        config.save_freq = 100_000
        return config, SACConfig(buffer_size=100_000)


# Example usage:
# from src.config.training_config import TrainingPresets
# config, sac_config = TrainingPresets.standard()
# model = SAC("MlpPolicy", env, **asdict(sac_config), 
#            policy_kwargs=dict(net_arch=config.network_arch))