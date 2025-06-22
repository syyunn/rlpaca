# Configuration Guide

## Overview

The RL trading system uses JSON configuration files for flexible hyperparameter management.

## Configuration Files

### 1. `configs/test.json`
- **Purpose**: Quick testing and debugging
- **Training time**: ~2 minutes (1,000 steps)
- **Network**: [256, 256]
- **Use case**: Verify setup, test changes

### 2. `configs/quick_test.json`
- **Purpose**: Fast iteration and development
- **Training time**: ~10 minutes (50,000 steps)
- **Network**: [512, 512, 256]
- **Use case**: Initial model development

### 3. `configs/default.json`
- **Purpose**: Standard training runs
- **Training time**: ~3 hours (1M steps)
- **Network**: [1024, 512, 256]
- **Use case**: Production-ready models

### 4. `configs/production.json`
- **Purpose**: Final production deployment
- **Training time**: ~15 hours (5M steps)
- **Network**: [2048, 1024, 512, 256]
- **Transaction cost**: 0.001 (0.1%)
- **Use case**: Real money trading

## Key Parameters

### Model Parameters
- `network_arch`: Neural network architecture
- `learning_rate`: Optimization step size (default: 3e-4)
- `batch_size`: Samples per gradient update
- `buffer_size`: Experience replay capacity
- `log_std_init`: Initial exploration (-1 recommended, -3 too low)

### Training Parameters
- `total_timesteps`: Total environment interactions
- `eval_freq`: Steps between evaluations
- `save_freq`: Steps between checkpoints

### Environment Parameters
- `transaction_cost`: Trading fees (0.0 for Alpaca)
- `initial_capital`: Starting portfolio value
- `decision_interval_seconds`: Time between actions (5 seconds)

## Usage

```bash
# Quick test
python src/rl/train.py --config configs/test.json

# Standard training
python src/rl/train.py --config configs/default.json

# Custom configuration
python src/rl/train.py --config my_config.json
```

## Creating Custom Configurations

1. Copy an existing config:
```bash
cp configs/default.json configs/my_config.json
```

2. Modify parameters as needed
3. Train with your config:
```bash
python src/rl/train.py --config configs/my_config.json
```

## Best Practices

1. **Start small**: Use `test.json` to verify changes work
2. **Scale up gradually**: test → quick_test → default → production
3. **Monitor tensorboard**: Check loss curves and returns
4. **Save often**: Set appropriate `save_freq` for long runs