# Real-Time ML Trading System

A production-ready ML trading system that uses real tick data and streaming architecture - demonstrating capabilities beyond traditional daily-bar systems like FinRL.

## Key Features

- **Real-time tick streaming** from Alpaca WebSocket
- **Apache Kafka** for event streaming (quotes, trades, bars)
- **SAC (Soft Actor-Critic)** RL model with long-only constraint
- **5,185-dimensional state space** capturing market microstructure
- **Production deployment** with Docker containers

## Quick Start

### 1. Setup Environment

```bash
# Clone and setup
git clone https://github.com/yourusername/poly-fka.git
cd poly-fka
pip install -r requirements.txt

# Configure Alpaca credentials
cp .env.example .env
# Edit .env with your Alpaca API keys
```

### 2. Download Data & Train Model

```bash
# Download historical data (ticks + minute bars)
python scripts/download_data.py

# Train SAC model (quick test - 100 steps)
python src/rl/train_long_only_model.py --quick-test

# Or full training (50,000 steps)
python src/rl/train_long_only_model.py --timesteps 50000
```

### 3. Deploy to Production

```bash
# Prepare model for deployment
cp long_only_sac_model.zip streaming_model.zip

# Start streaming infrastructure
docker-compose -f docker-compose.minimal-kafka.yml up -d
docker-compose -f docker-compose.minimal-alpaca.yml up -d
docker-compose -f docker-compose.sac-executor.yml up -d

# Monitor execution
docker-compose -f docker-compose.sac-executor.yml logs -f
```

## Architecture

```
Alpaca WebSocket → Kafka Topics → RL Executor → Trading Orders
                    ├── quotes
                    ├── trades  
                    └── bars
```

## Why Better Than FinRL?

1. **Tick-level data**: Captures real market microstructure
2. **Streaming architecture**: Built for production, not just backtesting
3. **Consistent pipeline**: Same data format in training and production
4. **Live execution**: Actually submits orders to broker
5. **Real-time decisions**: Acts every 5 seconds, not daily

## Documentation

- [Quick Start Guide](docs/QUICKSTART.md) - Complete walkthrough
- [SAC Executor Details](docs/SAC_EXECUTOR_README.md) - Model architecture

## Requirements

- Python 3.8+
- Docker & Docker Compose
- Alpaca account (free paper trading)
- ~2GB disk space

## Support

For issues or questions, please open a GitHub issue.