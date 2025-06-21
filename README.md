```
    ____  __                         
   / __ \/ /___  ____ __________ _   
  / /_/ / / __ \/ __ `/ ___/ __ `/   
 / _, _/ / /_/ / /_/ / /__/ /_/ /    
/_/ |_/_/ .___/\__,_/\___/\__,_/     
       /_/                           
       
  🤖 RL + 🦙 Alpaca = 📈 Real-Time Trading
```

# RLpaca: Real-Time RL Trading System

A production-ready ML trading system that uses real tick data and streaming architecture - demonstrating capabilities beyond traditional daily-bar systems like FinRL.

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│   📊 Market     │     │  🧠 RL Model │     │  💰 Trading     │
│     Data        │────▶│    (SAC)     │────▶│    Orders       │
│  (Tick-level)   │     │ 5,185 dims   │     │  (Fractional)   │
└─────────────────┘     └──────────────┘     └─────────────────┘
         │                      │                      │
         └──────────────────────┴──────────────────────┘
                        Every 5 seconds
```

## 🚀 Key Features

- **Real-time tick streaming** from Alpaca WebSocket
- **Apache Kafka** for event streaming (quotes, trades, bars)
- **SAC (Soft Actor-Critic)** RL model with long-only constraint
- **5,185-dimensional state space** capturing market microstructure
- **Production deployment** with Docker containers

## ⚡ Quick Start

### 1. Setup Environment

```bash
# Clone and setup
git clone https://github.com/syyunn/rlpaca.git
cd rlpaca
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
docker-compose -f docker-compose.minimal-alpaca.yml up -d
docker-compose -f docker-compose.sac-executor.yml up -d

# Monitor execution
docker-compose -f docker-compose.sac-executor.yml logs -f
```

## 🏗️ Architecture

```
        ┌─────────────────────────────────────────────────┐
        │                 Alpaca WebSocket                 │
        └────────────┬───────────┬──────────┬─────────────┘
                     │           │          │
                     ▼           ▼          ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │  Quotes  │ │  Trades  │ │   Bars   │
              └────┬─────┘ └────┬─────┘ └────┬─────┘
                   │            │            │
                   └────────────┴────────────┘
                                │
                          Apache Kafka
                                │
                     ┌──────────▼──────────┐
                     │   RL Executor       │
                     │  - Model Inference  │
                     │  - Risk Management  │
                     │  - Order Placement  │
                     └──────────┬──────────┘
                                │
                          Trading Orders
                                │
                     ┌──────────▼──────────┐
                     │   Alpaca Paper/Live │
                     │     Trading API     │
                     └─────────────────────┘
```

## 🎯 Why Better Than FinRL?

| Feature | RLpaca | FinRL |
|---------|---------|--------|
| Data Granularity | Tick-level (real-time) | Daily bars |
| Deployment | Production-ready streaming | Backtesting only |
| Execution | Real broker integration | Simulation only |
| Decision Frequency | Every 5 seconds | Daily |
| Architecture | Microservices + Kafka | Monolithic |
| Position Sizing | Flexible % of capital | Fixed share amounts |
| Budget Scaling | $1k to $1M+ with same model | Requires retraining |

## 📚 Documentation

- [Quick Start Guide](docs/QUICKSTART.md) - Complete walkthrough
- [Technical Details](docs/TECHNICAL_DETAILS.md) - Model architecture

## 🛠️ Requirements

- Python 3.8+
- Docker & Docker Compose
- Alpaca account (free paper trading)
- ~2GB disk space

## 📈 Performance & Flexibility

- Processes ~4,680 decisions per trading day
- Sub-100ms model inference latency
- Supports fractional share trading
- Long-only strategy with capital-based position sizing
- **Flexible Deployment**: Start with $1k, scale to $100k+ without retraining
- **No Fixed Limits**: Position size based on available capital, not arbitrary constraints

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

Built with ❤️ using:
- [Alpaca Markets API](https://alpaca.markets/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [Apache Kafka](https://kafka.apache.org/)

---

**⚠️ Disclaimer**: This is for educational purposes. Always test thoroughly with paper trading before using real money.
