```
    ____  __
   / __ \/ /___  ____ __________ _
  / /_/ / / __ \/ __ `/ ___/ __ `/
 / _, _/ / /_/ / /_/ / /__/ /_/ /
/_/ |_/_/ .___/\__,_/\___/\__,_/
       /_/

  Real-Time RL Trading System
```

# RLpaca: Reinforcement Learning for Quantitative Trading

A comprehensive study of reinforcement learning algorithms for high-frequency trading using real tick-level market data. This project evaluates **4 state-of-the-art RL algorithms** (SAC, PPO, TD3, DDPG) on 5 years of NVDA tick data (3.15B quotes).

## Key Results

| Algorithm | Sharpe Ratio | Max Drawdown | Win Rate | Training Time |
|-----------|--------------|--------------|----------|---------------|
| **TD3**   | **1.79** | **-5.9%** | **63.8%** | **2.1 hrs** |
| PPO       | 1.68 | -6.8% | 61.3% | 4.1 hrs |
| SAC       | 1.43 | -9.2% | 57.8% | 2.3 hrs |
| DDPG      | 1.18 | -14.3% | 54.2% | 2.4 hrs |

**Winner: TD3** achieves 52% better risk-adjusted returns than DDPG.

## Project Structure

```
rlpaca/
├── src/
│   ├── rl/                    # RL training code
│   │   ├── env/               # Trading environment
│   │   ├── train.py           # Main training script
│   │   └── train_long_only_model.py
│   ├── producer/              # Kafka data producers
│   │   └── alpaca_producer.py # Alpaca WebSocket → Kafka
│   ├── deployment/            # Production deployment
│   │   └── sac_streaming_executor.py
│   └── config/                # Configuration management
│
├── experiments/               # Experiment results
│   ├── configs/               # Algorithm configurations
│   │   ├── sac_nvda.yaml
│   │   ├── ppo_nvda.yaml
│   │   ├── td3_nvda.yaml
│   │   └── ddpg_nvda.yaml
│   ├── logs/                  # Training logs
│   │   ├── sac_training.log
│   │   ├── ppo_training.log
│   │   ├── td3_training.log
│   │   └── ddpg_training.log
│   └── results/               # Evaluation results
│       ├── algorithm_comparison.csv
│       ├── training_metrics.csv
│       └── test_set_evaluation.csv
│
├── scripts/                   # Utility scripts
│   └── download_data.py       # Historical data download
│
├── docker-compose*.yml        # Docker configurations
├── final_presentation.html    # Research presentation
└── README.md
```

## Data Infrastructure

### Alpaca Markets API
- Commission-free trading API for algorithmic trading
- Real-time market data via WebSocket
- Historical tick data (quotes, trades, bars)
- Paper trading for backtesting

### Apache Kafka
- Distributed streaming platform for real-time data
- High throughput: millions of messages/second
- Low latency: sub-millisecond delivery
- Buffers data between Alpaca stream and RL agent

### Data Pipeline
```
Alpaca WebSocket → Kafka Producer → Kafka Topic → RL Environment → Agent
```

### Dataset
- **Training**: 5 years (2020-2024), 1,260 trading days, 3.15B quotes, ~27.6 GB
- **Test**: Q1 2025, 63 days, 157M quotes, ~1.4 GB
- **Symbol**: NVDA (NVIDIA Corporation)

## State Space (14,548 dimensions)

| Component | Dimensions | Description |
|-----------|------------|-------------|
| Tick Window | 500 × 4 = 2,000 | Last 500 ticks (bid, ask, bid_size, ask_size) |
| Bar Window | 780 × 6 = 4,680 | 13 hours of minute bars (OHLCV + vwap) |
| Action History | 1,560 × 5 = 7,800 | Recent actions and outcomes |
| Portfolio State | 5 | Position, cash, PnL, etc. |
| Constraints | 3 | Trading constraints |

## Algorithms

### Off-Policy (uses replay buffer)

**SAC (Soft Actor-Critic)**
- Stochastic policy with entropy maximization
- Twin Q-networks for stability
- Best for: exploration, finding opportunities

**TD3 (Twin Delayed DDPG)**
- Deterministic policy
- Three innovations: twin critics, delayed updates, target smoothing
- Best for: production trading (highest Sharpe)

**DDPG (Deep Deterministic Policy Gradient)**
- Deterministic policy, single critic
- Prone to Q-value overestimation
- **Not recommended** for production

### On-Policy (fresh data only)

**PPO (Proximal Policy Optimization)**
- Stochastic policy with clipped updates
- Most stable training
- Best for: safety-critical applications

## Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/syyunn/rlpaca.git
cd rlpaca

# Install dependencies
pip install -r requirements.txt

# Configure credentials
cp .env.example .env
# Edit .env with your Alpaca API keys
```

### 2. Download Data & Train

```bash
# Download historical data
python scripts/download_data.py

# Train TD3 model (recommended)
python src/rl/train.py --algorithm TD3 --timesteps 100000

# Or train SAC
python src/rl/train.py --algorithm SAC --timesteps 100000
```

### 3. Evaluate

```bash
# Evaluate on test set
python src/rl/evaluate.py --model experiments/models/td3_nvda_final.zip
```

### 4. Deploy (Production)

```bash
# Start Kafka infrastructure
docker-compose -f docker-compose.minimal-alpaca.yml up -d

# Start trading executor
docker-compose -f docker-compose.sac-executor.yml up -d

# Monitor
docker-compose -f docker-compose.sac-executor.yml logs -f
```

## Key Research Insights

1. **Deterministic > Stochastic for Trading**: TD3's deterministic policy outperforms SAC/PPO's stochastic policies. Trading rewards consistent execution over exploration.

2. **Patient Updates Win**: TD3's delayed policy updates (d=2) prevent chasing noise. Wait for critics to stabilize before updating the actor.

3. **Twin Critics Reduce Risk**: Taking min(Q1, Q2) prevents overestimation. TD3: -5.9% drawdown vs DDPG: -14.3%.

4. **Trade Frequency Sweet Spot**: ~120 trades/day (TD3) balances opportunity capture with transaction costs.

5. **Modern Algorithms Matter**: DDPG (2015) fails catastrophically. TD3's 2018 improvements are transformative (+52% Sharpe).

## Algorithm Selection Guide

| Scenario | Recommendation | Reason |
|----------|----------------|--------|
| Production (real money) | **TD3** | Best Sharpe, lowest drawdown |
| Maximum returns | SAC | Highest daily return, most trades |
| Safety-critical | PPO | Most stable training |
| Fast training | TD3 | 2.1 hrs (off-policy efficient) |
| High-frequency | SAC | 156 trades/day, captures opportunities |
| Legacy systems | **AVOID DDPG** | Use modern algorithms |

## Presentation

View the full research presentation:
```bash
open final_presentation.html
```

15-slide presentation covering:
- Data infrastructure (Alpaca + Kafka)
- On-policy vs Off-policy learning
- Stochastic vs Deterministic policies
- Algorithm comparison and results
- Key insights and recommendations

## Requirements

- Python 3.8+
- Docker & Docker Compose
- Alpaca account (free paper trading available)
- ~30GB disk space (for full dataset)

## Tech Stack

- **ML Framework**: Stable Baselines3
- **Data Source**: Alpaca Markets API
- **Streaming**: Apache Kafka
- **Deployment**: Docker
- **Visualization**: TensorBoard

## Contributing

Contributions welcome! Please read the contributing guidelines before submitting PRs.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [Alpaca Markets](https://alpaca.markets/) - Commission-free trading API
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/) - RL algorithms
- [Apache Kafka](https://kafka.apache.org/) - Streaming platform

---

**Disclaimer**: This project is for educational and research purposes. Always test thoroughly with paper trading before using real money. Past performance does not guarantee future results.
