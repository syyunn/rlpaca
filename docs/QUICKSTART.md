# Complete Training Pipeline: Data → Model → Deployment

This guide walks through the entire process of training and deploying a real-time ML trading system using actual market data.

## Prerequisites

- Alpaca API credentials (free paper trading account)
- Docker and Docker Compose installed
- Python 3.8+ with required packages
- ~2GB disk space for data and models

## Step 1: Download Real Market Data (Including Bars)

Download historical tick data AND minute bars from Alpaca:

```bash
# Set your Alpaca credentials
export ALPACA_PAPER_API_KEY="your_paper_api_key"
export ALPACA_PAPER_API_SECRET="your_paper_secret_key"

# Download both tick data and minute bars
python scripts/download_data.py
```

**What this does:**
- Downloads official Alpaca minute bars to `data/alpaca_bars/`
- Downloads tick data to `data/ticks_fixed/`
- Ensures training and production use the same data source

**Expected output:**
```
📊 Downloading NVDA data for 2025-06-20
============================================================
📈 Downloading minute bars...
✅ Downloaded 389 minute bars
   Price range: $142.65 - $146.20
   Total volume: 113,435,660
💾 Saved to: data/alpaca_bars/NVDA_2025-06-20_bars.parquet
```

## Step 2: Train the Long-Only SAC Model

Train a Soft Actor-Critic (SAC) model with real market data:

```bash
python src/rl/train_long_only_model.py
```

**Training configuration:**
- Algorithm: SAC (Soft Actor-Critic)
- Environment: RealisticOfflineEnv
- Observation space: 5,185 dimensions
  - 500 dims: Recent tick data (quotes)
  - 4,680 dims: Alpaca minute bars (OHLCV + features)
  - 5 dims: Position state
- Action space: 2 dimensions (position delta, limit offset)
- Constraint: Long-only (no short selling)

**Training options:**

For quick testing (1,000 timesteps - 3 seconds):
```python
# Default in train_long_only_model.py line 124
total_timesteps = 1000
```

For real training (50,000 timesteps - 3 minutes):
```python
total_timesteps = 50000
```

## Step 4: Verify Model Compatibility

```bash
python -c "
from stable_baselines3 import SAC
model = SAC.load('long_only_sac_model.zip')
print(f'✅ Model observation space: {model.observation_space.shape}')
print(f'✅ Model action space: {model.action_space.shape}')
"
```

Expected: (5185,) and (2,) dimensions

## Step 5: Deploy Model for Streaming

### 5.1 Prepare model
```bash
cp long_only_sac_model.zip streaming_model.zip
```

### 5.2 Start streaming infrastructure
```bash
# Start Kafka and Alpaca streaming (includes bars topic)
docker-compose -f docker-compose.minimal-alpaca.yml up -d

# Start the RL executor
docker-compose -f docker-compose.sac-executor.yml up -d

# Monitor decisions
docker-compose -f docker-compose.sac-executor.yml logs -f
```

**Expected output:**
```
✅ Connected to Kafka
Consuming from: alpaca_market_data_paper_trades, alpaca_market_data_paper_quotes, alpaca_market_data_paper_bars
Data received - Trades: 5, Quotes: 10, Bars: 1
[02:16:30] DECISION TIME
Model action: position_delta=1.000, limit_offset=-10.0 bps
```

## Key Improvements in This Pipeline

1. **Consistent Data Source**: Both training and production use Alpaca's official minute bars
2. **No Manual Aggregation**: Eliminates discrepancies from computing bars differently
3. **Real Features in Production**: The executor now uses actual bar data instead of zeros
4. **Complete State**: Model sees the same 5,185-dimensional state in both environments

## Understanding the Data Flow

### Training:
```
Alpaca Historical API
    ├── Minute Bars → data/alpaca_bars/
    └── Tick Data → data/ticks_fixed/
                         ↓
                 Training Environment
                   (5,185 dims)
                         ↓
                    SAC Model
```

### Production:
```
Alpaca WebSocket
    ├── Bars → Kafka bars topic → 
    ├── Quotes → Kafka quotes topic → Executor
    └── Trades → Kafka trades topic →    ↓
                                    SAC Model
                                  (5,185 dims)
                                        ↓
                                  Trading Orders
```

## Monitoring

### Check data flow:
```bash
# Bars topic (new!)
docker exec kafka kafka-console-consumer \
  --bootstrap-server kafka:29092 \
  --topic alpaca_market_data_paper_bars \
  --max-messages 2
```

### Verify executor is using bars:
Look for "Bars: X" in the logs:
```
Data received - Trades: 100, Quotes: 200, Bars: 5
```

## Important Notes

- The model needs significant training (50k+ timesteps) for good performance
- Always verify data consistency between training and production
- Use FAKEPACA for 24/7 testing, NVDA for market hours only
- Monitor the Alpaca paper trading dashboard for executed orders

---

**Next Steps:**
1. Train on multiple days of data
2. Implement proper backtesting
3. Add performance metrics tracking
4. Scale to multiple symbols