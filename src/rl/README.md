# SAC Streaming Executor

Real-time reinforcement learning trading executor that consumes tick data from Kafka and makes trading decisions using a Soft Actor-Critic (SAC) model.

## What It Does

Every 5 seconds, the executor:
1. **Reads** latest market data from Kafka (quotes & trades)
2. **Constructs** a 5,185-dimensional observation vector
3. **Predicts** trading action using pre-trained SAC model
4. **Executes** trades via Alpaca API based on model output

## State Space: 5,185 Dimensions

```
┌─────────────────────────────────────────┐
│ Recent Ticks (500 dims)                 │
│ - Last 100 ticks × 5 features each      │
│ - Bid/Ask prices, sizes, timestamps     │
├─────────────────────────────────────────┤
│ Minute Bars (4,680 dims)                │
│ - 390 minutes × 12 features each        │  
│ - OHLCV, VWAP, returns, time features   │
├─────────────────────────────────────────┤
│ Position State (5 dims)                 │
│ - Current position, cash, market value  │
│ - Portfolio return, day progress        │
└─────────────────────────────────────────┘
```

## Action Space: 2 Dimensions

```
action[0]: Position Delta [-1.0 to 1.0]
           -1.0 = Sell everything
            0.0 = Hold
            1.0 = Buy maximum allowed

action[1]: Limit Offset [-10.0 to 10.0] 
           -10 bps = Demand discount
             0 bps = Trade at mid-price  
           +10 bps = Pay premium
```

### Why Maximum Allowed Position?

The "maximum allowed" design is crucial for practical deployment:

1. **Risk Management**: Prevents the model from taking excessive positions that could blow up the account. The maximum is a hard safety limit.

2. **Capital Efficiency**: By expressing actions as fractions of maximum, the model learns relative position sizing rather than absolute amounts. This makes the policy transferable across different account sizes.

3. **Regulatory Compliance**: Many brokers and regulations impose position limits. This design ensures we never exceed them.

4. **Training Stability**: During RL training, bounded actions prevent the agent from exploring catastrophic position sizes that would end episodes prematurely.

5. **Real-world Constraints**: In production, you have finite buying power. The model learns to work within realistic constraints rather than assuming infinite capital.

Example: If max_position=1000 shares and current_position=200:
- action[0]=1.0 → Buy 800 shares (to reach max)
- action[0]=0.5 → Buy 400 shares  
- action[0]=-0.5 → Sell 100 shares
- action[0]=-1.0 → Sell all 200 shares

This approach ensures the model's decisions are always executable and safe in live trading.

### Update: Flexible Capital-Based Deployment

The system now uses **percentage-based position sizing** instead of fixed limits:

1. **No Fixed Position Limit**: Removed the 1000-share maximum constraint
2. **Capital-Based Sizing**: Actions represent % of available buying power
3. **Gradual Scaling**: Deploy with small budgets ($1k) and scale up as confidence grows

Example with different budgets:
- **$1,000 account**: action=1.0 → Buy ~7 shares of NVDA
- **$10,000 account**: action=1.0 → Buy ~74 shares  
- **$100,000 account**: action=1.0 → Buy ~740 shares

The same trained model adapts to any account size automatically!

## Example Decision Flow

```
Time: 10:35:20
Data: 47 quotes, 23 trades in buffer
State: [134.56, 134.58, 100, 200, ..., 0.5, 50000, 67.28, 0.0001, 0.26]
       └─ Latest quote data ─┘         └─ Position state ─┘

Model Output: [0.75, 2.0]
Interpretation: Buy 75% of max size, pay up to 2 bps above mid

Order: BUY 7.5 shares of NVDA @ $134.59 limit
```

## Key Features

- **Long-only**: No short selling (enforced constraint)
- **Fractional shares**: Can trade 0.000001+ shares (Alpaca minimum)
- **Capital-based sizing**: No fixed position limits - uses % of available capital
- **24/7 capable**: Supports extended hours trading
- **Sub-second latency**: <100ms from data to decision
- **Budget flexible**: Same model works from $1k to $1M+ accounts

## Configuration

The executor uses these environment variables:
- `KAFKA_SYMBOL`: Symbol to read from Kafka (e.g., FAKEPACA)
- `TRADING_SYMBOL`: Symbol to trade on Alpaca (e.g., NVDA)
- `MODEL_PATH`: Path to trained SAC model
- `KAFKA_BOOTSTRAP_SERVERS`: Kafka connection string

## Files

- `sac_streaming_executor.py` - Main executor implementation
- `realistic_offline_env.py` - Training environment matching production
- `train_single_day.py` - Model training script
- `kafka_data_bridge.py` - Kafka consumer utilities

## Quick Start

```bash
# Ensure Kafka is running with market data
docker-compose -f docker-compose.minimal-alpaca.yml up -d

# Start the executor
docker-compose -f docker-compose.sac-executor.yml up -d

# Watch it trade
docker-compose -f docker-compose.sac-executor.yml logs -f
```

## Model Details

- **Algorithm**: Soft Actor-Critic (SAC)
- **Network**: MLP with [256, 128, 64] hidden units
- **Training objective**: 1% daily returns with risk control
- **Training data**: Historical tick data via TimescaleDB