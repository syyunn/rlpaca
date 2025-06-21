# SAC Streaming Executor Architecture

## Overview

The SAC (Soft Actor-Critic) Streaming Executor is a production-ready reinforcement learning trading system that consumes real-time market data from Kafka and makes trading decisions every 5 seconds. Unlike traditional backtesting frameworks, this system operates on live streaming data with sub-second latency.

## Key Features

- **Real-time Decision Making**: Processes streaming tick data and makes trading decisions every 5 seconds
- **Kafka Integration**: Consumes from multiple market data topics (quotes, trades, bars)
- **Long-only Strategy**: Enforces position constraints to prevent short selling
- **Fractional Shares**: Supports trading fractional shares via Alpaca API
- **Production Ready**: Containerized with proper error handling and monitoring

## State Space (Observation)

The RL model observes a **5,185-dimensional** feature vector composed of three main components:

### 1. Recent Tick Data (500 dimensions)
- **Structure**: 100 ticks × 5 features per tick
- **Features per tick**:
  - Bid price
  - Ask price  
  - Bid size
  - Ask size
  - Timestamp
- **Ordering**: Most recent tick first (reverse chronological)
- **Buffer**: Maintains rolling window of last 100 ticks

### 2. Minute Bar History (4,680 dimensions)
- **Structure**: 390 minute bars × 12 features per bar
- **Features per bar**:
  - OHLCV data (open, high, low, close, volume)
  - Trade count
  - VWAP (volume-weighted average price)
  - Intrabar return percentage
  - High-low range
  - Hour and minute timestamps
  - Normalized position in trading day
- **Coverage**: Full trading day (9:30 AM - 4:00 PM ET = 390 minutes)

### 3. Position State (5 dimensions)
- Current position (number of shares)
- Available cash
- Position market value
- Portfolio return (% gain/loss from initial capital)
- Trading progress (% of day completed)

### State Vector Structure
```python
observation = np.array([
    # Tick data (500 dims)
    bid_1, ask_1, bid_size_1, ask_size_1, timestamp_1,  # Most recent tick
    bid_2, ask_2, bid_size_2, ask_size_2, timestamp_2,  # Second most recent
    ... (98 more ticks)
    
    # Minute bars (4,680 dims)
    open_1, high_1, low_1, close_1, volume_1, trade_count_1, vwap_1, return_1, range_1, hour_1, minute_1, position_1,
    ... (389 more bars)
    
    # Position state (5 dims)
    position, cash, market_value, portfolio_return, day_progress
])
```

## Action Space

The model outputs a **2-dimensional continuous action**:

### Action Components
1. **Position Delta** (`action[0]`): Range [-1.0, 1.0]
   - Represents desired change in position as fraction of maximum allowed
   - Positive values = buy signal
   - Negative values = sell signal
   - Magnitude indicates confidence/size

2. **Limit Order Offset** (`action[1]`): Range [-10.0, 10.0] 
   - Basis points offset from mid-price for limit orders
   - Positive = willing to pay above mid (aggressive buy)
   - Negative = demand discount below mid (passive buy)
   - Used to balance execution probability vs price improvement

### Action Interpretation
```python
# Example action: [0.8, 2.5]
# Interpretation: 
# - Strong buy signal (80% of max position size)
# - Willing to pay 2.5 basis points above mid-price

# Example action: [-0.3, -5.0]
# Interpretation:
# - Moderate sell signal (30% of current position)
# - Only sell if can get 5 basis points below mid-price
```

## Trading Constraints

### Position Sizing
- **Long-only**: No short selling allowed
- **Capital-based**: Position size = action × available buying power
- **No fixed limits**: Model has full freedom within capital constraints
- **Fractional shares**: Minimum 0.000001 shares (Alpaca limit)

### Risk Management
- Risk controlled via capital allocation (start small, scale up)
- Commission-free trading (Alpaca has $0 commissions)
- End-of-day position closure enforced during training
- Market/extended hours trading supported

### Flexible Deployment Strategy
1. **Week 1**: Deploy with $1,000 (max loss = $1,000)
2. **Week 2**: Scale to $5,000 if profitable
3. **Month 1**: Increase to $10,000-$50,000
4. **Month 2+**: Full capital deployment

The model automatically adjusts position sizes based on available capital.

## Data Flow

```
1. Kafka Topics → Consumer
   - alpaca_market_data_paper_quotes
   - alpaca_market_data_paper_trades
   
2. Data Buffering
   - Maintain rolling windows of recent data
   - Update buffers with each new message
   
3. Decision Cycle (every 5 seconds)
   - Construct 5,185-dim observation
   - Run SAC model inference
   - Get 2-dim action output
   
4. Order Execution
   - Calculate position size from action[0]
   - Set limit price using action[1] offset
   - Submit order via Alpaca API
   
5. Monitoring
   - Log all decisions and trades
   - Track portfolio performance
   - Report session statistics
```

## Model Architecture

The SAC model uses separate neural networks for the policy (actor) and value functions (critics):

- **Policy Network**: Maps 5,185-dim state → 2-dim action
- **Q-Networks**: Two critics for stability (Double Q-learning)
- **Activation**: ReLU throughout
- **Hidden Layers**: [256, 128, 64] for both policy and Q-networks

## Training Objective

The model was trained to:
1. Achieve 1% daily returns consistently
2. Minimize drawdowns and risk
3. Close all positions by end of day
4. Adapt to varying market conditions

## Performance Characteristics

- **Inference latency**: <100ms per decision
- **Decision frequency**: Every 5 seconds (4,680 decisions/day)
- **Data throughput**: ~1000 ticks/minute during active trading
- **Memory usage**: ~500MB for model and buffers

## Configuration

Key environment variables:
```bash
KAFKA_SYMBOL=FAKEPACA      # Symbol to consume from Kafka
TRADING_SYMBOL=NVDA        # Symbol to trade on Alpaca
MODEL_PATH=/app/model.zip  # Trained SAC model
KAFKA_BOOTSTRAP_SERVERS=kafka:29092
ALPACA_PAPER_API_KEY=xxx
ALPACA_PAPER_API_SECRET=xxx
```

## Deployment

```bash
# Start the executor
docker-compose -f docker-compose.sac-executor.yml up -d

# Monitor logs
docker-compose -f docker-compose.sac-executor.yml logs -f

# Check Alpaca dashboard
# https://app.alpaca.markets/paper/dashboard/overview
```

## Why This Architecture?

1. **Rich State Representation**: 5,185 dimensions capture market microstructure, price trends, and portfolio state
2. **Real-time Adaptation**: Streaming architecture responds to market changes within seconds
3. **Production Ready**: Kafka provides reliability, scalability, and replay capabilities
4. **Risk Controlled**: Long-only constraint and position limits prevent dangerous trades
5. **Broker Integration**: Direct API connection for real order execution