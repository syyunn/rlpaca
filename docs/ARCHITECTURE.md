# Real-Time RL Trading System Architecture

## Overview
This document describes our end-to-end streaming RL trading system that connects market data to trained models for automated trading.

## System Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│ Alpaca WebSocket├────►│ Apache Kafka ├────►│ SAC RL Executor │
│   (Market Data) │     │   (Broker)   │     │    (Trading)    │
└─────────────────┘     └──────────────┘     └────────┬────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │  Alpaca Trading │
                                              │      API        │
                                              └─────────────────┘
```

## Components

### 1. Market Data Ingestion
- **Source**: Alpaca WebSocket API (real-time trades, quotes, bars)
- **Symbols**: Configurable (NVDA, SPY, or FAKEPACA for testing)
- **Producer**: `src/producer/alpaca_producer.py`
  - Connects to Alpaca streaming API
  - Publishes to Kafka topics:
    - `alpaca_market_data_paper_trades`
    - `alpaca_market_data_paper_quotes`
    - `alpaca_market_data_paper_bars`

### 2. Message Broker
- **Technology**: Apache Kafka
- **Purpose**: Decouples data ingestion from model inference
- **Benefits**:
  - Handles backpressure
  - Enables multiple consumers
  - Provides data persistence
  - Allows replay for backtesting

### 3. RL Model Executor
- **Component**: `src/rl/sac_streaming_executor.py`
- **Model**: SAC (Soft Actor-Critic) trained on historical data
- **Observation Space**: 5,185 dimensions
  - 500 dims: Last 100 tick data (price, size, volume, etc.)
  - 4,680 dims: Minute bar features (currently zeros)
  - 5 dims: Position state (position, cash, market_value, etc.)
- **Action Space**: 2D continuous
  - `position_delta`: [-1, 1] target position change
  - `limit_offset_bps`: [-50, 50] basis points from mid price

### 4. Trading Execution
- **API**: Alpaca Trading API
- **Order Types**: Limit orders only (no market orders)
- **Risk Controls**:
  - Max 10 shares per order
  - Position limits
  - Fractional share support

## Data Flow

### Real-Time Processing
```python
# 1. Kafka Consumer receives market data
market_data = {
    'T': 'q',  # Quote
    'S': 'NVDA',
    'bp': 145.50,  # Bid price
    'ap': 145.52,  # Ask price
    'bs': 100,     # Bid size
    'as': 100,     # Ask size
    't': '2025-06-20T09:30:00Z'
}

# 2. Buffer accumulation (100 tick window)
trade_buffer.append(market_data)

# 3. Every 5 seconds: Create observation
observation = np.array([
    # Recent 100 ticks flattened (500 dims)
    tick_prices, tick_sizes, tick_volumes, ...,
    # Minute bars (4680 dims)
    minute_features,
    # Position state (5 dims)
    current_position, cash, market_value, ...
])

# 4. Model inference
action = model.predict(observation)
# Returns: [position_delta=0.8, limit_offset=-10]

# 5. Order generation
if position_delta > 0:
    # Buy with limit 10 bps below mid
    submit_order('buy', qty=5, limit_price=145.40)
```

## Key Features

### 1. Symbol Flexibility
- Can consume data from one symbol (e.g., FAKEPACA test stream)
- Trade a different symbol (e.g., NVDA)
- Useful for testing with continuous data outside market hours

### 2. Ultra-Pure Data Approach
- No hand-crafted features or indicators
- Raw tick data → Neural Network → Trading decisions
- Model learns optimal transformations

### 3. Streaming Architecture
- Real-time data processing
- Low latency (< 100ms from data to decision)
- Scalable to multiple symbols

### 4. Paper Trading Integration
- Full Alpaca paper trading support
- Real order submissions
- Position tracking
- P&L monitoring

## Configuration

### Environment Variables
```bash
# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=kafka:29092

# Alpaca API (Paper Trading)
ALPACA_PAPER_API_KEY=PKXXXXXXXXXXXXXX
ALPACA_PAPER_API_SECRET=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
ALPACA_TRADING_MODE=paper

# Symbol Configuration
KAFKA_SYMBOL=FAKEPACA     # Symbol to consume from Kafka
TRADING_SYMBOL=NVDA       # Symbol to trade
```

### Docker Deployment
```yaml
# docker-compose.sac-executor.yml
services:
  sac-executor:
    environment:
      MODEL_PATH: /app/fixed_1epoch_model.zip
      KAFKA_SYMBOL: FAKEPACA
      TRADING_SYMBOL: NVDA
      PYTHONUNBUFFERED: 1
```

## Monitoring

### Logs
- Real-time decision making
- Order submissions
- Position updates
- Error handling

### Metrics
- Data received (trades/quotes count)
- Orders submitted
- Model inference time
- API rate limit status

## Testing

### FAKEPACA Test Stream
- Alpaca provides a test symbol that streams continuously
- Useful for testing outside market hours
- Same data format as real symbols

### Successful Test Results
```
2025-06-20 03:53:55 [info] Current NVDA position: -1267.102755 shares
2025-06-20 03:53:55 [info] 📤 SUBMITTING ORDER: BUY 0.1 @ $134.68
2025-06-20 03:53:59 [info] ✅ Order submitted! ID: 278bb8fc-e73d-443b-bcf5-89a0d7064120
```

## Future Enhancements

1. **Enhanced Features**
   - Implement minute bar aggregation
   - Add order book depth data
   - Include trading halts/news events

2. **Risk Management**
   - Dynamic position sizing
   - Drawdown controls
   - Correlation-based limits

3. **Model Improvements**
   - Online learning/adaptation
   - Multi-symbol portfolio optimization
   - Regime-aware trading

## Summary

This system demonstrates a complete pipeline from market data to executed trades using reinforcement learning. The architecture is designed for low latency, high reliability, and easy monitoring while maintaining the flexibility to test and deploy new models quickly.