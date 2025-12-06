#!/usr/bin/env python3
"""
Demo RL Trading Executor
Shows TD3 model consuming Kafka tick data and making trading decisions
(Mock trades - no real orders submitted)
"""

import json
import time
import os
import numpy as np
from datetime import datetime
from collections import deque
from kafka import KafkaConsumer

# Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:19092')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'nvda-quotes')
MODEL_TYPE = os.getenv('MODEL_TYPE', 'TD3')  # Best performing algorithm
INITIAL_CAPITAL = 100000.0
DECISION_INTERVAL = 50  # Make decision every 50 ticks (~5 seconds)

# State dimensions (simplified for demo)
TICK_WINDOW = 100
STATE_DIM = TICK_WINDOW * 4 + 5  # 100 ticks * 4 features + 5 portfolio features

class MockTD3Model:
    """
    Mock TD3 model that simulates realistic trading behavior
    In production, this would load the actual trained model
    """
    def __init__(self):
        self.name = "TD3"
        self.trade_count = 0
        self.last_action = 0
        print(f"🧠 Loaded {self.name} model (demo mode)")

    def predict(self, observation):
        """
        Simulate TD3's deterministic policy
        Returns action in [-1, 1] range (position delta)
        """
        # Extract recent price momentum from observation
        if len(observation) > 10:
            recent_prices = observation[:40:4]  # Sample bid prices
            if len(recent_prices) > 1:
                momentum = np.mean(np.diff(recent_prices))
            else:
                momentum = 0
        else:
            momentum = 0

        # TD3-style deterministic action with momentum following
        # Add small noise for exploration (target policy smoothing)
        base_action = np.tanh(momentum * 100)  # Scale momentum to action
        noise = np.random.normal(0, 0.1)  # Small exploration noise
        action = np.clip(base_action + noise, -1, 1)

        # TD3 tends to be more decisive - avoid tiny positions
        if abs(action) < 0.15:
            action = 0

        self.trade_count += 1
        self.last_action = action

        return action

class DemoExecutor:
    def __init__(self):
        self.tick_buffer = deque(maxlen=TICK_WINDOW)
        self.model = MockTD3Model()

        # Portfolio state
        self.cash = INITIAL_CAPITAL
        self.position = 0  # Number of shares
        self.avg_cost = 0
        self.total_pnl = 0
        self.trades_executed = 0
        self.winning_trades = 0

        # Statistics
        self.start_time = time.time()
        self.ticks_processed = 0
        self.decisions_made = 0

    def create_observation(self, current_price):
        """Create state observation from tick buffer"""
        obs = []

        # Tick features (bid, ask, bid_size, ask_size)
        for tick in list(self.tick_buffer):
            obs.extend([
                tick.get('bp', current_price),
                tick.get('ap', current_price),
                tick.get('bs', 100) / 1000,  # Normalize
                tick.get('as', 100) / 1000
            ])

        # Pad if not enough ticks
        while len(obs) < TICK_WINDOW * 4:
            obs.extend([current_price, current_price, 0.1, 0.1])

        # Portfolio features
        market_value = self.position * current_price if self.position > 0 else 0
        total_value = self.cash + market_value
        position_pct = market_value / total_value if total_value > 0 else 0

        obs.extend([
            position_pct,  # Current position as % of portfolio
            self.cash / INITIAL_CAPITAL,  # Normalized cash
            self.total_pnl / INITIAL_CAPITAL,  # Normalized PnL
            self.position / 100,  # Normalized position
            (current_price - 142.5) / 10  # Normalized price deviation
        ])

        return np.array(obs[:STATE_DIM], dtype=np.float32)

    def execute_action(self, action, current_price):
        """Execute trading action (mock - no real orders)"""
        if abs(action) < 0.1:
            return None  # No trade

        # Calculate target position change
        max_trade_value = self.cash * 0.1  # Max 10% of cash per trade
        trade_value = abs(action) * max_trade_value

        if action > 0:  # BUY
            shares_to_buy = int(trade_value / current_price)
            if shares_to_buy > 0 and self.cash >= shares_to_buy * current_price:
                cost = shares_to_buy * current_price
                self.cash -= cost
                self.avg_cost = ((self.avg_cost * self.position) + cost) / (self.position + shares_to_buy) if self.position > 0 else current_price
                self.position += shares_to_buy
                self.trades_executed += 1
                return {'side': 'BUY', 'shares': shares_to_buy, 'price': current_price, 'value': cost}

        else:  # SELL
            shares_to_sell = min(int(trade_value / current_price), self.position)
            if shares_to_sell > 0:
                revenue = shares_to_sell * current_price
                pnl = (current_price - self.avg_cost) * shares_to_sell
                self.total_pnl += pnl
                self.cash += revenue
                self.position -= shares_to_sell
                self.trades_executed += 1
                if pnl > 0:
                    self.winning_trades += 1
                return {'side': 'SELL', 'shares': shares_to_sell, 'price': current_price, 'value': revenue, 'pnl': pnl}

        return None

    def print_status(self, tick, trade=None):
        """Print colorful status update"""
        current_price = (tick['bp'] + tick['ap']) / 2
        market_value = self.position * current_price
        total_value = self.cash + market_value
        returns_pct = ((total_value / INITIAL_CAPITAL) - 1) * 100

        print("\n" + "=" * 70)
        print(f"🤖 TD3 TRADING DECISION #{self.decisions_made}")
        print("=" * 70)
        print(f"📊 Market: NVDA ${tick['bp']:.2f} / ${tick['ap']:.2f} (spread: ${tick['ap']-tick['bp']:.3f})")
        print(f"🧠 Model Action: {self.model.last_action:+.3f} ({'BUY' if self.model.last_action > 0 else 'SELL' if self.model.last_action < 0 else 'HOLD'})")

        if trade:
            if trade['side'] == 'BUY':
                print(f"✅ EXECUTED: BUY {trade['shares']} shares @ ${trade['price']:.2f} = ${trade['value']:.2f}")
            else:
                pnl_str = f"+${trade['pnl']:.2f}" if trade['pnl'] >= 0 else f"-${abs(trade['pnl']):.2f}"
                pnl_emoji = "📈" if trade['pnl'] >= 0 else "📉"
                print(f"✅ EXECUTED: SELL {trade['shares']} shares @ ${trade['price']:.2f} = ${trade['value']:.2f} ({pnl_emoji} {pnl_str})")
        else:
            print(f"⏸️  NO TRADE (action below threshold)")

        print("-" * 70)
        print(f"💰 Portfolio Status:")
        print(f"   Cash:        ${self.cash:,.2f}")
        print(f"   Position:    {self.position} shares (${market_value:,.2f})")
        print(f"   Total Value: ${total_value:,.2f}")
        print(f"   Returns:     {returns_pct:+.2f}%")
        print(f"   Total P&L:   ${self.total_pnl:+,.2f}")

        win_rate = (self.winning_trades / self.trades_executed * 100) if self.trades_executed > 0 else 0
        print(f"📊 Statistics:")
        print(f"   Trades: {self.trades_executed} | Win Rate: {win_rate:.1f}%")
        print(f"   Ticks Processed: {self.ticks_processed:,}")
        print("=" * 70)

    def run(self):
        """Main execution loop"""
        print("\n" + "=" * 70)
        print("🚀 RLpaca Demo Trading Executor")
        print("=" * 70)
        print(f"🧠 Model: {MODEL_TYPE} (Best Sharpe: 1.79)")
        print(f"💰 Initial Capital: ${INITIAL_CAPITAL:,.2f}")
        print(f"📡 Kafka: {KAFKA_BOOTSTRAP_SERVERS}")
        print(f"📨 Topic: {KAFKA_TOPIC}")
        print(f"⏱️  Decision Interval: Every {DECISION_INTERVAL} ticks")
        print("=" * 70)

        # Connect to Kafka
        print("\n⏳ Connecting to Kafka...")
        max_retries = 30
        consumer = None

        for attempt in range(max_retries):
            try:
                consumer = KafkaConsumer(
                    KAFKA_TOPIC,
                    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                    auto_offset_reset='latest',
                    api_version=(2, 5, 0)
                )
                print(f"✅ Connected to Kafka!")
                break
            except Exception as e:
                print(f"⏳ Waiting for Kafka... ({attempt + 1}/{max_retries})")
                time.sleep(2)

        if not consumer:
            raise Exception("❌ Could not connect to Kafka")

        print("\n🎯 Waiting for tick data...\n")

        try:
            for message in consumer:
                tick = message.value
                self.tick_buffer.append(tick)
                self.ticks_processed += 1

                # Make trading decision every N ticks
                if self.ticks_processed % DECISION_INTERVAL == 0 and len(self.tick_buffer) >= 10:
                    self.decisions_made += 1
                    current_price = (tick['bp'] + tick['ap']) / 2

                    # Create observation and get model prediction
                    observation = self.create_observation(current_price)
                    action = self.model.predict(observation)

                    # Execute trade
                    trade = self.execute_action(action, current_price)

                    # Print status
                    self.print_status(tick, trade)

        except KeyboardInterrupt:
            print("\n\n🛑 Stopping executor...")
            elapsed = time.time() - self.start_time
            print(f"\n📊 Final Summary:")
            print(f"   Runtime: {elapsed:.1f} seconds")
            print(f"   Ticks Processed: {self.ticks_processed:,}")
            print(f"   Decisions Made: {self.decisions_made}")
            print(f"   Trades Executed: {self.trades_executed}")
            print(f"   Final P&L: ${self.total_pnl:+,.2f}")

def main():
    executor = DemoExecutor()
    executor.run()

if __name__ == '__main__':
    main()
