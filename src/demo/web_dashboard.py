#!/usr/bin/env python3
"""
RLpaca Demo Web Dashboard
Real-time visualization of TD3 trading model execution
"""

import json
import time
import os
import threading
from datetime import datetime
from collections import deque
from flask import Flask, render_template_string, jsonify
from kafka import KafkaConsumer
import numpy as np

app = Flask(__name__)

# Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:19092')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'nvda-quotes')
INITIAL_CAPITAL = 100000.0
DECISION_INTERVAL = 20  # Make decision every 20 ticks (~2 seconds)

# Shared state
state = {
    'ticks': deque(maxlen=100),
    'trades': deque(maxlen=50),
    'portfolio': {
        'cash': INITIAL_CAPITAL,
        'position': 0,
        'avg_cost': 0,
        'total_pnl': 0,
        'total_value': INITIAL_CAPITAL,
        'returns_pct': 0
    },
    'current_price': 142.50,
    'bid': 142.50,
    'ask': 142.51,
    'ticks_processed': 0,
    'decisions_made': 0,
    'trades_executed': 0,
    'winning_trades': 0,
    'last_action': 0,
    'connected': False
}

# Lock for thread safety
state_lock = threading.Lock()

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>RLpaca TD3 Trading Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            min-height: 100vh;
            padding: 20px;
        }
        .header {
            text-align: center;
            padding: 20px;
            margin-bottom: 20px;
        }
        .header h1 {
            font-size: 2.5em;
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .header .subtitle {
            color: #888;
            margin-top: 5px;
        }
        .status-bar {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-bottom: 20px;
        }
        .status-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        .status-dot.connected { background: #00ff88; }
        .status-dot.disconnected { background: #ff4444; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .card h2 {
            font-size: 1.1em;
            color: #888;
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .price-display {
            text-align: center;
        }
        .price-main {
            font-size: 3em;
            font-weight: bold;
            color: #00d9ff;
        }
        .price-spread {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 10px;
        }
        .bid { color: #00ff88; }
        .ask { color: #ff6b6b; }
        .metric-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        .metric {
            text-align: center;
            padding: 10px;
            background: rgba(0,0,0,0.2);
            border-radius: 10px;
        }
        .metric-value {
            font-size: 1.8em;
            font-weight: bold;
        }
        .metric-label {
            font-size: 0.8em;
            color: #888;
            margin-top: 5px;
        }
        .positive { color: #00ff88; }
        .negative { color: #ff6b6b; }
        .neutral { color: #00d9ff; }
        .action-display {
            text-align: center;
            padding: 20px;
        }
        .action-value {
            font-size: 2.5em;
            font-weight: bold;
        }
        .action-bar {
            height: 20px;
            background: linear-gradient(90deg, #ff6b6b 0%, #888 50%, #00ff88 100%);
            border-radius: 10px;
            margin: 15px 0;
            position: relative;
        }
        .action-indicator {
            position: absolute;
            top: -5px;
            width: 30px;
            height: 30px;
            background: #fff;
            border-radius: 50%;
            transform: translateX(-50%);
            box-shadow: 0 0 10px rgba(0,0,0,0.5);
            transition: left 0.3s ease;
        }
        .trades-list {
            max-height: 300px;
            overflow-y: auto;
        }
        .trade-item {
            display: flex;
            justify-content: space-between;
            padding: 10px;
            margin: 5px 0;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            font-size: 0.9em;
        }
        .trade-buy { border-left: 3px solid #00ff88; }
        .trade-sell { border-left: 3px solid #ff6b6b; }
        .stats-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .stats-row:last-child { border-bottom: none; }
        .chart-placeholder {
            height: 150px;
            background: rgba(0,0,0,0.2);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #888;
        }
        .price-history {
            display: flex;
            align-items: flex-end;
            height: 100px;
            gap: 2px;
            padding: 10px;
            background: rgba(0,0,0,0.2);
            border-radius: 10px;
        }
        .price-bar {
            flex: 1;
            background: linear-gradient(180deg, #00d9ff, #0066ff);
            border-radius: 2px 2px 0 0;
            min-height: 5px;
            transition: height 0.3s ease;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>RLpaca TD3 Trading Dashboard</h1>
        <div class="subtitle">Real-time Reinforcement Learning Trading Execution</div>
    </div>

    <div class="status-bar">
        <div class="status-item">
            <div class="status-dot" id="kafka-status"></div>
            <span id="connection-text">Connecting to Kafka...</span>
        </div>
        <div class="status-item">
            <span>Ticks: <strong id="tick-count">0</strong></span>
        </div>
        <div class="status-item">
            <span>Decisions: <strong id="decision-count">0</strong></span>
        </div>
    </div>

    <div class="dashboard">
        <!-- Price Card -->
        <div class="card">
            <h2>NVDA Price</h2>
            <div class="price-display">
                <div class="price-main" id="current-price">$142.50</div>
                <div class="price-spread">
                    <span class="bid">Bid: $<span id="bid-price">142.49</span></span>
                    <span class="ask">Ask: $<span id="ask-price">142.51</span></span>
                </div>
            </div>
            <div class="price-history" id="price-chart"></div>
        </div>

        <!-- Portfolio Card -->
        <div class="card">
            <h2>Portfolio</h2>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-value neutral" id="total-value">$100,000</div>
                    <div class="metric-label">Total Value</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="returns-pct">0.00%</div>
                    <div class="metric-label">Returns</div>
                </div>
                <div class="metric">
                    <div class="metric-value neutral" id="cash">$100,000</div>
                    <div class="metric-label">Cash</div>
                </div>
                <div class="metric">
                    <div class="metric-value neutral" id="position">0</div>
                    <div class="metric-label">Shares</div>
                </div>
            </div>
        </div>

        <!-- Model Action Card -->
        <div class="card">
            <h2>TD3 Model Action</h2>
            <div class="action-display">
                <div class="action-value" id="action-text">HOLD</div>
                <div class="action-bar">
                    <div class="action-indicator" id="action-indicator" style="left: 50%;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; color: #888; font-size: 0.8em;">
                    <span>SELL (-1)</span>
                    <span>HOLD (0)</span>
                    <span>BUY (+1)</span>
                </div>
            </div>
            <div class="metric-grid" style="margin-top: 15px;">
                <div class="metric">
                    <div class="metric-value" id="total-pnl">$0.00</div>
                    <div class="metric-label">Total P&L</div>
                </div>
                <div class="metric">
                    <div class="metric-value neutral" id="win-rate">0%</div>
                    <div class="metric-label">Win Rate</div>
                </div>
            </div>
        </div>

        <!-- Trades Card -->
        <div class="card">
            <h2>Recent Trades</h2>
            <div class="trades-list" id="trades-list">
                <div style="text-align: center; color: #888; padding: 20px;">
                    Waiting for trades...
                </div>
            </div>
        </div>
    </div>

    <script>
        let priceHistory = [];

        function updateDashboard() {
            fetch('/api/state')
                .then(response => response.json())
                .then(data => {
                    // Connection status
                    const statusDot = document.getElementById('kafka-status');
                    const connectionText = document.getElementById('connection-text');
                    if (data.connected) {
                        statusDot.className = 'status-dot connected';
                        connectionText.textContent = 'Connected to Kafka';
                    } else {
                        statusDot.className = 'status-dot disconnected';
                        connectionText.textContent = 'Connecting to Kafka...';
                    }

                    // Counts
                    document.getElementById('tick-count').textContent = data.ticks_processed.toLocaleString();
                    document.getElementById('decision-count').textContent = data.decisions_made;

                    // Price
                    document.getElementById('current-price').textContent = '$' + data.current_price.toFixed(2);
                    document.getElementById('bid-price').textContent = data.bid.toFixed(2);
                    document.getElementById('ask-price').textContent = data.ask.toFixed(2);

                    // Price chart
                    priceHistory.push(data.current_price);
                    if (priceHistory.length > 50) priceHistory.shift();
                    updatePriceChart();

                    // Portfolio
                    document.getElementById('total-value').textContent = '$' + data.portfolio.total_value.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
                    document.getElementById('cash').textContent = '$' + data.portfolio.cash.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
                    document.getElementById('position').textContent = data.portfolio.position;

                    const returnsPct = document.getElementById('returns-pct');
                    returnsPct.textContent = (data.portfolio.returns_pct >= 0 ? '+' : '') + data.portfolio.returns_pct.toFixed(2) + '%';
                    returnsPct.className = 'metric-value ' + (data.portfolio.returns_pct >= 0 ? 'positive' : 'negative');

                    // P&L
                    const pnlEl = document.getElementById('total-pnl');
                    pnlEl.textContent = (data.portfolio.total_pnl >= 0 ? '+$' : '-$') + Math.abs(data.portfolio.total_pnl).toFixed(2);
                    pnlEl.className = 'metric-value ' + (data.portfolio.total_pnl >= 0 ? 'positive' : 'negative');

                    // Win rate
                    const winRate = data.trades_executed > 0 ? (data.winning_trades / data.trades_executed * 100) : 0;
                    document.getElementById('win-rate').textContent = winRate.toFixed(0) + '%';

                    // Model action
                    const actionText = document.getElementById('action-text');
                    const actionIndicator = document.getElementById('action-indicator');
                    const action = data.last_action;

                    if (action > 0.1) {
                        actionText.textContent = 'BUY';
                        actionText.className = 'action-value positive';
                    } else if (action < -0.1) {
                        actionText.textContent = 'SELL';
                        actionText.className = 'action-value negative';
                    } else {
                        actionText.textContent = 'HOLD';
                        actionText.className = 'action-value neutral';
                    }
                    actionIndicator.style.left = ((action + 1) / 2 * 100) + '%';

                    // Trades list
                    if (data.trades && data.trades.length > 0) {
                        const tradesHtml = data.trades.slice().reverse().map(trade => `
                            <div class="trade-item trade-${trade.side.toLowerCase()}">
                                <span>${trade.side} ${trade.shares} @ $${trade.price.toFixed(2)}</span>
                                <span>${trade.pnl !== undefined ? (trade.pnl >= 0 ? '+' : '') + '$' + trade.pnl.toFixed(2) : ''}</span>
                            </div>
                        `).join('');
                        document.getElementById('trades-list').innerHTML = tradesHtml;
                    }
                })
                .catch(err => console.error('Error fetching state:', err));
        }

        function updatePriceChart() {
            if (priceHistory.length < 2) return;

            const min = Math.min(...priceHistory);
            const max = Math.max(...priceHistory);
            const range = max - min || 1;

            const chartHtml = priceHistory.map(price => {
                const height = ((price - min) / range * 80) + 20;
                return `<div class="price-bar" style="height: ${height}%"></div>`;
            }).join('');

            document.getElementById('price-chart').innerHTML = chartHtml;
        }

        // Update every 500ms
        setInterval(updateDashboard, 500);
        updateDashboard();
    </script>
</body>
</html>
'''

class MockTD3Model:
    def __init__(self):
        self.name = "TD3"
        self.trade_count = 0
        self.last_action = 0
        self.position_bias = 0  # Track tendency

    def predict(self, observation):
        # More active trading simulation
        if len(observation) > 10:
            recent_prices = observation[:40:4]
            if len(recent_prices) > 1:
                momentum = float(np.mean(np.diff(recent_prices)))
            else:
                momentum = 0
        else:
            momentum = 0

        # TD3-style action with more variance for demo
        base_action = np.tanh(momentum * 200)  # Amplify momentum signal
        noise = np.random.normal(0, 0.3)  # More exploration

        # Add some randomness to make it more interesting
        random_impulse = np.random.choice([-0.5, 0, 0.5], p=[0.2, 0.5, 0.3])

        action = float(np.clip(base_action + noise + random_impulse, -1, 1))

        # Lower threshold for action
        if abs(action) < 0.05:
            action = 0.0

        self.trade_count += 1
        self.last_action = float(action)
        return action


def kafka_consumer_thread():
    """Background thread that consumes Kafka messages and updates state"""
    global state

    model = MockTD3Model()
    tick_buffer = deque(maxlen=100)

    print("Starting Kafka consumer thread...")

    max_retries = 60
    consumer = None

    for attempt in range(max_retries):
        try:
            consumer = KafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                consumer_timeout_ms=1000
            )
            print(f"Connected to Kafka at {KAFKA_BOOTSTRAP_SERVERS}")
            with state_lock:
                state['connected'] = True
            break
        except Exception as e:
            print(f"Waiting for Kafka... ({attempt + 1}/{max_retries}): {e}")
            time.sleep(2)

    if not consumer:
        print("Could not connect to Kafka")
        return

    while True:
        try:
            for message in consumer:
                tick = message.value
                tick_buffer.append(tick)

                with state_lock:
                    state['ticks_processed'] += 1
                    state['bid'] = float(tick.get('bp', 142.50))
                    state['ask'] = float(tick.get('ap', 142.51))
                    state['current_price'] = float((state['bid'] + state['ask']) / 2)
                    state['ticks'].append({
                        'price': float(state['current_price']),
                        'time': tick.get('t', '')
                    })

                # Make trading decision every N ticks
                if state['ticks_processed'] % DECISION_INTERVAL == 0 and len(tick_buffer) >= 10:
                    current_price = state['current_price']

                    # Create observation
                    obs = []
                    for t in list(tick_buffer):
                        obs.extend([
                            t.get('bp', current_price),
                            t.get('ap', current_price),
                            t.get('bs', 100) / 1000,
                            t.get('as', 100) / 1000
                        ])
                    while len(obs) < 400:
                        obs.extend([current_price, current_price, 0.1, 0.1])

                    # Portfolio features
                    with state_lock:
                        market_value = state['portfolio']['position'] * current_price
                        total_value = state['portfolio']['cash'] + market_value
                        position_pct = market_value / total_value if total_value > 0 else 0

                        obs.extend([
                            position_pct,
                            state['portfolio']['cash'] / INITIAL_CAPITAL,
                            state['portfolio']['total_pnl'] / INITIAL_CAPITAL,
                            state['portfolio']['position'] / 100,
                            (current_price - 142.5) / 10
                        ])

                    observation = np.array(obs[:405], dtype=np.float32)
                    action = model.predict(observation)

                    with state_lock:
                        state['decisions_made'] += 1
                        state['last_action'] = action

                        # Execute trade
                        trade = execute_action(action, current_price)
                        if trade:
                            state['trades'].append(trade)
                            state['trades_executed'] += 1
                            if trade.get('pnl', 0) > 0:
                                state['winning_trades'] += 1

                        # Update portfolio value
                        market_value = state['portfolio']['position'] * current_price
                        state['portfolio']['total_value'] = state['portfolio']['cash'] + market_value
                        state['portfolio']['returns_pct'] = ((state['portfolio']['total_value'] / INITIAL_CAPITAL) - 1) * 100

        except Exception as e:
            print(f"Kafka consumer error: {e}")
            time.sleep(1)


def execute_action(action, current_price):
    """Execute trading action and update portfolio"""
    global state

    if abs(action) < 0.1:
        return None

    max_trade_value = state['portfolio']['cash'] * 0.1
    trade_value = abs(action) * max_trade_value

    if action > 0:  # BUY
        shares_to_buy = int(trade_value / current_price)
        if shares_to_buy > 0 and state['portfolio']['cash'] >= shares_to_buy * current_price:
            cost = shares_to_buy * current_price
            state['portfolio']['cash'] -= cost
            if state['portfolio']['position'] > 0:
                state['portfolio']['avg_cost'] = (
                    (state['portfolio']['avg_cost'] * state['portfolio']['position']) + cost
                ) / (state['portfolio']['position'] + shares_to_buy)
            else:
                state['portfolio']['avg_cost'] = current_price
            state['portfolio']['position'] += shares_to_buy
            return {'side': 'BUY', 'shares': shares_to_buy, 'price': current_price, 'value': cost}

    else:  # SELL
        shares_to_sell = min(int(trade_value / current_price), state['portfolio']['position'])
        if shares_to_sell > 0:
            revenue = shares_to_sell * current_price
            pnl = (current_price - state['portfolio']['avg_cost']) * shares_to_sell
            state['portfolio']['total_pnl'] += pnl
            state['portfolio']['cash'] += revenue
            state['portfolio']['position'] -= shares_to_sell
            return {'side': 'SELL', 'shares': shares_to_sell, 'price': current_price, 'value': revenue, 'pnl': pnl}

    return None


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/state')
def get_state():
    with state_lock:
        return jsonify({
            'ticks_processed': int(state['ticks_processed']),
            'decisions_made': int(state['decisions_made']),
            'trades_executed': int(state['trades_executed']),
            'winning_trades': int(state['winning_trades']),
            'current_price': float(state['current_price']),
            'bid': float(state['bid']),
            'ask': float(state['ask']),
            'last_action': float(state['last_action']),
            'connected': state['connected'],
            'portfolio': {
                'cash': float(state['portfolio']['cash']),
                'position': int(state['portfolio']['position']),
                'avg_cost': float(state['portfolio']['avg_cost']),
                'total_pnl': float(state['portfolio']['total_pnl']),
                'total_value': float(state['portfolio']['total_value']),
                'returns_pct': float(state['portfolio']['returns_pct'])
            },
            'trades': [{
                'side': t['side'],
                'shares': int(t['shares']),
                'price': float(t['price']),
                'value': float(t['value']),
                'pnl': float(t.get('pnl', 0))
            } for t in state['trades']],
            'ticks': [{'price': float(t['price']), 'time': t['time']} for t in list(state['ticks'])[-20:]]
        })


def main():
    # Start Kafka consumer in background thread
    consumer_thread = threading.Thread(target=kafka_consumer_thread, daemon=True)
    consumer_thread.start()

    print("\n" + "=" * 60)
    print("RLpaca TD3 Trading Dashboard")
    print("=" * 60)
    print("Open http://localhost:5000 in your browser")
    print("=" * 60 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)


if __name__ == '__main__':
    main()
