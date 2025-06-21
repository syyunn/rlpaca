#!/usr/bin/env python3
"""
SAC Streaming Executor - Runs inside Docker with Kafka access
Consumes real-time market data and executes trades
"""

import os
import sys
import json
import time
import threading
from datetime import datetime
from collections import deque
import numpy as np
from kafka import KafkaConsumer
import alpaca_trade_api as tradeapi
from stable_baselines3 import SAC
import structlog

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config.trading_config import default_config

logger = structlog.get_logger()

class SACStreamingExecutor:
    def __init__(self):
        self.kafka_symbol = os.getenv('KAFKA_SYMBOL', 'FAKEPACA')  # Symbol to consume from Kafka
        self.trading_symbol = os.getenv('TRADING_SYMBOL', 'NVDA')  # Symbol to trade
        self.model_path = os.getenv('MODEL_PATH', '/app/fixed_1epoch_model.zip')
        self.running = True
        
        # Kafka settings - internal Docker network
        self.kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
        self.trade_topic = 'alpaca_market_data_paper_trades'
        self.quote_topic = 'alpaca_market_data_paper_quotes'
        self.bars_topic = 'alpaca_market_data_paper_bars'
        
        # Initialize components
        self._init_model()
        self._init_alpaca()
        self._init_buffers()
        
        # Decision making
        self.last_decision_time = time.time()
        self.decision_interval = 5  # seconds
        self.orders_submitted = []
        self.data_received = {'trades': 0, 'quotes': 0, 'bars': 0}
        
        # Trading constraints
        self.long_only = True  # NO SHORT SELLING
        
        logger.info(f"Initialized SAC Streaming Executor")
        logger.info(f"Kafka Symbol: {self.kafka_symbol}")
        logger.info(f"Trading Symbol: {self.trading_symbol}")
        logger.info(f"Model: {self.model_path}")
        logger.info(f"Kafka: {self.kafka_servers}")
        logger.info(f"Strategy: {'LONG-ONLY' if self.long_only else 'LONG/SHORT'}")
        logger.info("🚫 Short selling disabled - will only trade long positions")
        
    def _init_model(self):
        """Load trained SAC model"""
        logger.info(f"Loading model from {self.model_path}")
        try:
            self.model = SAC.load(self.model_path)
            logger.info("✅ Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
    def _init_alpaca(self):
        """Initialize Alpaca API"""
        self.api = tradeapi.REST(
            key_id=os.getenv('ALPACA_PAPER_API_KEY'),
            secret_key=os.getenv('ALPACA_PAPER_API_SECRET'),
            base_url='https://paper-api.alpaca.markets'
        )
        
        try:
            account = self.api.get_account()
            logger.info(f"✅ Connected to Alpaca Paper Trading")
            logger.info(f"Cash: ${float(account.cash):,.2f}")
            logger.info(f"Buying Power: ${float(account.buying_power):,.2f}")
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            raise
        
    def _init_buffers(self):
        """Initialize data buffers"""
        self.trade_buffer = deque(maxlen=100)
        self.quote_buffer = deque(maxlen=100)
        self.minute_bars = deque(maxlen=390)  # Store minute bars for the day
        
    def consume_market_data(self):
        """Consume market data from Kafka"""
        try:
            consumer = KafkaConsumer(
                self.trade_topic,
                self.quote_topic,
                self.bars_topic,
                bootstrap_servers=[self.kafka_servers],
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                group_id=f'sac_executor_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            )
            
            logger.info(f"✅ Connected to Kafka")
            logger.info(f"Consuming from: {self.trade_topic}, {self.quote_topic}, {self.bars_topic}")
            logger.info("Waiting for market data...")
            
            for message in consumer:
                if not self.running:
                    break
                    
                data = message.value
                
                # Filter for our Kafka symbol
                if data.get('S') != self.kafka_symbol:
                    continue
                
                # Route to appropriate buffer
                if message.topic == self.trade_topic:
                    self.trade_buffer.append({
                        'price': data.get('p'),
                        'size': data.get('s'),
                        'timestamp': data.get('t')
                    })
                    self.data_received['trades'] += 1
                    
                elif message.topic == self.quote_topic:
                    self.quote_buffer.append({
                        'bid_price': data.get('bp'),
                        'ask_price': data.get('ap'),
                        'bid_size': data.get('bs'),
                        'ask_size': data.get('as'),
                        'timestamp': data.get('t')
                    })
                    self.data_received['quotes'] += 1
                    
                elif message.topic == self.bars_topic:
                    # Store minute bar data
                    self.minute_bars.append({
                        'timestamp': data.get('t'),
                        'open': data.get('o'),
                        'high': data.get('h'),
                        'low': data.get('l'),
                        'close': data.get('c'),
                        'volume': data.get('v'),
                        'trade_count': data.get('n', 0),
                        'vwap': data.get('vw', data.get('c'))  # Use close if no VWAP
                    })
                    self.data_received['bars'] += 1
                
                # Log data flow periodically
                total_data = sum(self.data_received.values())
                if total_data % 100 == 0:
                    logger.info(f"Data received - Trades: {self.data_received['trades']}, Quotes: {self.data_received['quotes']}, Bars: {self.data_received['bars']}")
                    
                # Make trading decision at intervals
                self._check_trading_decision()
                    
        except Exception as e:
            logger.error(f"Kafka consumer error: {e}")
            raise
            
    def _check_trading_decision(self):
        """Check if it's time to make a trading decision"""
        current_time = time.time()
        
        if current_time - self.last_decision_time < self.decision_interval:
            return
            
        if len(self.trade_buffer) == 0:
            return
            
        logger.info(f"\n{'='*60}")
        logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] DECISION TIME")
        logger.info(f"Data in buffers - Trades: {len(self.trade_buffer)}, Quotes: {len(self.quote_buffer)}")
        
        # Show latest prices
        if self.trade_buffer:
            latest_trade = self.trade_buffer[-1]
            logger.info(f"Latest trade: ${latest_trade['price']:.2f}")
        
        if self.quote_buffer:
            latest_quote = self.quote_buffer[-1]
            logger.info(f"Latest quote: Bid ${latest_quote['bid_price']:.2f} / Ask ${latest_quote['ask_price']:.2f}")
        
        # Create observation
        obs = self._create_observation()
        
        # Get model prediction
        action, _ = self.model.predict(obs, deterministic=True)
        logger.info(f"Model action: position_delta={action[0]:.3f}, limit_offset={action[1]:.1f} bps")
        
        # Submit order
        self._submit_order(action)
        
        self.last_decision_time = current_time
        
    def _create_observation(self):
        """Create observation matching training environment"""
        features = []
        
        # Recent trades (tick_buffer_size * tick_features dims)
        tick_matrix = np.zeros((default_config.TICK_BUFFER_SIZE, default_config.TICK_FEATURES))
        trades = list(self.trade_buffer)
        
        for i in range(min(default_config.TICK_BUFFER_SIZE, len(trades))):
            t = trades[-(i+1)]
            tick_matrix[i] = [
                t.get('price', 0),
                t.get('size', 0),
                t.get('price', 0) * t.get('size', 0),
                0,
                time.time()
            ]
        features.extend(tick_matrix.flatten())
        
        # Minute bars from Alpaca (minute_bars_count * minute_bar_features dims)
        minute_matrix = np.zeros((default_config.minute_bars_count, default_config.MINUTE_BAR_FEATURES))
        bars = list(self.minute_bars)
        
        for i in range(min(default_config.minute_bars_count, len(bars))):
            bar = bars[i]  # Use chronological order for bars
            minute_matrix[i] = [
                bar.get('open', 0),
                bar.get('high', 0),
                bar.get('low', 0),
                bar.get('close', 0),
                bar.get('volume', 0),
                bar.get('trade_count', 0),
                bar.get('vwap', bar.get('close', 0)),
                # Derived features to match training
                (bar.get('close', 0) - bar.get('open', 0)) / max(bar.get('open', 1), 0.001) * 100,  # return %
                bar.get('high', 0) - bar.get('low', 0),  # range
                i // 60,  # hour approximation
                i % 60,   # minute approximation
                i / 390   # normalized position in day
            ]
        features.extend(minute_matrix.flatten())
        
        # Position state (5 dims)
        try:
            positions = self.api.list_positions()
            nvda_pos = next((p for p in positions if p.symbol == self.trading_symbol), None)
            position = float(nvda_pos.qty) if nvda_pos else 0
            market_value = float(nvda_pos.market_value) if nvda_pos else 0
            account = self.api.get_account()
            cash = float(account.cash)
        except Exception as e:
            logger.warning(f"Failed to get position info: {e}")
            position = 0
            market_value = 0
            cash = 100000
            
        features.extend([position, cash, market_value, 0.0, len(self.trade_buffer)])
        
        return np.array(features, dtype=np.float32)
        
    def _submit_order(self, action):
        """Submit order to Alpaca"""
        position_delta = action[0]
        limit_offset_bps = action[1]
        
        if abs(position_delta) < 0.1:
            logger.info("No significant action")
            return
            
        try:
            # Use FAKEPACA prices directly - the model doesn't care about the symbol
            if self.quote_buffer:
                latest_quote = self.quote_buffer[-1]
                mid_price = (latest_quote['bid_price'] + latest_quote['ask_price']) / 2
            else:
                # Fallback to trade price
                latest_trade = self.trade_buffer[-1]
                mid_price = latest_trade['price']
            
            # Get current position
            positions = self.api.list_positions()
            nvda_pos = next((p for p in positions if p.symbol == self.trading_symbol), None)
            current_position = float(nvda_pos.qty) if nvda_pos else 0
            logger.info(f"Current {self.trading_symbol} position: {current_position} shares")
            
            # Calculate order details
            if position_delta > 0:  # Buy
                limit_price = mid_price * (1 + limit_offset_bps * 0.0001)
                side = 'buy'
                # Calculate max shares based ONLY on buying power (no artificial limits)
                account = self.api.get_account()
                max_shares = float(account.buying_power) / limit_price
                
                # Calculate order size based on model's action
                desired_qty = abs(position_delta) * max_shares
                
                # Apply only Alpaca's minimum constraints
                if desired_qty * limit_price < default_config.MIN_ORDER_VALUE:
                    # Skip orders below $1 minimum
                    logger.info(f"Order value ${desired_qty * limit_price:.2f} below Alpaca minimum ${default_config.MIN_ORDER_VALUE}")
                    return
                    
                qty = max(default_config.MIN_ORDER_SIZE, 
                         round(desired_qty, default_config.ORDER_SIZE_PRECISION))
            else:  # Sell
                # LONG-ONLY: Can only sell if we have a position
                if current_position <= 0:
                    logger.info("No position to sell (long-only strategy)")
                    return
                    
                limit_price = mid_price * (1 - abs(limit_offset_bps) * 0.0001)
                side = 'sell'
                # Sell percentage of current position
                desired_qty = abs(position_delta) * current_position
                
                # Apply only Alpaca's minimum constraints  
                if desired_qty * limit_price < default_config.MIN_ORDER_VALUE:
                    logger.info(f"Order value ${desired_qty * limit_price:.2f} below Alpaca minimum ${default_config.MIN_ORDER_VALUE}")
                    return
                    
                qty = max(default_config.MIN_ORDER_SIZE, 
                         round(desired_qty, default_config.ORDER_SIZE_PRECISION))
                
            logger.info(f"📤 SUBMITTING ORDER: {side.upper()} {qty} @ ${limit_price:.2f}")
            logger.info(f"   Mid price: ${mid_price:.2f}, Offset: {limit_offset_bps:.1f} bps")
            
            # Check market status
            clock = self.api.get_clock()
            if not clock.is_open:
                logger.warning("Market is closed - order will be queued")
            
            # Format price as string with exactly 2 decimal places
            formatted_price = f"{limit_price:.2f}"
            
            order = self.api.submit_order(
                symbol=self.trading_symbol,
                qty=float(round(qty, 6)),  # Convert to native Python float
                side=side,
                type='limit',
                time_in_force='day',
                limit_price=formatted_price,
                extended_hours=True
            )
            
            logger.info(f"✅ Order submitted! ID: {order.id}")
            self.orders_submitted.append(order.id)
            
            # Don't check status immediately to reduce API calls
            logger.info(f"Order will be checked on next cycle to reduce API calls")
                
        except Exception as e:
            logger.error(f"Order submission failed: {e}")
            
    def run(self):
        """Main execution loop"""
        logger.info("\n" + "="*60)
        logger.info("🚀 STARTING SAC STREAMING EXECUTOR")
        logger.info("="*60)
        logger.info(f"Kafka Symbol: {self.kafka_symbol}")
        logger.info(f"Trading Symbol: {self.trading_symbol}")
        logger.info(f"Decision interval: {self.decision_interval} seconds")
        logger.info(f"Model: {self.model_path}")
        
        # Check market status
        try:
            clock = self.api.get_clock()
            logger.info(f"Market is {'OPEN' if clock.is_open else 'CLOSED'}")
            logger.info(f"Next open: {clock.next_open}")
        except:
            pass
        
        logger.info("\n📡 Connecting to Kafka stream...")
        
        try:
            # Run consumer (blocking)
            self.consume_market_data()
            
        except KeyboardInterrupt:
            logger.info("\n👋 Shutting down...")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
        finally:
            self.running = False
            self._print_summary()
            
    def _print_summary(self):
        """Print session summary"""
        logger.info("\n" + "="*60)
        logger.info("📊 SESSION SUMMARY")
        logger.info("="*60)
        
        # Data received
        logger.info(f"Data received - Trades: {self.data_received['trades']}, Quotes: {self.data_received['quotes']}, Bars: {self.data_received['bars']}")
        
        # Orders
        logger.info(f"Orders submitted: {len(self.orders_submitted)}")
        if self.orders_submitted:
            try:
                recent_orders = self.api.list_orders(status='all', limit=10)
                for order in recent_orders[:5]:
                    if order.id in self.orders_submitted:
                        logger.info(f"  {order.side.upper()} {order.qty} @ ${order.limit_price} - {order.status}")
            except:
                pass
        
        # Positions
        try:
            positions = self.api.list_positions()
            if positions:
                logger.info("\nFinal positions:")
                for pos in positions:
                    logger.info(f"  {pos.symbol}: {pos.qty} shares @ ${pos.avg_entry_price}")
        except:
            pass
                
        logger.info(f"\n📊 Check Alpaca Dashboard:")
        logger.info(f"   https://app.alpaca.markets/paper/dashboard/overview")


if __name__ == "__main__":
    executor = SACStreamingExecutor()
    executor.run()