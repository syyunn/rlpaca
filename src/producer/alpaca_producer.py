import json
import os
import signal
import sys
import time
from datetime import datetime
from typing import Optional, List, Dict, Any
import threading

import structlog
import websocket
from kafka import KafkaProducer
from kafka.errors import KafkaError
from dotenv import load_dotenv
from prometheus_client import Counter, Gauge, Histogram, start_http_server

load_dotenv()

logger = structlog.get_logger()

# Prometheus metrics
messages_received = Counter('alpaca_messages_received_total', 'Total messages received from Alpaca', ['message_type'])
messages_sent = Counter('kafka_messages_sent_total', 'Total messages sent to Kafka', ['topic', 'status'])
websocket_connections = Counter('websocket_connections_total', 'Total WebSocket connection attempts', ['status'])
websocket_active = Gauge('websocket_connection_active', 'WebSocket connection status')
kafka_send_duration = Histogram('kafka_send_duration_seconds', 'Time taken to send message to Kafka')
message_lag = Histogram('message_processing_lag_ms', 'Lag between message timestamp and processing time')


class AlpacaWebSocketProducer:
    def __init__(self):
        # ⚠️  CRITICAL: Check trading mode first
        self.trading_mode = os.getenv("ALPACA_TRADING_MODE", "paper").lower()
        
        if self.trading_mode not in ["paper", "live"]:
            raise ValueError(f"Invalid ALPACA_TRADING_MODE: {self.trading_mode}. Must be 'paper' or 'live'")
        
        # Load appropriate credentials based on mode
        if self.trading_mode == "paper":
            self.api_key = os.getenv("ALPACA_PAPER_API_KEY")
            self.api_secret = os.getenv("ALPACA_PAPER_API_SECRET")
            # Market data uses same endpoint for both paper and live
            self.ws_url_base = "wss://stream.data.alpaca.markets/v2"
            logger.warning("🧪 PAPER TRADING MODE - NO REAL MONEY 🧪")
        else:
            # Double-check for live trading
            confirm = os.getenv("ALPACA_LIVE_TRADING_CONFIRMED", "false").lower()
            if confirm != "true":
                raise ValueError(
                    "⚠️  LIVE TRADING requires ALPACA_LIVE_TRADING_CONFIRMED=true in .env ⚠️\n"
                    "This is real money trading. Set this only if you're absolutely sure!"
                )
            self.api_key = os.getenv("ALPACA_LIVE_API_KEY")
            self.api_secret = os.getenv("ALPACA_LIVE_API_SECRET")
            self.ws_url_base = "wss://stream.data.alpaca.markets/v2"
            logger.error("💸 LIVE TRADING MODE - REAL MONEY AT RISK! 💸")
        
        if not self.api_key or not self.api_secret:
            raise ValueError(f"API credentials not set for {self.trading_mode} mode")
        
        # Validate key format
        if self.trading_mode == "paper" and not self.api_key.startswith('PK'):
            raise ValueError("Paper trading key should start with 'PK'")
        elif self.trading_mode == "live" and not self.api_key.startswith('AK'):
            raise ValueError("Live trading key should start with 'AK'")
        
        # Data feed configuration
        self.data_feed = os.getenv("ALPACA_DATA_FEED", "iex")  # iex, sip, or delayed_sip
        self.use_test_stream = os.getenv("ALPACA_USE_TEST_STREAM", "false").lower() == "true"
        
        if self.use_test_stream:
            self.ws_url = f"{self.ws_url_base}/test"
            logger.info("Using Alpaca TEST stream with FAKEPACA symbol")
        else:
            self.ws_url = f"{self.ws_url_base}/{self.data_feed}"
        
        self.kafka_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        base_topic_prefix = os.getenv("KAFKA_TOPIC_PREFIX", "alpaca_market_data")
        self.topic_prefix = f"{base_topic_prefix}_{self.trading_mode}"
        
        self.symbols = os.getenv("SYMBOLS", "NVDA").split(",")
        self.subscription_types = os.getenv("SUBSCRIPTION_TYPES", "trades,quotes,bars").split(",")
        
        self.producer = self._create_kafka_producer()
        self.ws = None
        self.running = True
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Start Prometheus metrics server
        start_http_server(8000)
    
    def _signal_handler(self, signum, frame):
        logger.info("Received shutdown signal", signal=signum)
        self.running = False
        if self.ws:
            self.ws.close()
    
    def _create_kafka_producer(self) -> KafkaProducer:
        return KafkaProducer(
            bootstrap_servers=self.kafka_servers.split(","),
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            acks='all',
            retries=3,
            max_in_flight_requests_per_connection=1
        )
    
    def _get_topic_name(self, message_type: str) -> str:
        topic_map = {
            "t": "trades",
            "q": "quotes",
            "b": "bars",
            "d": "daily_bars",
            "u": "updated_bars",
            "c": "corrections",
            "x": "cancel_errors",
            "l": "lulds",
            "s": "trading_status"
        }
        return f"{self.topic_prefix}_{topic_map.get(message_type, 'unknown')}"
    
    def _send_to_kafka(self, message: Dict[str, Any]):
        try:
            message_type = message.get("T")
            if not message_type:
                return
                
            with kafka_send_duration.time():
                topic = self._get_topic_name(message_type)
                key = message.get("S", "unknown")  # Symbol
                
                # Calculate message lag if timestamp is available
                if "t" in message:
                    # Parse RFC-3339 timestamp
                    msg_time = datetime.fromisoformat(message["t"].replace("Z", "+00:00"))
                    current_time = datetime.now(msg_time.tzinfo)  # Use same timezone
                    lag = (current_time - msg_time).total_seconds() * 1000
                    message_lag.observe(lag)
                
                kafka_message = {
                    **message,
                    "processed_at": datetime.utcnow().isoformat(),
                    "producer_timestamp": int(datetime.utcnow().timestamp() * 1000)
                }
                
                future = self.producer.send(topic, key=key, value=kafka_message)
                
                future.add_callback(
                    lambda metadata: [
                        messages_sent.labels(topic=metadata.topic, status='success').inc(),
                        logger.debug(
                            "Message sent successfully",
                            topic=metadata.topic,
                            partition=metadata.partition,
                            offset=metadata.offset,
                            symbol=key,
                            type=message_type
                        )
                    ]
                )
                
                future.add_errback(
                    lambda error: [
                        messages_sent.labels(topic=topic, status='error').inc(),
                        logger.error(
                            "Failed to send message",
                            error=str(error),
                            topic=topic,
                            symbol=key
                        )
                    ]
                )
                
        except Exception as e:
            messages_sent.labels(topic='unknown', status='error').inc()
            logger.error("Error processing message", error=str(e), message=message)
    
    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            
            # Alpaca sends messages as arrays
            if isinstance(data, list):
                for item in data:
                    message_type = item.get("T", "unknown")
                    messages_received.labels(message_type=message_type).inc()
                    
                    if message_type == "success":
                        logger.info("Success message", msg=item.get("msg"))
                        if item.get("msg") == "authenticated":
                            self._subscribe()
                    elif message_type == "error":
                        logger.error("Error from server", msg=item.get("msg"), code=item.get("code"))
                    elif message_type == "subscription":
                        logger.info("Subscription confirmed", subscriptions=item)
                    else:
                        self._send_to_kafka(item)
                        
        except json.JSONDecodeError as e:
            logger.error("Failed to decode message", error=str(e), message=message)
        except Exception as e:
            logger.error("Unexpected error in message handler", error=str(e))
    
    def _authenticate(self):
        auth_message = {
            "action": "auth",
            "key": self.api_key,
            "secret": self.api_secret
        }
        self.ws.send(json.dumps(auth_message))
        logger.info("Authentication message sent")
    
    def _subscribe(self):
        subscription = {"action": "subscribe"}
        
        # Build subscription based on requested types
        if "trades" in self.subscription_types:
            subscription["trades"] = self.symbols
        if "quotes" in self.subscription_types:
            subscription["quotes"] = self.symbols
        if "bars" in self.subscription_types:
            subscription["bars"] = self.symbols
        if "daily_bars" in self.subscription_types:
            subscription["dailyBars"] = self.symbols
        if "statuses" in self.subscription_types:
            subscription["statuses"] = ["*"]  # All symbols
        
        self.ws.send(json.dumps(subscription))
        logger.info("Subscription request sent", subscription=subscription)
    
    def on_error(self, ws, error):
        logger.error("WebSocket error", error=str(error))
        websocket_connections.labels(status='error').inc()
    
    def on_close(self, ws, close_status_code, close_msg):
        logger.info("WebSocket closed", code=close_status_code, message=close_msg)
        websocket_active.set(0)
    
    def on_open(self, ws):
        logger.info("WebSocket connection opened")
        websocket_connections.labels(status='success').inc()
        websocket_active.set(1)
        # Alpaca requires authentication immediately after connection
        self._authenticate()
    
    def run(self):
        logger.info(f"Starting Alpaca WebSocket producer in {self.trading_mode.upper()} MODE", 
                   mode=self.trading_mode,
                   symbols=self.symbols, 
                   types=self.subscription_types,
                   feed=self.data_feed,
                   is_real_money=(self.trading_mode == "live"))
        
        while self.running:
            try:
                self.ws = websocket.WebSocketApp(
                    self.ws_url,
                    on_open=self.on_open,
                    on_message=self.on_message,
                    on_error=self.on_error,
                    on_close=self.on_close
                )
                
                self.ws.run_forever(ping_interval=30, ping_timeout=10)
                
                if self.running:
                    logger.info("WebSocket disconnected, reconnecting in 5 seconds...")
                    time.sleep(5)
                    
            except Exception as e:
                logger.error("Unexpected error in main loop", error=str(e))
                if self.running:
                    time.sleep(5)
        
        logger.info("Shutting down producer")
        self.producer.flush()
        self.producer.close()


if __name__ == "__main__":
    producer = AlpacaWebSocketProducer()
    producer.run()