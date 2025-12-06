#!/usr/bin/env python3
"""
Mock Tick Data Producer for Demo
Simulates real-time NVDA tick data being streamed to Kafka
"""

import json
import time
import random
import os
from datetime import datetime
from kafka import KafkaProducer

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:19092')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'nvda-quotes')

# NVDA base price (realistic)
BASE_PRICE = 142.50
TICK_INTERVAL = 0.1  # 100ms between ticks (10 ticks/second)

def create_producer():
    """Create Kafka producer with retries"""
    max_retries = 30
    for attempt in range(max_retries):
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                api_version=(2, 5, 0)
            )
            print(f"✅ Connected to Kafka at {KAFKA_BOOTSTRAP_SERVERS}")
            return producer
        except Exception as e:
            print(f"⏳ Waiting for Kafka... ({attempt + 1}/{max_retries})")
            time.sleep(2)
    raise Exception("❌ Could not connect to Kafka")

def generate_tick(base_price, tick_num):
    """Generate a realistic tick quote"""
    # Random walk for price movement
    price_change = random.gauss(0, 0.02)  # Small random changes
    mid_price = base_price + price_change

    # Bid-ask spread (typically 1-3 cents for NVDA)
    spread = random.uniform(0.01, 0.03)
    bid_price = round(mid_price - spread/2, 2)
    ask_price = round(mid_price + spread/2, 2)

    # Random sizes (100-1000 shares typical)
    bid_size = random.randint(1, 10) * 100
    ask_size = random.randint(1, 10) * 100

    timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

    return {
        'T': 'q',  # Quote type
        'S': 'NVDA',
        'bp': bid_price,
        'ap': ask_price,
        'bs': bid_size,
        'as': ask_size,
        't': timestamp,
        'tick_num': tick_num
    }

def main():
    print("=" * 60)
    print("🚀 RLpaca Mock Tick Data Producer")
    print("=" * 60)
    print(f"📊 Symbol: NVDA")
    print(f"💰 Base Price: ${BASE_PRICE}")
    print(f"📡 Kafka: {KAFKA_BOOTSTRAP_SERVERS}")
    print(f"📨 Topic: {KAFKA_TOPIC}")
    print(f"⏱️  Tick Interval: {TICK_INTERVAL}s")
    print("=" * 60)

    producer = create_producer()

    current_price = BASE_PRICE
    tick_num = 0
    ticks_sent = 0
    start_time = time.time()

    print("\n📤 Streaming tick data to Kafka...\n")

    try:
        while True:
            # Generate and send tick
            tick = generate_tick(current_price, tick_num)
            producer.send(KAFKA_TOPIC, tick)

            # Update price with random walk
            current_price += random.gauss(0, 0.01)
            current_price = max(100, min(200, current_price))  # Keep in range

            tick_num += 1
            ticks_sent += 1

            # Print progress every 10 ticks
            if tick_num % 10 == 0:
                elapsed = time.time() - start_time
                rate = ticks_sent / elapsed if elapsed > 0 else 0
                print(f"📈 Tick #{tick_num:,} | NVDA ${tick['bp']:.2f}/${tick['ap']:.2f} | "
                      f"Spread: ${tick['ap'] - tick['bp']:.3f} | "
                      f"Rate: {rate:.1f} ticks/sec")

            time.sleep(TICK_INTERVAL)

    except KeyboardInterrupt:
        print("\n\n🛑 Stopping producer...")
        producer.flush()
        producer.close()
        print(f"✅ Sent {ticks_sent:,} ticks total")

if __name__ == '__main__':
    main()
