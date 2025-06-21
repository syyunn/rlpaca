#!/usr/bin/env python3
"""
Download Alpaca data including minute bars for training
"""

import os
import pandas as pd
from datetime import datetime
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame

def download_nvda_with_bars(date='2025-06-20'):
    """Download NVDA data including minute bars from Alpaca"""
    
    print(f"📊 Downloading NVDA data for {date}")
    print("=" * 60)
    
    # Initialize Alpaca client
    api_key = os.getenv('ALPACA_PAPER_API_KEY', os.getenv('ALPACA_API_KEY'))
    secret_key = os.getenv('ALPACA_PAPER_API_SECRET', os.getenv('ALPACA_API_SECRET'))
    
    if not api_key or not secret_key:
        print("❌ Please set ALPACA_API_KEY and ALPACA_API_SECRET")
        return False
        
    api = tradeapi.REST(
        key_id=api_key,
        secret_key=secret_key,
        base_url='https://paper-api.alpaca.markets'
    )
    
    # Download minute bars directly from Alpaca
    print("📈 Downloading minute bars...")
    
    bars = api.get_bars(
        'NVDA',
        TimeFrame.Minute,
        start=f"{date}T00:00:00Z",
        end=f"{date}T23:59:59Z",
        adjustment='raw'
    ).df
    
    if bars.empty:
        print("❌ No bar data found")
        return False
        
    # Filter for market hours
    bars_market = bars.between_time('09:30', '16:00')
    
    print(f"✅ Downloaded {len(bars_market)} minute bars")
    print(f"   Price range: ${bars_market['low'].min():.2f} - ${bars_market['high'].max():.2f}")
    print(f"   Total volume: {bars_market['volume'].sum():,}")
    
    # Save bars
    os.makedirs('data/alpaca_bars', exist_ok=True)
    output_file = f'data/alpaca_bars/NVDA_{date}_bars.parquet'
    bars_market.to_parquet(output_file)
    
    print(f"\n💾 Saved to: {output_file}")
    
    # Also download quotes for tick data (sampled)
    print("\n💱 Downloading quote data...")
    
    # Use existing downloaded quotes if available
    quotes_file = 'data/ticks_fixed/NVDA_quotes_with_timestamps.parquet'
    if os.path.exists(quotes_file):
        quotes = pd.read_parquet(quotes_file)
        
        # Set timestamp as index
        if 'timestamp' in quotes.columns:
            quotes.set_index('timestamp', inplace=True)
            
        # Filter for market hours
        quotes_market = quotes.between_time('09:30', '16:00')
        
        # Resample to 1-minute for tick features
        quotes_1min = quotes_market.resample('1min').agg({
            'bid_price': 'last',
            'ask_price': 'last',
            'bid_size': 'last',
            'ask_size': 'last'
        }).dropna()
        
        print(f"✅ Processed {len(quotes_1min)} minute quote samples")
        
        # Save quote data
        os.makedirs('data/1min_ticks', exist_ok=True)
        quotes_1min.columns = ['bid', 'ask', 'bid_size', 'ask_size']
        quotes_1min['timestamp'] = quotes_1min.index
        quotes_1min.to_parquet(f'data/1min_ticks/NVDA_ticks_{date.replace("-", "")}_1min.parquet')
    
    return True

if __name__ == "__main__":
    download_nvda_with_bars()