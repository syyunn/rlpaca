#!/usr/bin/env python3
"""
Visualize Alpaca bars data to verify download
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

def visualize_alpaca_bars(date='2025-06-20'):
    """Load and visualize Alpaca bars"""
    
    # Check if bars file exists
    bars_file = f'data/alpaca_bars/NVDA_{date}_bars.parquet'
    
    if not os.path.exists(bars_file):
        print(f"❌ Bars file not found: {bars_file}")
        print("Please run: python scripts/download_data_with_bars.py")
        return
        
    # Load bars
    print(f"📊 Loading Alpaca bars for {date}")
    bars = pd.read_parquet(bars_file)
    
    print(f"\n📈 Data Summary:")
    print(f"Total bars: {len(bars)}")
    print(f"Time range: {bars.index[0]} to {bars.index[-1]}")
    print(f"Price range: ${bars['low'].min():.2f} - ${bars['high'].max():.2f}")
    print(f"Total volume: {bars['volume'].sum():,}")
    
    # Display first and last few bars
    print("\nFirst 5 bars:")
    print(bars.head().round(2))
    
    print("\nLast 5 bars:")
    print(bars.tail().round(2))
    
    # Create visualization
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'NVDA Alpaca Bars - {date}', fontsize=16)
    
    # Plot 1: OHLC Candlestick style
    for idx, row in bars.iterrows():
        color = 'green' if row['close'] >= row['open'] else 'red'
        # High-Low line
        ax1.plot([idx, idx], [row['low'], row['high']], color=color, linewidth=0.5, alpha=0.8)
        # Open-Close box
        height = abs(row['close'] - row['open'])
        bottom = min(row['open'], row['close'])
        ax1.bar(idx, height, bottom=bottom, width=0.0007, color=color, alpha=0.8)
    
    ax1.set_ylabel('Price ($)')
    ax1.set_title('OHLC Price Chart')
    ax1.grid(True, alpha=0.3)
    
    # Add some price levels
    ax1.axhline(bars['high'].max(), color='blue', linestyle='--', alpha=0.5, label=f"High: ${bars['high'].max():.2f}")
    ax1.axhline(bars['low'].min(), color='red', linestyle='--', alpha=0.5, label=f"Low: ${bars['low'].min():.2f}")
    ax1.axhline(bars['close'].mean(), color='orange', linestyle='--', alpha=0.5, label=f"Avg: ${bars['close'].mean():.2f}")
    ax1.legend()
    
    # Plot 2: Volume
    colors = ['green' if bars.iloc[i]['close'] >= bars.iloc[i]['open'] else 'red' for i in range(len(bars))]
    ax2.bar(bars.index, bars['volume'], width=0.0007, alpha=0.7, color=colors)
    ax2.set_ylabel('Volume')
    ax2.set_title('Volume')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: VWAP vs Close
    ax3.plot(bars.index, bars['close'], 'b-', label='Close', linewidth=1)
    if 'vwap' in bars.columns:
        ax3.plot(bars.index, bars['vwap'], 'r--', label='VWAP', linewidth=1)
    ax3.set_ylabel('Price ($)')
    ax3.set_xlabel('Time (ET)')
    ax3.set_title('Close Price and VWAP')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Format x-axis
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax3.xaxis.set_major_locator(mdates.HourLocator())
    ax3.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))
    
    plt.tight_layout()
    
    # Save plot
    output_file = f'alpaca_bars_{date}.png'
    plt.savefig(output_file, dpi=150)
    print(f"\n💾 Chart saved to: {output_file}")
    
    # Calculate some statistics
    print(f"\n📊 Trading Statistics:")
    print(f"Open: ${bars.iloc[0]['open']:.2f}")
    print(f"Close: ${bars.iloc[-1]['close']:.2f}")
    print(f"High: ${bars['high'].max():.2f}")
    print(f"Low: ${bars['low'].min():.2f}")
    
    daily_return = (bars.iloc[-1]['close'] / bars.iloc[0]['open'] - 1) * 100
    print(f"Daily Return: {daily_return:.2f}%")
    
    # Volatility
    returns = bars['close'].pct_change().dropna()
    print(f"Intraday Volatility: {returns.std() * 100:.3f}%")
    
    # Check data quality
    print(f"\n✅ Data Quality Check:")
    print(f"Missing values: {bars.isnull().sum().sum()}")
    print(f"Bars with zero volume: {(bars['volume'] == 0).sum()}")
    
    # Check if this matches our manually calculated bars
    print(f"\n🔍 Comparing with trade-based calculation:")
    trades_file = 'data/ticks_fixed/NVDA_trades_with_timestamps.parquet'
    if os.path.exists(trades_file):
        trades = pd.read_parquet(trades_file)
        if 'timestamp' in trades.columns:
            trades.set_index('timestamp', inplace=True)
        
        trades_market = trades.between_time('09:30', '16:00')
        manual_bars = trades_market.resample('1min').agg({
            'price': ['first', 'max', 'min', 'last'],
            'size': 'sum'
        })
        manual_bars.columns = ['open', 'high', 'low', 'close', 'volume']
        manual_bars = manual_bars.dropna()
        
        print(f"Alpaca bars: {len(bars)}")
        print(f"Manual bars from trades: {len(manual_bars)}")
        
        # Compare a few values
        if len(manual_bars) > 0:
            print(f"\nSample comparison (first bar):")
            print(f"Alpaca - Open: ${bars.iloc[0]['open']:.2f}, Close: ${bars.iloc[0]['close']:.2f}")
            print(f"Manual - Open: ${manual_bars.iloc[0]['open']:.2f}, Close: ${manual_bars.iloc[0]['close']:.2f}")
    
    plt.show()

if __name__ == "__main__":
    visualize_alpaca_bars()