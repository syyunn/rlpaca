#!/usr/bin/env python3
"""Check what price data we have"""

import pandas as pd
import numpy as np

# Load the tick data
ticks_df = pd.read_parquet('/Users/suyeolyun/gits/rlpaca/data/1min_ticks/NVDA_ticks_20250620_1min.parquet')
print("Tick data columns:", ticks_df.columns.tolist())
print("\nFirst 5 tick rows:")
print(ticks_df.head())

# Load the bar data  
bars_df = pd.read_parquet('/Users/suyeolyun/gits/rlpaca/data/alpaca_bars/NVDA_2025-06-20_bars.parquet')
print("\n\nBar data columns:", bars_df.columns.tolist())
print("\nFirst 5 bar rows:")
print(bars_df.head())

# Check if we have trade price
print("\n\nPrice Analysis:")
print(f"Tick data shape: {ticks_df.shape}")
print(f"Bar data shape: {bars_df.shape}")

# If we have VWAP (Volume Weighted Average Price) or close price, that's the market price
if 'vwap' in bars_df.columns:
    print(f"\nVWAP range: ${bars_df['vwap'].min():.2f} - ${bars_df['vwap'].max():.2f}")
if 'close' in bars_df.columns:
    print(f"Close price range: ${bars_df['close'].min():.2f} - ${bars_df['close'].max():.2f}")
    
# Plot minute bar prices to see the actual movement
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Plot close prices
times = bars_df.index
hours = [(t.hour + t.minute/60) for t in times]

ax1.plot(hours, bars_df['close'], 'b-', label='Close Price')
if 'vwap' in bars_df.columns:
    ax1.plot(hours, bars_df['vwap'], 'r--', alpha=0.7, label='VWAP')
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Price ($)')
ax1.set_title('NVDA Price Movement - 2025-06-20')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot tick bid/ask spread
ax2.plot(ticks_df.index, ticks_df['bid'], 'g-', label='Bid', alpha=0.7)
ax2.plot(ticks_df.index, ticks_df['ask'], 'r-', label='Ask', alpha=0.7)
ax2.set_xlabel('Time')
ax2.set_ylabel('Price ($)')
ax2.set_title('Bid-Ask Spread (1-min aggregated)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/suyeolyun/gits/rlpaca/nvda_actual_prices.png')
print(f"\nPrice chart saved as nvda_actual_prices.png")

# Calculate price statistics
price_return = (bars_df['close'].iloc[-1] - bars_df['close'].iloc[0]) / bars_df['close'].iloc[0] * 100
print(f"\nNVDA Performance on 2025-06-20:")
print(f"Open: ${bars_df['close'].iloc[0]:.2f}")
print(f"Close: ${bars_df['close'].iloc[-1]:.2f}")
print(f"Day Return: {price_return:+.2f}%")
print(f"High: ${bars_df['high'].max():.2f}")
print(f"Low: ${bars_df['low'].min():.2f}")
print(f"Range: ${bars_df['high'].max() - bars_df['low'].min():.2f}")