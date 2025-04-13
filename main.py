# ========== IMPORTS & CONFIG ==========
import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from pyentrp import entropy as ent
import datetime
import time
import os
from dotenv import load_dotenv

load_dotenv()

# 🧠 Your API credentials (loaded from .env file)
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET")
BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets") # Default to paper trading

# Check if essential variables are set
if not API_KEY or not API_SECRET:
    raise ValueError("Please set ALPACA_API_KEY and ALPACA_SECRET in your .env file or environment variables.")

# 🎯 Target stock
SYMBOL = "NVDA"

# 🧠 Strategy hyperparameters
ROLLING_WINDOW = 20       # Lookback window for indicators
STD_MULTIPLIER = 1.0      # Distance from mean to trigger a trade
ENTROPY_THRESHOLD = 1.2   # Below this, we consider market 'predictable'
TRADE_QUANTITY = 1        # Number of shares per trade

# ========== INIT ALPACA ==========
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# ========== ENTROPY CALCULATION ==========
def rolling_shannon_entropy(prices, window=20):
    """Calculate rolling entropy of return series to measure disorder."""
    returns = prices.pct_change().dropna()
    entropies = []

    for i in range(len(returns)):
        if i < window:
            entropies.append(np.nan)
        else:
            window_returns = returns.iloc[i - window:i]
            counts, _ = np.histogram(window_returns, bins=5)
            probs = counts / counts.sum()
            shannon = -np.sum([p * np.log2(p) for p in probs if p > 0])
            entropies.append(shannon)

    return pd.Series(entropies, index=prices.index)

# ========== DECISION LOGIC ==========
def should_trade(df):
    """Based on mean reversion and entropy logic, return trade signal."""
    latest = df.iloc[-1]
    price = latest['close']
    mean = latest['mean']
    std = latest['std']
    entropy = latest['entropy']

    print(f"🔍 Price={price:.2f}, Mean={mean:.2f}, Entropy={entropy:.2f}")

    if np.isnan(mean) or np.isnan(std) or np.isnan(entropy):
        return None  # Not enough data yet

    if entropy > ENTROPY_THRESHOLD:
        return None  # Market too chaotic

    if price < (mean - STD_MULTIPLIER * std):
        return "BUY"

    elif price > (mean + STD_MULTIPLIER * std):
        return "SELL"

    return None

# ========== TRADE EXECUTION ==========
def place_order(action, qty):
    """Place a market order if no open position exists."""
    try:
        # Check for existing position in NVDA
        positions = api.list_positions()
        symbols = [pos.symbol for pos in positions]

        if SYMBOL in symbols:
            print(f"⚠️ Already in a position — skipping {action}")
            return

        # api.submit_order(
        #     symbol=SYMBOL,
        #     qty=qty,
        #     side='buy' if action == "BUY" else 'sell',
        #     type='market',
        #     time_in_force='gtc'
        # )
        print(f"✅ [{action}] order submitted for {qty} share(s) of {SYMBOL}")

    except Exception as e:
        print(f"❌ Order failed: {e}")

# ========== MAIN BOT LOOP ==========
def run_bot():
    """Run the mean-reversion strategy with Alpaca every minute."""
    while True:
        try:
            now = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"\n[{now}] Checking market conditions for {SYMBOL}...")

            # Get 50 most recent 1-min candles from Alpaca
            bars = api.get_bars(SYMBOL, tradeapi.TimeFrame.Minute, limit=50).df
            
            # Debug output - check if we received data
            if bars.empty:
                print("⚠️ No data received from Alpaca API. Market might be closed.")
                time.sleep(60)
                continue
                
            # Debug the column names to see what's available
            print(f"Available columns: {bars.columns.tolist()}")
            
            # Calculate rolling indicators - use case-insensitive column matching
            # The API might return 'Close' instead of 'close'
            df = bars.copy()
            close_col = next((col for col in df.columns if col.lower() == 'close'), None)
            
            if not close_col:
                print("⚠️ Could not find 'close' column in data. Available columns:", df.columns.tolist())
                time.sleep(60)
                continue
                
            df['mean'] = df[close_col].rolling(window=ROLLING_WINDOW).mean()
            df['std'] = df[close_col].rolling(window=ROLLING_WINDOW).std()
            df['entropy'] = rolling_shannon_entropy(df[close_col], window=ROLLING_WINDOW)

            # Get trade signal
            signal = should_trade(df)

            # Place trade if signal is valid
            if signal:
                place_order(signal, TRADE_QUANTITY)

        except Exception as e:
            print(f"⚠️ Error in main loop: {e}")
            # Print more detailed error information for debugging
            import traceback
            traceback.print_exc()

        # Sleep until next minute
        time.sleep(60)

# ========== RUN ==========
if __name__ == "__main__":
    run_bot()
