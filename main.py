# ========== IMPORTS & CONFIG ==========
import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from pyentrp import entropy as ent
import datetime
import time
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

load_dotenv()

# 🧠 Your API credentials (loaded from .env file)
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET")
BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets") # Default to paper trading

# Check if essential variables are set
if not API_KEY or not API_SECRET:
    raise ValueError("Please set ALPACA_API_KEY and ALPACA_SECRET in your .env file or environment variables.")

# 🎯 Target stock and backtest parameters
SYMBOL = "NVDA"
INITIAL_CAPITAL = 10000
MAX_POSITION_SIZE = 0.9  # Maximum % of portfolio to use per trade

# 🧠 Strategy hyperparameters (can be optimized)
ROLLING_WINDOW = 20       # Lookback window for indicators
STD_MULTIPLIER = 1.0      # Distance from mean to trigger a trade
ENTROPY_THRESHOLD = 1.2   # Below this, we consider market 'predictable'

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

# ========== ALPACA DATA FETCHER ==========
def fetch_historical_data(symbol, timeframe, start_date, end_date=None, limit=10000):
    """
    Fetch historical data from Alpaca API with fallback to Yahoo Finance.
    
    Parameters:
    - symbol: Stock symbol (e.g., "NVDA")
    - timeframe: Time interval (e.g., "1Min", "1Hour", "1Day")
    - start_date: Start date as string "YYYY-MM-DD" or datetime
    - end_date: End date (optional)
    - limit: Maximum number of bars to fetch
    
    Returns:
    - Pandas DataFrame with historical data
    """
    print(f"Fetching {timeframe} data for {symbol} from {start_date} to {end_date or 'now'}...")
    
    if isinstance(timeframe, str):
        if timeframe == "1Min":
            tf = tradeapi.TimeFrame.Minute
        elif timeframe == "1Hour":
            tf = tradeapi.TimeFrame.Hour
        elif timeframe == "1Day":
            tf = tradeapi.TimeFrame.Day
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
    else:
        tf = timeframe
    
    try:
        # Convert string dates to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.Timestamp(start_date).date().isoformat()
        if end_date and isinstance(end_date, str):
            end_date = pd.Timestamp(end_date).date().isoformat()
            
        # Fetch the data
        if end_date:
            bars = api.get_bars(symbol, tf, start=start_date, end=end_date, limit=limit).df
        else:
            bars = api.get_bars(symbol, tf, start=start_date, limit=limit).df
            
        if bars.empty:
            print(f"No data available for {symbol} from Alpaca. Trying alternative source...")
            return fetch_from_yahoo(symbol, start_date, end_date)
            
        print(f"Retrieved {len(bars)} bars of {timeframe} data from Alpaca.")
        return bars
        
    except Exception as e:
        print(f"Error fetching data from Alpaca: {e}")
        print("Trying alternative data source (Yahoo Finance)...")
        return fetch_from_yahoo(symbol, start_date, end_date)

def fetch_from_yahoo(symbol, start_date, end_date=None):
    """Fetch historical data from Yahoo Finance as a fallback."""
    try:
        import yfinance as yf
        
        # Parse dates if they're strings
        if isinstance(start_date, str):
            start_date = pd.Timestamp(start_date)
        if end_date and isinstance(end_date, str):
            end_date = pd.Timestamp(end_date)
        else:
            end_date = datetime.now()
            
        # Download the data
        ticker_data = yf.download(symbol, start=start_date, end=end_date)
        
        if ticker_data.empty:
            print(f"No data available for {symbol} from Yahoo Finance.")
            return None
        
        # Convert column MultiIndex to regular index if needed
        if isinstance(ticker_data.columns, pd.MultiIndex):
            # If we have a multi-index from Yahoo Finance, use the second level
            print("Flattening multi-level columns from Yahoo Finance")
            ticker_data.columns = [col[1] if isinstance(col, tuple) else col for col in ticker_data.columns]
        
        # Rename columns to match Alpaca format (case-sensitive)
        column_mapping = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume'
        }
        ticker_data = ticker_data.rename(columns=column_mapping)
        
        # Print columns for debugging
        print(f"Yahoo Finance data columns: {ticker_data.columns.tolist()}")
        
        return ticker_data

    except Exception as e:
        print(f"Error fetching data from Yahoo Finance: {e}")
        import traceback
        traceback.print_exc()
        print("Please install yfinance with: pip install yfinance")
        return None

# ========== STRATEGY SIGNAL GENERATOR ==========
def generate_signals(df):
    """Apply our mean-reversion + entropy strategy to generate trading signals."""
    # Identify the correct column name for close price
    close_col = next((col for col in df.columns if col.lower() == 'close'), None)
    if not close_col:
        raise ValueError("Could not find 'close' column in data")
    
    # Calculate indicators
    df = df.copy()
    df['mean'] = df[close_col].rolling(window=ROLLING_WINDOW).mean()
    df['std'] = df[close_col].rolling(window=ROLLING_WINDOW).std()
    df['entropy'] = rolling_shannon_entropy(df[close_col], window=ROLLING_WINDOW)
    
    # Generate signals
    df['signal'] = None
    for i in range(ROLLING_WINDOW, len(df)):
        price = df.iloc[i][close_col]
        mean = df.iloc[i]['mean']
        std = df.iloc[i]['std']
        entropy = df.iloc[i]['entropy']
        
        # Skip if we have missing data
        if np.isnan(mean) or np.isnan(std) or np.isnan(entropy):
            continue
            
        # Skip if market is too chaotic
        if entropy > ENTROPY_THRESHOLD:
            continue
            
        # Mean-reversion logic
        if price < (mean - STD_MULTIPLIER * std):
            df.loc[df.index[i], 'signal'] = "BUY"
        elif price > (mean + STD_MULTIPLIER * std):
            df.loc[df.index[i], 'signal'] = "SELL"
    
    return df

# ========== BACKTEST ENGINE ==========
def run_backtest(df, initial_capital=INITIAL_CAPITAL):
    """Simulate trading based on signals and track portfolio performance."""
    if df is None or df.empty:
        print("No data to backtest")
        return None
        
    # Ensure we have the required columns
    close_col = next((col for col in df.columns if col.lower() == 'close'), None)
    if not close_col or 'signal' not in df.columns:
        raise ValueError("DataFrame must contain 'close' and 'signal' columns")
    
    # Initialize portfolio tracking
    df['position'] = 0      # Number of shares held
    df['cash'] = initial_capital  # Cash balance
    df['holdings_value'] = 0.0    # Value of stock holdings
    df['equity'] = initial_capital  # Total portfolio value
    
    position = 0
    last_buy_price = 0
    
    # Process each bar
    for i in range(1, len(df)):
        # Get current price and previous portfolio state
        prev_idx = i-1
        current_price = df.iloc[i][close_col]
        prev_cash = df.iloc[prev_idx]['cash']
        prev_position = df.iloc[prev_idx]['position']
        
        # Default: carry forward previous position
        df.loc[df.index[i], 'position'] = prev_position
        df.loc[df.index[i], 'cash'] = prev_cash
        
        # Check for trading signal
        signal = df.iloc[i]['signal']
        
        if signal == "BUY" and prev_position <= 0:
            # Calculate position size (90% of available cash)
            max_investment = prev_cash * MAX_POSITION_SIZE
            shares_to_buy = int(max_investment / current_price)
            
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                df.loc[df.index[i], 'cash'] = prev_cash - cost
                df.loc[df.index[i], 'position'] = shares_to_buy
                last_buy_price = current_price
                print(f"BUY: {df.index[i]} - {shares_to_buy} shares at ${current_price:.2f}")
                
        elif signal == "SELL" and prev_position > 0:
            # Sell all shares
            proceeds = prev_position * current_price
            df.loc[df.index[i], 'cash'] = prev_cash + proceeds
            df.loc[df.index[i], 'position'] = 0
            profit = proceeds - (prev_position * last_buy_price)
            print(f"SELL: {df.index[i]} - {prev_position} shares at ${current_price:.2f}, Profit: ${profit:.2f}")
        
        # Update holdings value and total equity
        df.loc[df.index[i], 'holdings_value'] = df.iloc[i]['position'] * current_price
        df.loc[df.index[i], 'equity'] = df.iloc[i]['cash'] + df.iloc[i]['holdings_value']
    
    # Calculate returns and performance metrics
    df['daily_return'] = df['equity'].pct_change()
    df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
    
    # Handle potential NaN values in the returns
    df['daily_return'] = df['daily_return'].fillna(0)
    df['cumulative_return'] = df['cumulative_return'].fillna(0)
    
    return df

# ========== PERFORMANCE ANALYSIS ==========
def analyze_performance(df):
    """Calculate and print key performance metrics from backtest results."""
    if df is None or df.empty:
        return None
        
    # Extract key metrics
    initial_equity = df['equity'].iloc[0]
    final_equity = df['equity'].iloc[-1]
    total_return = (final_equity / initial_equity - 1) * 100
    
    # Calculate daily stats (assuming the input data has daily frequency)
    daily_returns = df['daily_return'].dropna()
    
    if len(daily_returns) > 0:
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)  # Annualized Sharpe
        max_drawdown = (df['equity'] / df['equity'].cummax() - 1).min() * 100
        win_trades = df[df['daily_return'] > 0]['daily_return'].count()
        lose_trades = df[df['daily_return'] < 0]['daily_return'].count()
        win_rate = win_trades / (win_trades + lose_trades) * 100 if (win_trades + lose_trades) > 0 else 0
        
        # Count buy/sell signals
        buy_signals = df[df['signal'] == 'BUY']['signal'].count()
        sell_signals = df[df['signal'] == 'SELL']['signal'].count()
        
        # Print performance summary
        print("\n===== PERFORMANCE SUMMARY =====")
        print(f"Initial Capital: ${initial_equity:.2f}")
        print(f"Final Equity: ${final_equity:.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Total Signals: Buy={buy_signals}, Sell={sell_signals}")
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        }
    else:
        print("Not enough data to calculate performance metrics")
        return None

# ========== VISUALIZATION ==========
def visualize_backtest(df):
    """Create charts to visualize backtest results."""
    if df is None or df.empty:
        print("No data to visualize")
        return
        
    close_col = next((col for col in df.columns if col.lower() == 'close'), None)
    if not close_col:
        print("Could not find 'close' column for visualization")
        return
    
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Price with signals and bands
    plt.subplot(3, 1, 1)
    plt.plot(df.index, df[close_col], label='NVDA Price', color='blue', alpha=0.7)
    plt.plot(df.index, df['mean'], label='Rolling Mean', color='orange', linestyle='--', alpha=0.5)
    plt.plot(df.index, df['mean'] + STD_MULTIPLIER * df['std'], 
             label=f'Upper Band ({STD_MULTIPLIER} STD)', color='red', linestyle='--', alpha=0.5)
    plt.plot(df.index, df['mean'] - STD_MULTIPLIER * df['std'], 
             label=f'Lower Band ({STD_MULTIPLIER} STD)', color='green', linestyle='--', alpha=0.5)
    
    # Plot buy/sell markers
    buys = df[df['signal'] == 'BUY']
    sells = df[df['signal'] == 'SELL']
    
    plt.scatter(buys.index, buys[close_col], marker='^', color='green', s=100, label='Buy Signal')
    plt.scatter(sells.index, sells[close_col], marker='v', color='red', s=100, label='Sell Signal')
    
    plt.title('NVDA Price with Trading Signals and Bands')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Entropy
    plt.subplot(3, 1, 2)
    plt.plot(df.index, df['entropy'], label='Shannon Entropy', color='purple')
    plt.axhline(y=ENTROPY_THRESHOLD, color='red', linestyle='--', 
                label=f'Threshold ({ENTROPY_THRESHOLD})')
    plt.title('Market Entropy (Predictability)')
    plt.ylabel('Entropy')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Portfolio Equity
    plt.subplot(3, 1, 3)
    plt.plot(df.index, df['equity'], label='Portfolio Value', color='green')
    plt.title('Portfolio Equity Curve')
    plt.ylabel('Value ($)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'NVDA_Backtest_{datetime.now().strftime("%Y%m%d")}.png')
    plt.show()

# ========== PARAMETER OPTIMIZATION ==========
def optimize_parameters(df, rolling_windows=None, std_multipliers=None, entropy_thresholds=None):
    """Find optimal parameters by testing multiple combinations."""
    if df is None or df.empty:
        print("No data for optimization")
        return None
    
    # Default parameter ranges if not provided
    if rolling_windows is None:
        rolling_windows = [10, 15, 20, 25, 30]
    if std_multipliers is None:
        std_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]
    if entropy_thresholds is None:
        entropy_thresholds = [0.8, 1.0, 1.2, 1.4, 1.6]
    
    results = []
    close_col = next((col for col in df.columns if col.lower() == 'close'), None)
    
    print(f"Optimizing parameters with {len(rolling_windows) * len(std_multipliers) * len(entropy_thresholds)} combinations...")
    
    for window in rolling_windows:
        for multiplier in std_multipliers:
            for threshold in entropy_thresholds:
                # Set global parameters for this run
                global ROLLING_WINDOW, STD_MULTIPLIER, ENTROPY_THRESHOLD
                ROLLING_WINDOW = window
                STD_MULTIPLIER = multiplier
                ENTROPY_THRESHOLD = threshold
                
                # Run backtest with these parameters
                strategy_df = generate_signals(df)
                bt_df = run_backtest(strategy_df, INITIAL_CAPITAL)
                
                if bt_df is not None:
                    # Calculate performance metrics
                    metrics = analyze_performance(bt_df)
                    if metrics:
                        metrics.update({
                            'window': window,
                            'multiplier': multiplier,
                            'threshold': threshold
                        })
                        results.append(metrics)
                        print(f"Window={window}, Multiplier={multiplier}, Threshold={threshold} → "
                              f"Return={metrics['total_return']:.2f}%, Sharpe={metrics['sharpe_ratio']:.2f}")
    
    # Create DataFrame from results and sort by Sharpe ratio
    if results:
        results_df = pd.DataFrame(results)
        best_params = results_df.sort_values(by='sharpe_ratio', ascending=False).iloc[0]
        
        print("\n===== OPTIMIZATION RESULTS =====")
        print(f"Best parameters: Window={best_params['window']}, "
              f"Multiplier={best_params['multiplier']}, Threshold={best_params['threshold']}")
        print(f"Performance: Return={best_params['total_return']:.2f}%, "
              f"Sharpe={best_params['sharpe_ratio']:.2f}, Drawdown={best_params['max_drawdown']:.2f}%")
        
        # Update global parameters to best values
        ROLLING_WINDOW = best_params['window']
        STD_MULTIPLIER = best_params['multiplier']
        ENTROPY_THRESHOLD = best_params['threshold']
        
        return results_df
    else:
        print("No valid results from optimization")
        return None

# ========== MAIN FUNCTION ==========
def main():
    """Main function to run NVIDIA trading strategy backtest."""
    print("=" * 50)
    print("NVIDIA TRADING STRATEGY BACKTEST")
    print("=" * 50)
    
    # Ensure all dependencies are installed
    ensure_dependencies()
    
    # Ask user for backtest period
    print("\n1. Choose backtest period:")
    print("   a) Last 30 days")
    print("   b) Last 90 days")
    print("   c) Last 180 days")
    print("   d) Custom period")
    
    choice = input("Enter choice (a/b/c/d): ").lower()
    
    # Make sure we're using the correct datetime
    current_date = datetime.now()
    end_date = current_date.strftime("%Y-%m-%d")
    
    if choice == 'a':
        start_date = (current_date - timedelta(days=30)).strftime("%Y-%m-%d")
        print(f"Backtesting from {start_date} to {end_date}")
    elif choice == 'b':
        start_date = (current_date - timedelta(days=90)).strftime("%Y-%m-%d")
        print(f"Backtesting from {start_date} to {end_date}")
    elif choice == 'c':
        start_date = (current_date - timedelta(days=180)).strftime("%Y-%m-%d")
        print(f"Backtesting from {start_date} to {end_date}")
    elif choice == 'd':
        start_date = input("Enter start date (YYYY-MM-DD): ")
        end_input = input("Enter end date (YYYY-MM-DD) or press Enter for today: ")
        if not end_input.strip():
            end_date = current_date.strftime("%Y-%m-%d")
        else:
            end_date = end_input
        print(f"Backtesting from {start_date} to {end_date}")
    else:
        print("Invalid choice. Using last 30 days.")
        start_date = (current_date - timedelta(days=30)).strftime("%Y-%m-%d")
        print(f"Backtesting from {start_date} to {end_date}")
    
    # Choose timeframe
    print("\n2. Choose data timeframe:")
    print("   a) 1 Minute")
    print("   b) 1 Hour")
    print("   c) 1 Day")
    
    choice = input("Enter choice (a/b/c): ").lower()
    
    if choice == 'a':
        timeframe = tradeapi.TimeFrame.Minute
    elif choice == 'b':
        timeframe = tradeapi.TimeFrame.Hour
    elif choice == 'c':
        timeframe = tradeapi.TimeFrame.Day
    else:
        print("Invalid choice. Using 1 Hour timeframe.")
        timeframe = tradeapi.TimeFrame.Hour
    
    # Choose optimization
    print("\n3. Optimize strategy parameters?")
    print("   a) Yes - find optimal parameters")
    print("   b) No - use default parameters")
    
    optimize = input("Enter choice (a/b): ").lower() == 'a'
    
    # Fetch historical data
    historical_data = fetch_historical_data(SYMBOL, timeframe, start_date, end_date)
    
    if historical_data is None or historical_data.empty:
        print("No data available for the selected period. Exiting.")
        return
    
    # Run parameter optimization if requested
    if optimize:
        results_df = optimize_parameters(historical_data)
        
        if results_df is not None:
            print("\nTop 5 parameter combinations:")
            print(results_df.sort_values(by='sharpe_ratio', ascending=False).head(5))
    
    # Run backtest with current parameters
    print(f"\nRunning backtest with: Window={ROLLING_WINDOW}, "
          f"Multiplier={STD_MULTIPLIER}, Threshold={ENTROPY_THRESHOLD}")
    
    # Generate trading signals
    strategy_df = generate_signals(historical_data)
    
    # Run backtest
    backtest_results = run_backtest(strategy_df)
    
    # Analyze performance
    if backtest_results is not None:
        analyze_performance(backtest_results)
        visualize_backtest(backtest_results)
        
        # Save results to CSV
        output_file = f'NVDA_Backtest_Results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        backtest_results.to_csv(output_file)
        print(f"\nBacktest results saved to {output_file}")

def ensure_dependencies():
    """Ensure all required packages are installed."""
    try:
        import yfinance
        print("All dependencies are installed.")
    except ImportError:
        print("Installing yfinance package...")
        import subprocess
        subprocess.check_call(["pip", "install", "yfinance"])
        print("yfinance installed successfully.")

if __name__ == "__main__":
    main()
