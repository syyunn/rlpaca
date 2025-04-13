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

# Create a global API instance that can be used by all functions
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# ========== CONFIGURATION PARAMETERS ==========
# Backtest date range
START_DATE = "2023-01-01"  # Going back further in time
END_DATE = "2024-04-12"    # Current date

# Strategy parameters
PARAMETER_SETS = [
    {"name": "Ultra-Short", "window": 30, "multiplier": 0.3, "threshold": 0.8},
    {"name": "Short", "window": 60, "multiplier": 0.4, "threshold": 0.9},
    {"name": "Medium", "window": 120, "multiplier": 0.5, "threshold": 1.0}
]

# Symbols to test
SYMBOLS = ["NVDA", "AAPL", "MSFT", "GOOGL", "SPY"]

# Other settings
INITIAL_CAPITAL = 10000  # Starting capital for backtests
LONG_ONLY = True  # Whether to allow short positions

# Function to fetch data from Yahoo Finance as a fallback
def fetch_from_yahoo(symbol, start_date, end_date=None):
    """Fetch historical data from Yahoo Finance."""
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
        
        print(f"Retrieved {len(ticker_data)} bars of data from Yahoo Finance for {symbol}.")
        return ticker_data
        
    except Exception as e:
        print(f"Error fetching data from Yahoo Finance: {e}")
        import traceback
        traceback.print_exc()
        print("Please install yfinance with: pip install yfinance")
        return None

# ========== POSITION HANDLER CLASS ==========
class PositionHandler:
    """Position and trade handling for backtesting, similar to the approach in the Alpaca article."""
    
    def __init__(self, starting_balance=10000, live_trading=False):
        self.starting_balance = starting_balance
        self.cash_balance = starting_balance
        self.live_trading = live_trading
        self.positions = {}  # Symbol -> quantity
        self.open_orders = []
        self.trade_history = []
        self.position_history = {}  # Symbol -> list of positions over time
        self.cash_history = [starting_balance]  # Track cash balance over time
        self.equity_history = [starting_balance]  # Track total equity over time
        self.api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2') if live_trading else None
    
    def return_open_position(self, symbol):
        """Return quantity of open position for a symbol."""
        return self.positions.get(symbol, 0)
    
    def place_order(self, symbol, quantity, side, order_type, time_in_force, price=None, timestamp=None):
        """Place an order in the system."""
        if quantity == 0:
            return None
            
        # Record the timestamp
        if timestamp is None:
            timestamp = datetime.now()
            
        if self.live_trading:
            # Place a real order via Alpaca API
            try:
                if order_type == "market":
                    order = self.api.submit_order(
                        symbol=symbol,
                        qty=abs(quantity),
                        side=side,
                        type=order_type,
                        time_in_force=time_in_force
                    )
                    return order
                elif order_type == "limit" and price is not None:
                    order = self.api.submit_order(
                        symbol=symbol,
                        qty=abs(quantity),
                        side=side,
                        type=order_type,
                        time_in_force=time_in_force,
                        limit_price=price
                    )
                    return order
            except Exception as e:
                print(f"Error placing order: {e}")
                return None
        else:
            # Simulated order for backtesting
            order = {
                "symbol": symbol,
                "qty": abs(quantity),
                "side": side,
                "type": order_type,
                "time_in_force": time_in_force,
                "price": price,
                "timestamp": timestamp
            }
            self.open_orders.append(order)
            return order
    
    def execute_order(self, order, execution_price, timestamp=None):
        """Execute an order at the specified price (for backtesting)."""
        if timestamp is None:
            timestamp = datetime.now()
            
        symbol = order["symbol"]
        quantity = order["qty"]
        side = order["side"]
        
        # Calculate the cost of the trade
        cost = quantity * execution_price
        
        # Update positions and cash based on side
        if side == "buy":
            # Deduct cash, add to position
            self.cash_balance -= cost
            current_position = self.positions.get(symbol, 0)
            self.positions[symbol] = current_position + quantity
        else:  # side == "sell"
            # Add cash, reduce position
            self.cash_balance += cost
            current_position = self.positions.get(symbol, 0)
            self.positions[symbol] = current_position - quantity
        
        # Record the trade
        trade = {
            "symbol": symbol,
            "qty": quantity,
            "side": side,
            "price": execution_price,
            "timestamp": timestamp,
            "cost": cost
        }
        self.trade_history.append(trade)
        
        # Update position history
        if symbol not in self.position_history:
            self.position_history[symbol] = []
        self.position_history[symbol].append({
            "timestamp": timestamp,
            "position": self.positions.get(symbol, 0)
        })
        
        # Update cash history
        self.cash_history.append(self.cash_balance)
        
        # Remove the order from open orders
        if order in self.open_orders:
            self.open_orders.remove(order)
            
        return trade
    
    def update_equity(self, price_dict, timestamp=None):
        """Update equity value based on current positions and prices."""
        if timestamp is None:
            timestamp = datetime.now()
            
        equity = self.cash_balance
        
        # Add value of each position
        for symbol, quantity in self.positions.items():
            if symbol in price_dict:
                equity += quantity * price_dict[symbol]
        
        self.equity_history.append(equity)
        return equity
    
    def get_returns(self):
        """Calculate returns over the backtest period."""
        if len(self.equity_history) < 2:
            return 0
            
        initial = self.equity_history[0]
        final = self.equity_history[-1]
        return (final - initial) / initial

# ========== ENTROPY CALCULATION ==========
def rolling_shannon_entropy(prices, window=20):
    """Calculate rolling entropy of return series to measure disorder."""
    if len(prices) <= window:
        # Not enough data for meaningful entropy calculation
        return pd.Series([np.nan] * len(prices), index=prices.index)
    
    returns = prices.pct_change().dropna()
    # Create a Series of NaN values with the same index as prices
    entropies = pd.Series([np.nan] * len(prices), index=prices.index)
    
    # Calculate entropy for each window
    for i in range(window, len(returns) + 1):
        window_returns = returns.iloc[i - window:i]
        counts, _ = np.histogram(window_returns, bins=5)
        probs = counts / counts.sum()
        shannon = -np.sum([p * np.log2(p) for p in probs if p > 0])
        # Place the entropy value at the end of the window
        entropies.iloc[i - 1 + (len(prices) - len(returns))] = shannon

    return entropies

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
            
        # Fetch the data - explicitly using the IEX feed to avoid subscription errors
        if end_date:
            bars = api.get_bars(symbol, tf, start=start_date, end=end_date, limit=limit, feed='iex').df
        else:
            bars = api.get_bars(symbol, tf, start=start_date, limit=limit, feed='iex').df
            
        if bars.empty:
            print(f"No data available for {symbol} from Alpaca IEX. Trying Yahoo Finance...")
            return fetch_from_yahoo(symbol, start_date, end_date)
            
        print(f"Retrieved {len(bars)} bars of {timeframe} data from Alpaca IEX for {symbol}.")
        return bars
        
    except Exception as e:
        print(f"Error fetching data from Alpaca: {e}")
        print("Trying alternative data source (Yahoo Finance)...")
        return fetch_from_yahoo(symbol, start_date, end_date)

# ========== STRATEGY SIGNAL GENERATOR ==========
def generate_signals(df, symbol="STOCK", rolling_window=20, std_multiplier=1.0, entropy_threshold=1.2):
    """Apply our mean-reversion + entropy strategy to generate trading signals."""
    if df is None or df.empty:
        return None
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Find close column
    close_col = next((col for col in df.columns if col.lower() == 'close'), None)
    if not close_col:
        print(f"Could not find 'close' column. Available columns: {df.columns.tolist()}")
        raise ValueError("Could not find 'close' column in data")
    
    # Calculate indicators
    print(f"Calculating indicators with window={rolling_window}...")
    df['mean'] = df[close_col].rolling(window=rolling_window).mean()
    df['std'] = df[close_col].rolling(window=rolling_window).std()
    
    # Calculate entropy with padding to ensure same length
    entropy_series = rolling_shannon_entropy(df[close_col], window=rolling_window)
    
    # Verify lengths before assignment
    if len(entropy_series) != len(df):
        print(f"Warning: Entropy series length ({len(entropy_series)}) doesn't match DataFrame length ({len(df)})")
        # Fix by reindexing to match the DataFrame
        entropy_series = entropy_series.reindex(df.index, fill_value=np.nan)
    
    df['entropy'] = entropy_series
    
    # Print entropy statistics to diagnose issues
    valid_entropy = df['entropy'].dropna()
    if len(valid_entropy) > 0:
        min_entropy = valid_entropy.min()
        max_entropy = valid_entropy.max()
        mean_entropy = valid_entropy.mean()
        print(f"Entropy stats: Min={min_entropy:.2f}, Max={max_entropy:.2f}, Mean={mean_entropy:.2f}, Threshold={entropy_threshold}")
        
        # If all entropy values are above threshold, report this
        if min_entropy > entropy_threshold:
            print(f"WARNING: All entropy values are above threshold ({entropy_threshold})!")
            print(f"Consider increasing the threshold to at least {min_entropy + 0.1:.2f}")
    
    # Initialize signal column with None values
    df['signal'] = None
    
    # Count potential signals without entropy filter
    potential_buy = 0
    potential_sell = 0
    filtered_by_entropy = 0
    
    # Generate signals - process each row individually to avoid indexing errors
    signals_generated = 0
    for i in range(rolling_window, len(df)):
        try:
            row = df.iloc[i]
            price = row[close_col]
            mean_val = row['mean']
            std_val = row['std']
            entropy_val = row['entropy']
            
            # Skip if we have missing data
            if np.isnan(mean_val) or np.isnan(std_val) or np.isnan(entropy_val):
                continue
            
            # Mean-reversion logic - count potential signals first
            if price < (mean_val - std_multiplier * std_val):
                potential_buy += 1
                
                # Skip if market is too chaotic
                if entropy_val > entropy_threshold:
                    filtered_by_entropy += 1
                    continue
                    
                df.iloc[i, df.columns.get_loc('signal')] = "BUY"
                signals_generated += 1
                
            elif price > (mean_val + std_multiplier * std_val):
                potential_sell += 1
                
                # Skip if market is too chaotic
                if entropy_val > entropy_threshold:
                    filtered_by_entropy += 1
                    continue
                    
                df.iloc[i, df.columns.get_loc('signal')] = "SELL"
                signals_generated += 1
                
        except Exception as e:
            print(f"Error processing row {i}: {e}")
    
    print(f"Potential signals without entropy filter: BUY={potential_buy}, SELL={potential_sell}")
    print(f"Signals filtered by entropy: {filtered_by_entropy}")
    print(f"Final signals generated: {signals_generated}")
    
    # Add portfolio simulation to see $1 growth
    print("\nSimulating portfolio performance with $1 starting capital...")
    portfolio_results = simulate_portfolio(
        df, 
        symbol=symbol, 
        initial_capital=1.0,
        long_only=LONG_ONLY  # Use the global parameter
    )
    
    if portfolio_results:
        # Visualize portfolio performance
        visualize_portfolio(
            symbol=symbol,  # Pass the symbol explicitly
            portfolio=portfolio_results['portfolio'],
            trades_list=portfolio_results['trades'],
            params={
                'window': rolling_window,
                'multiplier': std_multiplier,
                'threshold': entropy_threshold
            }
        )
    
    return df

# ========== MULTI-SYMBOL BACKTESTER ==========
def run_multi_symbol_backtest(symbols, strategy, start_date, end_date=None, timeframe=tradeapi.TimeFrame.Day, position_sizing=0.2):
    """
    Run a backtest across multiple symbols.
    
    Parameters:
    - symbols: List of stock symbols to backtest
    - strategy: Strategy object with generate_signals method
    - start_date: Start date for the backtest
    - end_date: End date for the backtest (optional, defaults to today)
    - timeframe: Timeframe to use (1Day, 1Hour, etc)
    - position_sizing: Fraction of cash to allocate per position
    
    Returns:
    - Backtest results
    """
    # Initialize position handler with starting balance
    backtester = PositionHandler(starting_balance=10000, live_trading=False)
    
    # Fetch data for each symbol
    data = {}
    date_list = None
    
    # Try to get data for each symbol
    for symbol in symbols:
        try:
            symbol_data = fetch_historical_data(symbol, timeframe, start_date, end_date)
            print("symbol_data", symbol_data)
            if symbol_data is not None and not symbol_data.empty:
                # Apply strategy to generate signals
                symbol_data = strategy.generate_signals(symbol_data)
                data[symbol] = symbol_data
                
                # Keep track of the overall date list (use the first symbol's dates)
                if date_list is None:
                    date_list = symbol_data.index.tolist()
                    print(f"Got {len(date_list)} data points for {symbol}")
            else:
                print(f"No data available for {symbol}")
        except Exception as e:
            print(f"Error processing symbol {symbol}: {e}")
    
    if not data or date_list is None:
        print("No data available for backtesting. Please check your subscription or try different dates.")
        return None
    
    # If we made it here, we have data to work with
    print(f"Successfully retrieved data for {len(data)} symbols out of {len(symbols)}")
    
    # Process each time step
    print(f"Running backtest on {len(symbols)} symbols from {start_date} to {end_date or 'today'}...")
    
    benchmark_data = {}
    equity_data = {}
    
    for date_idx, current_date in enumerate(date_list):
        # Skip first few dates until we have enough data for indicators
        if date_idx < strategy.rolling_window:
            continue
        
        # Track prices for equity calculation
        current_prices = {}
        
        for symbol in symbols:
            if symbol not in data:
                continue
                
            symbol_data = data[symbol]
            
            # Check if this date exists in the symbol's data
            if current_date not in symbol_data.index:
                continue
            
            current_row = symbol_data.loc[current_date]
            
            # Store current price for equity calculation
            close_col = next((col for col in symbol_data.columns if col.lower() == 'close'), None)
            if close_col:
                current_prices[symbol] = current_row[close_col]
            
            # Get current signal
            signal = current_row.get('signal')
            
            if signal == "BUY":
                # Calculate position size based on available cash
                cash_to_use = backtester.cash_balance * position_sizing
                shares_to_buy = int(cash_to_use / current_prices[symbol])
                
                if shares_to_buy > 0:
                    # Place a buy order
                    order = backtester.place_order(
                        symbol=symbol,
                        quantity=shares_to_buy,
                        side="buy",
                        order_type="market",
                        time_in_force="day",
                        timestamp=current_date
                    )
                    
                    # Execute the order immediately at current price (simplified)
                    if order:
                        backtester.execute_order(order, current_prices[symbol], current_date)
                        print(f"{current_date}: BUY {shares_to_buy} shares of {symbol} at ${current_prices[symbol]:.2f}")
            
            elif signal == "SELL":
                # Check for existing position
                position = backtester.return_open_position(symbol)
                
                if position > 0:
                    # Place a sell order to close the position
                    order = backtester.place_order(
                        symbol=symbol,
                        quantity=position,
                        side="sell",
                        order_type="market",
                        time_in_force="day",
                        timestamp=current_date
                    )
                    
                    # Execute the order immediately at current price (simplified)
                    if order:
                        backtester.execute_order(order, current_prices[symbol], current_date)
                        print(f"{current_date}: SELL {position} shares of {symbol} at ${current_prices[symbol]:.2f}")
        
        # Update equity value with current prices
        backtester.update_equity(current_prices, current_date)
        
        # Store benchmark value for SPY (or first symbol as fallback)
        benchmark_symbol = "SPY" if "SPY" in data else symbols[0]
        if benchmark_symbol in current_prices:
            if date_idx == strategy.rolling_window:  # First valid date
                initial_benchmark_price = current_prices[benchmark_symbol]
                benchmark_data[current_date] = 1.0  # Normalized to 1.0
            else:
                # Calculate normalized benchmark value
                benchmark_data[current_date] = current_prices[benchmark_symbol] / initial_benchmark_price
        
        # Store equity data for comparison
        if date_idx == strategy.rolling_window:  # First valid date
            initial_equity = backtester.equity_history[-1]
            equity_data[current_date] = 1.0  # Normalized to 1.0
        else:
            # Calculate normalized equity value
            equity_data[current_date] = backtester.equity_history[-1] / initial_equity
    
    # Calculate final metrics
    final_equity = backtester.equity_history[-1]
    total_return = (final_equity - backtester.starting_balance) / backtester.starting_balance * 100
    
    print("\n===== BACKTEST RESULTS =====")
    print(f"Starting Balance: ${backtester.starting_balance:.2f}")
    print(f"Final Equity: ${final_equity:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    
    # Calculate more metrics
    equity_series = pd.Series(backtester.equity_history[strategy.rolling_window:], index=date_list[strategy.rolling_window:])
    daily_returns = equity_series.pct_change().dropna()
    
    if len(daily_returns) > 0:
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)  # Annualized
        max_drawdown = (equity_series / equity_series.cummax() - 1).min() * 100
        
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    # Plot strategy equity curve vs benchmark
    plt.subplot(2, 1, 1)
    plt.plot(equity_data.keys(), equity_data.values(), label="Strategy", color="blue")
    plt.plot(benchmark_data.keys(), benchmark_data.values(), label="Benchmark", color="green", alpha=0.7)
    plt.title(f"Strategy vs Benchmark ({', '.join(symbols)})")
    plt.legend()
    plt.grid(True)
    
    # Plot underwater equity (drawdowns)
    plt.subplot(2, 1, 2)
    underwater = (equity_series / equity_series.cummax() - 1) * 100
    plt.fill_between(underwater.index, underwater, 0, color="red", alpha=0.3)
    plt.title("Drawdowns")
    plt.ylabel("Drawdown %")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"backtest_results_{datetime.now().strftime('%Y%m%d')}.png")
    plt.show()
    
    return {
        "backtester": backtester,
        "equity_series": equity_series,
        "benchmark_data": pd.Series(benchmark_data),
        "symbols": symbols,
        "total_return": total_return,
        "sharpe": sharpe if 'sharpe' in locals() else None,
        "max_drawdown": max_drawdown if 'max_drawdown' in locals() else None
    }

# ========== PORTFOLIO BACKTESTING (LONG-ONLY) ==========
def simulate_portfolio(df, symbol="STOCK", initial_capital=1.0, commission=0.0, 
                      use_stop_loss=True, stop_loss_pct=10.0, 
                      use_time_exit=True, max_days_in_trade=10,
                      long_only=True):
    """
    Simulate a portfolio using the generated signals with risk management.
    
    Parameters:
    - df: DataFrame with price data and signals
    - symbol: Stock symbol (default "STOCK" if not specified)
    - initial_capital: Starting capital (default $1)
    - commission: Commission per trade (default 0)
    - use_stop_loss: Whether to use stop-loss exits
    - stop_loss_pct: Stop-loss percentage (default 10%)
    - use_time_exit: Whether to use time-based exits
    - max_days_in_trade: Maximum days to hold a position
    - long_only: If True, only allow long positions (no shorting)
    
    Returns:
    - Dictionary with portfolio performance metrics and data
    """
    if df is None or df.empty:
        return None
        
    # Find close column and signal column
    close_col = next((col for col in df.columns if col.lower() == 'close'), None)
    signal_col = 'signal'
    
    if not close_col or signal_col not in df.columns:
        print(f"Missing columns. Available: {df.columns.tolist()}")
        return None
    
    # Create a copy of the DataFrame
    portfolio = df.copy()
    
    # Initialize portfolio columns
    portfolio['position'] = 0.0  # Position in shares
    portfolio['cash'] = initial_capital  # Cash balance
    portfolio['holdings'] = 0.0  # Value of holdings
    portfolio['equity'] = initial_capital  # Total portfolio value
    portfolio['returns'] = 0.0  # Daily returns
    
    # Track trades
    trades = []
    
    # Process each day
    for i in range(1, len(portfolio)):
        # Default: carry forward previous position and cash
        portfolio.loc[portfolio.index[i], 'position'] = portfolio.loc[portfolio.index[i-1], 'position']
        portfolio.loc[portfolio.index[i], 'cash'] = portfolio.loc[portfolio.index[i-1], 'cash']
        
        # Check for signal
        signal = portfolio.loc[portfolio.index[i], signal_col]
        current_price = portfolio.loc[portfolio.index[i], close_col]
        prev_position = portfolio.loc[portfolio.index[i], 'position']
        
        # Execute trades based on signals
        if signal == 'BUY' and prev_position <= 0:
            # Close any short position first
            if prev_position < 0:
                close_cost = abs(prev_position) * current_price + commission
                portfolio.loc[portfolio.index[i], 'cash'] -= close_cost
                
                # Record the trade
                trades.append({
                    'date': portfolio.index[i],
                    'action': 'COVER',
                    'price': current_price,
                    'shares': abs(prev_position),
                    'value': close_cost,
                    'commission': commission
                })
            
            # Calculate how many shares we can buy
            available_cash = portfolio.loc[portfolio.index[i], 'cash']
            max_shares = available_cash / (current_price + commission) if commission > 0 else available_cash / current_price
            
            # Buy with all available cash
            portfolio.loc[portfolio.index[i], 'position'] = max_shares
            portfolio.loc[portfolio.index[i], 'cash'] = 0
            
            # Record the trade
            trades.append({
                'date': portfolio.index[i],
                'action': 'BUY',
                'price': current_price,
                'shares': max_shares,
                'value': max_shares * current_price,
                'commission': commission
            })
            
        elif signal == 'SELL' and prev_position >= 0:
            # Close any long position first
            if prev_position > 0:
                close_value = prev_position * current_price - commission
                portfolio.loc[portfolio.index[i], 'cash'] += close_value
                
                # Record the trade
                trades.append({
                    'date': portfolio.index[i],
                    'action': 'SELL',
                    'price': current_price,
                    'shares': prev_position,
                    'value': close_value,
                    'commission': commission
                })
            
            # Only go short if long_only is False
            if not long_only:
                # Calculate short position size (use all cash)
                available_cash = portfolio.loc[portfolio.index[i], 'cash']
                max_shares = available_cash / (current_price + commission) if commission > 0 else available_cash / current_price
                
                # Short sell
                portfolio.loc[portfolio.index[i], 'position'] = -max_shares
                portfolio.loc[portfolio.index[i], 'cash'] = available_cash + (max_shares * current_price)
                
                # Record the trade
                trades.append({
                    'date': portfolio.index[i],
                    'action': 'SHORT',
                    'price': current_price,
                    'shares': max_shares,
                    'value': max_shares * current_price,
                    'commission': commission
                })
        
        # Update holdings value and equity
        portfolio.loc[portfolio.index[i], 'holdings'] = portfolio.loc[portfolio.index[i], 'position'] * current_price
        portfolio.loc[portfolio.index[i], 'equity'] = portfolio.loc[portfolio.index[i], 'cash'] + portfolio.loc[portfolio.index[i], 'holdings']
        
        # Calculate daily returns
        prev_equity = portfolio.loc[portfolio.index[i-1], 'equity']
        current_equity = portfolio.loc[portfolio.index[i], 'equity']
        portfolio.loc[portfolio.index[i], 'returns'] = (current_equity / prev_equity) - 1
    
    # Calculate cumulative returns
    portfolio['cumulative_returns'] = (1 + portfolio['returns']).cumprod()
    
    # Calculate drawdowns
    portfolio['peak'] = portfolio['equity'].cummax()
    portfolio['drawdown'] = (portfolio['equity'] / portfolio['peak']) - 1
    
    # Calculate metrics
    start_equity = initial_capital
    end_equity = portfolio['equity'].iloc[-1]
    total_return_pct = ((end_equity / start_equity) - 1) * 100
    
    # Annualized return (assumes 252 trading days per year)
    days = (portfolio.index[-1] - portfolio.index[0]).days
    years = days / 365
    annualized_return = (((end_equity / start_equity) ** (1 / max(years, 1e-10))) - 1) * 100
    
    # Calculate Sharpe ratio (annualized, risk-free rate = 0)
    daily_returns = portfolio['returns'].dropna()
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5) if daily_returns.std() > 0 else 0
    
    # Max drawdown
    max_drawdown = portfolio['drawdown'].min() * 100
    
    # Calculate win/loss stats
    if trades:
        trades_df = pd.DataFrame(trades)
        # Group consecutive trades of the same type
        grouped_trades = []
        current_group = {'action': None, 'entry_date': None, 'entry_price': 0, 'exit_date': None, 'exit_price': 0, 'shares': 0, 'pnl': 0}
        
        for i, trade in enumerate(trades):
            if trade['action'] in ['BUY', 'SHORT']:
                # Start new trade
                if current_group['action'] is not None and current_group['exit_date'] is None:
                    # Close previous incomplete trade
                    grouped_trades.append(current_group.copy())
                
                current_group = {
                    'action': trade['action'],
                    'entry_date': trade['date'],
                    'entry_price': trade['price'],
                    'shares': trade['shares'],
                    'exit_date': None,
                    'exit_price': 0,
                    'pnl': 0
                }
            elif trade['action'] in ['SELL', 'COVER']:
                if (trade['action'] == 'SELL' and current_group['action'] == 'BUY') or \
                   (trade['action'] == 'COVER' and current_group['action'] == 'SHORT'):
                    # Close current trade
                    current_group['exit_date'] = trade['date']
                    current_group['exit_price'] = trade['price']
                    
                    # Calculate P&L
                    if current_group['action'] == 'BUY':
                        current_group['pnl'] = (current_group['exit_price'] - current_group['entry_price']) * current_group['shares']
                    else:  # SHORT
                        current_group['pnl'] = (current_group['entry_price'] - current_group['exit_price']) * current_group['shares']
                    
                    grouped_trades.append(current_group.copy())
                    current_group = {'action': None, 'entry_date': None, 'entry_price': 0, 'exit_date': None, 'exit_price': 0, 'shares': 0, 'pnl': 0}
        
        # Add any incomplete trade
        if current_group['action'] is not None:
            current_group['exit_date'] = portfolio.index[-1]
            current_group['exit_price'] = portfolio[close_col].iloc[-1]
            
            # Calculate P&L for incomplete trade
            if current_group['action'] == 'BUY':
                current_group['pnl'] = (current_group['exit_price'] - current_group['entry_price']) * current_group['shares']
            else:  # SHORT
                current_group['pnl'] = (current_group['entry_price'] - current_group['exit_price']) * current_group['shares']
                
            grouped_trades.append(current_group.copy())
        
        # Calculate win/loss metrics
        grouped_trades_df = pd.DataFrame(grouped_trades)
        if not grouped_trades_df.empty:
            winning_trades = grouped_trades_df[grouped_trades_df['pnl'] > 0]
            losing_trades = grouped_trades_df[grouped_trades_df['pnl'] < 0]
            
            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            total_trades = len(grouped_trades_df)
            
            win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0
            
            avg_win = winning_trades['pnl'].mean() if not winning_trades.empty else 0
            avg_loss = losing_trades['pnl'].mean() if not losing_trades.empty else 0
            
            profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if not losing_trades.empty and losing_trades['pnl'].sum() != 0 else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            win_count = 0
            loss_count = 0
            total_trades = 0
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
        win_count = 0
        loss_count = 0
        total_trades = 0
    
    # Print performance summary
    print("\n===== PORTFOLIO PERFORMANCE =====")
    print(f"Starting Capital: ${start_equity:.2f}")
    print(f"Final Equity: ${end_equity:.2f}")
    print(f"Total Return: {total_return_pct:.2f}%")
    print(f"Annualized Return: {annualized_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Win Rate: {win_rate:.2f}% ({win_count}/{total_trades})")
    print(f"Avg Win: ${avg_win:.2f}")
    print(f"Avg Loss: ${avg_loss:.2f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    
    # Return the portfolio DataFrame and performance metrics
    return {
        'portfolio': portfolio,
        'trades': trades,
        'symbol': symbol,
        'metrics': {
            'symbol': symbol,
            'total_return_pct': total_return_pct,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'stop_loss_exits': len(stop_loss_exits) if 'stop_loss_exits' in locals() else 0,
            'time_exits': len(time_exits) if 'time_exits' in locals() else 0
        }
    }

# ========== VISUALIZATION FUNCTION ==========
def visualize_portfolio(symbol, portfolio, trades_list=None, params=None):
    """
    Visualize portfolio performance over time.
    
    Parameters:
    - symbol: Stock symbol
    - portfolio: Portfolio DataFrame
    - trades_list: List of trade dictionaries
    - params: Strategy parameters
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    # Create figure and axis
    fig, axes = plt.subplots(3, 1, figsize=(16, 18), gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Find close column
    close_col = next((col for col in portfolio.columns if col.lower() == 'close'), None)
    
    # 1. Equity curve with price
    ax1 = axes[0]
    
    # Plot equity and price
    ax1_left = ax1
    ax1_right = ax1.twinx()
    
    equity_line, = ax1_left.plot(portfolio.index, portfolio['equity'], 'b-', linewidth=2, label='Portfolio Equity')
    price_line, = ax1_right.plot(portfolio.index, portfolio[close_col], 'gray', alpha=0.5, label=f'{symbol} Price')
    
    # Plot buy/sell markers if trades_list is provided
    if trades_list:
        buys = [t for t in trades_list if t['action'] == 'BUY']
        sells = [t for t in trades_list if t['action'] == 'SELL']
        shorts = [t for t in trades_list if t['action'] == 'SHORT']
        covers = [t for t in trades_list if t['action'] == 'COVER']
        
        if buys:
            buy_dates = [t['date'] for t in buys]
            buy_prices = [t['price'] for t in buys]
            ax1_right.scatter(buy_dates, buy_prices, marker='^', color='green', s=100, label='Buy')
        
        if sells:
            sell_dates = [t['date'] for t in sells]
            sell_prices = [t['price'] for t in sells]
            ax1_right.scatter(sell_dates, sell_prices, marker='v', color='red', s=100, label='Sell')
        
        if shorts:
            short_dates = [t['date'] for t in shorts]
            short_prices = [t['price'] for t in shorts]
            ax1_right.scatter(short_dates, short_prices, marker='v', color='purple', s=100, label='Short')
        
        if covers:
            cover_dates = [t['date'] for t in covers]
            cover_prices = [t['price'] for t in covers]
            ax1_right.scatter(cover_dates, cover_prices, marker='^', color='orange', s=100, label='Cover')
    
    # Add strategy parameters as text if provided
    if params:
        param_text = f"Symbol: {symbol} | Window: {params['window']}, Multiplier: {params['multiplier']}, Threshold: {params['threshold']}"
        ax1.text(0.02, 0.05, param_text, transform=ax1.transAxes, 
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    # Add legend
    lines_1, labels_1 = ax1_left.get_legend_handles_labels()
    lines_2, labels_2 = ax1_right.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
    
    # Set labels
    ax1_left.set_ylabel('Portfolio Value ($)', color='b')
    ax1_right.set_ylabel(f'{symbol} Price ($)', color='gray')
    ax1.set_title(f'{symbol} - Portfolio Performance vs. Price')
    ax1.grid(True)
    
    # 2. Drawdown chart
    ax2 = axes[1]
    ax2.fill_between(portfolio.index, 0, portfolio['drawdown'] * 100, color='red', alpha=0.3)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_title(f'{symbol} - Portfolio Drawdown')
    ax2.grid(True)
    
    # Add horizontal line at 0%
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    
    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    # 3. Position over time
    ax3 = axes[2]
    ax3.plot(portfolio.index, portfolio['position'], 'g-', label='Position Size')
    ax3.fill_between(portfolio.index, 0, portfolio['position'], where=portfolio['position'] >= 0, 
                      color='green', alpha=0.3, label='Long')
    ax3.fill_between(portfolio.index, 0, portfolio['position'], where=portfolio['position'] <= 0, 
                      color='red', alpha=0.3, label='Short')
    ax3.set_ylabel('Position Size (Shares)')
    ax3.set_title(f'{symbol} - Position Size Over Time')
    ax3.grid(True)
    ax3.legend(loc='upper left')
    
    # Format x-axis dates
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    filename = f"{symbol}_portfolio_performance.png"
    plt.savefig(filename, dpi=150)
    print(f"Portfolio visualization for {symbol} saved as {filename}")
    
    plt.show()

# ========== MAIN FUNCTION ==========
def main():
    """Main function to run the multi-symbol backtest."""
    print("=" * 50)
    print("MULTI-SYMBOL ENTROPY MEAN REVERSION BACKTEST")
    print("=" * 50)
    
    # Ensure yfinance is installed
    try:
        import yfinance
        print("yfinance is installed.")
    except ImportError:
        print("Installing yfinance...")
        import subprocess
        subprocess.check_call(["pip", "install", "yfinance"])
        print("yfinance installed successfully.")
    
    # Display configuration
    print(f"Backtesting from {START_DATE} to {END_DATE}")
    print(f"Testing {len(SYMBOLS)} symbols: {', '.join(SYMBOLS)}")
    print(f"Strategy mode: {'Long-only' if LONG_ONLY else 'Long-short'}")
    print(f"Initial capital: ${INITIAL_CAPITAL:.2f}")
    
    # Use daily timeframe for more stable signals
    timeframe = tradeapi.TimeFrame.Day
    
    # Process each symbol
    data = {}
    
    print("\nTesting multiple parameter combinations:")
    for params in PARAMETER_SETS:
        print(f"\n{params['name']} configuration:")
        print(f"  Window: {params['window']}, Multiplier: {params['multiplier']}, Threshold: {params['threshold']}")
        
        symbol_results = {}
        for symbol in SYMBOLS:
            try:
                # Fetch data (only need to do this once)
                if symbol not in data:
                    print(f"\nProcessing {symbol}...")
                    df = fetch_historical_data(symbol, timeframe, START_DATE, END_DATE)
                    print("df - shape", df.shape)
                    print("df - columns", df.columns)
                    if df is not None and not df.empty:
                        data[symbol] = df # store the dataframe in the data dictionary
                    else:
                        print(f"No data available for {symbol}")
                        continue
                
                # Generate signals with current parameter set
                signals_df = generate_signals(
                    data[symbol], # this is symobl level time series data; with all the columns - close high low trade-count open volume vwap
                    symbol=symbol,
                    rolling_window=params['window'],
                    std_multiplier=params['multiplier'],
                    entropy_threshold=params['threshold']
                )
                
                if signals_df is not None:
                    # Count signals
                    buy_signals = signals_df[signals_df['signal'] == 'BUY'].shape[0]
                    sell_signals = signals_df[signals_df['signal'] == 'SELL'].shape[0]
                    
                    symbol_results[symbol] = {
                        "buy": buy_signals,
                        "sell": sell_signals,
                        "total": buy_signals + sell_signals
                    }
                    
                    print(f"  {symbol}: {buy_signals} BUY, {sell_signals} SELL signals")
                else:
                    print(f"  Failed to generate signals for {symbol}")
            except Exception as e:
                print(f"  Error processing {symbol}: {e}")
        
        # Summarize results for this parameter set
        total_buy = sum(res["buy"] for res in symbol_results.values())
        total_sell = sum(res["sell"] for res in symbol_results.values())
        print(f"\n  Summary: {total_buy} BUY, {total_sell} SELL signals across all symbols")
        
        # If we have signals, display some examples
        if total_buy + total_sell > 0:
            # Find a symbol with signals
            for symbol, counts in symbol_results.items():
                if counts["total"] > 0:
                    signals_df = generate_signals(
                        data[symbol],
                        symbol=symbol,
                        rolling_window=params['window'],
                        std_multiplier=params['multiplier'],
                        entropy_threshold=params['threshold']
                    )
                    signals = signals_df[signals_df['signal'].notnull()]
                    
                    if len(signals) > 0:
                        print(f"\n  Example signals for {symbol}:")
                        example_signals = signals[['close', 'mean', 'std', 'entropy', 'signal']].head(3)
                        print(example_signals)
                    break
    
    # After processing all parameters and symbols, compare portfolio performance
    print("\nComparing portfolio performance across parameter sets:")
    
    portfolio_metrics = {}
    
    for params in PARAMETER_SETS:
        print(f"\n{params['name']} configuration:")
        
        for symbol in SYMBOLS:
            if symbol in data:
                # Generate signals with symbol parameter
                signals_df = generate_signals(
                    data[symbol],
                    symbol=symbol,
                    rolling_window=params['window'],
                    std_multiplier=params['multiplier'],
                    entropy_threshold=params['threshold']
                )
                
                if signals_df is not None:
                    # Simulate portfolio
                    results = simulate_portfolio(signals_df)
                    
                    if results:
                        key = f"{params['name']}_{symbol}"
                        portfolio_metrics[key] = results['metrics']
    
    # Print comparison table
    if portfolio_metrics:
        print("\n===== PORTFOLIO PERFORMANCE COMPARISON =====")
        print(f"{'Configuration':20} {'Symbol':6} {'Return %':10} {'Annual %':10} {'Sharpe':8} {'DrawDown %':10} {'Win Rate %':10}")
        print("-" * 80)
        
        for key, metrics in portfolio_metrics.items():
            param_name, symbol = key.split('_')
            print(f"{param_name:20} {symbol:6} {metrics['total_return_pct']:10.2f} {metrics['annualized_return']:10.2f} {metrics['sharpe_ratio']:8.2f} {metrics['max_drawdown']:10.2f} {metrics['win_rate']:10.2f}")
    
    print("\nBacktesting complete!")

if __name__ == "__main__":
    main() 