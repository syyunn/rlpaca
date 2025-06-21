"""
Realistic Offline Trading Environment with proper time-based tick feeding
"""

import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime, timedelta
from bisect import bisect_left, bisect_right
import gym
from gym import spaces
import random
import structlog

logger = structlog.get_logger()


class RealisticOfflineEnv(gym.Env):
    """
    Realistic offline environment that properly simulates:
    1. Time-based tick feeding (no look-ahead)
    2. Variable tick rates throughout the day
    3. Realistic order execution latencies
    4. Proper tick buffer management
    """
    
    def __init__(
        self,
        historical_ticks: list,
        historical_bars: pd.DataFrame,
        decision_interval_seconds: int = 5,
        initial_capital: float = 100000,
        max_position: float = None,  # None = no position limits
        transaction_cost: float = 0.001,
        max_ticks_in_buffer: int = 100,
        tick_buffer_max_age_seconds: int = 60,
        order_latency_ms: tuple = (5, 50),  # min, max milliseconds
    ):
        super().__init__()
        
        # Market data
        self.all_ticks = sorted(historical_ticks, key=lambda x: x['timestamp'])
        self.tick_timestamps = [t['timestamp'] for t in self.all_ticks]
        self.historical_bars = historical_bars
        
        # Timing - use UTC times since our data is in UTC
        self.decision_interval = timedelta(seconds=decision_interval_seconds)
        self.market_open = pd.Timestamp('13:30:00').time()  # 9:30 AM ET in UTC
        self.market_close = pd.Timestamp('20:00:00').time()  # 4:00 PM ET in UTC
        
        # Trading parameters
        self.initial_capital = initial_capital
        self.max_position = max_position  # None means no limit
        self.transaction_cost = transaction_cost
        self.order_latency_ms = order_latency_ms
        
        # Tick buffer management
        self.max_ticks = max_ticks_in_buffer
        self.tick_buffer_max_age = timedelta(seconds=tick_buffer_max_age_seconds)
        self.tick_buffer = deque()
        
        # State tracking
        self.current_time = None
        self.next_decision_time = None
        self.tick_pointer = 0  # Current position in all_ticks
        
        # Portfolio state
        self.capital = initial_capital
        self.position = 0.0
        self.trades_executed = []
        
        # Minute bars seen so far
        self.minute_bars_seen = []
        
        # Gym spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5185,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -10.0]), 
            high=np.array([1.0, 10.0]), 
            dtype=np.float32
        )
        
    def reset(self):
        """Reset for new trading day"""
        # Find first tick of the day
        day_date = self.all_ticks[0]['timestamp'].date()
        
        # Use pandas for timezone-aware datetime
        self.current_time = pd.Timestamp.combine(
            day_date, 
            self.market_open
        ).tz_localize(self.all_ticks[0]['timestamp'].tz)
        
        self.next_decision_time = self.current_time + self.decision_interval
        self.tick_pointer = 0
        
        # Reset portfolio
        self.capital = self.initial_capital
        self.position = 0.0
        self.trades_executed = []
        
        # Reset buffers
        self.tick_buffer.clear()
        self.minute_bars_seen = []
        
        # Feed initial ticks
        self._feed_ticks_until(self.current_time)
        
        return self._get_state()
        
    def step(self, action):
        """Execute one trading step"""
        # 1. Record decision time and simulate order latency
        decision_time = self.current_time
        latency_ms = random.uniform(*self.order_latency_ms)
        order_time = decision_time + timedelta(milliseconds=latency_ms)
        
        # 2. Get portfolio value before trade
        current_price = self._get_current_price()
        portfolio_before = self.capital + self.position * current_price
        
        # 3. Execute trade at realistic price (next tick after order_time)
        execution_price, execution_time = self._find_execution_price(order_time)
        if execution_price is not None:
            self._execute_trade(action, execution_price, execution_time)
        
        # 4. Feed all ticks up to next decision time
        self._feed_ticks_until(self.next_decision_time)
        
        # 5. Update minute bars seen
        self._update_minute_bars()
        
        # 6. Calculate reward
        new_price = self._get_current_price()
        portfolio_after = self.capital + self.position * new_price
        reward = (portfolio_after - portfolio_before) / portfolio_before * 100
        
        # Penalize constraint violations
        if hasattr(self, 'constraint_violations') and self.constraint_violations:
            constraint_penalty = -0.1 * len(self.constraint_violations)  # -0.1% per violation
            reward += constraint_penalty
        
        # 7. Check if done
        done = self._is_end_of_day()
        
        # Apply end-of-day penalty for open positions
        if done and abs(self.position) > 0.01:
            # Strong penalty for holding positions overnight
            position_penalty = -2.0 * abs(self.position) / self.max_position  # -2% per full position
            reward += position_penalty
            self.constraint_violations.append('overnight_position')
            logger.info(f"End of day penalty: {position_penalty:.2f}% for {self.position:.2f} shares")
        
        # 8. Advance time
        self.current_time = self.next_decision_time
        self.next_decision_time += self.decision_interval
        
        info = {
            'time': self.current_time,
            'portfolio_value': portfolio_after,
            'position': self.position,
            'capital': self.capital,
            'execution_price': execution_price,
            'ticks_in_interval': len([t for t in self.tick_buffer 
                                     if t['timestamp'] >= decision_time]),
            'total_ticks_seen': self.tick_pointer,
            'constraint_violations': getattr(self, 'constraint_violations', [])
        }
        
        return self._get_state(), reward, done, info
        
    def _feed_ticks_until(self, target_time):
        """Feed all ticks from current pointer until target time"""
        while self.tick_pointer < len(self.all_ticks):
            tick = self.all_ticks[self.tick_pointer]
            
            # Stop if we've reached future ticks
            if tick['timestamp'] > target_time:
                break
                
            # Add to buffer
            self.tick_buffer.append(tick)
            
            # Remove old ticks beyond max age
            cutoff_time = tick['timestamp'] - self.tick_buffer_max_age
            while self.tick_buffer and self.tick_buffer[0]['timestamp'] < cutoff_time:
                self.tick_buffer.popleft()
                
            # Maintain max buffer size
            while len(self.tick_buffer) > self.max_ticks:
                self.tick_buffer.popleft()
                
            self.tick_pointer += 1
            
    def _find_execution_price(self, order_time):
        """Find next tick after order_time for realistic execution"""
        # Search from current pointer forward
        for i in range(self.tick_pointer, len(self.all_ticks)):
            tick = self.all_ticks[i]
            if tick['timestamp'] >= order_time:
                # Use mid price if we have bid/ask, otherwise use trade price
                if 'bid' in tick and 'ask' in tick:
                    return (tick['bid'] + tick['ask']) / 2, tick['timestamp']
                else:
                    return tick['price'], tick['timestamp']
        return None, None
        
    def _execute_trade(self, action, execution_price, execution_time):
        """Execute trade at given price"""
        position_delta = action[0]  # -1 to 1
        
        # Track if we hit any constraints
        self.constraint_violations = []
        
        if abs(position_delta) < 0.1:  # No significant action
            return
            
        if position_delta > 0:  # Buy
            # FIX: Calculate actual affordable shares (no crazy leverage!)
            max_shares_affordable = self.capital / execution_price / (1 + self.transaction_cost)
            
            # Check if we're short and need to cover first
            if self.position < 0:
                # Buying to cover short position
                max_shares_allowed = abs(self.position)
            else:
                # Regular buy - only limited by capital
                max_shares_allowed = max_shares_affordable
            
            max_buy = max_shares_allowed
            target_delta = position_delta * max_buy
            
            if target_delta > 0.01:
                cost = target_delta * execution_price * (1 + self.transaction_cost)
                self.position += target_delta
                self.capital -= cost
                
                self.trades_executed.append({
                    'time': execution_time,
                    'side': 'buy',
                    'quantity': target_delta,
                    'price': execution_price,
                    'cost': cost
                })
                
        else:  # Sell
            if self.position > 0:
                # Selling long position only - NO SHORT SELLING
                target_delta = min(abs(position_delta) * self.position, self.position)
            else:
                # No short selling allowed - action does nothing
                self.constraint_violations.append('no_short_selling')
                target_delta = 0  # No change in position
            
            if target_delta > 0.01:
                proceeds = target_delta * execution_price * (1 - self.transaction_cost)
                self.position -= target_delta
                self.capital += proceeds
                
                self.trades_executed.append({
                    'time': execution_time,
                    'side': 'sell',
                    'quantity': target_delta,
                    'price': execution_price,
                    'proceeds': proceeds
                })
                
    def _get_current_price(self):
        """Get current price from tick buffer"""
        if not self.tick_buffer:
            return 0
            
        latest_tick = self.tick_buffer[-1]
        if 'bid' in latest_tick and 'ask' in latest_tick:
            return (latest_tick['bid'] + latest_tick['ask']) / 2
        return latest_tick.get('price', 0)
        
    def _update_minute_bars(self):
        """Update minute bars seen up to current time"""
        current_minute = self.current_time.replace(second=0, microsecond=0)
        
        # Get the last seen minute
        last_seen = self.minute_bars_seen[-1].name if self.minute_bars_seen else pd.Timestamp.min.tz_localize(self.current_time.tz)
        
        # Add any minute bars we haven't seen yet
        mask = (self.historical_bars.index <= current_minute) & (self.historical_bars.index > last_seen)
        
        new_bars = self.historical_bars[mask]
        for idx, bar in new_bars.iterrows():
            self.minute_bars_seen.append(bar)
            
    def _is_end_of_day(self):
        """Check if trading day is over"""
        return self.current_time.time() >= self.market_close
        
    def _get_state(self) -> np.ndarray:
        """
        Get current state - same as ultra-pure but with realistic data
        """
        features = []
        
        # 1. Recent tick data (500 dims)
        tick_matrix = np.zeros((100, 5))
        ticks = list(self.tick_buffer)
        
        for i in range(min(100, len(ticks))):
            t = ticks[-(i+1)]  # Most recent first
            tick_matrix[i] = [
                t.get('bid', 0),
                t.get('ask', 0),
                t.get('bid_size', 0),
                t.get('ask_size', 0),
                t['timestamp'].timestamp() if isinstance(t['timestamp'], pd.Timestamp) else t['timestamp']
            ]
        features.extend(tick_matrix.flatten())
        
        # 2. Minute bars (4,680 dims)
        minute_matrix = np.zeros((390, 12))
        
        for i in range(min(390, len(self.minute_bars_seen))):
            bar = self.minute_bars_seen[i]
            minute_matrix[i] = [
                bar.get('open', 0),
                bar.get('high', 0),
                bar.get('low', 0),
                bar.get('close', 0),
                bar.get('volume', 0),
                bar.get('trade_count', 0),
                bar.get('vwap', 0),
                (bar.get('close', 0) - bar.get('open', 0)) / bar.get('open', 1) * 100,  # return
                bar.get('high', 0) - bar.get('low', 0),  # range
                bar.name.hour if hasattr(bar.name, 'hour') else 0,
                bar.name.minute if hasattr(bar.name, 'minute') else 0,
                (i - 195) / 195  # normalized position in day
            ]
        features.extend(minute_matrix.flatten())
        
        # 3. Position state (5 dims)
        current_price = self._get_current_price()
        portfolio_value = self.capital + self.position * current_price
        
        position_features = [
            self.position,
            self.capital,
            self.position * current_price if current_price > 0 else 0,
            (portfolio_value - self.initial_capital) / self.initial_capital,
            len(self.minute_bars_seen) / 390  # Progress through day
        ]
        features.extend(position_features)
        
        return np.array(features, dtype=np.float32)
        
    def get_tick_distribution_stats(self):
        """Analyze tick distribution for current episode"""
        if not self.trades_executed:
            return {}
            
        # Group ticks by minute
        tick_counts = {}
        for i in range(self.tick_pointer):
            tick = self.all_ticks[i]
            minute = tick['timestamp'].replace(second=0, microsecond=0)
            tick_counts[minute] = tick_counts.get(minute, 0) + 1
            
        return {
            'total_ticks': self.tick_pointer,
            'avg_ticks_per_minute': np.mean(list(tick_counts.values())),
            'max_ticks_per_minute': max(tick_counts.values()),
            'min_ticks_per_minute': min(tick_counts.values()),
            'trades_executed': len(self.trades_executed)
        }