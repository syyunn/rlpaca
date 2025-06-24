"""
Realistic Offline Trading Environment with proper time-based tick feeding
"""

import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime, timedelta
from bisect import bisect_left, bisect_right
import gymnasium as gym
from gymnasium import spaces
import random
import structlog

logger = structlog.get_logger()

# Normalization constants with explanations
# NVDA typically trades 50-200M shares/day, so minute volumes are in millions
VOLUME_NORMALIZER = 1e6  # Convert to millions of shares

# NVDA tick sizes are typically 100-10,000 shares per quote
SIZE_NORMALIZER = 1e3  # Convert to thousands of shares  

# Trade counts per minute bar typically 100-5,000 for liquid stocks
TRADE_COUNT_NORMALIZER = 1e3  # Convert to thousands of trades

# Position sizes for $100k account at ~$140 stock = ~700 shares max
POSITION_NORMALIZER = 100  # Convert to hundreds of shares


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
        min_trade_value: float = 100,  # NEW: Minimum $ value for a trade
        enable_masking: bool = True,    # NEW: Enable action masking
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
        
        # NEW: Action masking parameters
        self.min_trade_value = min_trade_value
        self.enable_masking = enable_masking
        self.invalid_action_attempts = 0
        self.constraint_violations = []
        
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
        
        # Track ALL attempted actions (executed or not)
        self.action_history = []  # List of (action, executed_flag) for entire day
        
        # Minute bars seen so far
        self.minute_bars_seen = []
        
        # Gym spaces
        # Original: 5185 = 500 ticks + 4680 minute bars + 5 position
        # With action history: 5185 + 4680 action history * 2 = 14545
        # NEW: + 3 constraint features = 14548
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14548,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -10.0]), 
            high=np.array([1.0, 10.0]), 
            dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
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
        
        # Reset action history
        self.action_history = []
        
        # NEW: Reset constraint tracking
        self.invalid_action_attempts = 0
        self.constraint_violations = []
        
        # Store market open for timestamp normalization
        self.day_market_open = self.current_time
        
        # Reset buffers
        self.tick_buffer.clear()
        self.minute_bars_seen = []
        
        # Feed initial ticks
        self._feed_ticks_until(self.current_time)
        
        return self._get_state(), {}
        
    def step(self, action):
        """Execute one trading step"""
        # 1. Record decision time and simulate order latency
        decision_time = self.current_time
        latency_ms = random.uniform(*self.order_latency_ms)
        order_time = decision_time + timedelta(milliseconds=latency_ms)
        
        # 2. Get portfolio value before trade
        current_price = self._get_current_price()
        portfolio_before = self.capital + self.position * current_price
        
        # NEW: Apply environment-level constraints
        original_action = float(action[0])
        constrained_action, was_constrained, constraint_type = self._apply_constraints(original_action)
        action = np.array([constrained_action, action[1]], dtype=np.float32)
        
        # 3. Execute trade at realistic price (next tick after order_time)
        execution_price, execution_time = self._find_execution_price(order_time)
        
        # Track number of trades before execution
        trades_before = len(self.trades_executed)
        
        if execution_price is not None:
            self._execute_trade(action, execution_price, execution_time)
        
        # Check if trade was actually executed
        trade_executed = len(self.trades_executed) > trades_before
        
        # Record action and whether it was executed (using original action)
        self.action_history.append((original_action, 1.0 if trade_executed and not was_constrained else 0.0))
        
        # NEW: Apply proportional penalty for invalid action attempts
        constraint_penalty = 0.0
        if was_constrained and self.enable_masking:
            self.invalid_action_attempts += 1
            self.constraint_violations.append({
                'time': self.current_time,
                'type': constraint_type,
                'original_action': original_action,
                'capital': self.capital,
                'position': self.position
            })
            
            # Calculate proportional penalty based on constraint type and severity
            if constraint_type == 'insufficient_funds':
                # Penalty proportional to how much we're trying to overspend
                severity = min(abs(original_action), 1.0)  # Action magnitude
                constraint_penalty = -0.01 * severity  # -0.01% for full buy attempt with no money
            elif constraint_type == 'no_position':
                # Smaller penalty for trying to sell nothing
                severity = min(abs(original_action), 1.0)
                constraint_penalty = -0.005 * severity  # -0.005% for full sell with no position
            else:  # partial_funds
                # Minimal penalty since we still executed partially
                constraint_penalty = -0.001
        
        # 4. Feed all ticks up to next decision time
        self._feed_ticks_until(self.next_decision_time)
        
        # 5. Update minute bars seen
        self._update_minute_bars()
        
        # 6. Calculate reward
        new_price = self._get_current_price()
        portfolio_after = self.capital + self.position * new_price
        base_reward = (portfolio_after - portfolio_before) / portfolio_before * 100
        reward = base_reward + constraint_penalty
        
        # 7. Check if done
        done = self._is_end_of_day()
        
        # No end-of-day penalty - let the model learn naturally
        # If holding overnight is profitable, it should do it
        # If day trading is better, market returns will teach that
        
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
            'trade_executed': len(self.trades_executed) > 0 and self.trades_executed[-1]['time'] == execution_time if execution_time else False,
            # NEW: Constraint info
            'original_action': original_action,
            'executed_action': constrained_action,
            'was_constrained': was_constrained,
            'constraint_type': constraint_type,
            'constraint_penalty': constraint_penalty,
            'invalid_attempts': self.invalid_action_attempts
        }
        
        return self._get_state(), reward, done, False, info
    
    def _apply_constraints(self, action):
        """
        Apply environment-level constraints to action
        Returns: (constrained_action, was_constrained, constraint_type)
        """
        was_constrained = False
        constraint_type = None
        constrained_action = action
        
        # Get current price for calculations
        current_price = self._get_current_price()
        if current_price <= 0:
            return action, False, None
        
        # Constraint 1: Can't buy if insufficient capital
        if action > 0:  # Trying to buy
            if self.capital < self.min_trade_value:
                # No money to buy anything
                constrained_action = 0.0
                was_constrained = True
                constraint_type = "insufficient_funds"
            else:
                # Check if action would require more capital than available
                max_shares_affordable = self.capital / current_price / (1 + self.transaction_cost)
                
                # If short, can only buy to cover
                if self.position < 0:
                    max_buy_shares = min(abs(self.position), max_shares_affordable)
                else:
                    max_buy_shares = max_shares_affordable
                
                # Action of 1.0 means buy all affordable shares
                if action * max_buy_shares * current_price > self.capital:
                    # Scale down to what we can afford
                    constrained_action = self.capital / (max_buy_shares * current_price) if max_buy_shares > 0 else 0.0
                    was_constrained = True
                    constraint_type = "partial_funds"
        
        # Constraint 2: Can't sell if no position (no short selling)
        elif action < 0:  # Trying to sell
            if self.position <= 0:
                # No position to sell
                constrained_action = 0.0
                was_constrained = True
                constraint_type = "no_position"
        
        return constrained_action, was_constrained, constraint_type
        
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
        
        # No action threshold - tiny actions are important for large positions
        # e.g., action=0.01 on 1000 shares = 10 share adjustment (meaningful!)
        
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
            # Normalize timestamp to hours since market open (0-6.5 range)
            if isinstance(t['timestamp'], pd.Timestamp) and hasattr(self, 'day_market_open') and self.day_market_open:
                hours_since_open = (t['timestamp'] - self.day_market_open).total_seconds() / 3600
            else:
                hours_since_open = 0
                
            tick_matrix[i] = [
                t.get('bid', 0),
                t.get('ask', 0),
                t.get('bid_size', 0) / SIZE_NORMALIZER,
                t.get('ask_size', 0) / SIZE_NORMALIZER,  
                hours_since_open  # Now 0-6.5 instead of 1.75e9!
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
                bar.get('volume', 0) / VOLUME_NORMALIZER,
                bar.get('trade_count', 0) / TRADE_COUNT_NORMALIZER,
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
            self.position / POSITION_NORMALIZER,
            self.capital / self.initial_capital,  # Fraction of initial capital
            (self.position * current_price) / self.initial_capital if current_price > 0 else 0,  # Position value as fraction
            (portfolio_value - self.initial_capital) / self.initial_capital,  # Return as fraction
            len(self.minute_bars_seen) / 390  # Progress through day
        ]
        features.extend(position_features)
        
        # 4. Action history (9360 dims = 4680 steps * 2 features)
        # Pad with zeros for future steps
        action_matrix = np.zeros((4680, 2))  # Max possible 5-sec intervals in a day
        
        for i, (action, executed) in enumerate(self.action_history):
            if i >= 4680:
                break
            action_matrix[i] = [action, executed]
        
        features.extend(action_matrix.flatten())
        
        # NEW: 5. Constraint features (3 dims)
        # These help the agent learn what actions are valid
        cash_ratio = min(1.0, self.capital / (self.initial_capital * 0.2))  # Normalized to 20% of initial
        can_buy = 1.0 if self.capital >= self.min_trade_value else 0.0
        can_sell = 1.0 if self.position > 0 else 0.0
        
        features.extend([
            cash_ratio,  # How much cash available (0-1)
            can_buy,     # Binary: can execute buy
            can_sell     # Binary: can execute sell
        ])
        
        return np.array(features[:14548], dtype=np.float32)
        
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
    
    def get_constraint_stats(self):
        """Get statistics about constraint violations"""
        stats = {
            'total_invalid_attempts': self.invalid_action_attempts,
            'invalid_rate': self.invalid_action_attempts / max(1, len(self.action_history)),
            'violations_by_type': {}
        }
        
        # Count violations by type
        for violation in self.constraint_violations:
            vtype = violation['type']
            stats['violations_by_type'][vtype] = stats['violations_by_type'].get(vtype, 0) + 1
        
        return stats