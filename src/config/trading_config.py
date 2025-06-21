"""Trading configuration constants shared between training and deployment"""

from dataclasses import dataclass
from datetime import datetime, time
from typing import Optional


@dataclass
class TradingConfig:
    """Configuration for trading constraints and parameters"""
    
    # Position constraints  
    MIN_POSITION: float = 0.0     # Minimum position (0 = long-only, negative = allow shorts)
    # NOTE: No MAX_POSITION - model has full freedom to use available capital
    
    # Alpaca API constraints only
    MIN_ORDER_SIZE: float = 0.000001   # Alpaca minimum fractional shares  
    MIN_ORDER_VALUE: float = 1.0       # Alpaca minimum $1 order value
    ORDER_SIZE_PRECISION: int = 6      # Alpaca supports up to 6 decimal places
    
    # Capital and costs
    INITIAL_CAPITAL: float = 100000.0
    TRANSACTION_COST: float = 0.0      # Alpaca has commission-free trading
    # Note: Small regulatory fees (SEC/FINRA) may apply but are negligible for RL training
    
    # Timing
    DECISION_INTERVAL_SECONDS: int = 5
    ORDER_LATENCY_MS_MIN: int = 5
    ORDER_LATENCY_MS_MAX: int = 50
    
    # Market hours (UTC for US Eastern)
    MARKET_OPEN_UTC: str = "13:30:00"   # 9:30 AM ET
    MARKET_CLOSE_UTC: str = "20:00:00"  # 4:00 PM ET
    
    # State space dimensions
    TICK_BUFFER_SIZE: int = 100
    TICK_FEATURES: int = 5  # bid, ask, bid_size, ask_size, timestamp
    MINUTE_BAR_FEATURES: int = 12
    POSITION_STATE_FEATURES: int = 5
    
    @property
    def market_open_time(self) -> time:
        """Market open as time object"""
        return datetime.strptime(self.MARKET_OPEN_UTC, "%H:%M:%S").time()
    
    @property
    def market_close_time(self) -> time:
        """Market close as time object"""
        return datetime.strptime(self.MARKET_CLOSE_UTC, "%H:%M:%S").time()
    
    @property
    def minute_bars_count(self) -> int:
        """Calculate number of minute bars in a trading day"""
        # Convert to datetime for calculation
        open_dt = datetime.combine(datetime.today(), self.market_open_time)
        close_dt = datetime.combine(datetime.today(), self.market_close_time)
        
        # Calculate minutes between open and close
        minutes = int((close_dt - open_dt).total_seconds() / 60)
        return minutes
    
    @property
    def observation_dim(self) -> int:
        """Total observation space dimensions"""
        return (
            self.TICK_BUFFER_SIZE * self.TICK_FEATURES +  # 500
            self.minute_bars_count * self.MINUTE_BAR_FEATURES +  # 4680 (390 * 12)
            self.POSITION_STATE_FEATURES  # 5
        )  # Total: 5185
    


# Default configuration instance
default_config = TradingConfig()


# Optional: Environment-specific overrides
@dataclass
class PaperTradingConfig(TradingConfig):
    """Configuration for paper trading with conservative limits"""
    MAX_POSITION: float = 100.0  # Smaller position for testing
    

@dataclass 
class LiveTradingConfig(TradingConfig):
    """Configuration for live trading with production limits"""
    TRANSACTION_COST: float = 0.0025  # Higher for real trading costs