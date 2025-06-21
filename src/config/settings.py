from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # Polygon API
    polygon_api_key: str
    polygon_ws_url: str = "wss://socket.polygon.io/stocks"
    polygon_ws_delayed_url: str = "wss://delayed.polygon.io/stocks"
    use_delayed_feed: bool = False
    
    # Kafka
    kafka_bootstrap_servers: str = "localhost:19092"
    kafka_topic_prefix: str = "polygon_market_data"
    
    # Subscriptions
    symbols: List[str] = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    subscription_types: List[str] = ["T", "Q", "AM"]
    
    # Application
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        
    @property
    def symbols_str(self) -> str:
        return ",".join(self.symbols)
    
    @property
    def subscription_types_str(self) -> str:
        return ",".join(self.subscription_types)


settings = Settings()