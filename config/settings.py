"""配置管理"""
import os
from typing import List
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # 数据库
    database_url: str = "postgresql://sentiment:sentiment123@localhost:5432/sentiment_db"
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    
    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    
    # API Keys
    news_api_key: str = ""
    
    # Telegram
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # 雪球 Cookie（可选，用于突破访问限制）
    xueqiu_cookie: str = ""
    
    # 预警配置
    alert_sentiment_threshold: float = -0.7
    alert_volume_multiplier: float = 3.0
    alert_keywords: str = "暴跌,崩盘,爆仓,利空,违约"
    
    # 监控股票
    watch_stocks: str = "AAPL,NVDA,TSLA,MSFT,GOOGL"
    
    # 服务
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    workers: int = 4
    
    @property
    def watch_stock_list(self) -> List[str]:
        return [s.strip() for s in self.watch_stocks.split(",") if s.strip()]
    
    @property
    def alert_keyword_list(self) -> List[str]:
        return [k.strip() for k in self.alert_keywords.split(",") if k.strip()]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()