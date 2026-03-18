"""配置管理 - 纯本地爬虫版本"""
import os
from typing import List
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # 数据库
    database_url: str = "postgresql://sentiment:sentiment123@localhost:5432/sentiment_db"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Telegram (用于推送)
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # 雪球 Cookie（可选，用于突破访问限制）
    xueqiu_cookie: str = ""

    # 微博 Cookie（可选，用于获取更完整的热搜数据）
    weibo_cookie: str = ""

    # 预警配置
    alert_sentiment_threshold: float = -0.7
    alert_volume_multiplier: float = 3.0
    alert_keywords: str = "暴跌，崩盘，爆仓，利空，违约"

    # 监控股票（支持 A 股、港股、美股）
    watch_stocks: str = "600519,300750,002594,601318,600036,300059,00700.HK,09988.HK"

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

    # LLM 模型配置
    llm_model_name: str = "qwen2.5"  # "qwen2.5" 或 "roberta"
    llm_enabled: bool = True  # 是否启用 LLM 分析
    llm_use_gpu: bool = True  # 是否使用 GPU

    # RoBERTa 模型配置
    roberta_enabled: bool = True  # 是否启用 RoBERTa 分析
    roberta_use_cpu: bool = True  # 是否强制使用 CPU

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # 忽略额外字段


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
