"""采集器模块"""
from .news_collector import (
    BaseCollector,
    RSSCollector,
    NewsAPICollector,
    WeiboCollector,
    MultiSourceCollector,
)
from .xueqiu_collector import (
    XueqiuCollector,
    XueqiuHotTopicCollector,
    XueqiuStockCollector,
)

__all__ = [
    "BaseCollector",
    "RSSCollector",
    "NewsAPICollector",
    "WeiboCollector",
    "MultiSourceCollector",
    "XueqiuCollector",
    "XueqiuHotTopicCollector",
    "XueqiuStockCollector",
]
