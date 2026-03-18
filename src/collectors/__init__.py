"""采集器模块"""
from .news_collector import (
    BaseCollector,
    RSSCollector,
    WeiboCollector,
    MultiSourceCollector,
)
from .xueqiu_collector import (
    XueqiuCollector,
    XueqiuHotTopicCollector,
    XueqiuStockCollector,
)
from .eastmoney_collector import (
    EastmoneyCollector,
    GubaCollector,
    TonghuashunCollector,
)

__all__ = [
    "BaseCollector",
    "RSSCollector",
    "WeiboCollector",
    "MultiSourceCollector",
    "XueqiuCollector",
    "XueqiuHotTopicCollector",
    "XueqiuStockCollector",
    "EastmoneyCollector",
    "GubaCollector",
    "TonghuashunCollector",
]
