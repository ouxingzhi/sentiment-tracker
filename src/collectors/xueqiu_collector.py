"""雪球数据采集器 - 中国领先的投资交流社区"""
import asyncio
import hashlib
import httpx
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, AsyncIterator
from loguru import logger

from config.settings import settings


class XueqiuCollector:
    """雪球数据采集器

    雪球是中国领先的投资交流社区，提供丰富的股票讨论、组合管理和投资资讯。
    采集内容包括：
    - 个股讨论（按股票代码）
    - 热门话题/热帖
    - 组合动态
    - 实时快讯
    """

    BASE_URL = "https://xueqiu.com"
    API_BASE = "https://stock.xueqiu.com"

    # 默认热门股票列表（可按需配置）
    DEFAULT_HOT_STOCKS = [
        # A 股热门
        "SH600519",  # 贵州茅台
        "SH601318",  # 中国平安
        "SZ002594",  # 比亚迪
        "SZ300750",  # 宁德时代
        "SZ300059",  # 东方财富
        "SH600036",  # 招商银行
        # 港股热门
        "HK00700",   # 腾讯控股
        "HK09988",   # 阿里巴巴
        "HK03690",   # 美团
        "HK01810",   # 小米集团
        # 美股热门中概
        "BABA",      # 阿里巴巴
        "PDD",       # 拼多多
        "JD",        # 京东
        "NIO",       # 蔚来
    ]

    def __init__(self, custom_stocks: List[str] = None, use_cookie: bool = True):
        self.source_name = "xueqiu"
        self.use_cookie = use_cookie
        self.client = httpx.AsyncClient(
            timeout=30,
            follow_redirects=True,
            headers=self._get_headers()
        )
        self.stocks = custom_stocks or self.DEFAULT_HOT_STOCKS
        self._cookie = None

    def _get_headers(self) -> Dict:
        """获取请求头"""
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "X-Requested-With": "XMLHttpRequest",
            "Origin": "https://xueqiu.com",
            "Referer": "https://xueqiu.com/",
        }
        # 如果配置了 Cookie，添加更多必要的请求头
        if settings.xueqiu_cookie:
            headers["Cookie"] = settings.xueqiu_cookie
            headers["Accept-Encoding"] = "gzip, deflate, br"
        return headers

    async def _fetch_cookie(self) -> Optional[str]:
        """获取雪球 Cookie（用于突破访问限制）

        如果没有配置 Cookie，尝试自动获取匿名访问的 Cookie
        """
        if self._cookie:
            return self._cookie

        # 如果配置文件中已有，直接使用
        if settings.xueqiu_cookie:
            self._cookie = settings.xueqiu_cookie
            return self._cookie

        # 尝试自动获取匿名 Cookie
        try:
            response = await self.client.get(self.BASE_URL)
            if response.status_code == 200:
                self._cookie = response.headers.get("Set-Cookie")
                if self._cookie:
                    # 更新后续请求的 Cookie
                    self.client.headers["Cookie"] = self._cookie
                    logger.info("雪球：已获取匿名访问 Cookie")
        except Exception as e:
            logger.debug(f"雪球：获取 Cookie 失败 {e}")

        return self._cookie

    def generate_source_id(self, content: str) -> str:
        """生成唯一 ID"""
        hash_obj = hashlib.md5(content.encode())
        return f"{self.source_name}_{hash_obj.hexdigest()[:12]}"

    async def collect(self, keywords: List[str] = None) -> AsyncIterator[Dict]:
        """采集雪球数据

        Args:
            keywords: 关键词列表，用于过滤内容
        """
        # 先尝试获取 Cookie
        if self.use_cookie:
            await self._fetch_cookie()

        # 采集热帖
        async for post in self._collect_hot_posts():
            if self._match_keywords(post, keywords):
                yield post

        # 采集个股讨论
        for stock in self.stocks:
            async for post in self._collect_stock_discussions(stock):
                if self._match_keywords(post, keywords):
                    yield post
            await asyncio.sleep(0.3)  # 速率限制

    async def _collect_realtime_news(self) -> AsyncIterator[Dict]:
        """采集实时快讯（7x24 小时电报）"""
        try:
            url = f"{self.BASE_URL}/v4/statuses/realtime.json"
            params = {"size": 50}

            response = await self.client.get(url, params=params)
            if response.status_code != 200:
                return

            data = response.json()
            items = data.get("list", [])

            for item in items:
                yield {
                    "source": "xueqiu_realtime",
                    "source_id": self.generate_source_id(str(item.get("id", ""))),
                    "url": f"https://xueqiu.com/k/{item.get('id', '')}",
                    "title": f"[快讯] {item.get('title', '')[:50]}",
                    "content": item.get("content", item.get("title", "")),
                    "author": "雪球快讯",
                    "published_at": self._parse_timestamp(item.get("created_at", 0)),
                    "extra": {
                        "type": "realtime_news",
                        "source_name": item.get("source", ""),
                        "mentioned_stocks": self._extract_stocks_from_text(item.get("content", ""))
                    }
                }
        except Exception as e:
            logger.error(f"雪球实时快讯采集失败：{e}")

    async def _collect_trending_topics(self) -> AsyncIterator[Dict]:
        """采集热门话题"""
        try:
            url = f"{self.BASE_URL}/v4/trending/topics.json"
            params = {"size": 20}

            response = await self.client.get(url, params=params)
            if response.status_code != 200:
                return

            data = response.json()
            topics = data.get("list", [])

            for topic in topics:
                # 只采集财经相关话题
                if self._is_finance_topic(topic):
                    yield {
                        "source": "xueqiu_topic",
                        "source_id": self.generate_source_id(str(topic.get("id", ""))),
                        "url": f"https://xueqiu.com/t/{topic.get('name', '')}",
                        "title": f"[话题] {topic.get('name', '')}",
                        "content": topic.get("description", topic.get("name", "")),
                        "author": "雪球话题",
                        "published_at": datetime.utcnow(),
                        "extra": {
                            "type": "trending_topic",
                            "view_count": topic.get("view_count", 0),
                            "post_count": topic.get("post_count", 0)
                        }
                    }
        except Exception as e:
            logger.error(f"雪球热门话题采集失败：{e}")

    def _is_finance_topic(self, topic: Dict) -> bool:
        """判断是否财经相关话题"""
        finance_keywords = [
            "股", "市", "基金", "财经", "经济", "金融", "投资", "理财",
            "银行", "利率", "政策", "期货", "外汇", "债券", "保险",
            "crypto", "区块链", "数字货币", "IPO", "财报", "牛市", "熊市"
        ]
        text = topic.get("name", "") + topic.get("description", "")
        return any(kw in text.lower() for kw in finance_keywords)

    async def _collect_hot_posts(self) -> AsyncIterator[Dict]:
        """采集热帖"""
        try:
            # 雪球热帖 API
            url = f"{self.API_BASE}/v5/statuses/hot.json"
            params = {
                "size": 20,
                "since_id": 0
            }

            response = await self.client.get(url, params=params)
            if response.status_code != 200:
                logger.warning(f"雪球热帖 API 返回 {response.status_code}")
                return

            data = response.json()
            posts = data.get("list", [])

            for post in posts:
                yield self._parse_post(post)

        except Exception as e:
            logger.error(f"雪球热帖采集失败：{e}")

    async def _collect_stock_discussions(self, symbol: str) -> AsyncIterator[Dict]:
        """采集个股讨论

        Args:
            symbol: 股票代码，如 SH600519, SZ002594, HK00700, BABA
        """
        try:
            # 雪球个股讨论 API
            url = f"{self.API_BASE}/v5/stock/statuses/list.json"
            params = {
                "symbol": symbol,
                "size": 20,
                "sort": "time"  # 按时间排序
            }

            response = await self.client.get(url, params=params)
            if response.status_code != 200:
                return

            data = response.json()
            posts = data.get("list", [])

            for post in posts:
                yield self._parse_post(post, symbol)

        except Exception as e:
            logger.error(f"雪球个股讨论采集失败 {symbol}: {e}")

    def _parse_post(self, post: Dict, default_symbol: str = None) -> Dict:
        """解析雪球帖子"""
        # 提取股票代码
        mentioned_stocks = self._extract_stocks_from_text(post.get("text", ""))

        # 处理 HTML 标签
        content = self._clean_html(post.get("text", ""))
        title = content[:100] if len(content) > 100 else content

        # 获取作者信息
        user = post.get("user", {})
        author = user.get("screen_name", "匿名")

        # 计算热度
        hot_score = (
            post.get("retweets_count", 0) * 2 +
            post.get("replies_count", 0) +
            post.get("likes_count", 0) * 0.5
        )

        return {
            "source": "xueqiu",
            "source_id": self.generate_source_id(str(post.get("id", ""))),
            "url": f"https://xueqiu.com/{user.get('id', '')}/{post.get('id', '')}",
            "title": f"[雪球] {title}",
            "content": content,
            "author": author,
            "published_at": self._parse_timestamp(post.get("created_at", 0)),
            "extra": {
                "retweets": post.get("retweets_count", 0),
                "replies": post.get("replies_count", 0),
                "likes": post.get("likes_count", 0),
                "hot_score": hot_score,
                "stock_symbol": default_symbol,
                "mentioned_stocks": mentioned_stocks
            }
        }

    def _clean_html(self, html_text: str) -> str:
        """清理 HTML 标签"""
        if not html_text:
            return ""
        # 移除 HTML 标签
        text = re.sub(r'<[^>]+>', '', html_text)
        # 移除@用户
        text = re.sub(r'@\w+', '', text)
        # 移除话题标签
        text = re.sub(r'#\w+#', '', text)
        # 清理空白
        text = re.sub(r'\s+', ' ', text).strip()
        # 解码 HTML 实体
        text = text.replace('&nbsp;', ' ').replace('&quot;', '"')
        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        return text

    def _extract_stocks_from_text(self, text: str) -> List[str]:
        """从文本中提取股票代码"""
        stocks = []
        # 匹配$代码$格式（雪球特有）
        matches = re.findall(r'\$([A-Z]{2,6})\$', text)
        stocks.extend(matches)
        # 匹配 A 股代码格式
        matches = re.findall(r'([036]\d{5}\.S[H|Z])', text, re.IGNORECASE)
        stocks.extend(matches)
        # 匹配港股代码
        matches = re.findall(r'HK(\d{5})', text, re.IGNORECASE)
        stocks.extend([f"HK{m}" for m in matches])
        # 匹配常见股票名称（中文）
        stock_names = {
            "茅台": "SH600519", "贵州茅台": "SH600519",
            "腾讯": "HK00700", "腾讯控股": "HK00700",
            "阿里": "HK09988", "阿里巴巴": "HK09988",
            "美团": "HK03690", "小米": "HK01810",
            "宁德": "SZ300750", "宁德时代": "SZ300750",
            "比亚迪": "SZ002594", "平安": "SH601318",
        }
        for name, symbol in stock_names.items():
            if name in text:
                stocks.append(symbol)
        return list(set(stocks))

    def _parse_timestamp(self, timestamp: int) -> Optional[datetime]:
        """解析时间戳"""
        if not timestamp:
            return None
        try:
            # 雪球使用毫秒时间戳
            if timestamp > 1e12:
                timestamp = timestamp / 1000
            return datetime.fromtimestamp(timestamp)
        except:
            return None

    def _match_keywords(self, article: Dict, keywords: List[str]) -> bool:
        """检查是否匹配关键词"""
        if not keywords:
            return True
        text = f"{article.get('title', '')} {article.get('content', '')}"
        return any(kw.lower() in text.lower() for kw in keywords)

    async def close(self):
        """关闭连接"""
        await self.client.aclose()


class XueqiuHotTopicCollector(XueqiuCollector):
    """雪球热门话题采集器"""

    async def collect(self, keywords: List[str] = None) -> AsyncIterator[Dict]:
        """只采集热门话题"""
        async for post in self._collect_hot_posts():
            if self._match_keywords(post, keywords):
                yield post


class XueqiuStockCollector(XueqiuCollector):
    """雪球个股讨论采集器"""

    def __init__(self, symbols: List[str]):
        super().__init__()
        self.symbols = symbols

    async def collect(self, keywords: List[str] = None) -> AsyncIterator[Dict]:
        """只采集指定股票的讨论"""
        for symbol in self.symbols:
            async for post in self._collect_stock_discussions(symbol):
                if self._match_keywords(post, keywords):
                    yield post
            await asyncio.sleep(0.3)
