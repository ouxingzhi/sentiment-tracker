"""东方财富数据采集器 - 中国领先的财经门户网站"""
import asyncio
import hashlib
import httpx
import re
from datetime import datetime
from typing import List, Dict, Optional, AsyncIterator
from loguru import logger

from config.settings import settings


class EastmoneyCollector:
    """东方财富数据采集器

    东方财富是中国领先的财经门户网站，提供丰富的财经新闻、股吧社区、
    行情数据等。采集内容包括：
    - 财经快讯（7x24 小时）
    - 个股新闻
    - 热门资讯
    - 研究报告
    """

    BASE_URL = "https://www.eastmoney.com"
    API_BASE = "https://api.eastmoney.com"

    def __init__(self, use_cookie: bool = False):
        self.source_name = "eastmoney"
        self.use_cookie = use_cookie
        self.client = httpx.AsyncClient(
            timeout=30,
            follow_redirects=True,
            headers=self._get_headers()
        )

    def _get_headers(self) -> Dict:
        """获取请求头"""
        return {
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.eastmoney.com/",
        }

    def generate_source_id(self, content: str) -> str:
        """生成唯一 ID"""
        hash_obj = hashlib.md5(content.encode())
        return f"{self.source_name}_{hash_obj.hexdigest()[:12]}"

    async def collect(self, keywords: List[str] = None) -> AsyncIterator[Dict]:
        """采集东方财富数据"""
        # 采集财经快讯
        async for news in self._collect_realtime_news():
            if self._match_keywords(news, keywords):
                yield news

        # 采集热门资讯
        async for news in self._collect_hot_news():
            if self._match_keywords(news, keywords):
                yield news

        await asyncio.sleep(0.5)

    async def _collect_realtime_news(self) -> AsyncIterator[Dict]:
        """采集实时快讯（7x24 小时电报）"""
        try:
            # 东方财富快讯 API
            url = "https://api.eastmoney.com/api/GetKXList"
            params = {
                "callback": "jQuery",
                "pageindex": 1,
                "pagesize": 50,
                "_": int(datetime.utcnow().timestamp() * 1000)
            }

            response = await self.client.get(url, params=params)
            if response.status_code != 200:
                return

            # 处理 JSONP 响应
            text = response.text
            match = re.search(r'jQuery\((.+)\)', text)
            if match:
                data = eval(match.group(1))
                items = data.get("List", [])

                for item in items:
                    yield {
                        "source": "eastmoney_realtime",
                        "source_id": self.generate_source_id(str(item.get("ID", ""))),
                        "url": f"https://kx.eastmoney.com/{item.get('ID', '')}",
                        "title": f"[快讯] {item.get('Title', '')[:50]}",
                        "content": item.get("Content", item.get("Title", "")),
                        "author": "东方财富快讯",
                        "published_at": self._parse_timestamp(item.get("ShowTime", 0)),
                        "extra": {
                            "type": "realtime_news",
                            "source_name": "东方财富",
                        }
                    }
        except Exception as e:
            logger.error(f"东方财富实时快讯采集失败：{e}")

    async def _collect_hot_news(self) -> AsyncIterator[Dict]:
        """采集热门资讯"""
        try:
            # 东方财富热门新闻 API
            url = "https://api.eastmoney.com/api/GetZDList"
            params = {
                "callback": "jQuery",
                "pageindex": 1,
                "pagesize": 30,
                "_": int(datetime.utcnow().timestamp() * 1000)
            }

            response = await self.client.get(url, params=params)
            if response.status_code != 200:
                return

            # 处理 JSONP 响应
            text = response.text
            match = re.search(r'jQuery\((.+)\)', text)
            if match:
                data = eval(match.group(1))
                items = data.get("List", [])

                for item in items:
                    yield {
                        "source": "eastmoney",
                        "source_id": self.generate_source_id(str(item.get("ID", ""))),
                        "url": item.get("Url", ""),
                        "title": item.get("Title", ""),
                        "content": item.get("Digest", item.get("Title", "")),
                        "author": item.get("Source", "东方财富"),
                        "published_at": self._parse_timestamp(item.get("ShowTime", 0)),
                        "extra": {
                            "type": "hot_news",
                            "source_name": "东方财富",
                            "read_count": item.get("ReadCount", 0),
                        }
                    }
        except Exception as e:
            logger.error(f"东方财富热门资讯采集失败：{e}")

    def _parse_timestamp(self, timestamp: int) -> Optional[datetime]:
        """解析时间戳"""
        if not timestamp:
            return None
        try:
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


class GubaCollector:
    """股吧采集器 - 东方财富旗下股票社区

    股吧是中国最大的股票投资者社区，提供：
    - 个股讨论区
    - 热门帖子
    - 最新回复
    - 财富号文章
    """

    BASE_URL = "https://guba.eastmoney.com"
    API_BASE = "https://app.guba.eastmoney.com"

    # 默认热门股票股吧
    DEFAULT_STOCKS = [
        "600519",  # 贵州茅台
        "300750",  # 宁德时代
        "002594",  # 比亚迪
        "601318",  # 中国平安
        "600036",  # 招商银行
        "300059",  # 东方财富
        "000858",  # 五粮液
        "000333",  # 美的集团
        "601127",  # 赛力斯
        "002475",  # 立讯精密
    ]

    def __init__(self, custom_stocks: List[str] = None):
        self.source_name = "guba"
        self.stocks = custom_stocks or self.DEFAULT_STOCKS
        self.client = httpx.AsyncClient(
            timeout=30,
            follow_redirects=True,
            headers=self._get_headers()
        )

    def _get_headers(self) -> Dict:
        """获取请求头"""
        return {
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://guba.eastmoney.com/",
        }

    def generate_source_id(self, content: str) -> str:
        """生成唯一 ID"""
        hash_obj = hashlib.md5(content.encode())
        return f"{self.source_name}_{hash_obj.hexdigest()[:12]}"

    async def collect(self, keywords: List[str] = None) -> AsyncIterator[Dict]:
        """采集股吧数据"""
        # 采集热门帖子
        async for post in self._collect_hot_posts():
            if self._match_keywords(post, keywords):
                yield post

        # 采集个股讨论
        for stock in self.stocks:
            async for post in self._collect_stock_posts(stock):
                if self._match_keywords(post, keywords):
                    yield post
            await asyncio.sleep(0.3)

    async def _collect_hot_posts(self) -> AsyncIterator[Dict]:
        """采集热门帖子"""
        try:
            url = f"{self.API_BASE}/ws/HotPostSet.ashx"
            params = {"jsname": "callback"}

            response = await self.client.get(url, params=params)
            if response.status_code != 200:
                return

            # 处理 JSONP 响应
            text = response.text
            match = re.search(r'callback\((.+)\)', text)
            if match:
                data = eval(match.group(1))
                posts = data.get("hot_post_set", [])

                for post in posts:
                    yield self._parse_post(post)
        except Exception as e:
            logger.error(f"股吧热门帖子采集失败：{e}")

    async def _collect_stock_posts(self, stock_code: str) -> AsyncIterator[Dict]:
        """采集个股讨论"""
        try:
            # 确定市场
            if stock_code.startswith("6"):
                market = "SH"
            else:
                market = "SZ"

            url = f"{self.API_BASE}/ws/PostList.ashx"
            params = {
                "Symbol": f"{market}{stock_code}",
                "pn": 1,  # 页码
                "ps": 20,  # 每页数量
                "co": 1,  # 排序方式
            }

            response = await self.client.get(url, params=params)
            if response.status_code != 200:
                return

            # 处理 JSONP 响应
            text = response.text
            match = re.search(r'callback\((.+)\)', text)
            if match:
                data = eval(match.group(1))
                posts = data.get("re_post", [])

                for post in posts:
                    yield self._parse_post(post, stock_code)
        except Exception as e:
            logger.error(f"股吧个股讨论采集失败 {stock_code}: {e}")

    def _parse_post(self, post: Dict, default_stock: str = None) -> Dict:
        """解析股吧帖子"""
        # 清理 HTML
        content = self._clean_html(post.get("content", post.get("Title", "")))
        title = content[:100] if len(content) > 100 else content

        return {
            "source": "guba",
            "source_id": self.generate_source_id(str(post.get("post_id", post.get("ID", "")))),
            "url": f"{self.BASE_URL}/news,{post.get('stock_code', default_stock, '')},{post.get('post_id', post.get('ID', ''))}.html",
            "title": f"[股吧] {title}",
            "content": content,
            "author": post.get("user_name", "匿名"),
            "published_at": self._parse_date(post.get("post_time", post.get("created_time"))),
            "extra": {
                "read_count": post.get("read_number", post.get("click_count", 0)),
                "reply_count": post.get("reply_number", post.get("comment_count", 0)),
                "stock_code": default_stock,
            }
        }

    def _clean_html(self, html_text: str) -> str:
        """清理 HTML 标签"""
        if not html_text:
            return ""
        # 移除 HTML 标签
        text = re.sub(r'<[^>]+>', '', html_text)
        # 清理空白
        text = re.sub(r'\s+', ' ', text).strip()
        # 解码 HTML 实体
        text = text.replace('&nbsp;', ' ').replace('&quot;', '"')
        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        return text

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """解析日期字符串"""
        if not date_str:
            return None
        try:
            # 尝试多种格式
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S"]:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            return datetime.utcnow()
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


class TonghuashunCollector:
    """同花顺数据采集器

    同花顺是中国领先的金融信息服务商，提供：
    - 财经要闻
    - 个股资讯
    - 研报中心
    - 问财数据
    """

    BASE_URL = "https://www.10jqka.com.cn"
    API_BASE = "https://data.10jqka.com.cn"

    def __init__(self):
        self.source_name = "tonghuashun"
        self.client = httpx.AsyncClient(
            timeout=30,
            follow_redirects=True,
            headers=self._get_headers()
        )

    def _get_headers(self) -> Dict:
        """获取请求头"""
        return {
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.10jqka.com.cn/",
        }

    def generate_source_id(self, content: str) -> str:
        """生成唯一 ID"""
        hash_obj = hashlib.md5(content.encode())
        return f"{self.source_name}_{hash_obj.hexdigest()[:12]}"

    async def collect(self, keywords: List[str] = None) -> AsyncIterator[Dict]:
        """采集同花顺数据"""
        # 采集财经要闻
        async for news in self._collect_news():
            if self._match_keywords(news, keywords):
                yield news

    async def _collect_news(self) -> AsyncIterator[Dict]:
        """采集财经要闻"""
        try:
            # 同花顺要闻 API
            url = "https://data.10jqka.com.cn/wg/ajax/getWgRemoteList.php"
            params = {
                "type": "1",
                "page": "1",
                "pagesize": "30",
            }

            response = await self.client.get(url, params=params)
            if response.status_code != 200:
                return

            data = response.json()
            items = data.get("list", [])

            for item in items:
                yield {
                    "source": "tonghuashun",
                    "source_id": self.generate_source_id(str(item.get("id", ""))),
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "content": item.get("summary", item.get("title", "")),
                    "author": item.get("source", "同花顺"),
                    "published_at": self._parse_date(item.get("uptime", "")),
                    "extra": {
                        "source_name": "同花顺",
                    }
                }
        except Exception as e:
            logger.error(f"同花顺要闻采集失败：{e}")

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """解析日期字符串"""
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str.replace(" ", "T"))
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
