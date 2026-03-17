"""财经新闻采集器 - 多源数据采集"""
import asyncio
import hashlib
import httpx
import feedparser
from datetime import datetime, timedelta
from typing import List, Dict, Optional, AsyncIterator
from abc import ABC, abstractmethod
from loguru import logger
from newspaper import Article as NewspaperArticle

from config.settings import settings


class BaseCollector(ABC):
    """采集器基类"""
    
    def __init__(self):
        self.source_name = self.__class__.__name__.lower().replace("collector", "")
        self.client = httpx.AsyncClient(timeout=30, follow_redirects=True)
    
    @abstractmethod
    async def collect(self, keywords: List[str] = None) -> AsyncIterator[Dict]:
        """采集数据，返回文章生成器"""
        pass
    
    def generate_source_id(self, content: str) -> str:
        """生成唯一ID"""
        hash_obj = hashlib.md5(content.encode())
        return f"{self.source_name}_{hash_obj.hexdigest()[:12]}"
    
    async def close(self):
        await self.client.aclose()


class RSSCollector(BaseCollector):
    """RSS新闻采集器"""
    
    RSS_FEEDS = [
        # 中文财经媒体
        "https://www.caixin.com/rss/rss_finance.xml",
        "https://www.yicai.com/rss/news.xml",
        "https://www.jiemian.com/rss/news.xml",
        "https://www.ftchinese.com/rss/news",
        # 英文财经
        "https://feeds.bloomberg.com/markets/news.rss",
        "https://www.cnbc.com/id/10000664/device/rss/rss.html",
        "https://feeds.reuters.com/reuters/businessNews",
        # 科技媒体
        "https://www.36kr.com/feed",
        "https://techcrunch.com/feed/",
    ]
    
    def __init__(self, custom_feeds: List[str] = None):
        super().__init__()
        self.feeds = custom_feeds or self.RSS_FEEDS
    
    async def collect(self, keywords: List[str] = None) -> AsyncIterator[Dict]:
        """采集RSS feeds"""
        for feed_url in self.feeds:
            try:
                feed = await self._fetch_feed(feed_url)
                for entry in feed.entries[:20]:  # 每个源最多20条
                    article = await self._parse_entry(entry, feed_url)
                    if article and self._match_keywords(article, keywords):
                        yield article
            except Exception as e:
                logger.error(f"RSS采集失败 {feed_url}: {e}")
    
    async def _fetch_feed(self, url: str) -> feedparser.FeedParserDict:
        """获取RSS内容"""
        response = await self.client.get(url)
        return feedparser.parse(response.content)
    
    async def _parse_entry(self, entry, feed_url: str) -> Optional[Dict]:
        """解析RSS条目"""
        try:
            # 尝试提取正文
            content = entry.get("summary", "")
            if entry.get("link"):
                try:
                    article = NewspaperArticle(entry.link)
                    article.download()
                    article.parse()
                    content = article.text or content
                except:
                    pass
            
            return {
                "source": "rss",
                "source_id": self.generate_source_id(entry.get("link", "")),
                "url": entry.get("link"),
                "title": entry.get("title", ""),
                "content": content,
                "author": entry.get("author", ""),
                "published_at": self._parse_date(entry.get("published")),
                "extra": {"feed_url": feed_url}
            }
        except Exception as e:
            logger.error(f"解析RSS条目失败: {e}")
            return None
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """解析日期字符串"""
        if not date_str:
            return None
        try:
            from dateutil import parser
            return parser.parse(date_str)
        except:
            return None
    
    def _match_keywords(self, article: Dict, keywords: List[str]) -> bool:
        """检查是否匹配关键词"""
        if not keywords:
            return True
        text = f"{article.get('title', '')} {article.get('content', '')}"
        return any(kw.lower() in text.lower() for kw in keywords)


class NewsAPICollector(BaseCollector):
    """NewsAPI采集器 (需要API Key)"""
    
    BASE_URL = "https://newsapi.org/v2"
    
    async def collect(self, keywords: List[str] = None) -> AsyncIterator[Dict]:
        """从NewsAPI采集新闻"""
        if not settings.news_api_key:
            logger.warning("NewsAPI未配置API Key")
            return
        
        keywords = keywords or settings.watch_stock_list
        
        for keyword in keywords[:5]:  # 限制关键词数量
            try:
                articles = await self._search_news(keyword)
                for article in articles:
                    yield article
                await asyncio.sleep(1)  # 速率限制
            except Exception as e:
                logger.error(f"NewsAPI采集失败 {keyword}: {e}")
    
    async def _search_news(self, query: str) -> List[Dict]:
        """搜索新闻"""
        params = {
            "q": query,
            "apiKey": settings.news_api_key,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 20
        }
        
        response = await self.client.get(
            f"{self.BASE_URL}/everything",
            params=params
        )
        data = response.json()
        
        if data.get("status") != "ok":
            logger.error(f"NewsAPI错误: {data}")
            return []
        
        articles = []
        for item in data.get("articles", []):
            articles.append({
                "source": "newsapi",
                "source_id": self.generate_source_id(item.get("url", "")),
                "url": item.get("url"),
                "title": item.get("title", ""),
                "content": item.get("content") or item.get("description", ""),
                "author": item.get("author", ""),
                "published_at": self._parse_date(item.get("publishedAt")),
                "extra": {"source_name": item.get("source", {}).get("name")}
            })
        
        return articles
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except:
            return None


class WeiboCollector(BaseCollector):
    """微博热搜采集器"""
    
    HOT_SEARCH_URL = "https://weibo.com/ajax/side/hotSearch"
    
    async def collect(self, keywords: List[str] = None) -> AsyncIterator[Dict]:
        """采集微博热搜"""
        try:
            hot_search = await self._fetch_hot_search()
            for item in hot_search[:30]:
                # 过滤财经相关
                if self._is_finance_related(item):
                    article = await self._process_weibo_topic(item)
                    if article:
                        yield article
        except Exception as e:
            logger.error(f"微博采集失败: {e}")
    
    async def _fetch_hot_search(self) -> List[Dict]:
        """获取微博热搜"""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        if settings.weibo_cookie:
            headers["Cookie"] = settings.weibo_cookie
        
        response = await self.client.get(
            self.HOT_SEARCH_URL,
            headers=headers
        )
        data = response.json()
        
        if data.get("ok") != 1:
            return []
        
        return data.get("data", {}).get("realtime", [])
    
    def _is_finance_related(self, item: Dict) -> bool:
        """判断是否财经相关"""
        finance_keywords = [
            "股", "市", "涨", "跌", "基金", "财经", "经济", "金融",
            "基金", "投资", "理财", "银行", "利率", "政策"
        ]
        text = item.get("note", "") + item.get("word", "")
        return any(kw in text for kw in finance_keywords)
    
    async def _process_weibo_topic(self, item: Dict) -> Optional[Dict]:
        """处理微博话题"""
        return {
            "source": "weibo",
            "source_id": self.generate_source_id(item.get("word", "")),
            "url": f"https://s.weibo.com/weibo?q=%23{item.get('word', '')}%23",
            "title": f"微博热搜: {item.get('word', '')}",
            "content": item.get("note", item.get("word", "")),
            "published_at": datetime.utcnow(),
            "extra": {
                "rank": item.get("rank"),
                "hot": item.get("raw_hot"),
                "label": item.get("icon_desc")
            }
        }


class MultiSourceCollector:
    """多源采集器管理"""
    
    def __init__(self):
        self.collectors = [
            RSSCollector(),
            NewsAPICollector(),
            WeiboCollector(),
        ]
    
    async def collect_all(self, keywords: List[str] = None) -> AsyncIterator[Dict]:
        """并行采集所有源"""
        tasks = []
        for collector in self.collectors:
            tasks.append(collector.collect(keywords))
        
        # 并行执行
        async for article in self._merge_collectors(tasks):
            yield article
    
    async def _merge_collectors(self, collectors: List) -> AsyncIterator[Dict]:
        """合并多个采集器的输出"""
        queue = asyncio.Queue()
        running = len(collectors)
        
        async def producer(collector):
            nonlocal running
            try:
                async for article in collector:
                    await queue.put(article)
            finally:
                running -= 1
                if running == 0:
                    await queue.put(None)  # 结束信号
        
        # 启动所有生产者
        for col in collectors:
            asyncio.create_task(producer(col))
        
        # 消费队列
        while True:
            article = await queue.get()
            if article is None:
                break
            yield article
    
    async def close(self):
        """关闭所有采集器"""
        for collector in self.collectors:
            await collector.close()