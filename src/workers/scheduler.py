"""定时任务调度器"""
import asyncio
import schedule
import time
from loguru import logger
from threading import Thread

from src.collectors.news_collector import MultiSourceCollector
from src.analyzers.sentiment_analyzer import HybridAnalyzer
from src.analyzers.alert_engine import AlertEngine, AlertNotifier
from src.database import async_session
from src.models import Article
from sqlalchemy import select
from datetime import datetime, timedelta


class Scheduler:
    """定时任务调度器"""
    
    def __init__(self):
        self.collector = MultiSourceCollector()
        self.analyzer = HybridAnalyzer()
        self.alert_engine = AlertEngine()
        self.notifier = AlertNotifier()
    
    async def collect_and_analyze(self):
        """采集和分析"""
        logger.info("开始采集任务...")
        
        count = 0
        try:
            async with async_session() as session:
                async for article_data in self.collector.collect_all():
                    try:
                        # 分析情感
                        result = await self.analyzer.analyze(article_data["content"])
                        
                        # 检查是否已存在
                        existing = await session.scalar(
                            select(Article).where(
                                Article.source_id == article_data["source_id"]
                            )
                        )
                        
                        if existing:
                            continue
                        
                        # 创建文章
                        article = Article(
                            source=article_data["source"],
                            source_id=article_data["source_id"],
                            url=article_data.get("url"),
                            title=article_data.get("title", "")[:500],
                            content=article_data["content"][:5000],
                            author=article_data.get("author"),
                            published_at=article_data.get("published_at"),
                            sentiment_score=result.score,
                            sentiment_label=result.label,
                            stocks_mentioned=result.entities.get("stocks", []),
                            companies=result.entities.get("companies", []),
                            people=result.entities.get("people", []),
                            is_processed=True
                        )
                        
                        session.add(article)
                        await session.commit()
                        count += 1
                        
                    except Exception as e:
                        logger.error(f"处理文章失败: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"采集任务失败: {e}")
        
        logger.info(f"采集完成，共 {count} 篇文章")
    
    async def check_alerts(self):
        """检查预警"""
        logger.info("开始预警检查...")
        
        try:
            async with async_session() as session:
                # 获取最近文章
                articles = await session.execute(
                    select(Article).where(
                        Article.is_processed == True,
                        Article.is_alert_sent == False,
                        Article.collected_at >= datetime.utcnow() - timedelta(hours=1)
                    ).limit(100)
                )
                articles = articles.scalars().all()
                
                if not articles:
                    logger.info("没有需要检查的文章")
                    return
                
                # 检查预警
                alerts = await self.alert_engine.check_alerts(list(articles), session)
                
                # 发送预警
                for alert in alerts:
                    sent = await self.notifier.send_alert(alert)
                    if sent:
                        logger.info(f"预警已发送: {alert.title}")
                
                # 标记已检查
                for article in articles:
                    article.is_alert_sent = True
                await session.commit()
        
        except Exception as e:
            logger.error(f"预警检查失败: {e}")
    
    async def daily_summary(self):
        """每日摘要"""
        logger.info("发送每日摘要...")
        
        try:
            async with async_session() as session:
                today = datetime.utcnow().replace(hour=0, minute=0, second=0)
                
                # 统计
                from sqlalchemy import func, and_
                
                total = await session.scalar(
                    select(func.count(Article.id)).where(Article.collected_at >= today)
                )
                
                positive = await session.scalar(
                    select(func.count(Article.id)).where(
                        and_(
                            Article.collected_at >= today,
                            Article.sentiment_label == "positive"
                        )
                    )
                )
                
                negative = await session.scalar(
                    select(func.count(Article.id)).where(
                        and_(
                            Article.collected_at >= today,
                            Article.sentiment_label == "negative"
                        )
                    )
                )
                
                await self.notifier.send_daily_summary({
                    "total_articles": total or 0,
                    "positive": positive or 0,
                    "negative": negative or 0,
                    "neutral": (total or 0) - (positive or 0) - (negative or 0),
                    "alerts": 0,
                    "top_stocks": []
                })
        
        except Exception as e:
            logger.error(f"每日摘要失败: {e}")
    
    def run_async(self, coro):
        """在同步上下文中运行异步函数"""
        asyncio.run(coro)
    
    def start(self):
        """启动调度器"""
        logger.info("启动定时任务调度器...")
        
        # 定时任务
        schedule.every(30).minutes.do(lambda: self.run_async(self.collect_and_analyze()))
        schedule.every(10).minutes.do(lambda: self.run_async(self.check_alerts()))
        schedule.every().day.at("20:00").do(lambda: self.run_async(self.daily_summary()))
        
        # 启动时立即执行一次采集
        logger.info("执行初始采集...")
        self.run_async(self.collect_and_analyze())
        
        # 运行调度循环
        while True:
            schedule.run_pending()
            time.sleep(60)


if __name__ == "__main__":
    scheduler = Scheduler()
    scheduler.start()