"""后台任务处理"""
from celery import Celery
from datetime import datetime, timedelta
from loguru import logger
import asyncio

from config.settings import settings

# Celery应用
celery_app = Celery(
    "sentiment_worker",
    broker=settings.redis_url,
    backend=settings.redis_url
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,
    task_routes={
        "src.workers.tasks.collect_news": {"queue": "collect"},
        "src.workers.tasks.analyze_sentiment": {"queue": "analyze"},
        "src.workers.tasks.check_alerts": {"queue": "alert"},
    }
)


@celery_app.task
def collect_news():
    """采集新闻任务"""
    from src.collectors.news_collector import MultiSourceCollector
    from src.analyzers.sentiment_analyzer import HybridAnalyzer
    from src.database import async_session
    from src.models import Article
    
    async def _collect():
        collector = MultiSourceCollector()
        analyzer = HybridAnalyzer()
        
        try:
            async with async_session() as session:
                async for article_data in collector.collect_all():
                    # 分析情感
                    result = await analyzer.analyze(article_data["content"])
                    
                    # 创建文章记录
                    article = Article(
                        source=article_data["source"],
                        source_id=article_data["source_id"],
                        url=article_data.get("url"),
                        title=article_data.get("title", "")[:500],
                        content=article_data["content"],
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
                    
                    logger.info(f"采集: {article.title[:30]}... 情感: {result.label}")
        finally:
            await collector.close()
    
    asyncio.run(_collect())


@celery_app.task
def analyze_sentiment(article_id: int):
    """分析单篇文章情感"""
    async def _analyze():
        from src.database import async_session
        from src.models import Article
        from src.analyzers.sentiment_analyzer import HybridAnalyzer
        
        async with async_session() as session:
            result = await session.execute(
                select(Article).where(Article.id == article_id)
            )
            article = result.scalar_one_or_none()
            
            if not article or article.is_processed:
                return
            
            analyzer = HybridAnalyzer()
            sentiment = await analyzer.analyze(article.content, use_llm=True)
            
            article.sentiment_score = sentiment.score
            article.sentiment_label = sentiment.label
            article.stocks_mentioned = sentiment.entities.get("stocks", [])
            article.companies = sentiment.entities.get("companies", [])
            article.people = sentiment.entities.get("people", [])
            article.is_processed = True
            
            await session.commit()
    
    asyncio.run(_analyze())


@celery_app.task
def check_alerts():
    """检查预警任务"""
    async def _check():
        from src.database import async_session
        from src.models import Article, SentimentAlert
        from src.analyzers.alert_engine import AlertEngine, AlertNotifier
        
        async with async_session() as session:
            # 获取最近未检查的文章
            result = await session.execute(
                select(Article).where(
                    Article.is_processed == True,
                    Article.is_alert_sent == False,
                    Article.collected_at >= datetime.utcnow() - timedelta(hours=1)
                ).limit(100)
            )
            articles = result.scalars().all()
            
            if not articles:
                return
            
            # 检查预警
            alert_engine = AlertEngine()
            alerts = await alert_engine.check_alerts(list(articles), session)
            
            # 发送预警
            notifier = AlertNotifier()
            for alert in alerts:
                # 保存预警
                db_alert = SentimentAlert(
                    alert_type=alert.alert_type,
                    severity=alert.severity,
                    title=alert.title,
                    description=alert.description,
                    stocks=alert.stocks,
                    article_ids=alert.article_ids
                )
                session.add(db_alert)
                
                # 发送通知
                await notifier.send_alert(alert)
            
            await session.commit()
    
    asyncio.run(_check())


@celery_app.task
def send_daily_summary():
    """发送每日摘要"""
    async def _send():
        from src.database import async_session
        from src.models import Article, SentimentAlert
        from sqlalchemy import func, and_
        from src.analyzers.alert_engine import AlertNotifier
        
        async with async_session() as session:
            today = datetime.utcnow().replace(hour=0, minute=0, second=0)
            
            # 统计数据
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
            
            alerts = await session.scalar(
                select(func.count(SentimentAlert.id)).where(SentimentAlert.created_at >= today)
            )
            
            notifier = AlertNotifier()
            await notifier.send_daily_summary({
                "total_articles": total or 0,
                "positive": positive or 0,
                "negative": negative or 0,
                "neutral": (total or 0) - (positive or 0) - (negative or 0),
                "alerts": alerts or 0,
                "top_stocks": []  # TODO: 实现热门股票统计
            })
    
    asyncio.run(_send())