"""FastAPI 主应用"""
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Depends, Query, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select, func, desc, and_, cast, Date, case
from sqlalchemy.types import DateTime
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
import logging

from src.database import get_session, init_db
from src.models import Article, SentimentAlert, StockMention
from src.collectors.news_collector import MultiSourceCollector
from src.collectors.xueqiu_collector import XueqiuCollector, XueqiuHotTopicCollector
from src.analyzers.sentiment_analyzer import HybridAnalyzer, StockEntityRecognizer
from src.analyzers.alert_engine import AlertEngine, AlertNotifier
from config.settings import settings

logger = logging.getLogger(__name__)

app = FastAPI(
    title="财经舆情分析系统",
    description="实时监控财经新闻和社交媒体，分析市场情绪，预警异常动向",
    version="1.1.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ Pydantic Models ============

class ArticleResponse(BaseModel):
    id: int
    source: str
    title: str
    content: str
    sentiment_score: Optional[float]
    sentiment_label: Optional[str]
    stocks_mentioned: List[dict]
    published_at: Optional[datetime]
    url: Optional[str]

    class Config:
        from_attributes = True


class AlertResponse(BaseModel):
    id: int
    alert_type: str
    severity: str
    title: str
    description: str
    stocks: List[str]
    created_at: datetime
    is_read: bool

    class Config:
        from_attributes = True


class StatsResponse(BaseModel):
    total_articles: int
    positive: int
    negative: int
    neutral: int
    avg_sentiment: float
    top_stocks: List[dict]
    alerts_today: int


class BatchAnalyzeRequest(BaseModel):
    """批量分析请求"""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="待分析文本列表")
    use_llm: bool = Field(False, description="是否使用 LLM 进行更精确分析")


class BatchAnalyzeResponse(BaseModel):
    """批量分析响应"""
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]


class SentimentTrendResponse(BaseModel):
    """情感趋势响应"""
    symbol: Optional[str]
    period: str
    data: List[Dict[str, Any]]
    summary: Dict[str, Any]


class StockAnalysisResponse(BaseModel):
    """股票分析响应"""
    symbol: str
    name: Optional[str]
    current_sentiment: float
    sentiment_label: str
    mention_count: int
    trend_7d: List[Dict]
    recent_news: List[ArticleResponse]
    alert_count: int


# ============ API Endpoints ============

@app.on_event("startup")
async def startup():
    """启动时初始化"""
    await init_db()


@app.get("/")
async def root():
    return {"message": "财经舆情分析系统 API", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow()}


@app.get("/articles", response_model=List[ArticleResponse])
async def get_articles(
    source: Optional[str] = None,
    stock: Optional[str] = None,
    sentiment: Optional[str] = Query(None, regex="^(positive|negative|neutral)$"),
    days: int = Query(7, ge=1, le=30),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session)
):
    """获取文章列表"""
    query = select(Article).where(
        Article.collected_at >= datetime.utcnow() - timedelta(days=days)
    )
    
    if source:
        query = query.where(Article.source == source)
    
    if sentiment:
        query = query.where(Article.sentiment_label == sentiment)
    
    if stock:
        query = query.where(Article.stocks_mentioned.contains([{"symbol": stock}]))
    
    query = query.order_by(desc(Article.collected_at)).offset(offset).limit(limit)
    
    result = await session.execute(query)
    articles = result.scalars().all()
    
    return articles


@app.get("/articles/{article_id}", response_model=ArticleResponse)
async def get_article(
    article_id: int,
    session: AsyncSession = Depends(get_session)
):
    """获取单篇文章"""
    result = await session.execute(
        select(Article).where(Article.id == article_id)
    )
    article = result.scalar_one_or_none()
    
    if not article:
        return {"error": "Article not found"}
    
    return article


@app.get("/stats", response_model=StatsResponse)
async def get_stats(
    days: int = Query(7, ge=1, le=30),
    session: AsyncSession = Depends(get_session)
):
    """获取统计数据"""
    since = datetime.utcnow() - timedelta(days=days)
    
    # 总数和情感分布
    total = await session.scalar(
        select(func.count(Article.id)).where(Article.collected_at >= since)
    )
    
    positive = await session.scalar(
        select(func.count(Article.id)).where(
            and_(
                Article.collected_at >= since,
                Article.sentiment_label == "positive"
            )
        )
    )
    
    negative = await session.scalar(
        select(func.count(Article.id)).where(
            and_(
                Article.collected_at >= since,
                Article.sentiment_label == "negative"
            )
        )
    )
    
    neutral = await session.scalar(
        select(func.count(Article.id)).where(
            and_(
                Article.collected_at >= since,
                Article.sentiment_label == "neutral"
            )
        )
    )
    
    # 平均情感分数
    avg_sentiment = await session.scalar(
        select(func.avg(Article.sentiment_score)).where(
            and_(
                Article.collected_at >= since,
                Article.sentiment_score.isnot(None)
            )
        )
    ) or 0.0
    
    # 热门股票
    # 简化版本，实际应该解析JSONB
    top_stocks = []
    
    # 今日预警数
    alerts_today = await session.scalar(
        select(func.count(SentimentAlert.id)).where(
            SentimentAlert.created_at >= datetime.utcnow().replace(hour=0, minute=0, second=0)
        )
    )
    
    return StatsResponse(
        total_articles=total or 0,
        positive=positive or 0,
        negative=negative or 0,
        neutral=neutral or 0,
        avg_sentiment=round(avg_sentiment, 3),
        top_stocks=top_stocks,
        alerts_today=alerts_today or 0
    )


@app.get("/alerts", response_model=List[AlertResponse])
async def get_alerts(
    severity: Optional[str] = Query(None, regex="^(low|medium|high|critical)$"),
    is_read: Optional[bool] = None,
    limit: int = Query(50, ge=1, le=200),
    session: AsyncSession = Depends(get_session)
):
    """获取预警列表"""
    query = select(SentimentAlert)
    
    if severity:
        query = query.where(SentimentAlert.severity == severity)
    
    if is_read is not None:
        query = query.where(SentimentAlert.is_read == is_read)
    
    query = query.order_by(desc(SentimentAlert.created_at)).limit(limit)
    
    result = await session.execute(query)
    alerts = result.scalars().all()
    
    return alerts


@app.post("/alerts/{alert_id}/read")
async def mark_alert_read(
    alert_id: int,
    session: AsyncSession = Depends(get_session)
):
    """标记预警为已读"""
    result = await session.execute(
        select(SentimentAlert).where(SentimentAlert.id == alert_id)
    )
    alert = result.scalar_one_or_none()
    
    if not alert:
        return {"error": "Alert not found"}
    
    alert.is_read = True
    await session.commit()
    
    return {"message": "Alert marked as read"}


@app.post("/collect")
async def trigger_collect(
    background_tasks: BackgroundTasks,
    keywords: Optional[str] = None
):
    """手动触发采集"""
    keyword_list = keywords.split(",") if keywords else None
    
    async def collect_task():
        collector = MultiSourceCollector()
        analyzer = HybridAnalyzer()
        alert_engine = AlertEngine()
        notifier = AlertNotifier()
        
        try:
            async for article_data in collector.collect_all(keyword_list):
                # 分析情感
                result = await analyzer.analyze(
                    article_data["content"], 
                    use_llm=True
                )
                
                # TODO: 保存到数据库
                logger.info(f"采集: {article_data['title'][:30]}... 情感: {result.label}")
        finally:
            await collector.close()
    
    background_tasks.add_task(collect_task)
    
    return {"message": "采集任务已启动", "keywords": keyword_list}


@app.post("/analyze")
async def analyze_text(
    text: str,
    use_llm: bool = False
):
    """分析文本情感"""
    analyzer = HybridAnalyzer()
    result = await analyzer.analyze(text, use_llm=use_llm)

    return {
        "score": result.score,
        "label": result.label,
        "confidence": result.confidence,
        "keywords": result.keywords,
        "entities": result.entities
    }


@app.post("/analyze/batch", response_model=BatchAnalyzeResponse)
async def batch_analyze_text(request: BatchAnalyzeRequest):
    """批量分析文本情感

    支持一次分析最多 100 条文本，返回每条结果和汇总统计
    """
    analyzer = HybridAnalyzer()

    # 并发分析所有文本
    results = await analyzer.batch_analyze(request.texts, use_llm=request.use_llm)

    # 构建响应
    result_list = []
    positive_count = negative_count = neutral_count = 0
    total_score = 0.0

    for r in results:
        result_list.append({
            "score": r.score,
            "label": r.label,
            "confidence": r.confidence,
            "keywords": r.keywords,
            "entities": r.entities
        })
        total_score += r.score
        if r.label == "positive":
            positive_count += 1
        elif r.label == "negative":
            negative_count += 1
        else:
            neutral_count += 1

    count = len(results)
    summary = {
        "total": count,
        "positive": positive_count,
        "negative": negative_count,
        "neutral": neutral_count,
        "positive_pct": round(positive_count / count * 100, 2) if count else 0,
        "negative_pct": round(negative_count / count * 100, 2) if count else 0,
        "avg_score": round(total_score / count, 3) if count else 0,
        "overall_sentiment": "positive" if total_score > 0 else "negative" if total_score < 0 else "neutral"
    }

    return BatchAnalyzeResponse(results=result_list, summary=summary)


@app.get("/sentiment/trend", response_model=SentimentTrendResponse)
async def get_sentiment_trend(
    symbol: Optional[str] = Query(None, description="股票代码，不传则分析整体趋势"),
    days: int = Query(7, ge=1, le=30, description="查询天数"),
    group_by: str = Query("day", regex="^(day|hour)$", description="按天或按小时聚合"),
    session: AsyncSession = Depends(get_session)
):
    """获取情感趋势

    支持查询单只股票或整体市场的情感趋势
    """
    since = datetime.utcnow() - timedelta(days=days)

    # 按时间聚合情感数据
    if group_by == "day":
        date_cast = cast(Article.published_at, Date)
    else:
        # 按小时聚合（简化处理，使用 published_at 的小时部分）
        date_cast = cast(func.date_trunc("hour", Article.published_at), DateTime)

    query = select(
        date_cast.label("date"),
        func.count(Article.id).label("count"),
        func.avg(Article.sentiment_score).label("avg_score"),
        func.sum(case((Article.sentiment_label == "positive", 1), else_=0)).label("positive"),
        func.sum(case((Article.sentiment_label == "negative", 1), else_=0)).label("negative"),
        func.sum(case((Article.sentiment_label == "neutral", 1), else_=0)).label("neutral"),
    ).where(Article.published_at >= since)

    if symbol:
        query = query.where(Article.stocks_mentioned.contains([{"symbol": symbol}]))

    query = query.group_by(date_cast).order_by(date_cast)

    result = await session.execute(query)
    rows = result.fetchall()

    # 构建趋势数据
    trend_data = []
    total_positive = total_negative = total_neutral = 0
    total_score = 0.0
    total_count = 0

    for row in rows:
        trend_data.append({
            "date": row.date.isoformat() if row.date else None,
            "count": row.count or 0,
            "avg_sentiment": round(row.avg_score, 3) if row.avg_score else 0,
            "positive": row.positive or 0,
            "negative": row.negative or 0,
            "neutral": row.neutral or 0,
        })
        total_count += row.count or 0
        total_positive += row.positive or 0
        total_negative += row.negative or 0
        total_neutral += row.neutral or 0
        if row.avg_score and row.count:
            total_score += row.avg_score * row.count

    summary = {
        "total_articles": total_count,
        "avg_sentiment": round(total_score / total_count, 3) if total_count else 0,
        "positive_pct": round(total_positive / total_count * 100, 2) if total_count else 0,
        "negative_pct": round(total_negative / total_count * 100, 2) if total_count else 0,
        "neutral_pct": round(total_neutral / total_count * 100, 2) if total_count else 0,
    }

    return SentimentTrendResponse(
        symbol=symbol,
        period=f"{days}d",
        data=trend_data,
        summary=summary
    )


@app.get("/stocks/{symbol}/analysis", response_model=StockAnalysisResponse)
async def get_stock_analysis(
    symbol: str,
    days: int = Query(7, ge=1, le=30),
    session: AsyncSession = Depends(get_session)
):
    """获取股票综合分析

    包含当前情感、提及统计、趋势和相关新闻
    """
    since = datetime.utcnow() - timedelta(days=days)

    # 查询该股票的文章
    stock_query = select(Article).where(
        and_(
            Article.published_at >= since,
            Article.stocks_mentioned.contains([{"symbol": symbol}])
        )
    ).order_by(desc(Article.published_at)).limit(50)

    result = await session.execute(stock_query)
    articles = result.scalars().all()

    if not articles:
        # 尝试从 StockMention 表获取统计数据
        mention_query = select(StockMention).where(
            StockMention.stock_symbol == symbol
        ).order_by(desc(StockMention.date)).limit(7)

        result = await session.execute(mention_query)
        mentions = result.scalars().all()

        if mentions:
            latest = mentions[0]
            current_sentiment = latest.avg_sentiment or 0
        else:
            current_sentiment = 0

        return StockAnalysisResponse(
            symbol=symbol,
            name=None,
            current_sentiment=current_sentiment,
            sentiment_label="neutral",
            mention_count=0,
            trend_7d=[],
            recent_news=[],
            alert_count=0
        )

    # 计算当前情感
    total_score = sum(a.sentiment_score or 0 for a in articles)
    avg_score = total_score / len(articles) if articles else 0

    # 统计标签
    positive = sum(1 for a in articles if a.sentiment_label == "positive")
    negative = sum(1 for a in articles if a.sentiment_label == "negative")

    if avg_score > 0.1:
        label = "positive"
    elif avg_score < -0.1:
        label = "negative"
    else:
        label = "neutral"

    # 获取 7 天趋势（简化版）
    trend_data = []
    for i in range(7):
        day_start = datetime.utcnow() - timedelta(days=i+1)
        day_end = datetime.utcnow() - timedelta(days=i)

        day_articles = [
            a for a in articles
            if a.published_at and day_start <= a.published_at < day_end
        ]
        if day_articles:
            day_avg = sum(a.sentiment_score or 0 for a in day_articles) / len(day_articles)
            trend_data.append({
                "date": day_start.strftime("%Y-%m-%d"),
                "count": len(day_articles),
                "avg_sentiment": round(day_avg, 3)
            })

    # 查询相关预警
    alert_query = select(func.count(SentimentAlert.id)).where(
        and_(
            SentimentAlert.created_at >= since,
            SentimentAlert.stocks.contains([symbol])
        )
    )
    alert_count = await session.scalar(alert_query) or 0

    # 获取股票名称
    recognizer = StockEntityRecognizer()
    name = recognizer.STOCK_ALIASES.get(symbol, None)

    return StockAnalysisResponse(
        symbol=symbol,
        name=name,
        current_sentiment=round(avg_score, 3),
        sentiment_label=label,
        mention_count=len(articles),
        trend_7d=trend_data[:7],
        recent_news=[ArticleResponse.model_validate(a) for a in articles[:10]],
        alert_count=alert_count
    )


@app.get("/stocks/{symbol}/mentions")
async def get_stock_mentions(
    symbol: str,
    days: int = Query(7, ge=1, le=30),
    session: AsyncSession = Depends(get_session)
):
    """获取股票提及历史"""
    since = datetime.utcnow() - timedelta(days=days)
    
    result = await session.execute(
        select(StockMention).where(
            and_(
                StockMention.stock_symbol == symbol,
                StockMention.date >= since
            )
        ).order_by(desc(StockMention.date))
    )
    
    mentions = result.scalars().all()
    
    return {
        "symbol": symbol,
        "mentions": [
            {
                "date": m.date,
                "count": m.mention_count,
                "avg_sentiment": m.avg_sentiment,
                "trend": m.sentiment_trend
            }
            for m in mentions
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)