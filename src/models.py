"""数据库模型"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class Article(Base):
    """新闻/文章表"""
    __tablename__ = "articles"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # 来源信息
    source = Column(String(50), nullable=False, index=True)  # 微博/雪球/新闻等
    source_id = Column(String(200), unique=True, nullable=False)  # 原文ID
    url = Column(String(500))
    title = Column(String(500))
    content = Column(Text, nullable=False)
    author = Column(String(100))
    
    # 时间
    published_at = Column(DateTime, index=True)
    collected_at = Column(DateTime, default=func.now(), index=True)
    
    # 分析结果
    sentiment_score = Column(Float, index=True)  # -1 到 1
    sentiment_label = Column(String(20))  # positive/negative/neutral
    
    # 实体识别
    stocks_mentioned = Column(JSONB, default=list)  # [{"symbol": "AAPL", "name": "Apple", "relevance": 0.9}]
    companies = Column(JSONB, default=list)
    people = Column(JSONB, default=list)
    
    # 元数据
    extra = Column(JSONB, default=dict)
    
    # 状态
    is_processed = Column(Boolean, default=False, index=True)
    is_alert_sent = Column(Boolean, default=False)
    
    __table_args__ = (
        Index('idx_articles_published_source', 'published_at', 'source'),
        Index('idx_articles_sentiment', 'sentiment_score'),
    )


class SentimentAlert(Base):
    """舆情预警表"""
    __tablename__ = "sentiment_alerts"
    
    id = Column(Integer, primary_key=True)
    
    # 预警类型
    alert_type = Column(String(50), nullable=False)  # sentiment_spike/volume_spike/keyword_match
    severity = Column(String(20), default="medium")  # low/medium/high/critical
    
    # 关联文章
    article_ids = Column(JSONB, default=list)
    
    # 预警内容
    title = Column(String(500), nullable=False)
    description = Column(Text)
    
    # 相关股票
    stocks = Column(JSONB, default=list)
    
    # 时间
    created_at = Column(DateTime, default=func.now())
    
    # 状态
    is_read = Column(Boolean, default=False)
    is_resolved = Column(Boolean, default=False)


class StockMention(Base):
    """股票提及统计表（按天聚合）"""
    __tablename__ = "stock_mentions"
    
    id = Column(Integer, primary_key=True)
    
    stock_symbol = Column(String(20), nullable=False, index=True)
    stock_name = Column(String(100))
    date = Column(DateTime, nullable=False, index=True)
    
    # 统计数据
    mention_count = Column(Integer, default=0)
    positive_count = Column(Integer, default=0)
    negative_count = Column(Integer, default=0)
    neutral_count = Column(Integer, default=0)
    
    avg_sentiment = Column(Float)
    sentiment_trend = Column(Float)  # 相比前一天的变化
    
    # 热门文章
    top_articles = Column(JSONB, default=list)
    
    __table_args__ = (
        Index('idx_stock_mentions_symbol_date', 'stock_symbol', 'date', unique=True),
    )


class SystemLog(Base):
    """系统日志表"""
    __tablename__ = "system_logs"
    
    id = Column(Integer, primary_key=True)
    
    level = Column(String(20), default="INFO")
    module = Column(String(50))
    message = Column(Text)
    data = Column(JSONB, default=dict)
    created_at = Column(DateTime, default=func.now())