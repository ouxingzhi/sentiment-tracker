"""预警系统"""
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
from loguru import logger
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import Article, SentimentAlert, StockMention
from config.settings import settings


@dataclass
class Alert:
    """预警数据"""
    alert_type: str
    severity: str
    title: str
    description: str
    stocks: List[str]
    article_ids: List[int]
    data: Dict


class AlertEngine:
    """预警引擎"""
    
    def __init__(self):
        self.threshold = settings.alert_sentiment_threshold
        self.volume_multiplier = settings.alert_volume_multiplier
        self.keywords = settings.alert_keyword_list
    
    async def check_alerts(
        self, 
        articles: List[Article], 
        session: AsyncSession
    ) -> List[Alert]:
        """检查多类预警"""
        alerts = []
        
        # 1. 极端情感预警
        for article in articles:
            if article.sentiment_score and article.sentiment_score < self.threshold:
                alert = Alert(
                    alert_type="sentiment_spike",
                    severity="high" if article.sentiment_score < -0.8 else "medium",
                    title=f"负面舆情: {article.title[:50]}...",
                    description=f"情感分数: {article.sentiment_score:.2f}\n来源: {article.source}",
                    stocks=[s["symbol"] for s in (article.stocks_mentioned or [])],
                    article_ids=[article.id],
                    data={"sentiment_score": article.sentiment_score}
                )
                alerts.append(alert)
        
        # 2. 关键词预警
        for article in articles:
            content = f"{article.title} {article.content}"
            matched_keywords = [kw for kw in self.keywords if kw in content]
            if matched_keywords:
                alert = Alert(
                    alert_type="keyword_match",
                    severity="high",
                    title=f"关键词预警: {', '.join(matched_keywords)}",
                    description=f"标题: {article.title}\n来源: {article.source}",
                    stocks=[s["symbol"] for s in (article.stocks_mentioned or [])],
                    article_ids=[article.id],
                    data={"keywords": matched_keywords}
                )
                alerts.append(alert)
        
        # 3. 股票提及量异常
        volume_alerts = await self._check_volume_spike(session)
        alerts.extend(volume_alerts)
        
        return alerts
    
    async def _check_volume_spike(self, session: AsyncSession) -> List[Alert]:
        """检查提及量异常"""
        alerts = []
        
        try:
            # 获取过去24小时的提及统计
            now = datetime.utcnow()
            yesterday = now - timedelta(hours=24)
            prev_day = now - timedelta(hours=48)
            
            for symbol in settings.watch_stock_list[:10]:  # 限制查询数量
                # 当前24小时
                current_count = await session.scalar(
                    select(func.count(Article.id)).where(
                        and_(
                            Article.collected_at >= yesterday,
                            Article.stocks_mentioned.contains([{"symbol": symbol}])
                        )
                    )
                )
                
                # 前24小时
                prev_count = await session.scalar(
                    select(func.count(Article.id)).where(
                        and_(
                            Article.collected_at >= prev_day,
                            Article.collected_at < yesterday,
                            Article.stocks_mentioned.contains([{"symbol": symbol}])
                        )
                    )
                )
                
                # 检查是否异常
                if prev_count > 0:
                    multiplier = current_count / prev_count
                    if multiplier >= self.volume_multiplier:
                        alerts.append(Alert(
                            alert_type="volume_spike",
                            severity="medium",
                            title=f"{symbol} 讨论量激增",
                            description=f"过去24h提及{current_count}次，相比之前增长{multiplier:.1f}倍",
                            stocks=[symbol],
                            article_ids=[],
                            data={
                                "current_count": current_count,
                                "prev_count": prev_count,
                                "multiplier": multiplier
                            }
                        ))
        except Exception as e:
            logger.error(f"检查提及量异常失败: {e}")
        
        return alerts


class AlertNotifier:
    """预警通知器"""
    
    def __init__(self):
        self.telegram_token = settings.telegram_bot_token
        self.telegram_chat_id = settings.telegram_chat_id
    
    async def send_alert(self, alert: Alert) -> bool:
        """发送预警通知"""
        # 优先Telegram
        if self.telegram_token and self.telegram_chat_id:
            return await self._send_telegram(alert)
        
        # 其他渠道可以在这里扩展
        logger.info(f"预警: [{alert.severity}] {alert.title}")
        return False
    
    async def _send_telegram(self, alert: Alert) -> bool:
        """发送Telegram通知"""
        import httpx
        
        # 构建消息
        severity_emoji = {
            "low": "ℹ️",
            "medium": "⚠️",
            "high": "🔴",
            "critical": "🚨"
        }
        
        text = f"""{severity_emoji.get(alert.severity, '⚠️')} *舆情预警*

*类型*: {alert.alert_type}
*标题*: {alert.title}

{alert.description}

*相关股票*: {', '.join(alert.stocks) if alert.stocks else '无'}
*时间*: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"https://api.telegram.org/bot{self.telegram_token}/sendMessage",
                    json={
                        "chat_id": self.telegram_chat_id,
                        "text": text,
                        "parse_mode": "Markdown"
                    }
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Telegram通知发送失败: {e}")
            return False
    
    async def send_daily_summary(self, stats: Dict) -> bool:
        """发送每日摘要"""
        if not self.telegram_token or not self.telegram_chat_id:
            return False
        
        import httpx
        
        text = f"""📊 *每日舆情摘要*

*今日统计*:
- 文章数: {stats.get('total_articles', 0)}
- 正面: {stats.get('positive', 0)} ({stats.get('positive_pct', 0):.1f}%)
- 负面: {stats.get('negative', 0)} ({stats.get('negative_pct', 0):.1f}%)
- 中性: {stats.get('neutral', 0)}

*热门股票*:
{self._format_top_stocks(stats.get('top_stocks', []))}

*预警数*: {stats.get('alerts', 0)}
"""
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"https://api.telegram.org/bot{self.telegram_token}/sendMessage",
                    json={
                        "chat_id": self.telegram_chat_id,
                        "text": text,
                        "parse_mode": "Markdown"
                    }
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"每日摘要发送失败: {e}")
            return False
    
    def _format_top_stocks(self, stocks: List[Dict]) -> str:
        """格式化热门股票"""
        if not stocks:
            return "暂无数据"
        
        lines = []
        for stock in stocks[:5]:
            sentiment_emoji = "📈" if stock.get('avg_sentiment', 0) > 0 else "📉" if stock.get('avg_sentiment', 0) < 0 else "➡️"
            lines.append(f"{sentiment_emoji} {stock['symbol']}: {stock['mentions']}次")
        
        return "\n".join(lines)