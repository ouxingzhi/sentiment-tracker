"""RoBERTa 预训练模型情感分析器 - 针对中文财经文本优化"""
import os
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

@dataclass
class RoBERTaSentimentResult:
    """RoBERTa 情感分析结果"""
    score: float
    label: str
    confidence: float
    reasoning: str
    entities: Dict[str, List[str]]
    keywords: List[str]


class RoBERTaAnalyzer:
    """使用预训练 RoBERTa 模型进行情感分析

    模型选择策略:
    1. 首选：lxyuan/distilbert-base-uncased-finetuned-sentiment (轻量，~200MB)
    2. 备选：hfl/chinese-roberta-wwm-ext (中文优化，~400MB)
    3. 回退：hfl/chinese-roberta-wwm-ext-large (更大模型，~1.3GB)
    4. 最终回退：规则分析器

    GTX 1060 (6GB) 可运行所有模型
    """

    _instance = None
    _lock = threading.Lock()

    # 模型配置 - 按优先级排序
    MODEL_CONFIGS = [
        {
            "name": "lxyuan/distilbert-base-uncased-finetuned-sentiment",
            "type": "distilbert",
            "size": "~250MB",
            "description": "轻量级情感分类模型"
        },
        {
            "name": "uer/roberta-base-finetuned-jd-binary-chinese",
            "type": "roberta",
            "size": "~400MB",
            "description": "中文商品评论情感分析"
        },
        {
            "name": "hfl/chinese-roberta-wwm-ext",
            "type": "roberta",
            "size": "~400MB",
            "description": "哈工大中文 RoBERTa"
        },
        {
            "name": "hfl/chinese-roberta-wwm-ext-large",
            "type": "roberta-large",
            "size": "~1.3GB",
            "description": "哈工大中文 RoBERTa Large"
        },
    ]

    def __init__(self, model_name: Optional[str] = None, use_cpu: bool = True):
        """初始化 RoBERTa 分析器

        Args:
            model_name: 指定模型名称，None 则自动选择
            use_cpu: 是否强制使用 CPU (GTX 1060 显存有限，建议 True)
        """
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.model_name = None
        self.use_cpu = use_cpu
        self._load_model(model_name)

    @classmethod
    def get_instance(cls, model_name: Optional[str] = None, use_cpu: bool = True):
        """获取单例实例"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(model_name=model_name, use_cpu=use_cpu)
        return cls._instance

    def _load_model(self, specified_model: Optional[str] = None):
        """加载模型"""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
            import torch

            # 确定要使用的模型
            if specified_model:
                model_configs = [c for c in self.MODEL_CONFIGS if c["name"] == specified_model]
                if not model_configs:
                    model_configs = [{"name": specified_model, "type": "custom", "size": "?", "description": "自定义模型"}]
            else:
                model_configs = self.MODEL_CONFIGS

            # 设置设备
            if self.use_cpu:
                device = -1  # CPU
                logger.info("使用 CPU 模式")
            else:
                if torch.cuda.is_available():
                    device = 0
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    logger.info(f"使用 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
                else:
                    device = -1
                    logger.warning("GPU 不可用，回退到 CPU")

            # 尝试加载模型
            for config in model_configs:
                model_path = config["name"]
                logger.info(f"尝试加载模型：{model_path} ({config.get('description', '')})")

                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_path,
                        trust_remote_code=True
                    )

                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        model_path,
                        trust_remote_code=True
                    )

                    if device >= 0:
                        self.model.to(device)

                    # 创建 pipeline
                    self.pipeline = pipeline(
                        "sentiment-analysis",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        device=device,
                        return_all_scores=False,
                        truncation=True,
                        max_length=512
                    )

                    self.model_name = model_path
                    logger.info(f"模型加载成功：{model_path} ({config.get('size', 'unknown')})")
                    return

                except Exception as e:
                    logger.warning(f"模型加载失败 {model_path}: {e}")
                    continue

            # 所有模型都失败
            logger.error("所有模型加载失败，将使用回退模式")

        except ImportError as e:
            logger.error(f"transformers 库未安装：{e}")
        except Exception as e:
            logger.error(f"模型加载异常：{e}")

    def analyze(self, text: str) -> RoBERTaSentimentResult:
        """分析文本情感 - 融合规则和模型"""
        if not text or not text.strip():
            return self._fallback(text)

        # 1. 先用规则分析器获取基准（对财经文本更准确）
        from .financial_sentiment import FinancialSentimentAnalyzer
        rule_analyzer = FinancialSentimentAnalyzer()
        rule_result = rule_analyzer.analyze(text)

        # 2. 如果模型不可用，仅使用规则
        if not self.pipeline:
            return RoBERTaSentimentResult(
                score=rule_result.score,
                label=rule_result.label,
                confidence=rule_result.confidence,
                reasoning="规则分析器",
                entities=rule_result.entities,
                keywords=rule_result.keywords
            )

        try:
            # 3. 模型分析
            if len(text) > 512:
                text = text[:512]

            model_result = self.pipeline(text)[0]
            model_label = model_result["label"].lower()
            model_conf = model_result["score"]

            # 转换模型分数
            if "positive" in model_label or "pos" in model_label:
                model_score = model_conf
            elif "negative" in model_label or "neg" in model_label:
                model_score = -model_conf
            else:
                model_score = 0.0

            # 4. 融合：规则权重0.7，模型权重0.3
            # 规则对财经更准确
            final_score = rule_result.score * 0.7 + model_score * 0.3

            # 确定标签
            if final_score >= 0.2:
                label = "positive"
            elif final_score <= -0.2:
                label = "negative"
            else:
                label = "neutral"

            logger.info(f"融合分析：规则({rule_result.score:.2f}) + 模型({model_score:.2f}) = {final_score:.2f}")

            return RoBERTaSentimentResult(
                score=final_score,
                label=label,
                confidence=0.85,
                reasoning=f"规则+模型融合",
                entities=rule_result.entities,
                keywords=rule_result.keywords
            )

        except Exception as e:
            logger.warning(f"模型分析失败：{e}")
            return RoBERTaSentimentResult(
                score=rule_result.score,
                label=rule_result.label,
                confidence=rule_result.confidence,
                reasoning="规则分析器（模型失败）",
                entities=rule_result.entities,
                keywords=rule_result.keywords
            )

    def batch_analyze(self, texts: List[str]) -> List[RoBERTaSentimentResult]:
        """批量分析文本"""
        if not self.pipeline:
            return [self._fallback(text) for text in texts]

        try:
            results = self.pipeline(texts)
            return [self._parse_single_result(r, t) for r, t in zip(results, texts)]
        except Exception as e:
            logger.error(f"批量分析失败：{e}")
            return [self._fallback(text) for text in texts]

    def _parse_single_result(self, result: dict, text: str) -> RoBERTaSentimentResult:
        """解析单个结果"""
        label = result["label"].lower()
        score = result["score"]

        if "positive" in label or "pos" in label or label in ["1", "positive"]:
            sentiment_label = "positive"
            sentiment_score = score
        elif "negative" in label or "neg" in label or label in ["0", "negative"]:
            sentiment_label = "negative"
            sentiment_score = -score
        else:
            sentiment_label = "neutral"
            sentiment_score = 0.0

        return RoBERTaSentimentResult(
            score=sentiment_score,
            label=sentiment_label,
            confidence=score,
            reasoning=f"RoBERTa: {result}",
            entities=self._extract_entities(text),
            keywords=self._extract_keywords(text)
        )

    def _extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """提取关键词"""
        try:
            import jieba.analyse
            return jieba.analyse.extract_tags(text, topK=top_k)
        except Exception:
            # 简单分词
            words = [w for w in text if len(w.strip()) > 1]
            return words[:top_k]

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """提取实体 (股票、公司等)"""
        entities = {"stocks": [], "companies": [], "people": []}

        try:
            # 使用现有的实体识别
            from .sentiment_analyzer import StockEntityRecognizer
            recognizer = StockEntityRecognizer()
            stocks = recognizer.extract_stocks(text)
            entities["stocks"] = [s["symbol"] for s in stocks]
        except Exception:
            pass

        return entities

    def _fallback(self, text: str) -> RoBERTaSentimentResult:
        """回退到规则分析"""
        try:
            from .financial_sentiment import FinancialSentimentAnalyzer
            analyzer = FinancialSentimentAnalyzer()
            result = analyzer.analyze(text)

            return RoBERTaSentimentResult(
                score=result.score,
                label=result.label,
                confidence=result.confidence,
                reasoning="回退到金融情感分析器",
                entities=result.entities,
                keywords=result.keywords
            )
        except Exception as e:
            logger.error(f"回退分析失败：{e}")
            return RoBERTaSentimentResult(
                score=0.0,
                label="neutral",
                confidence=0.5,
                reasoning="分析器不可用",
                entities={},
                keywords=[]
            )

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "loaded": self.pipeline is not None,
            "use_cpu": self.use_cpu,
            "available_models": [c["name"] for c in self.MODEL_CONFIGS]
        }


# 测试
if __name__ == "__main__":
    import asyncio

    print("=" * 60)
    print("RoBERTa 情感分析器测试")
    print("=" * 60)

    # 测试用例
    test_cases = [
        ("贵州茅台业绩大增，净利润同比增长 50%", "positive"),
        ("恒大暴雷，债券违约风险加剧", "negative"),
        ("微软发布财报，业绩符合预期", "neutral"),
        ("特斯拉股价暴跌，市值蒸发千亿", "negative"),
        ("宁德时代获大额订单，产能扩张", "positive"),
        ("苹果新品发布会，市场反应平平", "neutral"),
        ("腾讯游戏业务增长强劲", "positive"),
        ("美联储加息预期升温", "negative"),
    ]

    # 测试模型
    analyzer = RoBERTaAnalyzer.get_instance(use_cpu=True)

    print(f"\n模型信息：{analyzer.get_model_info()}")
    print("\n测试结果:")
    print("-" * 60)

    for text, expected in test_cases:
        result = analyzer.analyze(text)
        status = "✓" if result.label == expected else "?"
        print(f"{status} [{result.label:8}] {result.score:+.3f} | 置信度：{result.confidence:.2f}")
        print(f"  文本：{text}")
        print(f"  说明：{result.reasoning}")
        if result.keywords:
            print(f"  关键词：{result.keywords}")
        print()
