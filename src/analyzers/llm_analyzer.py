"""LLM情感分析器 - 使用Qwen2.5生成模型"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 强制CPU模式（GTX 1060不兼容）

from typing import Dict, List, Optional
from dataclasses import dataclass
from loguru import logger
import threading

@dataclass
class LLMSentimentResult:
    """LLM情感分析结果"""
    score: float
    label: str
    confidence: float
    reasoning: str
    entities: Dict[str, List[str]]
    keywords: List[str]


class LLMAnalyzer:
    """使用Qwen2.5生成模型进行情感分析"""
    
    _instance = None
    _lock = threading.Lock()
    _model_cache = {}
    
    # Few-shot示例
    FEW_SHOT_EXAMPLES = [
        {"text": "贵州茅台业绩大增，净利润同比增长50%", "label": "positive", "score": 0.85},
        {"text": "恒大暴雷，债券违约风险加剧", "label": "negative", "score": -0.80},
        {"text": "微软发布财报，业绩符合预期", "label": "neutral", "score": 0.1},
        {"text": "特斯拉股价暴跌，市值蒸发千亿", "label": "negative", "score": -0.75},
        {"text": "宁德时代获大额订单，产能扩张", "label": "positive", "score": 0.7},
    ]
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    @classmethod
    def get_instance(cls):
        """获取单例实例"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def _load_model(self):
        """加载模型"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            model_path = "Qwen/Qwen2.5-1.5B-Instruct"
            logger.info(f"加载模型: {model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True
            )
            
            logger.info("模型加载完成")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self.model = None
            self.tokenizer = None
    
    def analyze(self, text: str) -> LLMSentimentResult:
        """分析文本情感"""
        if not self.model or not self.tokenizer:
            return self._fallback(text)
        
        try:
            # 构建prompt
            prompt = self._build_prompt(text)
            
            # 生成
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.3,
                do_sample=False
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 解析结果
            return self._parse_response(response, text)
            
        except Exception as e:
            logger.warning(f"LLM分析失败: {e}")
            return self._fallback(text)
    
    def _build_prompt(self, text: str) -> str:
        """构建few-shot prompt"""
        examples_text = "\n".join([
            f"文本: {ex['text']}\n情感: {ex['label']}\n分数: {ex['score']}"
            for ex in self.FEW_SHOT_EXAMPLES
        ])
        
        return f"""你是一个财经情感分析专家。分析文本的情感倾向，返回JSON格式。

示例:
{examples_text}

分析以下文本:
{text}

返回JSON (只返回JSON，不要其他内容):
{{"label": "positive/negative/neutral", "score": -1到1的数字, "reasoning": "简短理由"}}"""
    
    def _parse_response(self, response: str, text: str) -> LLMSentimentResult:
        """解析模型响应"""
        import json
        import re
        
        # 提取最后一个JSON部分（模型可能继续生成多个）
        json_matches = re.findall(r'\{"label":\s*"[^"]+",\s*"score":\s*[^,}]+,\s*"reasoning":\s*"[^"]+"\}', response)
        if json_matches:
            try:
                # 取最后一个匹配（当前新闻的分析结果）
                result = json.loads(json_matches[-1])
                label = result.get("label", "neutral")
                score = float(result.get("score", 0))
                reasoning = result.get("reasoning", "")
                
                logger.info(f"LLM分析成功: {label} ({score})")
                
                return LLMSentimentResult(
                    score=score,
                    label=label,
                    confidence=0.85,
                    reasoning=f"LLM: {reasoning}",
                    entities={},
                    keywords=[]
                )
            except Exception as e:
                logger.warning(f"JSON解析失败: {e}")
        
        return self._fallback(text)
    
    def _fallback(self, text: str) -> LLMSentimentResult:
        """回退到规则分析"""
        from .financial_sentiment import FinancialSentimentAnalyzer
        
        analyzer = FinancialSentimentAnalyzer()
        result = analyzer.analyze(text)
        
        return LLMSentimentResult(
            score=result.score,
            label=result.label,
            confidence=result.confidence,
            reasoning="使用规则分析器",
            entities=result.entities,
            keywords=result.keywords
        )


# 测试
if __name__ == "__main__":
    analyzer = LLMAnalyzer.get_instance()
    
    tests = [
        "贵州茅台业绩大增，净利润同比增长50%",
        "恒大暴雷，债券违约风险加剧",
        "特斯拉股价暴跌，市值蒸发千亿",
    ]
    
    for text in tests:
        result = analyzer.analyze(text)
        print(f"[{result.label:8}] {result.score:+.3f} | {text[:30]}...")
        print(f"  推理: {result.reasoning}")
        print()