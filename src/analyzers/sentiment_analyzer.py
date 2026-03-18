"""情感分析引擎 - 针对中文财经文本优化（纯本地版本）"""
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re
from loguru import logger
from snownlp import SnowNLP
import jieba
import jieba.analyse
from .financial_sentiment import FinancialSentimentAnalyzer as FinAnalyzer


@dataclass
class SentimentResult:
    """情感分析结果"""
    score: float  # -1 到 1
    label: str  # positive/negative/neutral
    confidence: float
    keywords: List[str]
    entities: Dict[str, List[str]]  # {"stocks": [], "companies": [], "people": []}


class BaseAnalyzer(ABC):
    """分析器基类"""

    @abstractmethod
    async def analyze(self, text: str) -> SentimentResult:
        pass


class RuleBasedAnalyzer(BaseAnalyzer):
    """规则情感分析器 (快速，无 API 调用)

    针对中文财经文本优化的情感分析器：
    - 扩展的金融领域情感词库
    - 中文否定词处理
    - 程度副词权重
    - 成语/俗语识别
    - 金融术语专业词库
    """

    # 金融领域情感词库 - 超扩展版
    POSITIVE_WORDS = {
        # 基础利好
        "利好", "上涨", "涨停", "牛市", "盈利", "增长", "突破", "新高",
        "反弹", "走强", "拉升", "买入", "增持", "回购", "分红",
        "业绩大增", "超预期", "看多", "推荐", "龙头", "领涨",
        # 扩展利好
        "飙升", "暴涨", "井喷", "放量", "加仓", "建仓", "抄底",
        "主升浪", "连板", "一字板", "封板", "吸金", "吸筹",
        "放量突破", "量价齐升", "强势上涨", "加速上扬", "再创新高",
        "业绩爆棚", "利润暴增", "营收大增", "订单爆满", "产能释放",
        "政策利好", "行业景气", "需求旺盛", "供不应求", "市占率提升",
        "技术突破", "产品大卖", "签约", "合作", "重组", "并购",
        "高分红", "高送转", "股权激励", "员工持股", "定增获批",
        "获准", "获批", "通过", "中标", "签约大单", "战略合作",
        "价值投资", "长期看好", "买入评级", "强烈推荐", "目标价上调",
        # 金融专业术语 - 正面
        "降准", "降息", "宽松", "刺激", "扶持", "补贴", "税收优惠",
        "融资成功", "发债成功", "授信", "注资", "增持股份", "回购股份",
        "业绩预增", "扭亏为盈", "环比增长", "同比增长", "复合增长",
        "毛利率提升", "净利率提升", "ROE 提升", "现金流改善", "负债率下降",
        "资产增值", "投资回报", "股息率", "市盈率合理", "估值修复",
        "基金增持", "机构买入", "外资流入", "北向资金", "主力资金流入",
        "金叉", "突破阻力", "量价配合", "底部放量", "空中加油",
        "分红派息", "转增股本", "股份回购", "大股东增持", "管理层增持",
        # 成语类
        "蒸蒸日上", "如日中天", "欣欣向荣", "蓬勃发展", "方兴未艾",
        "势如破竹", "扶摇直上", "步步高升", "财源广进", "日进斗金",
        # A 股特色词汇
        "开门红", "红盘", "飘红", "翻红", "拉升", "直线拉升",
        "秒板", "封死涨停", "涨停板", "跌停板打开", "地天板",
        "反包", "弱转强", "分歧转一致", "主力进场", "机构加持",
        # 政策利好
        "稳增长", "促发展", "高质量发展", "政策底", "市场底",
        "估值底", "三底共振", "利好兑现", "预期改善",
    }

    NEGATIVE_WORDS = {
        # 基础利空
        "利空", "下跌", "跌停", "熊市", "亏损", "下滑", "暴跌", "新低",
        "跳水", "走弱", "砸盘", "卖出", "减持", "套现", "暴雷",
        "业绩下滑", "不及预期", "看空", "风险", "踩雷", "领跌",
        # 扩展利空
        "崩盘", "爆仓", "穿仓", "爆雷", "退市", "ST", "*ST",
        "闪崩", "杀跌", "破发", "破净", "阴跌", "阴跌不止", "连续下跌",
        "业绩暴雷", "利润下滑", "营收下降", "订单减少", "产能过剩",
        "政策利空", "行业低迷", "需求萎缩", "供过于求", "市占率下滑",
        "技术失败", "产品滞销", "违约", "欠债", "负债累累", "资金链断裂",
        "被调查", "被处罚", "被立案", "涉嫌违法", "财务造假", "欺诈",
        "减持潮", "解禁潮", "质押爆仓", "强行平仓", "被动减持",
        "卖出评级", "下调目标价", "不及预期", "业绩预警", "亏损扩大",
        "债务危机", "流动性危机", "经营困难", "倒闭", "破产清算",
        # 金融专业术语 - 负面
        "加息", "收紧", "调控", "限购", "限售", "去杠杆", "强监管",
        "融资失败", "发债失败", "抽贷", "断贷", "债务违约", "信用违约",
        "业绩预减", "由盈转亏", "环比下降", "同比下降", "负增长",
        "毛利率下降", "净利率下降", "ROE 下降", "现金流恶化", "负债率上升",
        "资产减值", "投资亏损", "商誉减值", "存货跌价", "坏账增加",
        "基金减持", "机构卖出", "外资流出", "北向资金流出", "主力资金流出",
        "死叉", "跌破支撑", "量价背离", "高位放量", "顶部背离",
        "股份减持", "股东减持", "高管减持", "解禁减持", "质押平仓",
        "立案调查", "行政处罚", "监管函", "问询函", "警示函",
        "审计意见", "非标意见", "无法表示意见", "保留意见",
        # 成语类
        "一落千丈", "江河日下", "日薄西山", "每况愈下", "岌岌可危",
        "风雨飘摇", "四面楚歌", "难以为继", "入不敷出", "资不抵债",
        "量价齐跌", "价跌量增", "资金出逃", "一片绿",
        # A 股特色词汇
        "绿盘", "翻绿", "飘绿", "大盘跳水", "千股跌停",
        "跌停潮", "关灯吃面", "核按钮", "天地板", "一字跌停",
        "流动性枯竭", "踩踏", "多杀多", "获利盘出逃", "主力出逃",
        # 政策利空
        "强监管", "去杠杆", "防风险", "调控加码", "政策收紧",
    }

    # 程度副词 - 影响情感强度
    INTENSIFIERS = {
        "大": 1.5, "暴": 1.8, "狂": 1.8, "疯": 1.8, "猛": 1.5,
        "巨": 1.6, "超": 1.5, "极": 1.8, "特": 1.6, "十分": 1.5,
        "非常": 1.5, "特别": 1.5, "极其": 1.8, "异常": 1.6,
        "大幅": 1.5, "暴跌": 1.8, "暴涨": 1.8, "飙升": 1.6,
        "直线": 1.5, "直线上升": 1.6, "直线下降": 1.6,
        "连续": 1.3, "持续": 1.2, "不断": 1.2,
        # 新增财经场景程度词
        "翻倍": 2.0, "腰斩": 1.8, "历史新高": 1.7, "多年新低": 1.7,
        "创纪录": 1.6, "前所未有": 1.8, "罕见": 1.5,
    }

    # 否定词 - 反转情感
    NEGATION_WORDS = {
        "不", "没有", "未", "未能", "无法", "不能", "不可",
        "难", "难以", "并未", "尚未", "不再", "不用", "不必",
    }

    # 股票代码正则
    STOCK_PATTERN = re.compile(
        r'(?:\$([A-Z]{1,5})\b)|'  # $AAPL 格式
        r'(?:([0-9]{6})(?:\.SH|\.SZ)?)|'  # A 股代码
        r'(?:股票 [代码]?:\s*([0-9]{6}))|'
        r'(?:([A-Z]{2,4})(?:\s+)?(?:股份 | 集团 | 公司))'
    )

    async def analyze(self, text: str) -> SentimentResult:
        """快速规则分析 - 针对中文优化"""
        # 分词
        words = list(jieba.cut(text))

        # 统计情感词
        pos_count = 0
        neg_count = 0

        # 检查成语（直接文本匹配，权重更高）
        for idiom in self.POSITIVE_WORDS:
            if idiom in ["蒸蒸日上", "如日中天", "欣欣向荣", "蓬勃发展", "方兴未艾",
                         "势如破竹", "扶摇直上", "步步高升", "财源广进", "日进斗金"]:
                if idiom in text:
                    pos_count += 2

        for idiom in self.NEGATIVE_WORDS:
            if idiom in ["一落千丈", "江河日下", "日薄西山", "每况愈下", "岌岌可危",
                         "风雨飘摇", "四面楚歌", "难以为继", "入不敷出", "资不抵债"]:
                if idiom in text:
                    neg_count += 2

        # 检查普通情感词（带否定词和程度副词处理）
        for i, word in enumerate(words):
            # 检查是否有否定词在前（简单的前后窗口检查）
            has_negation = False
            for j in range(max(0, i-3), i):
                if words[j] in self.NEGATION_WORDS:
                    has_negation = True
                    break

            # 获取程度副词权重
            intensifier_weight = 1.0
            if i > 0 and words[i-1] in self.INTENSIFIERS:
                intensifier_weight = self.INTENSIFIERS[words[i-1]]

            if word in self.POSITIVE_WORDS:
                weight = 1.0 * intensifier_weight
                if has_negation:
                    neg_count += weight  # 否定变负面
                else:
                    pos_count += weight
            elif word in self.NEGATIVE_WORDS:
                weight = 1.0 * intensifier_weight
                if has_negation:
                    pos_count += weight  # 否定变正面
                else:
                    neg_count += weight

        # 计算基础分数
        total = pos_count + neg_count
        if total == 0:
            score = 0.0
        else:
            score = (pos_count - neg_count) / max(total, 1)

        # SnowNLP 补充（针对中文优化）
        try:
            snownlp_score = SnowNLP(text).sentiments
            # 综合分数 - 规则和 SnowNLP 加权
            score = score * 0.6 + (snownlp_score - 0.5) * 0.4
        except Exception as e:
            logger.debug(f"SnowNLP 分析失败：{e}")

        # 提取关键词（中文优化）
        try:
            keywords = jieba.analyse.extract_tags(text, topK=10)
        except Exception as e:
            logger.debug(f"关键词提取失败：{e}")
            keywords = [w for w in words if len(w) >= 2 and w not in {"的", "了", "是", "在", "有", "和", "与", "或"}][:10]

        # 实体识别
        entities = self._extract_entities(text)

        # 确定标签
        if score >= 0.2:
            label = "positive"
        elif score <= -0.2:
            label = "negative"
        else:
            label = "neutral"

        return SentimentResult(
            score=max(-1, min(1, score)),
            label=label,
            confidence=min(abs(score) + 0.5, 1.0),
            keywords=list(keywords),
            entities=entities
        )

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """提取实体 - 中文优化版

        支持：
        - 股票代码（A 股、港股、美股）
        - 公司名（中文简称）
        - 人名（简单提取）
        """
        entities = {
            "stocks": [],
            "companies": [],
            "people": []
        }

        # 提取股票代码 - 支持多种格式
        # $AAPL 格式（美股）
        for match in re.finditer(r'\$([A-Z]{1,5})\b', text):
            entities["stocks"].append(match.group(1))

        # A 股代码 600519.SH 或 600519
        for match in re.finditer(r'(\d{6})(?:\.(SH|SZ|HK))?', text):
            code = match.group(1)
            suffix = match.group(2) or ('SH' if code.startswith('6') else 'SZ')
            entities["stocks"].append(f"{code}.{suffix}")

        # 港股代码 HK00700 或 00700.HK
        for match in re.finditer(r'(?:HK)?(\d{5})\.(?:HK|COM)', text, re.IGNORECASE):
            entities["stocks"].append(f"HK{match.group(1)}")

        # 使用预定义的中文股票别名映射
        for name, symbol in StockEntityRecognizer.STOCK_ALIASES.items():
            if name in text:
                entities["stocks"].append(f"{name}({symbol})")

        # 提取公司名 - 中文模式
        company_patterns = [
            r'([^\s,.]{2,20}公司)',
            r'([^\s,.]{2,20}集团)',
            r'([^\s,.]{2,20}股份)',
            r'([^\s,.]{2,20}科技)',
        ]
        for pattern in company_patterns:
            for match in re.finditer(pattern, text):
                company = match.group(1)
                if company not in entities["companies"] and len(company) >= 4:
                    entities["companies"].append(company)

        # 提取人名 - 简单中文名字模式
        person_patterns = [
            r'([张王李赵钱孙周吴郑冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华][^\s,.]{1,3})(?:表示 | 称 | 说 | 指出 | 认为)',
        ]
        for pattern in person_patterns:
            for match in re.finditer(pattern, text):
                person = match.group(1)
                if 2 <= len(person) <= 4:
                    entities["people"].append(person)

        # 去重
        entities["stocks"] = list(set(entities["stocks"]))
        entities["companies"] = list(set(entities["companies"]))
        entities["people"] = list(set(entities["people"]))

        return entities


class HybridAnalyzer:
    """混合分析器 - 纯本地版本，结合规则、SnowNLP 和金融情感词典"""

    def __init__(self, use_financial_dict: bool = True):
        """初始化混合分析器

        Args:
            use_financial_dict: 是否使用金融情感词典增强分析
        """
        self.rule_analyzer = RuleBasedAnalyzer()
        self.use_financial_dict = use_financial_dict
        if use_financial_dict:
            self.financial_analyzer = FinAnalyzer()

    async def analyze(self, text: str, use_llm: bool = False) -> SentimentResult:
        """分析情感（纯本地，use_llm 参数已废弃）

        Args:
            text: 待分析文本
            use_llm: 已废弃，保留用于兼容性

        Returns:
            SentimentResult: 情感分析结果
        """
        if self.use_financial_dict:
            # 使用金融情感分析器进行分析（更准确）
            fin_result = self.financial_analyzer.analyze(text)
            return SentimentResult(
                score=fin_result.score,
                label=fin_result.label,
                confidence=fin_result.confidence,
                keywords=fin_result.keywords,
                entities=fin_result.entities
            )
        else:
            # 使用规则分析器进行分析（已内置 SnowNLP 补充）
            return await self.rule_analyzer.analyze(text)

    async def batch_analyze(self, texts: List[str], use_llm: bool = False) -> List[SentimentResult]:
        """批量分析"""
        return await asyncio.gather(*[
            self.analyze(text, use_llm) for text in texts
        ])

    def analyze_sync(self, text: str) -> SentimentResult:
        """同步分析接口（适用于金融词典模式）"""
        if self.use_financial_dict:
            fin_result = self.financial_analyzer.analyze(text)
            return SentimentResult(
                score=fin_result.score,
                label=fin_result.label,
                confidence=fin_result.confidence,
                keywords=fin_result.keywords,
                entities=fin_result.entities
            )
        raise RuntimeError("金融词典模式未启用，请使用 async analyze 方法")


# 股票实体识别器
class StockEntityRecognizer:
    """股票实体识别 - 中文优化版

    支持：
    - A 股：茅台、宁德时代、比亚迪等
    - 港股：腾讯、阿里巴巴、美团等
    - 美股：苹果、特斯拉、英伟达等
    """

    # 常见股票映射 - 超扩展版
    STOCK_ALIASES = {
        # ================= 美股 =================
        "苹果": "AAPL", "Apple": "AAPL", "AAPL": "AAPL",
        "特斯拉": "TSLA", "Tesla": "TSLA", "电车": "TSLA", "马斯克": "TSLA",
        "英伟达": "NVDA", "Nvidia": "NVDA", "老黄": "NVDA", "皮衣刀客": "NVDA",
        "微软": "MSFT", "Microsoft": "MSFT", "纳德拉": "MSFT",
        "谷歌": "GOOGL", "Google": "GOOGL", "Alphabet": "GOOGL",
        "亚马逊": "AMZN", "Amazon": "AMZN", "贝索斯": "AMZN",
        "Meta": "META", "Facebook": "META", "脸书": "META", "小扎": "META", "扎克伯格": "META",
        "Netflix": "NFLX", "奈飞": "NFLX",
        "英特尔": "INTC", "Intel": "INTC", "牙膏厂": "INTC",
        "AMD": "AMD", "苏妈": "AMD", "超威": "AMD",
        "波音": "BA", "Boeing": "BA",
        "高盛": "GS", "Goldman": "GS", "大摩": "MS", "摩根士丹利": "MS",
        "花旗": "C", "Citigroup": "C", "摩根大通": "JPM", "小摩": "JPM",
        "伯克希尔": "BRK.A", "巴菲特": "BRK.A", "股神": "BRK.A",
        "辉瑞": "PFE", "Pfizer": "PFE",
        "强生": "JNJ", "Johnson": "JNJ",
        "沃尔玛": "WMT", "Walmart": "WMT",
        "迪士尼": "DIS", "Disney": "DIS",
        "奈飞": "NFLX", "网飞": "NFLX",
        "可口可乐": "KO", "CocaCola": "KO",
        "百事可乐": "PEP", "Pepsi": "PEP",
        "耐克": "NKE", "Nike": "NKE",
        "星巴克": "SBUX", "Starbucks": "SBUX",
        "麦当劳": "MCD", "McDonald": "MCD",
        "Visa": "V", "万事达": "MA", "Mastercard": "MA",
        "Salesforce": "CRM", "甲骨文": "ORCL", "Oracle": "ORCL",
        "IBM": "IBM", "思科": "CSCO", "Cisco": "CSCO",
        "高通": "QCOM", "Qualcomm": "QCOM",
        "德州仪器": "TXN", "TI": "TXN",
        "阿斯麦": "ASML", "ASML": "ASML", "光刻机": "ASML",
        "台积电": "TSM", "TSMC": "TSM",
        "三星": "005930.KS", "Samsung": "005930.KS",
        "索尼": "SONY", "Sony": "SONY",
        "丰田": "TM", "Toyota": "TM",
        "礼来": "LLY", "诺和诺德": "NVO",
        "Moderna": "MRNA", "生科": "MRNA",
        "PayPal": "PYPL", "贝宝": "PYPL",
        "Square": "SQ", "Block": "SQ",
        "Shopify": "SHOP",
        "Uber": "UBER", "优步": "UBER",
        "Lyft": "LYFT",
        "Airbnb": "ABNB",
        "Palantir": "PLTR",
        "Snowflake": "SNOW",
        "Coinbase": "COIN",
        "Roblox": "RBLX",
        "Unity": "U",

        # ================= A 股 =================
        # 白酒
        "茅台": "600519.SH", "贵州茅台": "600519.SH", "酱香科技": "600519.SH", "茅指数": "600519.SH",
        "五粮液": "000858.SZ", "普五": "000858.SZ",
        "泸州老窖": "000568.SZ", "国窖": "000568.SZ",
        "山西汾酒": "600809.SH", "汾酒": "600809.SH",
        "洋河股份": "002304.SZ", "洋河": "002304.SZ",
        "古井贡酒": "000596.SZ", "古井": "000596.SZ",
        # 金融
        "宁德时代": "300750.SZ", "宁王": "300750.SZ", "电池茅": "300750.SZ", "CATL": "300750.SZ",
        "比亚迪": "002594.SZ", "比雅迪": "002594.SZ", "王传福": "002594.SZ",
        "平安": "601318.SH", "中国平安": "601318.SH", "马明哲": "601318.SH",
        "招商银行": "600036.SH", "招行": "600036.SH", "零售之王": "600036.SH",
        "工商银行": "601398.SH", "工行": "601398.SH", "宇宙行": "601398.SH",
        "建设银行": "601939.SH", "建行": "601939.SH",
        "农业银行": "601288.SH", "农行": "601288.SH",
        "中国银行": "601988.SH", "中行": "601988.SH",
        "交通银行": "601328.SH", "交行": "601328.SH",
        "中信证券": "600030.SH", "中信": "600030.SH", "券商一哥": "600030.SH",
        "东方财富": "300059.SZ", "东财": "300059.SZ", "券茅": "300059.SZ",
        "华泰证券": "601688.SH", "华泰": "601688.SH",
        "海通证券": "600837.SH", "海通": "600837.SH",
        "国泰君安": "601211.SH", "国泰": "601211.SH",
        "中国人寿": "601628.SH", "国寿": "601628.SH",
        "新华保险": "601336.SH", "新华": "601336.SH",
        "宁波银行": "002142.SZ", "宁行": "002142.SZ",
        "兴业银行": "601166.SH", "兴业": "601166.SH",
        "民生银行": "600016.SH", "民生": "600016.SH",
        # 科技
        "工业富联": "601138.SH", "富士康": "601138.SH",
        "立讯精密": "002475.SZ", "立讯": "002475.SZ", "果链龙头": "002475.SZ",
        "京东方": "000725.SZ", "京东方 A": "000725.SZ", "面板龙头": "000725.SZ",
        "中芯国际": "688981.SH", "中芯": "688981.SH", "国产光刻": "688981.SH",
        "海康威视": "002415.SZ", "海康": "002415.SZ", "安防龙头": "002415.SZ",
        "大华股份": "002236.SZ", "大华": "002236.SZ",
        "中兴通讯": "000063.SZ", "中兴": "000063.SZ",
        "工业富联": "601138.SH", "工业": "601138.SH",
        "浪潮信息": "000977.SZ", "浪潮": "000977.SZ", "服务器": "000977.SZ",
        "中科曙光": "603019.SH", "曙光": "603019.SH",
        "寒武纪": "688256.SH", "AI 芯片": "688256.SH",
        "韦尔股份": "603501.SH", "韦尔": "603501.SH", "芯片": "603501.SH",
        "卓胜微": "300782.SZ", "射频": "300782.SZ",
        "兆易创新": "603986.SH", "兆易": "603986.SH", "存储": "603986.SH",
        # 医药
        "恒瑞医药": "600276.SH", "恒瑞": "600276.SH", "药茅": "600276.SH",
        "药明康德": "603259.SH", "药明": "603259.SH", "CXO": "603259.SH",
        "泰格医药": "300347.SZ", "泰格": "300347.SZ",
        "康龙化成": "300759.SZ", "康龙": "300759.SZ",
        "片仔癀": "600436.SH", "药中茅台": "600436.SH",
        "云南白药": "000538.SZ", "白药": "000538.SZ",
        "同仁堂": "600085.SH", "同仁": "600085.SH",
        "迈瑞医疗": "300760.SZ", "迈瑞": "300760.SZ", "器械茅": "300760.SZ",
        "爱尔眼科": "300015.SZ", "爱尔": "300015.SZ",
        "通策医疗": "600763.SH", "通策": "600763.SH", "牙茅": "600763.SH",
        # 消费
        "美的集团": "000333.SZ", "美的": "000333.SZ", "方洪波": "000333.SZ",
        "格力电器": "000651.SZ", "格力": "000651.SZ", "董明珠": "000651.SZ",
        "海尔智家": "600690.SH", "海尔": "600690.SH",
        "海天味业": "603288.SH", "海天": "603288.SH", "酱油茅": "603288.SH",
        "中炬高新": "600872.SH", "厨邦": "600872.SH",
        "伊利股份": "600887.SH", "伊利": "600887.SH", "奶茅": "600887.SH",
        "蒙牛乳业": "02319.HK", "蒙牛": "02319.HK",
        "牧原股份": "002714.SZ", "牧原": "002714.SZ", "猪茅": "002714.SZ",
        "温氏股份": "300498.SZ", "温氏": "300498.SZ",
        "万华化学": "600309.SH", "万华": "600309.SH", "化工茅": "600309.SH",
        # 地产
        "万科": "000002.SZ", "万科 A": "000002.SZ", "郁亮": "000002.SZ",
        "保利发展": "600048.SH", "保利": "600048.SH",
        "招商蛇口": "001979.SZ", "招蛇": "001979.SZ",
        "碧桂园": "02007.HK", "碧桂园": "02007.HK",
        # 新能源
        "隆基绿能": "601012.SH", "隆基": "601012.SH", "光伏茅": "601012.SH",
        "通威股份": "600438.SH", "通威": "600438.SH", "硅料": "600438.SH",
        "阳光电源": "300274.SZ", "阳光": "300274.SZ", "逆变器": "300274.SZ",
        "亿纬锂能": "300014.SZ", "亿纬": "300014.SZ",
        "赣锋锂业": "002460.SZ", "赣锋": "002460.SZ", "锂矿": "002460.SZ",
        "天齐锂业": "002466.SZ", "天齐": "002466.SZ",
        "华友钴业": "603799.SH", "华友": "603799.SH", "钴": "603799.SH",
        "寒锐钴业": "300618.SZ", "寒锐": "300618.SZ",
        # 其他
        "中国建筑": "601668.SH", "中建": "601668.SH",
        "中国中免": "601888.SH", "中免": "601888.SH", "免税": "601888.SH",
        "顺丰控股": "002352.SZ", "顺丰": "002352.SZ", "快递": "002352.SZ",
        "京沪高铁": "601816.SH", "京沪": "601816.SH", "高铁": "601816.SH",
        "三峡能源": "600905.SH", "三峡": "600905.SH",

        # ================= 港股 =================
        "腾讯": "00700.HK", "腾讯控股": "00700.HK", "企鹅": "00700.HK", "马化腾": "00700.HK",
        "阿里巴巴": "09988.HK", "阿里": "09988.HK", "蔡崇信": "09988.HK",
        "京东": "09618.HK", "京东健康": "06618.HK", "刘强东": "09618.HK",
        "拼多多": "PDD", "黄峥": "PDD",
        "美团": "03690.HK", "美団": "03690.HK", "王兴": "03690.HK", "外卖": "03690.HK",
        "小米": "01810.HK", "小米集团": "01810.HK", "雷军": "01810.HK", "su7": "01810.HK",
        "百度": "BIDU", "百度集团": "09888.HK", "李彦宏": "BIDU",
        "网易": "NTES", "网易-S": "09999.HK", "丁磊": "NTES",
        "快手": "01024.HK", "快手续": "01024.HK", "宿华": "01024.HK",
        "哔哩哔哩": "BILI", "B 站": "BILI", "陈睿": "BILI",
        "理想汽车": "LI", "李想": "LI",
        "小鹏汽车": "XPEV", "小鹏": "XPEV", "何小鹏": "XPEV",
        "蔚来": "NIO", "李斌": "NIO", "斌哥": "NIO",
        "吉利汽车": "00175.HK", "吉利": "00175.HK",
        "长城汽车": "02333.HK", "长城": "02333.HK",
        "比亚迪股份": "01211.HK", "比亚迪": "01211.HK",
        "中芯国际": "00981.HK", "中芯": "00981.HK",
        "华虹半导体": "01347.HK", "华虹": "01347.HK",
        "联想集团": "00992.HK", "联想": "00992.HK",
        "惠普": "00992.HK",
        "中兴通讯": "00763.HK", "中兴": "00763.HK",
        "中国移动": "00941.HK", "移动": "00941.HK",
        "中国电信": "00728.HK", "电信": "00728.HK",
        "中国联通": "00762.HK", "联通": "00762.HK",
        "港交所": "00388.HK", "港交所": "00388.HK", "李小加": "00388.HK",
        "汇丰控股": "00005.HK", "汇丰": "00005.HK",
        "友邦保险": "01299.HK", "友邦": "01299.HK",
        "安踏体育": "02020.HK", "安踏": "02020.HK",
        "李宁": "02331.HK", "李宁": "02331.HK",
        "华润啤酒": "00291.HK", "华润": "00291.HK", "啤酒": "00291.HK",
        "农夫山泉": "09633.HK", "农夫": "09633.HK", "钟睒睒": "09633.HK",
        "药明生物": "02269.HK", "药明": "02269.HK",
        "信达生物": "01801.HK", "信达": "01801.HK",
        "百济神州": "06160.HK", "百济": "06160.HK",
        "龙湖集团": "00960.HK", "龙湖": "00960.HK",
        "华润置地": "01109.HK", "华润置地": "01109.HK",
        "长和": "00001.HK", "长江": "00001.HK", "李嘉诚": "00001.HK",
        "银河娱乐": "00027.HK", "银河": "00027.HK", "澳门": "00027.HK",
        "金沙中国": "01928.HK", "金沙": "01928.HK",
        "新东方": "09901.HK", "新东方": "09901.HK", "俞敏洪": "09901.HK",
        "好未来": "TAL", "学而思": "TAL",

        # ================= 其他热门中概 =================
        "贝壳": "BEKE", "链家": "BEKE",
        "中通快递": "ZTO", "中通": "ZTO",
        "好未来": "TAL",
        "新东方": "EDX",
        "携程": "TCOM", "Trip": "TCOM",
        "百度": "BIDU",
        "搜狐": "SOHU",
        "新浪": "SINA",
        "微博": "WB",
        "58 同城": "WUBA",
        "汽车之家": "ATHM",
        "虎牙": "HUYA",
        "斗鱼": "DOYU",
        "B 站": "BILI",
        "爱奇艺": "IQ",
        "腾讯视频": "00772.HK",
    }

    def extract_stocks(self, text: str) -> List[Dict]:
        """从文本中提取股票实体"""
        found = {}

        # 检查别名
        for name, symbol in self.STOCK_ALIASES.items():
            if name in text:
                if symbol not in found:
                    found[symbol] = {
                        "symbol": symbol,
                        "name": name,
                        "mentions": 0
                    }
                found[symbol]["mentions"] += 1

        # 检查股票代码格式
        # 美股 $AAPL 格式
        for match in re.finditer(r'\$([A-Z]{1,5})\b', text):
            symbol = match.group(1)
            if symbol not in found:
                found[symbol] = {"symbol": symbol, "name": "", "mentions": 1}
            else:
                found[symbol]["mentions"] += 1

        # A 股代码
        for match in re.finditer(r'([036]\d{5})(?:\.(?:SH|SZ))?', text):
            code = match.group(1)
            symbol = f"{code}.SH" if code.startswith('6') else f"{code}.SZ"
            if symbol not in found:
                found[symbol] = {"symbol": symbol, "name": "", "mentions": 1}
            else:
                found[symbol]["mentions"] += 1

        return list(found.values())
