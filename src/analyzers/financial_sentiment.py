"""金融情感词典分析器 - 针对中文财经文本的专业情感分析"""
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
from snownlp import SnowNLP
import jieba
import jieba.analyse


@dataclass
class FinancialSentimentResult:
    """金融情感分析结果"""
    score: float  # -1 到 1
    label: str  # positive/negative/neutral
    confidence: float
    keywords: List[str]
    matched_words: Dict[str, List[str]]  # {"positive": [], "negative": [], "intensifiers": []}
    entities: Dict[str, List[str]]  # {"stocks": [], "companies": [], "people": []}


class FinancialSentimentDictionary:
    """金融情感词典 - 专业版"""

    # ==================== 正面词汇 ====================
    POSITIVE_WORDS = {
        # 核心利好词汇
        "利好", "上涨", "涨停", "突破", "新高", "反弹", "走强", "拉升",
        "增持", "回购", "分红", "超预期", "业绩大增", "牛市", "看多",
        "龙头", "领涨", "飙升", "暴涨", "井喷", "放量", "加仓", "建仓",
        "抄底", "主升浪", "连板", "一字板", "封板", "吸金", "吸筹",

        # 业绩增长类
        "业绩爆棚", "利润暴增", "营收大增", "订单爆满", "产能释放",
        "扭亏为盈", "环比增长", "同比增长", "复合增长", "持续盈利",
        "盈利改善", "利润翻倍", "收入激增", "毛利率提升", "净利率提升",

        # 市场表现类
        "量价齐升", "强势上涨", "加速上扬", "再创新高", "突破阻力",
        "空中加油", "底部放量", "底部反转", "底部启动", "底部抬升",
        "红盘", "飘红", "翻红", "开门红", "满堂红",

        # 资本运作类
        "并购", "重组", "收购", "注资", "资产注入", "整体上市",
        "分拆上市", "IPO 获批", "定增获批", "配股", "可转债",
        "股权激励", "员工持股", "股份回购", "大股东增持", "管理层增持",

        # 政策利好类
        "政策利好", "行业景气", "需求旺盛", "供不应求", "市占率提升",
        "稳增长", "促发展", "高质量发展", "政策扶持", "税收优惠",
        "财政补贴", "产业扶持", "区域利好", "自贸区", "一带一路",

        # 金融专业术语 - 正面
        "降准", "降息", "宽松", "融资成功", "授信", "发债成功",
        "资产增值", "投资回报", "股息率高", "市盈率合理", "估值修复",
        "价值重估", "价值投资", "长期看好", "买入评级", "强烈推荐",
        "目标价上调", "基金增持", "机构买入", "外资流入", "北向资金流入",
        "主力资金流入", "融资买入", "杠杆资金", "金叉", "MACD 金叉",

        # 技术突破类
        "技术突破", "产品大卖", "新品发布", "专利获批", "研发成功",
        "投产", "产能扩张", "项目落地", "合作签约", "战略签约",
        "大客户", "订单饱满", "中标", "入围", "通过认证",

        # 成语类 - 正面
        "蒸蒸日上", "如日中天", "欣欣向荣", "蓬勃发展", "方兴未艾",
        "势如破竹", "扶摇直上", "步步高升", "财源广进", "日进斗金",
        "蒸蒸日上", "如日中天", "欣欣向荣", "蓬勃发展", "锦上添花",

        # A 股特色词汇
        "秒板", "封死涨停", "涨停板", "地天板", "反包", "弱转强",
        "分歧转一致", "主力进场", "机构加持", "游资接力", "连板龙头",
        "趋势龙头", "价值龙头", "成长龙头", "周期龙头",
    }

    # ==================== 负面词汇 ====================
    NEGATIVE_WORDS = {
        # 核心利空词汇
        "利空", "下跌", "跌停", "暴跌", "新低", "跳水", "走弱", "砸盘",
        "减持", "套现", "暴雷", "业绩下滑", "不及预期", "熊市", "看空",
        "风险", "踩雷", "领跌", "崩盘", "爆仓", "违约", "闪崩", "杀跌",

        # 业绩下滑类
        "业绩暴雷", "利润下滑", "营收下降", "订单减少", "产能过剩",
        "亏损", "亏损扩大", "由盈转亏", "环比下降", "同比下降",
        "负增长", "业绩预警", "业绩预亏", "利润腰斩", "收入锐减",

        # 市场表现类
        "破发", "破净", "阴跌", "阴跌不止", "连续下跌", "放量下跌",
        "高位放量", "顶部背离", "量价背离", "死叉", "MACD 死叉",
        "跌破支撑", "绿色", "翻绿", "飘绿", "一片绿", "千股跌停",

        # 资本运作类 - 负面
        "减持潮", "解禁潮", "质押爆仓", "强行平仓", "被动减持",
        "股东减持", "高管减持", "大额减持", "清仓式减持", "减持计划",
        "终止上市", "暂停上市", "退市风险", "强制退市", "破产重整",

        # 风险事件类
        "财务造假", "欺诈发行", "虚假陈述", "内幕交易", "操纵股价",
        "被调查", "被立案", "被处罚", "监管函", "警示函", "问询函",
        "涉嫌违法", "违法违规", "被起诉", "诉讼", "仲裁", "纠纷",

        # 债务风险类
        "债务危机", "流动性危机", "资金链断裂", "负债累累", "资不抵债",
        "债务违约", "信用违约", "债券违约", "信托违约", "理财违约",
        "抽贷", "断贷", "催收", "冻结资产", "失信", "老赖",

        # 经营风险类
        "经营困难", "倒闭", "破产清算", "停产", "裁员", "欠薪",
        "产品滞销", "库存积压", "客户流失", "市场份额下滑", "竞争恶化",
        "原材料涨价", "成本上升", "价格战", "行业寒冬", "需求萎缩",

        # 政策利空类
        "政策利空", "行业低迷", "强监管", "去杠杆", "防风险",
        "调控加码", "政策收紧", "限购", "限售", "限价", "限贷",

        # 金融专业术语 - 负面
        "加息", "收紧", "融资失败", "发债失败", "担保代偿",
        "资产减值", "投资亏损", "商誉减值", "存货跌价", "坏账增加",
        "毛利率下降", "净利率下降", "ROE 下降", "现金流恶化", "负债率上升",
        "基金减持", "机构卖出", "外资流出", "北向资金流出", "主力资金流出",
        "融资卖出", "杠杆爆仓", "爆仓平仓", "穿仓",

        # 成语类 - 负面
        "一落千丈", "江河日下", "日薄西山", "每况愈下", "岌岌可危",
        "风雨飘摇", "四面楚歌", "难以为继", "入不敷出", "雪上加霜",
        "火上浇油", "屋漏偏逢连夜雨", "祸不单行",

        # A 股特色词汇
        "跌停板", "一字跌停", "跌停潮", "关灯吃面", "核按钮",
        "天地板", "大面", "被埋", "套牢", "深套", "割肉",
        "流动性枯竭", "踩踏", "多杀多", "获利盘出逃", "主力出逃",
        "机构砸盘", "北向砸盘", "融资盘爆仓",
    }

    # ==================== 程度副词 ====================
    INTENSIFIERS = {
        # 基础程度词
        "大": 1.5, "暴": 1.8, "狂": 1.8, "疯": 1.8, "猛": 1.5,
        "巨": 1.6, "超": 1.5, "极": 1.8, "特": 1.6, "十分": 1.5,
        "非常": 1.5, "特别": 1.5, "极其": 1.8, "异常": 1.6,
        "大幅": 1.5, "显著": 1.4, "明显": 1.3, "明显": 1.3,

        # 财经场景程度词
        "翻倍": 2.0, "腰斩": 1.8, "历史新高": 1.7, "多年新低": 1.7,
        "创纪录": 1.6, "前所未有": 1.8, "罕见": 1.5, "空前": 1.7,
        "直线": 1.5, "直线上升": 1.6, "直线下降": 1.6,
        "连续": 1.3, "持续": 1.2, "不断": 1.2, "反复": 1.2,
        "加速": 1.4, "急剧": 1.6, "骤": 1.6, "骤然": 1.6,
        "断崖式": 1.8, "断崖": 1.8, "雪崩式": 1.9, "雪崩": 1.9,
        "井喷式": 1.7, "井喷": 1.7, "爆发式": 1.6, "爆发": 1.6,
        "暴跌": 1.8, "暴涨": 1.8, "飙升": 1.6, "狂飙": 1.7,
        " plummeting": 1.8, "skyrocketing": 1.8,

        # 复合程度词
        "大幅上涨": 1.6, "大幅下跌": 1.6,
        "快速上涨": 1.4, "快速下跌": 1.4,
        "稳步上涨": 1.2, "稳步下跌": 1.2,
        "强势上涨": 1.5, "弱势下跌": 1.5,
    }

    # ==================== 否定词 ====================
    NEGATION_WORDS = {
        "不", "没有", "未", "未能", "无法", "不能", "不可",
        "难", "难以", "并未", "尚未", "不再", "不用", "不必",
        "非", "勿", "毋", "莫", "无", "没", "别", "休",
        "不至于", "谈不上", "说不上", "谈不上好",
    }

    # ==================== 转折词 ====================
    TRANSITION_WORDS = {
        "但是": 0.7, "但": 0.7, "然而": 0.7, "却": 0.6, "反而": 0.8,
        "相反": 0.8, "不过": 0.5, "只是": 0.5, "可是": 0.7,
        "尽管": 0.6, "虽然": 0.6, "即使": 0.5, "即便": 0.5,
    }


class FinancialSentimentAnalyzer:
    """金融情感分析器

    结合金融词典规则和 SnowNLP，对金融文本进行更准确的情感判断。

    特性：
    - 专业金融情感词典（正面/负面/程度副词）
    - 否定词处理
    - 程度副词权重
    - 成语识别
    - 转折词处理
    - SnowNLP 补充
    """

    def __init__(self, dictionary: Optional[FinancialSentimentDictionary] = None):
        """初始化分析器

        Args:
            dictionary: 金融情感词典，如果为 None 则使用默认词典
        """
        self.dictionary = dictionary or FinancialSentimentDictionary()

        # 预编译正则
        self.stock_pattern = re.compile(
            r'(?:\$([A-Z]{1,5})\b)|'  # $AAPL 格式
            r'(?:([0-9]{6})(?:\.SH|\.SZ)?)|'  # A 股代码
            r'(?:股票 [代码]?:\s*([0-9]{6}))|'
            r'(?:([A-Z]{2,4})(?:股份 | 集团 | 公司))'
        )

    def analyze(self, text: str) -> FinancialSentimentResult:
        """分析金融文本情感

        Args:
            text: 待分析的文本

        Returns:
            FinancialSentimentResult: 情感分析结果
        """
        if not text or not text.strip():
            return FinancialSentimentResult(
                score=0.0,
                label="neutral",
                confidence=0.5,
                keywords=[],
                matched_words={"positive": [], "negative": [], "intensifiers": []},
                entities={"stocks": [], "companies": [], "people": []}
            )

        # 分词
        words = list(jieba.cut(text))

        # 统计情感词
        pos_count = 0.0
        neg_count = 0.0
        matched_positive = []
        matched_negative = []
        matched_intensifiers = []

        # 1. 先检查成语（直接文本匹配，权重更高）
        for idiom in self.dictionary.POSITIVE_WORDS:
            if idiom in ["蒸蒸日上", "如日中天", "欣欣向荣", "蓬勃发展", "方兴未艾",
                         "势如破竹", "扶摇直上", "步步高升", "财源广进", "日进斗金",
                         "锦上添花"]:
                if idiom in text:
                    pos_count += 2.0
                    matched_positive.append(idiom)

        for idiom in self.dictionary.NEGATIVE_WORDS:
            if idiom in ["一落千丈", "江河日下", "日薄西山", "每况愈下", "岌岌可危",
                         "风雨飘摇", "四面楚歌", "难以为继", "入不敷出", "雪上加霜",
                         "火上浇油", "祸不单行"]:
                if idiom in text:
                    neg_count += 2.0
                    matched_negative.append(idiom)

        # 2. 检查普通情感词（带否定词和程度副词处理）
        for i, word in enumerate(words):
            # 检查程度副词
            if word in self.dictionary.INTENSIFIERS:
                matched_intensifiers.append(word)

            # 检查否定词
            has_negation = False
            negation_distance = 3  # 否定词窗口大小
            for j in range(max(0, i - negation_distance), i):
                if words[j] in self.dictionary.NEGATION_WORDS:
                    has_negation = True
                    break

            # 获取程度副词权重
            intensifier_weight = 1.0
            prev_word = words[i - 1] if i > 0 else ""
            if prev_word in self.dictionary.INTENSIFIERS:
                intensifier_weight = self.dictionary.INTENSIFIERS[prev_word]

            # 匹配正面词
            if word in self.dictionary.POSITIVE_WORDS:
                weight = 1.0 * intensifier_weight
                if has_negation:
                    neg_count += weight
                else:
                    pos_count += weight
                if word not in matched_positive:
                    matched_positive.append(word)

            # 匹配负面词
            elif word in self.dictionary.NEGATIVE_WORDS:
                weight = 1.0 * intensifier_weight
                if has_negation:
                    pos_count += weight
                else:
                    neg_count += weight
                if word not in matched_negative:
                    matched_negative.append(word)

        # 3. 检查复合词（如"大幅上涨"）
        for compound_word, weight in self.dictionary.INTENSIFIERS.items():
            if compound_word in text:
                matched_intensifiers.append(compound_word)
                # 检查复合词后面是否跟着情感词
                for pos_word in self.dictionary.POSITIVE_WORDS:
                    if compound_word.endswith(pos_word) or pos_word in compound_word:
                        pos_count += weight
                        if pos_word not in matched_positive:
                            matched_positive.append(pos_word)
                        break
                for neg_word in self.dictionary.NEGATIVE_WORDS:
                    if compound_word.endswith(neg_word) or neg_word in compound_word:
                        neg_count += weight
                        if neg_word not in matched_negative:
                            matched_negative.append(neg_word)
                        break

        # 4. 计算基础分数
        total = pos_count + neg_count
        if total == 0:
            score = 0.0
        else:
            score = (pos_count - neg_count) / total

        # 5. SnowNLP 补充
        try:
            snownlp_score = SnowNLP(text).sentiments
            # 综合分数 - 规则和 SnowNLP 加权
            # 规则权重更高，因为金融词典更专业
            score = score * 0.7 + (snownlp_score - 0.5) * 0.3
        except Exception as e:
            logger.debug(f"SnowNLP 分析失败：{e}")

        # 6. 转折词处理
        transition_offset = 0.0
        for transition, weight in self.dictionary.TRANSITION_WORDS.items():
            if transition in text:
                # 转折词后的内容更重要
                idx = text.find(transition)
                if idx != -1:
                    after_text = text[idx + len(transition):]
                    if len(after_text) > 0:
                        try:
                            after_score = SnowNLP(after_text).sentiments
                            # 转折后内容权重更高
                            score = score * 0.4 + (after_score - 0.5) * 0.6
                        except:
                            pass
                        break

        # 7. 提取关键词
        keywords = self._extract_keywords(text, words)

        # 8. 实体识别
        entities = self._extract_entities(text)

        # 9. 确定标签
        if score >= 0.2:
            label = "positive"
        elif score <= -0.2:
            label = "negative"
        else:
            label = "neutral"

        # 10. 计算置信度
        confidence = min(abs(score) + 0.5, 1.0)
        # 匹配词越多，置信度越高
        word_bonus = min((len(matched_positive) + len(matched_negative)) * 0.05, 0.2)
        confidence = min(confidence + word_bonus, 1.0)

        return FinancialSentimentResult(
            score=max(-1.0, min(1.0, score)),
            label=label,
            confidence=confidence,
            keywords=keywords,
            matched_words={
                "positive": matched_positive,
                "negative": matched_negative,
                "intensifiers": matched_intensifiers
            },
            entities=entities
        )

    def _extract_keywords(self, text: str, words: List[str]) -> List[str]:
        """提取关键词"""
        try:
            keywords = jieba.analyse.extract_tags(text, topK=10)
            return list(keywords)
        except Exception as e:
            logger.debug(f"关键词提取失败：{e}")
            # 备用方案：使用分词结果
            return [w for w in words if len(w) >= 2 and w not in
                    {"的", "了", "是", "在", "有", "和", "与", "或", "等", "而", "及"}][:10]

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """提取实体（股票、公司、人物）"""
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
        for match in re.finditer(r'(\d{6})(?:\.(SH|SZ))?', text):
            code = match.group(1)
            suffix = match.group(2) or ('SH' if code.startswith('6') else 'SZ')
            entities["stocks"].append(f"{code}.{suffix}")

        # 港股代码
        for match in re.finditer(r'(?:HK)?(\d{5})\.(?:HK|COM)', text, re.IGNORECASE):
            entities["stocks"].append(f"HK{match.group(1)}")

        # 提取公司名
        company_patterns = [
            r'([^\s,.;:,.]{2,20}公司)',
            r'([^\s,.;:,.]{2,20}集团)',
            r'([^\s,.;:,.]{2,20}股份)',
            r'([^\s,.;:,.]{2,20}科技)',
            r'([^\s,.;:,.]{2,20}有限)',
        ]
        for pattern in company_patterns:
            for match in re.finditer(pattern, text):
                company = match.group(1)
                if company and len(company) >= 4 and company not in entities["companies"]:
                    entities["companies"].append(company)

        # 提取人名（简单模式）
        person_patterns = [
            r'([张王李赵钱孙周吴郑冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华][^\s,.]{1,3})(?:表示 | 称 | 说 | 指出 | 认为 | 透露)',
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


class FinancialHybridAnalyzer:
    """混合金融情感分析器

    结合词典规则、SnowNLP 和实体识别，提供更全面的金融情感分析。
    """

    def __init__(self):
        self.financial_analyzer = FinancialSentimentAnalyzer()

    def analyze(self, text: str) -> FinancialSentimentResult:
        """分析文本情感

        Args:
            text: 待分析的文本

        Returns:
            FinancialSentimentResult: 情感分析结果
        """
        return self.financial_analyzer.analyze(text)

    def batch_analyze(self, texts: List[str]) -> List[FinancialSentimentResult]:
        """批量分析"""
        return [self.financial_analyzer.analyze(text) for text in texts]

    def compare(self, text1: str, text2: str) -> Dict:
        """比较两段文本的情感

        Args:
            text1: 第一段文本
            text2: 第二段文本

        Returns:
            包含比较结果的字典
        """
        result1 = self.analyze(text1)
        result2 = self.analyze(text2)

        return {
            "text1": {
                "score": result1.score,
                "label": result1.label,
                "confidence": result1.confidence
            },
            "text2": {
                "score": result2.score,
                "label": result2.label,
                "confidence": result2.confidence
            },
            "comparison": {
                "score_diff": result1.score - result2.score,
                "more_positive": "text1" if result1.score > result2.score else "text2"
            }
        }


# 便捷函数
def analyze_financial_sentiment(text: str) -> FinancialSentimentResult:
    """便捷函数：分析金融文本情感"""
    analyzer = FinancialSentimentAnalyzer()
    return analyzer.analyze(text)


def get_sentiment_label(score: float) -> str:
    """根据分数返回情感标签"""
    if score >= 0.2:
        return "positive"
    elif score <= -0.2:
        return "negative"
    else:
        return "neutral"
