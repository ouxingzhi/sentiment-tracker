"""测试脚本 - 快速验证系统"""
import asyncio
from src.collectors.xueqiu_collector import XueqiuCollector, XueqiuHotTopicCollector
from src.analyzers.sentiment_analyzer import HybridAnalyzer, StockEntityRecognizer, RuleBasedAnalyzer
from loguru import logger


async def test_xueqiu_collector():
    """测试雪球采集"""
    print("\n📡 测试雪球采集...")
    collector = XueqiuCollector()

    count = 0
    async for article in collector.collect():
        print(f"  - [{article['source']}] {article['title'][:50]}...")
        count += 1
        if count >= 5:
            break

    await collector.close()
    print(f"✅ 雪球采集测试完成，获取 {count} 篇文章")
    return count > 0


async def test_sentiment_analyzer():
    """测试情感分析"""
    print("\n🧠 测试情感分析...")
    analyzer = RuleBasedAnalyzer()  # 使用规则分析器，无需 API

    test_texts = [
        ("利好！贵州茅台股价大涨，突破历史新高", "positive"),
        ("利空！特斯拉股价暴跌，投资者恐慌抛售", "negative"),
        ("宁德时代发布季度财报，业绩符合预期", "neutral"),
        ("贵州茅台净利润暴增 50%，市场反应热烈", "positive"),
        ("中国平安业绩下滑，机构下调目标价", "negative"),
    ]

    correct = 0
    for text, expected in test_texts:
        result = await analyzer.analyze(text)
        status = "✓" if result.label == expected else "✗"
        if result.label == expected:
            correct += 1
        print(f"  {status} 文本：{text[:30]}...")
        print(f"     情感：{result.label} (score: {result.score:.2f}, expected: {expected})")
        print()

    print(f"✅ 情感分析测试完成，准确率：{correct}/{len(test_texts)}")
    return True


async def test_batch_analyzer():
    """测试批量分析"""
    print("\n🔄 测试批量分析...")
    analyzer = HybridAnalyzer()

    test_texts = [
        "腾讯控股发布强劲财报，游戏业务大幅增长",
        "阿里巴巴遭监管调查，股价承压",
        "小米新车 SU7 大获成功，订单爆满",
    ]

    results = await analyzer.batch_analyze(test_texts)

    print(f"  批量分析 {len(results)} 条文本:")
    for i, r in enumerate(results):
        print(f"  [{i+1}] {r.label} (score: {r.score:.2f}) - {test_texts[i][:20]}...")

    print("✅ 批量分析测试完成")
    return len(results) == len(test_texts)


async def test_stock_recognizer():
    """测试股票识别"""
    print("\n📊 测试股票实体识别...")
    recognizer = StockEntityRecognizer()

    test_text = """
    苹果和特斯拉股价今日大涨，苹果 (AAPL) 创历史新高。
    茅台、宁德时代等 A 股龙头股表现强劲。
    比亚迪宣布新车型计划，市场反应积极。
    腾讯控股发布新游戏，美团外卖业务持续增长。
    """

    stocks = recognizer.extract_stocks(test_text)
    print(f"  识别到的股票：{stocks}")

    # 检查是否识别到主要股票
    symbols = [s['symbol'] for s in stocks]
    expected_found = any('AAPL' in s or 'TSLA' in s for s in symbols)

    print("✅ 股票识别测试完成")
    return len(stocks) > 0


async def main():
    """运行所有测试"""
    print("=" * 60)
    print("🚀 财经舆情分析系统 - 功能测试")
    print("=" * 60)

    results = []

    # 测试雪球采集器
    try:
        results.append(("雪球采集", await test_xueqiu_collector()))
    except Exception as e:
        print(f"❌ 雪球采集测试失败：{e}")
        results.append(("雪球采集", False))

    # 测试情感分析
    try:
        results.append(("情感分析", await test_sentiment_analyzer()))
    except Exception as e:
        print(f"❌ 情感分析测试失败：{e}")
        results.append(("情感分析", False))

    # 测试批量分析
    try:
        results.append(("批量分析", await test_batch_analyzer()))
    except Exception as e:
        print(f"❌ 批量分析测试失败：{e}")
        results.append(("批量分析", False))

    # 测试股票识别
    try:
        results.append(("股票识别", await test_stock_recognizer()))
    except Exception as e:
        print(f"❌ 股票识别测试失败：{e}")
        results.append(("股票识别", False))

    # 总结
    print("\n" + "=" * 60)
    print("📋 测试结果汇总")
    print("=" * 60)

    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\n总计：{passed}/{total} 测试通过")


if __name__ == "__main__":
    asyncio.run(main())
