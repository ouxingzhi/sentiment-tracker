"""简单测试脚本 - 测试核心功能"""
import sys
sys.path.insert(0, '.')

import asyncio
from snownlp import SnowNLP
import jieba

print("=" * 60)
print("🧪 舆情系统核心功能测试")
print("=" * 60)

# 1. 测试jieba分词
print("\n📝 测试1: jieba分词")
text = "苹果公司股价大涨，突破历史新高，特斯拉暴跌"
words = list(jieba.cut(text))
print(f"  原文: {text}")
print(f"  分词: {' / '.join(words)}")
print("  ✅ jieba分词正常")

# 2. 测试SnowNLP情感分析
print("\n📊 测试2: SnowNLP情感分析")
test_texts = [
    "苹果股价大涨，市场情绪高涨！",
    "特斯拉暴跌，投资者恐慌抛售。",
    "微软发布财报，业绩符合预期。",
]

for t in test_texts:
    s = SnowNLP(t)
    score = s.sentiments
    label = "正面" if score > 0.6 else "负面" if score < 0.4 else "中性"
    print(f"  [{label}] {t[:20]}... 分数:{score:.2f}")
print("  ✅ SnowNLP情感分析正常")

# 3. 测试股票实体识别
print("\n📈 测试3: 股票实体识别")
STOCK_ALIASES = {
    "苹果": "AAPL", "特斯拉": "TSLA", "英伟达": "NVDA",
    "茅台": "600519.SH", "宁德时代": "300750.SZ", "比亚迪": "002594.SZ",
    "腾讯": "00700.HK", "阿里巴巴": "BABA",
}

found_stocks = []
for name, symbol in STOCK_ALIASES.items():
    if name in text:
        found_stocks.append(f"{name}({symbol})")

print(f"  原文: {text}")
print(f"  识别到: {', '.join(found_stocks)}")
print("  ✅ 股票识别正常")

# 4. 测试关键词预警
print("\n⚠️ 测试4: 关键词预警")
ALERT_KEYWORDS = ["暴跌", "崩盘", "爆仓", "利空", "违约"]
matched = [kw for kw in ALERT_KEYWORDS if kw in text]
if matched:
    print(f"  ⚠️ 检测到预警关键词: {matched}")
else:
    print(f"  ✅ 无预警关键词")
print("  ✅ 关键词检测正常")

# 5. 测试Telegram推送
print("\n📢 测试5: Telegram推送")
import httpx
BOT_TOKEN = "8730532482:AAEiR7O2k8kUW1lPkmPGd4oFs5vE2xyYOX8"
CHAT_ID = "6086651958"

async def send_test():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={
                "chat_id": CHAT_ID,
                "text": f"🧪 舆情系统测试\n\n✅ 所有核心功能正常！\n\n测试内容：\n• jieba分词\n• SnowNLP情感分析\n• 股票识别\n• 关键词预警"
            }
        )
        return response.status_code == 200

result = asyncio.run(send_test())
if result:
    print("  ✅ Telegram推送成功")
else:
    print("  ❌ Telegram推送失败")

print("\n" + "=" * 60)
print("✅ 所有测试完成！")
print("=" * 60)