"""启动脚本"""
#!/bin/bash

# 启动舆情分析系统

echo "🚀 启动财经舆情分析系统..."

# 检查.env文件
if [ ! -f .env ]; then
    echo "⚠️  .env文件不存在，从模板复制..."
    cp .env.example .env
    echo "📝 请编辑 .env 文件配置API密钥"
    exit 1
fi

# 创建必要的目录
mkdir -p logs

# 启动服务
echo "🐳 启动Docker服务..."
docker-compose up -d

echo "✅ 服务已启动！"
echo ""
echo "📋 可用接口:"
echo "  - API: http://localhost:8000"
echo "  - API文档: http://localhost:8000/docs"
echo "  - PostgreSQL: localhost:5432"
echo "  - Redis: localhost:6379"
echo ""
echo "📊 查看日志:"
echo "  docker-compose logs -f api"
echo ""
echo "🛑 停止服务:"
echo "  docker-compose down"