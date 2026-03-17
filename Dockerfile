FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 创建非root用户
RUN useradd -m sentiment && chown -R sentiment:sentiment /app
USER sentiment

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]