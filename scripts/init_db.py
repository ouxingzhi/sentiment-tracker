"""初始化脚本"""
import asyncio
from src.database import init_db
from loguru import logger


async def main():
    """初始化数据库"""
    logger.info("初始化数据库...")
    await init_db()
    logger.info("数据库初始化完成！")


if __name__ == "__main__":
    asyncio.run(main())