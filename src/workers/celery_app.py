"""启动Celery Worker"""
from src.workers.tasks import celery_app

if __name__ == "__main__":
    celery_app.start()