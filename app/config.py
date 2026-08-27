import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

class Config:
    BASE_DIR = Path(__file__).parent.parent
    MODEL_DIR = BASE_DIR / 'models'
    DATA_DIR = BASE_DIR / 'data'
    LOG_DIR = BASE_DIR / 'logs'
    ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY', '')
    NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
    FRED_API_KEY = os.getenv('FRED_API_KEY', '')
    TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN', '')
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('REDIS_DB', 0))
    REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', '')
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://admin:changeme@localhost:5432/predictions')
    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'mlruns')
    SALES_MODEL_PATH = MODEL_DIR / 'sales_ensemble.pkl'
    STOCK_MODEL_PATH = MODEL_DIR / 'stock_ensemble.pkl'
    CACHE_TTL = 3600
    MAX_CONCURRENT = 1000
    PREDICTION_TIMEOUT = 0.2

    @classmethod
    def ensure_dirs(cls):
        for d in [cls.MODEL_DIR, cls.DATA_DIR, cls.LOG_DIR]:
            d.mkdir(parents=True, exist_ok=True)
config = Config()
config.ensure_dirs()