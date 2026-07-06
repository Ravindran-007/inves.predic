# app/config.py
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class Config:
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    MODEL_DIR = BASE_DIR / "models"
    DATA_DIR = BASE_DIR / "data"
    LOG_DIR = BASE_DIR / "logs"
    
    # API Keys
    ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")
    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
    FRED_API_KEY = os.getenv("FRED_API_KEY", "")
    
    # Redis
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB = int(os.getenv("REDIS_DB", 0))
    
    # Model parameters
    SALES_MODEL_PATH = MODEL_DIR / "sales_ensemble.pkl"
    STOCK_MODEL_PATH = MODEL_DIR / "stock_ensemble.pkl"
    
    # Performance
    CACHE_TTL = 3600  # 1 hour
    MAX_CONCURRENT = 1000
    PREDICTION_TIMEOUT = 0.2  # 200ms
    
    @classmethod
    def ensure_dirs(cls):
        for dir_path in [cls.MODEL_DIR, cls.DATA_DIR, cls.LOG_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)

config = Config()
config.ensure_dirs()