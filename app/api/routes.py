import json
import logging
import os
from typing import Optional
import redis
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from ..config import config
from app.models.sales_forecast import SalesForecaster
from app.models.stock_predictor import StockPredictor
from app.risk.risk_manager import RiskManager
from app.strategies.trading_engine import TradingEngine
from app.data.data_fusion import DataFusion
from app.mlops.mlops_pipeline import MLOpsPipeline
logger = logging.getLogger(__name__)
try:
    redis_client = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, password=config.REDIS_PASSWORD, db=config.REDIS_DB, decode_responses=True)
    redis_client.ping()
    logger.info('✅ Redis connected successfully')
    REDIS_AVAILABLE = True
except Exception as e:
    logger.warning(f'Redis unavailable — caching disabled: {e}')
    REDIS_AVAILABLE = False
    redis_client = None
app = FastAPI(title='Investment & Sales Prediction API', version='2.0.0')
try:
    sales_predictor = SalesForecaster()
    logger.info('✅ Sales model initialized')
except Exception as e:
    logger.warning(f'Sales model not available: {e}')
    sales_predictor = None
try:
    stock_predictor = StockPredictor()
    logger.info('✅ Stock model initialized')
except Exception as e:
    logger.warning(f'Stock model not available: {e}')
    stock_predictor = None

def _load_model(name: str, fallback_prefix: str, model_dir):
    from pathlib import Path
    try:
        return mlops.load_model(name)
    except Exception:
        candidates = sorted(Path(model_dir).glob(f'{fallback_prefix}_*.pkl'))
        if candidates:
            import pickle
            with open(candidates[-1], 'rb') as f:
                logger.warning(f'Loaded fallback model: {candidates[-1].name}')
                return pickle.load(f)
        logger.warning(f'{name}.pkl not found — train first')
        return None
_loaded_sales = _load_model('sales_ensemble', 'sales', config.MODEL_DIR)
_loaded_stock = _load_model('stock_ensemble', 'stock', config.MODEL_DIR)
if _loaded_sales is not None:
    sales_predictor = _loaded_sales
if _loaded_stock is not None:
    stock_predictor = _loaded_stock

def _cache_get(key: str):
    if redis_client is None:
        return None
    try:
        val = redis_client.get(key)
        return json.loads(val) if val else None
    except Exception:
        return None

def _cache_set(key: str, value, ttl: int=3600):
    if redis_client is None:
        return
    try:
        redis_client.setex(key, ttl, json.dumps(value, default=str))
    except Exception:
        pass

class StockPredictionRequest(BaseModel):
    symbol: str = 'AAPL'
    horizon: int = 30
    period: str = '2y'

class SalesPredictionRequest(BaseModel):
    symbol: str = 'NVDA'
    periods: int = 90
    period: str = '2y'

class BacktestRequest(BaseModel):
    symbol: str = 'AAPL'
    strategy: str = 'ensemble'
    period: str = '2y'
    initial_capital: float = 10000

class RiskRequest(BaseModel):
    symbol: str = 'AAPL'
    period: str = '1y'
    confidence: float = 0.95

class TrainRequest(BaseModel):
    symbol: str
    model_type: str = 'stock'
    period: str = '2y'

class DriftRequest(BaseModel):
    symbol: str
    reference_period: str = '1y'
    current_days: int = 30

@app.get('/health')
async def health_check():
    """Health check with Redis status"""
    redis_status = False
    try:
        redis_client.ping()
        redis_status = True
    except:
        pass
    return {'status': 'healthy', 'version': '2.0.0', 'redis': redis_status, 'sales_model_ready': sales_predictor is not None, 'stock_model_ready': stock_predictor is not None}

@app.post('/predict/stock')
async def predict_stock(request: StockPredictionRequest):
    """Stock prediction with Redis caching"""
    try:
        cache_key = f'invest:stock:{request.symbol}:{request.horizon}'
        cached = _cache_get(cache_key)
        if cached:
            logger.info(f'✅ Cache hit for {request.symbol}')
            return json.loads(cached)
        logger.info(f'🔄 Computing prediction for {request.symbol}...')
        if stock_predictor is None:
            return {'status': 'error', 'message': 'Stock predictor not available'}
        result = stock_predictor.predict_future(request.horizon)
        _cache_set(cache_key, result, ttl=3600)
        logger.info(f'✅ Cached prediction for {request.symbol}')
        return result
    except Exception as e:
        logger.error(f'Prediction error: {e}')
        if stock_predictor is not None:
            return stock_predictor.predict_future(request.horizon)
        return {'status': 'error', 'message': str(e)}

@app.post('/predict/sales')
async def predict_sales(request: SalesPredictionRequest):
    """Sales forecast with Redis caching"""
    try:
        cache_key = f'invest:sales:{request.symbol}:{request.periods}'
        cached = _cache_get(cache_key)
        if cached:
            logger.info(f'✅ Cache hit for {request.symbol} sales')
            return json.loads(cached)
        logger.info(f'🔄 Computing sales forecast for {request.symbol}...')
        if sales_predictor is None:
            return {'status': 'error', 'message': 'Sales predictor not available'}
        result = sales_predictor.predict_future(request.periods)
        _cache_set(cache_key, result, ttl=3600)
        logger.info(f'✅ Cached sales forecast for {request.symbol}')
        return result
    except Exception as e:
        logger.error(f'Sales prediction error: {e}')
        if sales_predictor is not None:
            return sales_predictor.predict_future(request.periods)
        return {'status': 'error', 'message': str(e)}

@app.post('/backtest')
async def backtest_strategy(request: BacktestRequest):
    """Backtest a trading strategy"""
    try:
        if trading_engine is None:
            return {'status': 'error', 'message': 'Trading engine not available'}
        result = trading_engine.backtest(symbol=request.symbol, strategy=request.strategy, period=request.period, initial_capital=request.initial_capital)
        return {'status': 'success', 'result': result}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

@app.post('/risk/analysis')
async def risk_analysis(request: RiskRequest):
    """Calculate VaR and risk metrics"""
    try:
        if risk_manager is None:
            return {'status': 'error', 'message': 'Risk manager not available'}
        result = risk_manager.analyze(symbol=request.symbol, period=request.period, confidence=request.confidence)
        return {'status': 'success', 'result': result}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

@app.get('/risk/kelly/{symbol}')
async def get_kelly(symbol: str, period: str='2y'):
    """Calculate Kelly Criterion position sizing"""
    try:
        if risk_manager is None:
            return {'status': 'error', 'message': 'Risk manager not available'}
        result = risk_manager.kelly_criterion(symbol=symbol, period=period)
        return {'status': 'success', 'result': result}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

@app.get('/data/enriched/{symbol}')
async def get_enriched_data(symbol: str, period: str='2y'):
    """Get enriched data from multiple sources"""
    try:
        if data_fusion is None:
            return {'status': 'error', 'message': 'Data fusion not available'}
        result = data_fusion.get_enriched_data(symbol=symbol, period=period)
        return {'status': 'success', 'result': result}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

@app.post('/mlops/drift')
async def detect_drift(request: DriftRequest):
    """Detect data drift in predictions"""
    try:
        if mlops is None:
            return {'status': 'error', 'message': 'MLOps pipeline not available'}
        result = mlops.detect_drift(symbol=request.symbol, reference_period=request.reference_period, current_days=request.current_days)
        return {'status': 'success', 'result': result}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

@app.post('/mlops/retrain')
async def trigger_retrain(request: TrainRequest):
    """Trigger model retraining"""
    try:
        if mlops is None:
            return {'status': 'error', 'message': 'MLOps pipeline not available'}
        result = mlops.retrain_model(symbol=request.symbol, model_type=request.model_type, period=request.period)
        return {'status': 'success', 'result': result}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

@app.get('/metrics/performance')
async def get_performance_metrics():
    """Get model performance metrics"""
    try:
        metrics = {'sales_model': sales_predictor.performance if sales_predictor else None, 'stock_model': stock_predictor.performance if stock_predictor else None, 'redis_available': REDIS_AVAILABLE}
        return {'status': 'success', 'metrics': metrics}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

@app.get('/market/summary/{symbol}')
async def get_market_summary(symbol: str):
    """Get live market summary for a symbol"""
    try:
        if data_fusion is None:
            return {'status': 'error', 'message': 'Data fusion not available'}
        result = data_fusion.get_market_summary(symbol=symbol)
        return {'status': 'success', 'result': result}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}