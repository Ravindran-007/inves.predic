# app/api/routes.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis
import json
import logging
from typing import Optional
from ..config import config
from ..models.sales_forecast import SalesForecaster
from ..models.stock_predictor import StockPredictor

logger = logging.getLogger(__name__)

app = FastAPI(title="Investment & Sales Prediction API", version="1.0.0")
redis_client = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, db=config.REDIS_DB)

# Models
sales_model = SalesForecaster()
stock_model = StockPredictor()

# Try loading trained models
try:
    sales_model = SalesForecaster.load_model()
    logger.info("✅ Sales model loaded")
except:
    logger.warning("⚠️ Sales model not found. Train first!")

# Schemas
class SalesPredictionRequest(BaseModel):
    periods: int = 90

class StockPredictionRequest(BaseModel):
    symbol: str
    horizon: int = 7

class PredictionResponse(BaseModel):
    status: str
    data: dict
    confidence_interval: Optional[dict]

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

@app.post("/predict/sales", response_model=PredictionResponse)
async def predict_sales(request: SalesPredictionRequest):
    """Predict sales for future periods"""
    try:
        # Check cache
        cache_key = f"sales_prediction:{request.periods}"
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Generate prediction
        if not sales_model.is_trained:
            raise HTTPException(status_code=400, detail="Model not trained")
        
        prediction = sales_model.predict_future(periods=request.periods)
        response = {
            "status": "success",
            "data": prediction.to_dict(orient="records"),
            "confidence_interval": {
                "lower": prediction['yhat_lower'].tolist(),
                "upper": prediction['yhat_upper'].tolist()
            }
        }
        
        # Cache
        redis_client.setex(cache_key, config.CACHE_TTL, json.dumps(response))
        return response
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/stock", response_model=PredictionResponse)
async def predict_stock(request: StockPredictionRequest):
    """Predict stock price for given symbol"""
    try:
        cache_key = f"stock_prediction:{request.symbol}:{request.horizon}"
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
        
        prediction = stock_model.predict_next_day(request.symbol)
        response = {
            "status": "success",
            "data": prediction
        }
        
        redis_client.setex(cache_key, config.CACHE_TTL, json.dumps(response))
        return response
    
    except Exception as e:
        logger.error(f"Stock prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/performance")
async def get_performance():
    """Get model performance metrics"""
    return {
        "sales": sales_model.performance,
        "stock": stock_model.performance if hasattr(stock_model, 'performance') else {}
    }