# app/models/sales_forecast.py
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import joblib
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SalesForecaster:
    """Sales forecasting with Prophet"""
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.performance = {}
    
    def fit(self, df):
        """Train Prophet model"""
        logger.info("🚀 Training sales forecast model...")
        
        # Prepare data for Prophet
        prophet_df = df[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
        
        # Train Prophet
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            seasonality_mode='multiplicative'
        )
        self.model.add_country_holidays(country_name='US')
        self.model.fit(prophet_df)
        
        # Evaluate on held-out last 20% (chronological)
        split = int(len(df) * 0.8)
        test_df = df.iloc[split:]
        future = self.model.make_future_dataframe(periods=len(test_df))
        forecast = self.model.predict(future)
        pred = forecast['yhat'].values[-len(test_df):]
        actual = test_df['sales'].values

        mape = mean_absolute_percentage_error(actual, pred) * 100
        r2 = r2_score(actual, pred)

        self.performance = {'mape': mape, 'r2': r2, 'accuracy': max(0, 100 - mape)}
        self.is_trained = True

        logger.info(f"✅ Out-of-sample MAPE: {mape:.2f}%, R²: {r2:.4f}")
        
        # Save model
        joblib.dump(self, 'models/sales_ensemble.pkl')
        return self
    
    def predict_future(self, periods=90):
        """Predict future sales"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        last_periods = forecast.tail(periods)
        
        return pd.DataFrame({
            'date': last_periods['ds'],
            'yhat': last_periods['yhat'],
            'yhat_lower': last_periods['yhat_lower'],
            'yhat_upper': last_periods['yhat_upper']
        })
    
    @classmethod
    def load_model(cls):
        """Load saved model"""
        try:
            return joblib.load('models/sales_ensemble.pkl')
        except FileNotFoundError:
            logger.warning("Model not found. Train first with: python train.py")
            return None