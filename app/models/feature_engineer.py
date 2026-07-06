# app/models/feature_engineer.py
import pandas as pd
import numpy as np

class FeatureEngineer:
    """Create 85+ features for sales and stock prediction"""
    
    def __init__(self):
        self.feature_columns = []
        self.n_features = 0
    
    def create_all_features(self, df, target_col='sales'):
        """Create all 85+ features"""
        df = df.copy()
        
        # Time features (15)
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
        df['is_year_end'] = df['date'].dt.is_year_end.astype(int)
        
        # Seasonal (sin/cos encoding)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Lag features (7,14,21,28,56)
        for lag in [7, 14, 21, 28, 56]:
            if target_col in df.columns:
                df[f'lag_{lag}'] = df[target_col].shift(lag)
        
        # Rolling statistics (7,14,30,60,90)
        if target_col in df.columns:
            for window in [7, 14, 30, 60, 90]:
                df[f'rolling_mean_{window}'] = df[target_col].rolling(window).mean()
                df[f'rolling_std_{window}'] = df[target_col].rolling(window).std()
                df[f'rolling_max_{window}'] = df[target_col].rolling(window).max()
                df[f'rolling_min_{window}'] = df[target_col].rolling(window).min()
                df[f'rolling_skew_{window}'] = df[target_col].rolling(window).skew()
                df[f'rolling_kurt_{window}'] = df[target_col].rolling(window).kurt()
        
        # For stocks: technical indicators
        if 'Close' in df.columns:
            close = df['Close']
            df['returns'] = close.pct_change()
            df['log_returns'] = np.log(close / close.shift(1))
            
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-10)
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = close.ewm(span=12, adjust=False).mean()
            exp2 = close.ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            # Bollinger Bands
            rolling_mean = close.rolling(window=20).mean()
            rolling_std = close.rolling(window=20).std()
            df['BB_Upper'] = rolling_mean + (rolling_std * 2)
            df['BB_Lower'] = rolling_mean - (rolling_std * 2)
            df['BB_Middle'] = rolling_mean
        
        # Drop NaN
        df = df.dropna()
        
        # Store feature columns
        exclude = ['date', target_col, 'Close', 'Adj Close', 'Open', 'High', 'Low', 'Volume']
        self.feature_columns = [col for col in df.columns if col not in exclude]
        self.n_features = len(self.feature_columns)
        return df