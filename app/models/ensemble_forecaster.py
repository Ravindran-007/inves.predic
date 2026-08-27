import pandas as pd
import numpy as np
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import logging
import warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_time_features(df):
    dates = pd.to_datetime(df['date'])
    df['day'] = dates.dt.day
    df['day_of_week'] = dates.dt.dayofweek
    df['month'] = dates.dt.month
    df['quarter'] = dates.dt.quarter
    df['year'] = dates.dt.year
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
    return df

def add_price_features(df):
    close = df['sales'].copy()
    for p in [1, 3, 5, 10, 20]:
        df[f'ret_{p}d'] = close.pct_change(p)
    df['vol_5d'] = df['ret_1d'].rolling(5).std()
    df['vol_10d'] = df['ret_1d'].rolling(10).std()
    df['vol_20d'] = df['ret_1d'].rolling(20).std()
    return df

def add_lag_features(df):
    close = df['sales'].copy()
    for lag in [1, 3, 7, 14, 21, 28]:
        df[f'lag_{lag}d'] = close.shift(lag)
    return df

def add_rolling_stats(df):
    close = df['sales'].copy()
    for w in [5, 7, 14, 21, 30]:
        df[f'roll_mean_{w}d'] = close.rolling(w).mean()
        df[f'roll_std_{w}d'] = close.rolling(w).std()
        df[f'roll_max_{w}d'] = close.rolling(w).max()
        df[f'roll_min_{w}d'] = close.rolling(w).min()
    return df

def prepare_features(df):
    df = add_time_features(df)
    df = add_price_features(df)
    df = add_lag_features(df)
    df = add_rolling_stats(df)
    return df
FEATURE_COLS = ['day', 'day_of_week', 'month', 'quarter', 'year', 'day_sin', 'day_cos', 'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos', 'ret_1d', 'ret_3d', 'ret_5d', 'ret_10d', 'ret_20d', 'vol_5d', 'vol_10d', 'vol_20d', 'lag_1d', 'lag_3d', 'lag_7d', 'lag_14d', 'lag_21d', 'lag_28d', 'roll_mean_5d', 'roll_std_5d', 'roll_max_5d', 'roll_min_5d', 'roll_mean_7d', 'roll_std_7d', 'roll_max_7d', 'roll_min_7d', 'roll_mean_14d', 'roll_std_14d', 'roll_max_14d', 'roll_min_14d', 'roll_mean_21d', 'roll_std_21d', 'roll_max_21d', 'roll_min_21d', 'roll_mean_30d', 'roll_std_30d', 'roll_max_30d', 'roll_min_30d']

class EnsembleForecaster:

    def __init__(self, prophet_weight=0.3, xgb_weight=0.4, lstm_weight=0.3):
        self.prophet_weight = prophet_weight
        self.xgb_weight = xgb_weight
        self.lstm_weight = lstm_weight
        self.prophet_model = None
        self.xgb_model = None
        self.scaler = RobustScaler()
        self.is_trained = False
        self.performance = {}
        self.data = None
        self.symbol = None
        self.lstm_model = None
        self.lstm_scaler = StandardScaler()
        self.sequence_length = 60

    def _build_prophet(self, changepoint_prior=0.01, seasonality_prior=10.0, holidays_prior=10.0):
        return Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, seasonality_mode='multiplicative', changepoint_prior_scale=changepoint_prior, seasonality_prior_scale=seasonality_prior, holidays_prior_scale=holidays_prior, interval_width=0.95)

    def _build_xgb(self):
        return XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=6, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=0)

    def _build_lstm(self):
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.optimizers import Adam
            model = Sequential([LSTM(64, return_sequences=True, input_shape=(self.sequence_length, 1)), Dropout(0.2), LSTM(32, return_sequences=True), Dropout(0.2), LSTM(16, return_sequences=False), Dropout(0.2), Dense(1)])
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            return model
        except ImportError:
            logger.warning('TensorFlow not available, LSTM will be skipped')
            return None

    def _prepare_lstm_sequences(self, prices):
        prices = np.array(prices).reshape(-1, 1)
        scaled = self.lstm_scaler.fit_transform(prices)
        X, y = ([], [])
        for i in range(self.sequence_length, len(scaled)):
            X.append(scaled[i - self.sequence_length:i])
            y.append(scaled[i])
        return (np.array(X), np.array(y))

    def fit(self, df, symbol=None):
        logger.info(f'Training ensemble model for {symbol or 'unknown'}...')
        self.symbol = symbol
        self.data = df[['date', 'sales']].copy()
        if df['sales'].isna().any():
            df = df.dropna(subset=['sales'])
        if (df['sales'] <= 0).any():
            logger.warning('Negative prices detected, clipping to small positive value')
            df['sales'] = df['sales'].clip(lower=0.01)
        feat_df = prepare_features(df.copy())
        feat_df = feat_df.dropna()
        if len(feat_df) < 100:
            raise ValueError(f'Insufficient data: {len(feat_df)} samples')
        split = int(len(feat_df) * 0.8)
        train_df = feat_df.iloc[:split]
        test_df = feat_df.iloc[split:]
        prophet_df = df[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
        self.prophet_model = self._build_prophet()
        self.prophet_model.fit(prophet_df)
        X_train = self.scaler.fit_transform(train_df[FEATURE_COLS])
        X_test = self.scaler.transform(test_df[FEATURE_COLS])
        y_train = train_df['sales'].values
        y_test = test_df['sales'].values
        self.xgb_model = self._build_xgb()
        self.xgb_model.fit(X_train, y_train)
        if self.lstm_model is None:
            self.lstm_model = self._build_lstm()
        if self.lstm_model:
            X_seq, y_seq = self._prepare_lstm_sequences(df['sales'].values)
            if len(X_seq) > 0:
                split_seq = int(len(X_seq) * 0.8)
                self.lstm_model.fit(X_seq[:split_seq], y_seq[:split_seq], epochs=50, batch_size=32, validation_data=(X_seq[split_seq:], y_seq[split_seq:]), verbose=0)
        prophet_pred = self._predict_prophet(len(test_df))
        xgb_pred = self.xgb_model.predict(X_test)
        if self.lstm_model:
            lstm_pred = self._predict_lstm(len(test_df))
            ensemble_pred = self.prophet_weight * prophet_pred + self.xgb_weight * xgb_pred + self.lstm_weight * lstm_pred
        else:
            ensemble_pred = self.prophet_weight * prophet_pred + self.xgb_weight * xgb_pred
        mape = mean_absolute_percentage_error(y_test, ensemble_pred) * 100
        r2 = r2_score(y_test, ensemble_pred)
        pred_trend = np.diff(ensemble_pred) > 0
        actual_trend = np.diff(y_test) > 0
        direction_acc = np.mean(pred_trend == actual_trend) * 100
        returns = np.diff(ensemble_pred) / (ensemble_pred[:-1] + 1e-10)
        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
        self.performance = {'mape': float(mape), 'r2': float(r2), 'accuracy': float(max(0, 100 - mape)), 'direction_accuracy': float(direction_acc), 'sharpe_ratio': float(sharpe)}
        self.is_trained = True
        logger.info(f'Ensemble - MAPE: {mape:.2f}% | R2: {r2:.4f} | Direction: {direction_acc:.1f}% | Sharpe: {sharpe:.2f}')
        return self

    def _predict_prophet(self, periods):
        future = self.prophet_model.make_future_dataframe(periods=periods)
        forecast = self.prophet_model.predict(future)
        return forecast['yhat'].values[-periods:]

    def _predict_lstm(self, periods):
        if not self.lstm_model:
            return np.zeros(periods)
        prices = self.data['sales'].values[-self.sequence_length:]
        prices_scaled = self.lstm_scaler.transform(prices.reshape(-1, 1))
        predictions = []
        current_seq = prices_scaled.copy()
        for _ in range(periods):
            pred = self.lstm_model.predict(current_seq.reshape(1, self.sequence_length, 1), verbose=0)
            predictions.append(pred[0, 0])
            current_seq = np.vstack([current_seq[1:], pred])
        return self.lstm_scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    def predict_future(self, periods=90):
        if not self.is_trained:
            raise ValueError('Model not trained. Call fit() first.')
        prophet_pred = self._predict_prophet(periods)
        xgb_pred = self.xgb_model.predict(self.scaler.transform(prepare_features(self.data.copy()).iloc[-periods:][FEATURE_COLS]))
        if self.lstm_model:
            lstm_pred = self._predict_lstm(periods)
            ensemble_pred = self.prophet_weight * prophet_pred + self.xgb_weight * xgb_pred + self.lstm_weight * lstm_pred
        else:
            ensemble_pred = self.prophet_weight * prophet_pred + self.xgb_weight * xgb_pred
        pred_std = np.std([prophet_pred, xgb_pred]) * 1.96
        yhat_lower = ensemble_pred - pred_std
        yhat_upper = ensemble_pred + pred_std
        yhat_lower = np.minimum(yhat_lower, ensemble_pred)
        yhat_upper = np.maximum(yhat_upper, ensemble_pred)
        yhat_lower = np.maximum(yhat_lower, 0.01)
        yhat_upper = np.maximum(yhat_upper, 0.01)
        last_date = self.data['date'].iloc[-1]
        dates = []
        day_offset = 1
        while len(dates) < periods:
            next_date = last_date + pd.Timedelta(days=day_offset)
            if next_date.weekday() < 5:
                dates.append(next_date)
            day_offset += 1
        return pd.DataFrame({'date': dates, 'yhat': np.round(ensemble_pred, 2).astype(float), 'yhat_lower': np.round(yhat_lower, 2).astype(float), 'yhat_upper': np.round(yhat_upper, 2).astype(float)})