import logging
import warnings
import numpy as np
import pandas as pd
from prophet import Prophet
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    close = d['sales']
    d['target'] = np.log(close.shift(-1) / close)
    for p in [1, 2, 3, 5, 7, 10, 14, 20, 30]:
        d[f'ret_{p}d'] = close.pct_change(p)
    d['log_ret_1d'] = np.log(close / close.shift(1))
    d['log_ret_5d'] = np.log(close / close.shift(5))
    for lag in [1, 2, 3, 5, 7, 10, 14, 20, 30, 60]:
        d[f'lag_ret_{lag}'] = d['ret_1d'].shift(lag)
    for w in [5, 10, 20, 60]:
        d[f'vol_{w}'] = d['ret_1d'].rolling(w).std()
    for w in [10, 20, 50, 200]:
        sma = close.rolling(w).mean()
        d[f'dist_sma_{w}'] = close / (sma + 1e-10) - 1
    for span in [9, 21]:
        ema = close.ewm(span=span, adjust=False).mean()
        d[f'dist_ema_{span}'] = close / (ema + 1e-10) - 1
    for w in [7, 14, 21]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(w).mean()
        loss = (-delta.clip(upper=0)).rolling(w).mean()
        d[f'rsi_{w}'] = 100 - 100 / (1 + gain / (loss + 1e-10))
    d['rsi_slope'] = d['rsi_14'] - d['rsi_14'].shift(3)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    d['macd_norm'] = macd / (close + 1e-10)
    d['macd_hist'] = (macd - sig) / (close + 1e-10)
    d['macd_cross'] = (macd > sig).astype(int)
    for w in [10, 20]:
        bb_mean = close.rolling(w).mean()
        bb_std = close.rolling(w).std().clip(lower=1e-10)
        d[f'bb_z_{w}'] = (close - bb_mean) / (2 * bb_std)
        d[f'bb_width_{w}'] = 4 * bb_std / (bb_mean + 1e-10)
    for p in [5, 10, 20]:
        d[f'mom_{p}'] = close / close.shift(p) - 1
    for w in [5, 10, 20]:
        ret = d['ret_1d']
        d[f'roll_ret_mean_{w}'] = ret.rolling(w).mean()
        d[f'roll_ret_std_{w}'] = ret.rolling(w).std()
        d[f'roll_ret_skew_{w}'] = ret.rolling(w).skew()
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    d['above_sma20'] = (close > sma20).astype(int)
    d['above_sma50'] = (close > sma50).astype(int)
    d['trend_strength'] = (sma20 - sma50) / (sma50 + 1e-10)
    d['vol_regime'] = d['vol_20'] / (d['vol_20'].rolling(60).mean() + 1e-10)
    dates = pd.to_datetime(d['date'])
    d['day_of_week'] = dates.dt.dayofweek
    d['month'] = dates.dt.month
    d['quarter'] = dates.dt.quarter
    d['is_month_end'] = dates.dt.is_month_end.astype(int)
    d['month_sin'] = np.sin(2 * np.pi * d['month'] / 12)
    d['month_cos'] = np.cos(2 * np.pi * d['month'] / 12)
    return d.dropna()
_EXCLUDE = {'date', 'sales', 'target'}

def _tune_prophet(train_df: pd.DataFrame, val_df: pd.DataFrame, n_trials: int=50) -> dict:
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning('optuna not installed — pip install optuna')
        return {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'}

    def objective(trial):
        params = {'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.001, 0.5, log=True), 'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.01, 20.0, log=True), 'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative'])}
        try:
            m = Prophet(yearly_seasonality=True, weekly_seasonality=True, interval_width=0.95, **params)
            m.add_seasonality('quarterly', period=91.25, fourier_order=5)
            m.add_seasonality('monthly', period=30.5, fourier_order=3)
            m.fit(train_df)
            fc = m.predict(m.make_future_dataframe(periods=len(val_df)))
            pred = fc['yhat'].values[-len(val_df):]
            return mean_absolute_percentage_error(val_df['y'].values, pred)
        except Exception:
            return 1.0
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(f'Optuna best: {study.best_params}')
    return study.best_params

class SalesForecaster:

    def __init__(self, tune: bool=False, n_optuna_trials: int=50):
        self.tune = tune
        self.n_optuna_trials = n_optuna_trials
        self.prophet = None
        self.xgb = None
        self.lgb = None
        self.scaler = RobustScaler()
        self.is_trained = False
        self.performance = {}
        self._feat_cols: list = []

    def fit(self, df: pd.DataFrame) -> 'SalesForecaster':
        df = df[['date', 'sales']].copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna(subset=['sales'])
        df['sales'] = df['sales'].clip(lower=0.01)
        z = np.abs((df['sales'] - df['sales'].mean()) / df['sales'].std())
        n_before = len(df)
        df = df[z < 3].reset_index(drop=True)
        if len(df) < n_before:
            logger.info(f'Removed {n_before - len(df)} outliers')
        cv = df['sales'].std() / df['sales'].mean()
        prophet_df = df.rename(columns={'date': 'ds', 'sales': 'y'})
        split_p = int(len(prophet_df) * 0.8)
        train_p = prophet_df.iloc[:split_p]
        val_p = prophet_df.iloc[split_p:]
        if self.tune:
            best = _tune_prophet(train_p, val_p, self.n_optuna_trials)
        else:
            best = {'changepoint_prior_scale': 0.005, 'seasonality_prior_scale': 0.001, 'seasonality_mode': 'additive' if cv <= 0.15 else 'multiplicative'}
        self.prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.01, seasonality_prior_scale=0.01, holidays_prior_scale=0.01, seasonality_mode='additive', interval_width=0.95)
        self.prophet.add_seasonality('quarterly', period=91.25, fourier_order=5)
        self.prophet.add_seasonality('monthly', period=30.5, fourier_order=3)
        self.prophet.fit(prophet_df)
        feat_df = _build_features(df)
        self._feat_cols = [c for c in feat_df.columns if c not in _EXCLUDE]
        X = self.scaler.fit_transform(feat_df[self._feat_cols].values)
        y = feat_df['target'].values
        prices = feat_df['sales'].values
        split_ml = int(len(feat_df) * 0.8)
        X_tr, X_te = (X[:split_ml], X[split_ml:])
        y_tr, y_te = (y[:split_ml], y[split_ml:])
        prices_te = prices[split_ml:]
        self.xgb = XGBRegressor(n_estimators=600, learning_rate=0.02, max_depth=4, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_alpha=0.05, reg_lambda=1.5, random_state=42, n_jobs=-1, verbosity=0)
        self.xgb.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        self.lgb = LGBMRegressor(n_estimators=600, learning_rate=0.02, max_depth=4, subsample=0.8, colsample_bytree=0.8, min_child_samples=20, reg_alpha=0.05, reg_lambda=1.5, random_state=42, n_jobs=-1, verbose=-1)
        self.lgb.fit(X_tr, y_tr)
        pred_log_ret = (self.xgb.predict(X_te) + self.lgb.predict(X_te)) / 2
        actual_prices = prices_te * np.exp(y_te)
        pred_prices = prices_te * np.exp(pred_log_ret)
        mape = mean_absolute_percentage_error(actual_prices, pred_prices) * 100
        r2 = r2_score(actual_prices, pred_prices)
        dir_acc = np.mean((pred_log_ret > 0) == (y_te > 0)) * 100
        self._last_price = float(prices[-1])
        self._last_feat_row = feat_df[self._feat_cols].iloc[[-1]].values
        self.performance = {'mape': float(mape), 'r2': float(r2), 'accuracy': float(max(0, 100 - mape)), 'direction_accuracy': float(dir_acc), 'test_samples': int(len(actual_prices)), 'seasonality_mode': best.get('seasonality_mode'), 'cv': round(float(cv), 3)}
        self.is_trained = True
        logger.info(f'MAPE: {mape:.2f}% | R²: {r2:.4f} | Direction: {dir_acc:.1f}%')
        return self

    def predict_future(self, periods: int=90) -> pd.DataFrame:
        if not self.is_trained:
            raise ValueError('Model not trained. Call fit() first.')
        future = self.prophet.make_future_dataframe(periods=periods)
        forecast = self.prophet.predict(future)
        tail = forecast.tail(periods)
        prophet_yhat = tail['yhat'].values.astype(float)
        x_last = self.scaler.transform(self._last_feat_row)
        ml_ret = float((self.xgb.predict(x_last)[0] + self.lgb.predict(x_last)[0]) / 2)
        ml_ret = np.clip(ml_ret, -0.05, 0.05)
        ml_prices = np.array([self._last_price * np.exp(ml_ret * 0.85 ** i * (i + 1)) for i in range(periods)])
        yhat = np.maximum(0.01, 0.6 * prophet_yhat + 0.4 * ml_prices)
        yhat_lower = np.maximum(np.minimum(tail['yhat_lower'].values.astype(float), yhat), 0.01)
        yhat_upper = np.maximum(tail['yhat_upper'].values.astype(float), yhat)
        return pd.DataFrame({'date': tail['ds'].values, 'yhat': np.round(yhat, 2), 'yhat_lower': np.round(yhat_lower, 2), 'yhat_upper': np.round(yhat_upper, 2)})