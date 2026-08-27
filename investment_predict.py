import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def get_stock_data(ticker='AAPL', period='2y'):
    df = yf.download(ticker, period=period, auto_adjust=True)
    df.reset_index(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    return df

def build_features(df):
    close = df['Close']
    df['returns_1d'] = close.pct_change()
    df['returns_5d'] = close.pct_change(5)
    df['log_returns'] = np.log(close / close.shift(1))
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df['rsi'] = 100 - 100 / (1 + gain / (loss + 1e-10))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    bb_mean = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df['bb_pct'] = (close - (bb_mean - 2 * bb_std)) / (4 * bb_std + 1e-10)
    df['atr'] = close.rolling(14).std()
    for w in [5, 10, 20, 50]:
        df[f'sma_{w}'] = close.rolling(w).mean()
        df[f'price_vs_sma_{w}'] = close / df[f'sma_{w}'] - 1
    for lag in [1, 2, 3, 5, 10]:
        df[f'lag_{lag}'] = close.shift(lag)
    for w in [5, 10, 20]:
        df[f'vol_{w}'] = close.rolling(w).std()
    if 'Volume' in df.columns:
        df['vol_change'] = df['Volume'].pct_change()
        df['vol_ma5'] = df['Volume'].rolling(5).mean()
        df['vol_ratio'] = df['Volume'] / (df['vol_ma5'] + 1)
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['target'] = close.shift(-1)
    return df.dropna()
FEATURES = ['returns_1d', 'returns_5d', 'log_returns', 'rsi', 'macd', 'macd_signal', 'bb_pct', 'atr', 'price_vs_sma_5', 'price_vs_sma_10', 'price_vs_sma_20', 'price_vs_sma_50', 'lag_1', 'lag_2', 'lag_3', 'lag_5', 'lag_10', 'vol_5', 'vol_10', 'vol_20', 'day_of_week', 'month']
VOLUME_FEATURES = ['vol_change', 'vol_ratio']

def predict_investment(ticker='AAPL'):
    df = get_stock_data(ticker)
    df = build_features(df)
    feature_cols = FEATURES.copy()
    if 'vol_change' in df.columns:
        feature_cols += VOLUME_FEATURES
    split = int(len(df) * 0.8)
    train, test = (df.iloc[:split], df.iloc[split:])
    X_train, y_train = (train[feature_cols].values, train['target'].values)
    X_test, y_test = (test[feature_cols].values, test['target'].values)
    model = XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=0)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    preds = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, preds) * 100
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(np.mean((y_test - preds) ** 2))
    actual_dir = np.sign(np.diff(y_test))
    pred_dir = np.sign(preds[1:] - y_test[:-1])
    dir_acc = np.mean(actual_dir == pred_dir) * 100
    return (preds, y_test, rmse, df, {'mape': mape, 'r2': r2, 'rmse': rmse, 'direction_accuracy': dir_acc})