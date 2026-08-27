import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score, accuracy_score
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_technical_indicators(df):
    close = df['sales'].copy()
    cols = {}
    for p in [1, 2, 3, 5, 10, 20]:
        cols[f'ret_{p}d'] = close.pct_change(p)
    cols['log_ret_1d'] = np.log(close / close.shift(1))
    cols['log_ret_5d'] = np.log(close / close.shift(5))
    for w in [7, 14, 21]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(w).mean()
        loss = (-delta.clip(upper=0)).rolling(w).mean()
        cols[f'rsi_{w}'] = 100 - 100 / (1 + gain / (loss + 1e-10))
    cols['rsi_slope'] = cols['rsi_14'] - cols['rsi_14'].shift(3)
    cols['rsi_overbought'] = (cols['rsi_14'] > 70).astype(int)
    cols['rsi_oversold'] = (cols['rsi_14'] < 30).astype(int)
    cols['rsi_mid_cross'] = ((cols['rsi_14'] > 50) & (cols['rsi_14'].shift(1) <= 50)).astype(int)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_raw = ema12 - ema26
    macd_sig = macd_raw.ewm(span=9, adjust=False).mean()
    macd_hist = (macd_raw - macd_sig) / (close + 1e-10)
    cols['macd_norm'] = macd_raw / (close + 1e-10)
    cols['macd_signal_norm'] = macd_sig / (close + 1e-10)
    cols['macd_hist_norm'] = macd_hist
    cols['macd_hist_accel'] = macd_hist.diff()
    cols['macd_cross'] = (macd_raw > macd_sig).astype(int)
    cols['macd_cross_chg'] = cols['macd_cross'].diff()
    cols['macd_crossover'] = ((macd_raw > macd_sig) & (macd_raw.shift(1) <= macd_sig.shift(1))).astype(int)
    cols['macd_crossunder'] = ((macd_raw < macd_sig) & (macd_raw.shift(1) >= macd_sig.shift(1))).astype(int)
    for w in [10, 20]:
        bb_mean = close.rolling(w).mean()
        bb_std = close.rolling(w).std().clip(lower=1e-10)
        cols[f'bb_z_{w}'] = (close - bb_mean) / (2 * bb_std)
        cols[f'bb_width_{w}'] = 4 * bb_std / (bb_mean + 1e-10)
    bb_mean20 = close.rolling(20).mean()
    bb_std20 = close.rolling(20).std().clip(lower=1e-10)
    bb_upper = bb_mean20 + 2 * bb_std20
    bb_lower = bb_mean20 - 2 * bb_std20
    cols['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
    cols['bb_compression'] = (bb_upper - bb_lower) / (bb_mean20 + 1e-10)
    cols['bb_squeeze'] = (cols['bb_compression'] < cols['bb_compression'].rolling(20).mean()).astype(int)
    for w in [7, 14]:
        cols[f'atr_pct_{w}'] = close.rolling(w).std() / (close + 1e-10)
    cols['vol_ratio_5_20'] = cols['atr_pct_7'] / (cols['atr_pct_14'].rolling(20).mean() + 1e-10)
    smas, emas = ({}, {})
    for w in [5, 10, 20, 50, 60]:
        smas[w] = close.rolling(w).mean()
        cols[f'dist_sma_{w}'] = close / (smas[w] + 1e-10) - 1
        cols[f'sma_{w}'] = smas[w]
    for w in [9, 21]:
        emas[w] = close.ewm(span=w, adjust=False).mean()
        cols[f'dist_ema_{w}'] = close / (emas[w] + 1e-10) - 1
        cols[f'ema_{w}'] = emas[w]
    cols['cross_5_20'] = (smas[5] > smas[20]).astype(int)
    cols['cross_10_50'] = (smas[10] > smas[50]).astype(int)
    cols['cross_20_60'] = (smas[20] > smas[60]).astype(int)
    cols['cross_ema_9_21'] = (emas[9] > emas[21]).astype(int)
    cols['above_sma20'] = (close > smas[20]).astype(int)
    cols['above_sma50'] = (close > smas[50]).astype(int)
    cols['trend_strength'] = (smas[20] - smas[50]) / (smas[50] + 1e-10)
    cols['vol_regime'] = cols['atr_pct_14'] / (cols['atr_pct_14'].rolling(20).mean() + 1e-10)
    cols['mom_5'] = close / close.shift(5) - 1
    cols['mom_10'] = close / close.shift(10) - 1
    cols['mom_20'] = close / close.shift(20) - 1
    cols['roc_3'] = close.diff(3) / (close.shift(3) + 1e-10)
    low14 = close.rolling(14).min()
    high14 = close.rolling(14).max()
    stoch_k = (close - low14) / (high14 - low14 + 1e-10) * 100
    cols['stoch_k'] = stoch_k
    cols['stoch_d'] = stoch_k.rolling(3).mean()
    cols['stoch_cross'] = (stoch_k > cols['stoch_d']).astype(int)
    cols['williams_r'] = (high14 - close) / (high14 - low14 + 1e-10) * -100
    for lag in [1, 2, 3, 5, 10, 20]:
        cols[f'lag_{lag}'] = close.shift(lag)
    ret_1d = close.pct_change(1)
    for lag in [1, 2, 3, 4, 5, 7, 10]:
        cols[f'lag_ret_{lag}'] = ret_1d.shift(lag)
    for w in [5, 10, 20]:
        cols[f'roll_std_{w}'] = close.rolling(w).std()
        cols[f'roll_max_{w}'] = close.rolling(w).max()
        cols[f'roll_min_{w}'] = close.rolling(w).min()
        cols[f'roll_ret_mean_{w}'] = ret_1d.rolling(w).mean()
        cols[f'roll_ret_std_{w}'] = ret_1d.rolling(w).std()
        cols[f'roll_ret_skew_{w}'] = ret_1d.rolling(w).skew()
        cols[f'roll_pmax_{w}'] = close.rolling(w).max() / (close + 1e-10) - 1
        cols[f'roll_pmin_{w}'] = close.rolling(w).min() / (close + 1e-10) - 1
    cols['pct_from_52w_high'] = close / (close.rolling(252).max() + 1e-10) - 1
    cols['pct_from_52w_low'] = close / (close.rolling(252).min() + 1e-10) - 1
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        vol = df['Volume'].replace(0, np.nan).ffill()
        cols['volume_ratio'] = vol / (vol.rolling(20).mean() + 1e-10)
        cols['volume_spike'] = (vol > vol.rolling(20).mean() * 1.5).astype(int)
        cols['volume_trend'] = (vol > vol.shift(1)).astype(int)
        cols['volume_ret_corr'] = ret_1d.rolling(10).corr(vol.pct_change())
    dates = pd.to_datetime(df['date'])
    cols['day_of_week'] = dates.dt.dayofweek
    cols['month'] = dates.dt.month
    cols['is_month_end'] = dates.dt.is_month_end.astype(int)
    cols['quarter'] = dates.dt.quarter
    return pd.concat([df[['date', 'sales']], pd.DataFrame(cols, index=df.index)], axis=1)
REG_FEATURES = ['ret_1d', 'ret_2d', 'ret_3d', 'ret_5d', 'ret_10d', 'ret_20d', 'log_ret_1d', 'log_ret_5d', 'lag_ret_1', 'lag_ret_2', 'lag_ret_3', 'lag_ret_4', 'lag_ret_5', 'lag_ret_7', 'lag_ret_10', 'rsi_7', 'rsi_14', 'rsi_21', 'rsi_slope', 'rsi_overbought', 'rsi_oversold', 'rsi_mid_cross', 'macd_norm', 'macd_signal_norm', 'macd_hist_norm', 'macd_hist_accel', 'macd_cross', 'macd_cross_chg', 'macd_crossover', 'macd_crossunder', 'bb_z_10', 'bb_z_20', 'bb_width_10', 'bb_width_20', 'bb_position', 'bb_compression', 'bb_squeeze', 'atr_pct_7', 'atr_pct_14', 'vol_ratio_5_20', 'dist_sma_5', 'dist_sma_10', 'dist_sma_20', 'dist_sma_50', 'dist_ema_9', 'dist_ema_21', 'cross_5_20', 'cross_10_50', 'cross_20_60', 'cross_ema_9_21', 'mom_5', 'mom_10', 'mom_20', 'roc_3', 'stoch_k', 'stoch_d', 'stoch_cross', 'williams_r', 'roll_ret_mean_5', 'roll_ret_std_5', 'roll_ret_skew_5', 'roll_ret_mean_10', 'roll_ret_std_10', 'roll_ret_skew_10', 'roll_ret_mean_20', 'roll_ret_std_20', 'roll_ret_skew_20', 'roll_pmax_5', 'roll_pmin_5', 'roll_pmax_10', 'roll_pmin_10', 'above_sma20', 'above_sma50', 'trend_strength', 'vol_regime', 'pct_from_52w_high', 'pct_from_52w_low', 'day_of_week', 'month', 'is_month_end', 'quarter']

class StockPredictor:

    def __init__(self):
        self.reg_xgb = None
        self.reg_lgb = None
        self.scaler_reg = RobustScaler()
        self.is_trained = False
        self.performance = {}
        self.symbol = None
        self.data = None

    def fetch_data(self, symbol, period='2y', interval='1d'):
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            raise ValueError(f'No data found for symbol: {symbol}')
        df.reset_index(inplace=True)
        df.rename(columns={'Date': 'date', 'Close': 'sales'}, inplace=True)
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        self.symbol = symbol
        self.data = df[['date', 'sales']].copy()
        logger.info(f'Fetched {len(df)} days of data for {symbol}')
        return df

    def _prepare(self, df):
        keep = ['date', 'sales'] + [c for c in ['Volume'] if c in df.columns]
        out = add_technical_indicators(df[keep].copy())
        out['target'] = out['sales'].shift(-1) / out['sales'] - 1
        out['direction'] = (out['target'] > 0).astype(int)
        return out.dropna()

    def fit(self, df=None, symbol=None):
        if df is None and symbol:
            df = self.fetch_data(symbol)
        elif df is None:
            raise ValueError('Either df or symbol must be provided')
        logger.info(f'Training ensemble for {self.symbol or 'unknown'}...')
        prepared = self._prepare(df)
        split = max(int(len(prepared) * 0.8), len(prepared) - 120)
        train, test = (prepared.iloc[:split], prepared.iloc[split:])
        feat_cols = [f for f in REG_FEATURES if f in prepared.columns]
        X_train = train[feat_cols].values
        X_test = test[feat_cols].values
        y_tr_reg, y_te_reg = (train['target'].values, test['target'].values)
        self.reg_xgb = XGBRegressor(n_estimators=800, learning_rate=0.02, max_depth=4, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_alpha=0.05, reg_lambda=1.5, random_state=42, n_jobs=-1, verbosity=0)
        self.reg_xgb.fit(X_train, y_tr_reg, eval_set=[(X_test, y_te_reg)], verbose=False)
        self.reg_lgb = LGBMRegressor(n_estimators=800, learning_rate=0.02, max_depth=4, subsample=0.8, colsample_bytree=0.8, min_child_samples=20, reg_alpha=0.05, reg_lambda=1.5, random_state=42, n_jobs=-1, verbose=-1)
        self.reg_lgb.fit(X_train, y_tr_reg)
        self._feat_cols = feat_cols
        self.scaler_reg.fit(X_train)
        reg_preds_ret = (self.reg_xgb.predict(X_test) + self.reg_lgb.predict(X_test)) / 2
        reg_preds_price = test['sales'].values * (1 + reg_preds_ret)
        actual_price = test['sales'].values * (1 + y_te_reg)
        dir_preds = (reg_preds_ret > 0).astype(int)
        dir_actual = (y_te_reg > 0).astype(int)
        mape = mean_absolute_percentage_error(actual_price, reg_preds_price) * 100
        r2 = r2_score(actual_price, reg_preds_price)
        dir_acc = accuracy_score(dir_actual, dir_preds) * 100
        self.performance = {'mape': mape, 'r2': r2, 'accuracy': max(0, 100 - mape), 'direction_accuracy': dir_acc, 'test_samples': len(y_te_reg)}
        self.is_trained = True
        logger.info(f'MAPE: {mape:.2f}% | R2: {r2:.4f} | Direction: {dir_acc:.1f}%')
        return self

    def predict_future(self, periods=5):
        if not self.is_trained:
            raise ValueError('Model not trained. Call fit() first.')
        feat_df = add_technical_indicators(self.data.copy()).dropna()
        if feat_df.empty:
            return pd.DataFrame()
        last_row = feat_df.iloc[[-1]]
        last_real_price = float(self.data['sales'].iloc[-1])
        last_date = self.data['date'].iloc[-1]
        x = last_row[self._feat_cols].values
        day1_return = float((self.reg_xgb.predict(x)[0] + self.reg_lgb.predict(x)[0]) / 2)
        day1_return = max(-0.03, min(0.03, day1_return))
        day1_price = last_real_price * (1 + day1_return)
        dir_prob = 1.0 if day1_return > 0 else 0.0
        results, price, trading_days, day_offset = ([], last_real_price, 0, 1)
        while trading_days < periods:
            next_date = last_date + pd.Timedelta(days=day_offset)
            if next_date.weekday() >= 5:
                day_offset += 1
                continue
            price = day1_price if trading_days == 0 else price * (1 + day1_return * 0.5 ** trading_days)
            horizon_prob = 0.5 + (dir_prob - 0.5) * 0.85 ** trading_days
            results.append({'date': next_date, 'yhat': round(price, 2), 'up_probability': round(horizon_prob, 3)})
            trading_days += 1
            day_offset += 1
        return pd.DataFrame(results)

    def get_current_price(self):
        if self.data is not None and (not self.data.empty):
            return self.data['sales'].iloc[-1]
        return None