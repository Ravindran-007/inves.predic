# app/models/stock_predictor.py
import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score, accuracy_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_technical_indicators(df):
    """Build features. Returns df with both price-level and return-based columns."""
    close = df['sales'].copy()

    # ── Returns ───────────────────────────────────────────────────────────────
    for p in [1, 2, 3, 5, 10, 20]:
        df[f'ret_{p}d'] = close.pct_change(p)
    df['log_ret_1d'] = np.log(close / close.shift(1))
    df['log_ret_5d'] = np.log(close / close.shift(5))

    # ── RSI ───────────────────────────────────────────────────────────────────
    for w in [7, 14, 21]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(w).mean()
        loss = (-delta.clip(upper=0)).rolling(w).mean()
        df[f'rsi_{w}'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    df['rsi_slope'] = df['rsi_14'] - df['rsi_14'].shift(3)

    # ── MACD ──────────────────────────────────────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_raw = ema12 - ema26
    macd_sig = macd_raw.ewm(span=9, adjust=False).mean()
    df['macd_norm']        = macd_raw / (close + 1e-10)
    df['macd_signal_norm'] = macd_sig / (close + 1e-10)
    df['macd_hist_norm']   = (macd_raw - macd_sig) / (close + 1e-10)
    df['macd_cross']       = (macd_raw > macd_sig).astype(int)
    df['macd_cross_chg']   = df['macd_cross'].diff()

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    for w in [10, 20]:
        bb_mean = close.rolling(w).mean()
        bb_std  = close.rolling(w).std().clip(lower=1e-10)
        df[f'bb_z_{w}']     = (close - bb_mean) / (2 * bb_std)
        df[f'bb_width_{w}'] = (4 * bb_std) / (bb_mean + 1e-10)

    # ── ATR % ─────────────────────────────────────────────────────────────────
    for w in [7, 14]:
        df[f'atr_pct_{w}'] = close.rolling(w).std() / (close + 1e-10)

    # ── Moving averages — distance (for classifier) ───────────────────────────
    smas, emas = {}, {}
    for w in [5, 10, 20, 50, 60]:
        smas[w] = close.rolling(w).mean()
        df[f'dist_sma_{w}'] = close / (smas[w] + 1e-10) - 1
    for w in [9, 21]:
        emas[w] = close.ewm(span=w, adjust=False).mean()
        df[f'dist_ema_{w}'] = close / (emas[w] + 1e-10) - 1

    # ── Moving averages — raw (for regressor) ────────────────────────────────
    for w in [5, 10, 20, 50, 60]:
        df[f'sma_{w}'] = smas[w]
    for w in [9, 21]:
        df[f'ema_{w}'] = emas[w]

    # ── MA crossovers ─────────────────────────────────────────────────────────
    df['cross_5_20']     = (smas[5]  > smas[20]).astype(int)
    df['cross_10_50']    = (smas[10] > smas[50]).astype(int)
    df['cross_20_60']    = (smas[20] > smas[60]).astype(int)
    df['cross_ema_9_21'] = (emas[9]  > emas[21]).astype(int)

    # ── Momentum ──────────────────────────────────────────────────────────────
    df['mom_5']  = close / close.shift(5)  - 1
    df['mom_10'] = close / close.shift(10) - 1
    df['mom_20'] = close / close.shift(20) - 1
    df['roc_3']  = close.diff(3) / (close.shift(3) + 1e-10)

    # ── Stochastic ────────────────────────────────────────────────────────────
    low14  = close.rolling(14).min()
    high14 = close.rolling(14).max()
    df['stoch_k']    = (close - low14) / (high14 - low14 + 1e-10) * 100
    df['stoch_d']    = df['stoch_k'].rolling(3).mean()
    df['stoch_cross'] = (df['stoch_k'] > df['stoch_d']).astype(int)
    df['williams_r'] = (high14 - close) / (high14 - low14 + 1e-10) * -100

    # ── Lag raw prices (for regressor) ───────────────────────────────────────
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f'lag_{lag}'] = close.shift(lag)

    # ── Lag returns (for classifier) ─────────────────────────────────────────
    for lag in [1, 2, 3, 4, 5, 7, 10]:
        df[f'lag_ret_{lag}'] = df['ret_1d'].shift(lag)

    # ── Rolling price stats (for regressor) ──────────────────────────────────
    for w in [5, 10, 20]:
        df[f'roll_std_{w}']  = close.rolling(w).std()
        df[f'roll_max_{w}']  = close.rolling(w).max()
        df[f'roll_min_{w}']  = close.rolling(w).min()

    # ── Rolling return stats (for classifier) ────────────────────────────────
    for w in [5, 10, 20]:
        ret = df['ret_1d']
        df[f'roll_ret_mean_{w}'] = ret.rolling(w).mean()
        df[f'roll_ret_std_{w}']  = ret.rolling(w).std()
        df[f'roll_ret_skew_{w}'] = ret.rolling(w).skew()
        df[f'roll_pmax_{w}']     = close.rolling(w).max() / (close + 1e-10) - 1
        df[f'roll_pmin_{w}']     = close.rolling(w).min() / (close + 1e-10) - 1

    # ── Regime ────────────────────────────────────────────────────────────────
    df['above_sma20']    = (close > smas[20]).astype(int)
    df['above_sma50']    = (close > smas[50]).astype(int)
    df['trend_strength'] = (smas[20] - smas[50]) / (smas[50] + 1e-10)
    df['vol_regime']     = df['atr_pct_14'] / (df['atr_pct_14'].rolling(20).mean() + 1e-10)

    # ── Time ──────────────────────────────────────────────────────────────────
    dates = pd.to_datetime(df['date'])
    df['day_of_week']  = dates.dt.dayofweek
    df['month']        = dates.dt.month
    df['is_month_end'] = dates.dt.is_month_end.astype(int)
    df['quarter']      = dates.dt.quarter

    return df


# Regressor predicts next-day return — uses normalised/relative features only
REG_FEATURES = [
    'ret_1d', 'ret_2d', 'ret_3d', 'ret_5d', 'ret_10d', 'ret_20d',
    'log_ret_1d', 'log_ret_5d',
    'lag_ret_1', 'lag_ret_2', 'lag_ret_3', 'lag_ret_4', 'lag_ret_5', 'lag_ret_7', 'lag_ret_10',
    'rsi_7', 'rsi_14', 'rsi_21', 'rsi_slope',
    'macd_norm', 'macd_signal_norm', 'macd_hist_norm', 'macd_cross',
    'bb_z_10', 'bb_z_20', 'bb_width_10', 'bb_width_20',
    'atr_pct_7', 'atr_pct_14',
    'dist_sma_5', 'dist_sma_10', 'dist_sma_20', 'dist_sma_50',
    'dist_ema_9', 'dist_ema_21',
    'cross_5_20', 'cross_10_50', 'cross_20_60', 'cross_ema_9_21',
    'mom_5', 'mom_10', 'mom_20', 'roc_3',
    'stoch_k', 'stoch_d', 'williams_r',
    'roll_ret_mean_5', 'roll_ret_std_5',
    'roll_ret_mean_10', 'roll_ret_std_10',
    'roll_ret_mean_20', 'roll_ret_std_20',
    'above_sma20', 'above_sma50', 'trend_strength', 'vol_regime',
    'day_of_week', 'month',
]

# Classifier uses return-based features only (no raw prices)
CLF_FEATURES = [
    'ret_1d', 'ret_2d', 'ret_3d', 'ret_5d', 'ret_10d', 'ret_20d',
    'log_ret_1d', 'log_ret_5d',
    'lag_ret_1', 'lag_ret_2', 'lag_ret_3', 'lag_ret_4', 'lag_ret_5', 'lag_ret_7', 'lag_ret_10',
    'rsi_7', 'rsi_14', 'rsi_21', 'rsi_slope',
    'macd_norm', 'macd_signal_norm', 'macd_hist_norm', 'macd_cross', 'macd_cross_chg',
    'bb_z_10', 'bb_z_20', 'bb_width_10', 'bb_width_20',
    'atr_pct_7', 'atr_pct_14',
    'dist_sma_5', 'dist_sma_10', 'dist_sma_20', 'dist_sma_50',
    'dist_ema_9', 'dist_ema_21',
    'cross_5_20', 'cross_10_50', 'cross_20_60', 'cross_ema_9_21',
    'mom_5', 'mom_10', 'mom_20', 'roc_3',
    'stoch_k', 'stoch_d', 'stoch_cross', 'williams_r',
    'roll_ret_mean_5', 'roll_ret_std_5', 'roll_ret_skew_5',
    'roll_ret_mean_10', 'roll_ret_std_10', 'roll_ret_skew_10',
    'roll_ret_mean_20', 'roll_ret_std_20', 'roll_ret_skew_20',
    'roll_pmax_5', 'roll_pmin_5', 'roll_pmax_10', 'roll_pmin_10',
    'above_sma20', 'above_sma50', 'trend_strength', 'vol_regime',
    'day_of_week', 'month', 'is_month_end', 'quarter',
]


def _build_feature_cols(df):
    exclude = {'date', 'sales', 'target', 'direction'}
    return [c for c in df.columns if c not in exclude]


class StockPredictor:
    """Ensemble: XGB+LGBM regressor for price, XGB+LGBM classifier for direction."""

    def __init__(self):
        self.reg_xgb = None
        self.reg_lgb = None
        self.clf_xgb = None
        self.clf_lgb = None
        self.scaler_reg = RobustScaler()
        self.scaler_clf = RobustScaler()
        self.is_trained = False
        self.performance = {}
        self.symbol = None
        self.data = None

    def fetch_data(self, symbol, period='1y', interval='1d'):
        logger.info(f"Fetching data for {symbol}...")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            raise ValueError(f"No data found for symbol: {symbol}")
        df.reset_index(inplace=True)
        df.rename(columns={'Date': 'date', 'Close': 'sales'}, inplace=True)
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        self.symbol = symbol
        self.data = df[['date', 'sales']].copy()
        logger.info(f"Fetched {len(df)} days of data for {symbol}")
        return df

    def _prepare(self, df):
        df = add_technical_indicators(df[['date', 'sales']].copy())
        # Target: next-day return (not raw price) — generalises across price levels
        df['target']    = df['sales'].shift(-1) / df['sales'] - 1
        df['direction'] = (df['target'] > 0).astype(int)
        return df.dropna()

    def fit(self, df=None, symbol=None):
        if df is None and symbol:
            df = self.fetch_data(symbol)
        elif df is None:
            raise ValueError("Either df or symbol must be provided")

        logger.info(f"Training ensemble for {self.symbol or 'unknown'}...")
        prepared = self._prepare(df)
        split = int(len(prepared) * 0.8)
        train, test = prepared.iloc[:split], prepared.iloc[split:]

        X_tr_r = self.scaler_reg.fit_transform(train[REG_FEATURES])
        X_te_r = self.scaler_reg.transform(test[REG_FEATURES])
        X_tr_c = self.scaler_clf.fit_transform(train[CLF_FEATURES])
        X_te_c = self.scaler_clf.transform(test[CLF_FEATURES])

        y_tr_reg, y_te_reg = train['target'].values, test['target'].values
        y_tr_clf, y_te_clf = train['direction'].values, test['direction'].values

        self.reg_xgb = XGBRegressor(
            n_estimators=800, learning_rate=0.02, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, reg_alpha=0.05, reg_lambda=1.5,
            random_state=42, n_jobs=-1, verbosity=0,
        )
        self.reg_xgb.fit(X_tr_r, y_tr_reg, eval_set=[(X_te_r, y_te_reg)], verbose=False)

        self.reg_lgb = LGBMRegressor(
            n_estimators=800, learning_rate=0.02, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, reg_alpha=0.05, reg_lambda=1.5,
            random_state=42, n_jobs=-1, verbose=-1,
        )
        self.reg_lgb.fit(X_tr_r, y_tr_reg)

        self.clf_xgb = XGBClassifier(
            n_estimators=800, learning_rate=0.02, max_depth=3,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=8, reg_alpha=0.1, reg_lambda=2.0,
            random_state=42, n_jobs=-1, eval_metric='logloss', verbosity=0,
        )
        self.clf_xgb.fit(X_tr_c, y_tr_clf, eval_set=[(X_te_c, y_te_clf)], verbose=False)

        self.clf_lgb = LGBMClassifier(
            n_estimators=800, learning_rate=0.02, max_depth=3,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=2.0,
            random_state=42, n_jobs=-1, verbose=-1,
        )
        self.clf_lgb.fit(X_tr_c, y_tr_clf)

        reg_preds_ret = (self.reg_xgb.predict(X_te_r) + self.reg_lgb.predict(X_te_r)) / 2
        # Convert return predictions back to prices for MAPE
        reg_preds_price = test['sales'].values * (1 + reg_preds_ret)
        actual_price    = test['sales'].values * (1 + y_te_reg)
        clf_probs = (self.clf_xgb.predict_proba(X_te_c)[:, 1] +
                     self.clf_lgb.predict_proba(X_te_c)[:, 1]) / 2
        clf_preds = (clf_probs >= 0.5).astype(int)

        mape    = mean_absolute_percentage_error(actual_price, reg_preds_price) * 100
        r2      = r2_score(actual_price, reg_preds_price)
        dir_acc = accuracy_score(y_te_clf, clf_preds) * 100

        self.performance = {
            'mape': mape, 'r2': r2,
            'accuracy': max(0, 100 - mape),
            'direction_accuracy': dir_acc,
            'test_samples': len(y_te_reg),
        }
        self.is_trained = True
        logger.info(f"MAPE: {mape:.2f}% | R2: {r2:.4f} | Direction: {dir_acc:.1f}%")
        return self

    def predict_future(self, periods=5):
        """
        Predict next N trading days.
        - Day 1: direct model prediction from real features (most accurate)
        - Day 2+: apply the same predicted daily return, dampened toward 0
          so long-horizon predictions stay near the last known price.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")

        feat_df = add_technical_indicators(self.data.copy())
        feat_df = feat_df.dropna()
        if feat_df.empty:
            return pd.DataFrame()

        last_row        = feat_df.iloc[[-1]]
        last_real_price = float(self.data['sales'].iloc[-1])
        last_date       = self.data['date'].iloc[-1]

        x_r = self.scaler_reg.transform(last_row[REG_FEATURES])
        x_c = self.scaler_clf.transform(last_row[CLF_FEATURES])

        # Model now predicts return, convert to price
        day1_return = float(
            (self.reg_xgb.predict(x_r)[0] + self.reg_lgb.predict(x_r)[0]) / 2
        )
        day1_return = max(-0.03, min(0.03, day1_return))   # clamp to ±3%
        day1_price  = last_real_price * (1 + day1_return)

        dir_prob = float(
            (self.clf_xgb.predict_proba(x_c)[0, 1] +
             self.clf_lgb.predict_proba(x_c)[0, 1]) / 2
        )

        results = []
        price        = last_real_price
        trading_days = 0
        day_offset   = 1

        while trading_days < periods:
            next_date = last_date + pd.Timedelta(days=day_offset)
            if next_date.weekday() >= 5:   # skip weekends
                day_offset += 1
                continue

            if trading_days == 0:
                price = day1_price
            else:
                dampen = 0.5 ** trading_days
                price  = price * (1 + day1_return * dampen)

            # Direction confidence decays toward 0.5 over horizon
            horizon_prob = 0.5 + (dir_prob - 0.5) * (0.85 ** trading_days)

            results.append({
                'date':           next_date,
                'yhat':           round(price, 2),
                'up_probability': round(horizon_prob, 3),
            })
            trading_days += 1
            day_offset   += 1

        return pd.DataFrame(results)

    def get_current_price(self):
        if self.data is not None and not self.data.empty:
            return self.data['sales'].iloc[-1]
        return None

