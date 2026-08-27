import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_percentage_error, accuracy_score
import warnings
warnings.filterwarnings('ignore')
from app.models.stock_predictor import add_technical_indicators, REG_FEATURES, CLF_FEATURES

def validate_symbol(symbol, initial_train_days=300, test_days=14, max_tests=15):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period='3y')
    if df.empty:
        return None
    df.reset_index(inplace=True)
    df.rename(columns={'Date': 'date', 'Close': 'sales'}, inplace=True)
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    all_actual, all_pred = ([], [])
    all_clf_actual, all_clf_pred, all_clf_probs = ([], [], [])
    for i in range(max_tests):
        train_start = i * test_days
        train_end = train_start + initial_train_days
        test_end = train_end + test_days
        if test_end > len(df):
            break
        combined = df.iloc[train_start:test_end].copy()
        feat_df = add_technical_indicators(combined[['date', 'sales']].copy())
        feat_df['target'] = feat_df['sales'].shift(-1)
        feat_df['direction'] = (feat_df['target'] > feat_df['sales']).astype(int)
        feat_df = feat_df.dropna()
        n_train = len(feat_df) - test_days
        if n_train < 40:
            continue
        train_f, test_f = (feat_df.iloc[:n_train], feat_df.iloc[n_train:])
        try:
            sr = RobustScaler()
            sc = RobustScaler()
            X_tr_r = sr.fit_transform(train_f[REG_FEATURES])
            X_te_r = sr.transform(test_f[REG_FEATURES])
            X_tr_c = sc.fit_transform(train_f[CLF_FEATURES])
            X_te_c = sc.transform(test_f[CLF_FEATURES])
            y_tr_r, y_te_r = (train_f['target'].values, test_f['target'].values)
            y_tr_c, y_te_c = (train_f['direction'].values, test_f['direction'].values)
            reg_xgb = XGBRegressor(n_estimators=600, learning_rate=0.02, max_depth=4, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_alpha=0.05, reg_lambda=1.5, random_state=42, n_jobs=-1, verbosity=0)
            reg_xgb.fit(X_tr_r, y_tr_r, verbose=False)
            reg_lgb = LGBMRegressor(n_estimators=600, learning_rate=0.02, max_depth=4, subsample=0.8, colsample_bytree=0.8, min_child_samples=10, reg_alpha=0.05, reg_lambda=1.5, random_state=42, n_jobs=-1, verbose=-1)
            reg_lgb.fit(X_tr_r, y_tr_r)
            clf_xgb = XGBClassifier(n_estimators=600, learning_rate=0.02, max_depth=3, subsample=0.8, colsample_bytree=0.8, min_child_weight=8, reg_alpha=0.1, reg_lambda=2.0, random_state=42, n_jobs=-1, eval_metric='logloss', verbosity=0)
            clf_xgb.fit(X_tr_c, y_tr_c, verbose=False)
            clf_lgb = LGBMClassifier(n_estimators=600, learning_rate=0.02, max_depth=3, subsample=0.8, colsample_bytree=0.8, min_child_samples=10, reg_alpha=0.1, reg_lambda=2.0, random_state=42, n_jobs=-1, verbose=-1)
            clf_lgb.fit(X_tr_c, y_tr_c)
        except Exception:
            continue
        reg_preds = (reg_xgb.predict(X_te_r) + reg_lgb.predict(X_te_r)) / 2
        clf_probs = (clf_xgb.predict_proba(X_te_c)[:, 1] + clf_lgb.predict_proba(X_te_c)[:, 1]) / 2
        all_actual.extend(y_te_r.tolist())
        all_pred.extend(reg_preds.tolist())
        all_clf_actual.extend(y_te_c.tolist())
        all_clf_pred.extend((clf_probs >= 0.5).astype(int).tolist())
        all_clf_probs.extend(clf_probs.tolist())
    if not all_actual:
        return None
    mape = mean_absolute_percentage_error(all_actual, all_pred) * 100
    price_acc = 100 - mape
    probs_arr = np.array(all_clf_probs)
    actual_arr = np.array(all_clf_actual)
    pred_arr = np.array(all_clf_pred)
    conf_mask = (probs_arr >= 0.6) | (probs_arr <= 0.4)
    dir_acc_all = accuracy_score(actual_arr, pred_arr) * 100
    dir_acc_conf = accuracy_score(actual_arr[conf_mask], pred_arr[conf_mask]) * 100 if conf_mask.sum() > 0 else 0
    conf_pct = conf_mask.sum() / len(probs_arr) * 100
    return {'symbol': symbol, 'n_preds': len(all_actual), 'mape': mape, 'price_acc': price_acc, 'dir_acc_all': dir_acc_all, 'dir_acc_conf': dir_acc_conf, 'conf_pct': conf_pct}

def run_multi_symbol(symbols):
    rows = []
    for sym in symbols:
        res = validate_symbol(sym)
        if res is not None:
            rows.append(res)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    total_preds = df['n_preds'].sum()
    weighted_acc = (df['price_acc'] * df['n_preds']).sum() / total_preds
    weighted_dir = (df['dir_acc_conf'] * df['n_preds']).sum() / total_preds
    return {'results': df, 'weighted_acc': weighted_acc, 'weighted_dir': weighted_dir, 'passed': weighted_acc >= 96}