import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, r2_score, accuracy_score
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def walk_forward_validate(df, model_class, train_window=90, test_window=30, step=30):
    logger.info(f'Running walk-forward validation (train={train_window}, test={test_window})...')
    if len(df) < train_window + test_window:
        raise ValueError(f'Insufficient data: need at least {train_window + test_window} days')
    results = {'mape': [], 'r2': [], 'direction_accuracy': [], 'sharpe_ratio': []}
    start_idx = 0
    while start_idx + train_window + test_window <= len(df):
        train_df = df.iloc[start_idx:start_idx + train_window].copy()
        test_df = df.iloc[start_idx + train_window:start_idx + train_window + test_window].copy()
        model = model_class()
        try:
            model.fit(train_df)
            pred_df = model.predict_future(periods=test_window)
            if pred_df is not None and len(pred_df) > 0:
                actual = test_df['sales'].values[:len(pred_df)]
                pred = pred_df['yhat'].values
                mape = mean_absolute_percentage_error(actual, pred) * 100
                r2 = r2_score(actual, pred)
                pred_trend = np.diff(pred) > 0
                actual_trend = np.diff(actual) > 0
                direction_acc = np.mean(pred_trend == actual_trend) * 100
                returns = np.diff(pred) / (pred[:-1] + 1e-10)
                sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
                results['mape'].append(mape)
                results['r2'].append(r2)
                results['direction_accuracy'].append(direction_acc)
                results['sharpe_ratio'].append(sharpe)
                logger.info(f'Window {start_idx}: MAPE={mape:.2f}% | R2={r2:.4f} | Dir={direction_acc:.1f}%')
        except Exception as e:
            logger.warning(f'Window {start_idx} failed: {e}')
        start_idx += step
    if results['mape']:
        aggregated = {'mape_mean': float(np.mean(results['mape'])), 'mape_std': float(np.std(results['mape'])), 'r2_mean': float(np.mean(results['r2'])), 'r2_std': float(np.std(results['r2'])), 'direction_accuracy_mean': float(np.mean(results['direction_accuracy'])), 'direction_accuracy_std': float(np.std(results['direction_accuracy'])), 'sharpe_ratio_mean': float(np.mean(results['sharpe_ratio'])), 'sharpe_ratio_std': float(np.std(results['sharpe_ratio'])), 'n_windows': len(results['mape'])}
        logger.info(f'Walk-forward results: MAPE={aggregated['mape_mean']:.2f}% | R2={aggregated['r2_mean']:.4f} | Dir={aggregated['direction_accuracy_mean']:.1f}%')
        return aggregated
    return None

def quick_backtest(df, model_class, n_splits=5):
    logger.info(f'Running quick backtest with {n_splits} splits...')
    results = {'mape': [], 'r2': [], 'direction_accuracy': []}
    split_size = len(df) // (n_splits + 1)
    for i in range(n_splits):
        train_end = len(df) - (n_splits - i) * split_size
        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[train_end:train_end + split_size].copy()
        if len(test_df) < 10:
            continue
        model = model_class()
        try:
            model.fit(train_df)
            pred_df = model.predict_future(periods=len(test_df))
            if pred_df is not None and len(pred_df) > 0:
                actual = test_df['sales'].values[:len(pred_df)]
                pred = pred_df['yhat'].values
                mape = mean_absolute_percentage_error(actual, pred) * 100
                r2 = r2_score(actual, pred)
                pred_trend = np.diff(pred) > 0
                actual_trend = np.diff(actual) > 0
                direction_acc = np.mean(pred_trend == actual_trend) * 100
                results['mape'].append(mape)
                results['r2'].append(r2)
                results['direction_accuracy'].append(direction_acc)
        except Exception as e:
            logger.warning(f'Split {i} failed: {e}')
    if results['mape']:
        return {'mape_mean': float(np.mean(results['mape'])), 'r2_mean': float(np.mean(results['r2'])), 'direction_accuracy_mean': float(np.mean(results['direction_accuracy']))}
    return None