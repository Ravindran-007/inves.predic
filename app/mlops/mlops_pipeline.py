import os
os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'
os.environ['MLFLOW_TRACKING_URI'] = 'sqlite:///mlflow.db'
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)
try:
    import mlflow
    import mlflow.sklearn
    _MLFLOW = True
except ImportError:
    _MLFLOW = False
    logger.warning('mlflow not installed — experiment tracking disabled')
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    _EVIDENTLY = True
except ImportError:
    _EVIDENTLY = False
    logger.warning('evidently not installed — drift detection disabled')

class MLOpsPipeline:
    EXPERIMENT_NAME = 'investment_sales_prediction'
    DRIFT_THRESHOLD = 0.15
    RETRAIN_MAPE_THRESHOLD = 10.0

    def __init__(self, model_dir: str='models', tracking_uri: str='mlruns'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        if _MLFLOW:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(self.EXPERIMENT_NAME)

    def log_training_run(self, model_name: str, params: dict, metrics: dict, model_obj=None) -> Optional[str]:
        if not _MLFLOW:
            logger.info(f'[MLflow disabled] {model_name} | metrics={metrics}')
            return None
        with mlflow.start_run(run_name=f'{model_name}_{datetime.utcnow():%Y%m%d_%H%M}') as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.set_tag('model_name', model_name)
            mlflow.set_tag('trained_at', datetime.utcnow().isoformat())
            if model_obj is not None:
                try:
                    mlflow.sklearn.log_model(model_obj, artifact_path='model')
                except Exception:
                    pass
            return run.info.run_id

    def save_model(self, model, name: str) -> Path:
        path = self.model_dir / f'{name}.pkl'
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f'Model saved: {path}')
        return path

    def load_model(self, name: str):
        path = self.model_dir / f'{name}.pkl'
        if not path.exists():
            raise FileNotFoundError(f'Model not found: {path}')
        with open(path, 'rb') as f:
            return pickle.load(f)

    def detect_drift(self, reference: pd.DataFrame, current: pd.DataFrame, target_col: str='close') -> dict:
        if not _EVIDENTLY:
            return self._manual_drift_check(reference, current, target_col)
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        result = report.as_dict()
        drift_detected = result['metrics'][0]['result']['dataset_drift']
        drift_share = result['metrics'][0]['result']['share_of_drifted_columns']
        if drift_detected:
            logger.warning(f'DATA DRIFT DETECTED — {drift_share:.0%} of features drifted. Retraining recommended within 48 hours.')
        return {'drift_detected': drift_detected, 'drift_share': drift_share, 'checked_at': datetime.utcnow().isoformat(), 'early_warning': drift_detected}

    def _manual_drift_check(self, reference: pd.DataFrame, current: pd.DataFrame, target_col: str) -> dict:

        def psi(expected, actual, buckets=10):
            mn, mx = (expected.min(), expected.max())
            bins = np.linspace(mn, mx, buckets + 1)
            e_pct = np.histogram(expected, bins=bins)[0] / len(expected) + 1e-06
            a_pct = np.histogram(actual, bins=bins)[0] / len(actual) + 1e-06
            return float(np.sum((e_pct - a_pct) * np.log(e_pct / a_pct)))
        cols = [c for c in reference.columns if c in current.columns and pd.api.types.is_numeric_dtype(reference[c])]
        scores = {c: psi(reference[c].dropna().values, current[c].dropna().values) for c in cols}
        max_psi = max(scores.values()) if scores else 0.0
        drift = max_psi > self.DRIFT_THRESHOLD
        if drift:
            logger.warning(f'DRIFT DETECTED (PSI={max_psi:.3f}). Retraining recommended within 48 hours.')
        return {'drift_detected': drift, 'max_psi': round(max_psi, 4), 'feature_psi': {k: round(v, 4) for k, v in scores.items()}, 'checked_at': datetime.utcnow().isoformat(), 'early_warning': drift}

    def should_retrain(self, live_mape: float, last_trained: datetime, drift_report: Optional[dict]=None) -> bool:
        age_days = (datetime.utcnow() - last_trained).days
        mape_trigger = live_mape > self.RETRAIN_MAPE_THRESHOLD
        drift_trigger = drift_report is not None and drift_report.get('drift_detected', False)
        age_trigger = age_days >= 7
        reasons = []
        if mape_trigger:
            reasons.append(f'MAPE={live_mape:.1f}%')
        if drift_trigger:
            reasons.append('data drift')
        if age_trigger:
            reasons.append(f'age={age_days}d')
        if reasons:
            logger.info(f'Retraining triggered: {', '.join(reasons)}')
        return bool(reasons)

    def run_daily_pipeline(self, symbol: str, model_class, fetch_fn, reference_df: pd.DataFrame) -> dict:
        logger.info(f'Daily pipeline starting for {symbol}...')
        result = {'symbol': symbol, 'timestamp': datetime.utcnow().isoformat()}
        try:
            current_df = fetch_fn(symbol)
        except Exception as e:
            result['error'] = str(e)
            return result
        drift = self.detect_drift(reference_df, current_df)
        result['drift'] = drift
        try:
            model = self.load_model(symbol)
            live_mape = model.performance.get('mape', 999)
            last_trained = datetime.fromisoformat(model.performance.get('trained_at', '2000-01-01'))
        except Exception:
            live_mape = 999
            last_trained = datetime(2000, 1, 1)
        if self.should_retrain(live_mape, last_trained, drift):
            new_model = model_class()
            new_model.fit(current_df)
            new_model.performance['trained_at'] = datetime.utcnow().isoformat()
            run_id = self.log_training_run(model_name=symbol, params={'symbol': symbol, 'n_samples': len(current_df)}, metrics=new_model.performance, model_obj=new_model)
            self.save_model(new_model, symbol)
            result.update({'retrained': True, 'run_id': run_id, 'new_metrics': new_model.performance})
        else:
            result.update({'retrained': False, 'current_mape': live_mape})
        logger.info(f'Daily pipeline complete for {symbol}: {result}')
        return result