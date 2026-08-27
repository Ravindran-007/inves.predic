import optuna
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def objective_prophet(trial, train_df, val_df):
    changepoint_prior = trial.suggest_float('changepoint_prior_scale', 0.001, 0.5, log=True)
    seasonality_prior = trial.suggest_float('seasonality_prior_scale', 0.01, 20.0, log=True)
    holidays_prior = trial.suggest_float('holidays_prior_scale', 0.01, 20.0, log=True)
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, seasonality_mode='multiplicative', changepoint_prior_scale=changepoint_prior, seasonality_prior_scale=seasonality_prior, holidays_prior_scale=holidays_prior, interval_width=0.95)
    try:
        model.fit(train_df)
        future = model.make_future_dataframe(periods=len(val_df))
        forecast = model.predict(future)
        pred = forecast['yhat'].values[-len(val_df):]
        actual = val_df['y'].values
        mape = mean_absolute_percentage_error(actual, pred)
        r2 = r2_score(actual, pred)
        score = mape - 0.1 * r2
        return score
    except Exception as e:
        logger.warning(f'Trial failed: {e}')
        return float('inf')

def tune_prophet(df, n_trials=50):
    logger.info(f'Starting Optuna tuning with {n_trials} trials...')
    prophet_df = df[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
    split = int(len(prophet_df) * 0.8)
    train_df = prophet_df.iloc[:split]
    val_df = prophet_df.iloc[split:]
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective_prophet(trial, train_df, val_df), n_trials=n_trials, n_jobs=1)
    best_params = study.best_params
    logger.info(f'Best Prophet params: {best_params}')
    return best_params

def tune_xgb(df, n_trials=30):
    from xgboost import XGBRegressor
    from sklearn.preprocessing import RobustScaler
    logger.info(f'Starting XGBoost tuning with {n_trials} trials...')
    from app.models.ensemble_forecaster import prepare_features, FEATURE_COLS
    feat_df = prepare_features(df.copy())
    feat_df = feat_df.dropna()
    split = int(len(feat_df) * 0.8)
    train_df = feat_df.iloc[:split]
    test_df = feat_df.iloc[split:]
    scaler = RobustScaler()
    X_train = scaler.fit_transform(train_df[FEATURE_COLS])
    X_test = scaler.transform(test_df[FEATURE_COLS])
    y_train = train_df['sales'].values
    y_test = test_df['sales'].values

    def objective_xgb(trial):
        params = {'n_estimators': trial.suggest_int('n_estimators', 100, 1000), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True), 'max_depth': trial.suggest_int('max_depth', 3, 10), 'subsample': trial.suggest_float('subsample', 0.6, 1.0), 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0), 'min_child_weight': trial.suggest_int('min_child_weight', 1, 10), 'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0), 'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0), 'random_state': 42, 'n_jobs': -1, 'verbosity': 0}
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        return mape - 0.1 * r2
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_xgb, n_trials=n_trials)
    best_params = study.best_params
    logger.info(f'Best XGB params: {best_params}')
    return best_params