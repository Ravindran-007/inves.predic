import argparse
import logging
import os
import sys
import pandas as pd
import yfinance as yf
sys.path.insert(0, os.getcwd())
os.makedirs('models', exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)
from app.models.sales_forecast import SalesForecaster
from app.models.stock_predictor import StockPredictor
from app.mlops.mlops_pipeline import MLOpsPipeline

def fetch(symbol: str, period: str) -> pd.DataFrame:
    df = yf.Ticker(symbol).history(period=period)
    if df.empty:
        raise ValueError(f'No data returned for {symbol}')
    df.reset_index(inplace=True)
    df.rename(columns={'Date': 'date', 'Close': 'sales'}, inplace=True)
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    return df[['date', 'sales']].copy()

def train_sales(symbol: str, period: str, pipeline: MLOpsPipeline, tune: bool=False):
    logger.info(f'Training SalesForecaster for {symbol} (tune={tune})...')
    df = fetch(symbol, period)
    model = SalesForecaster(tune=tune)
    model.fit(df)
    pipeline.log_training_run(model_name=f'sales_{symbol}', params={'symbol': symbol, 'period': period, 'model': 'Prophet'}, metrics=model.performance)
    pipeline.save_model(model, f'sales_{symbol}')
    logger.info(f'Sales model saved — MAPE: {model.performance['mape']:.2f}%')
    return model

def train_stock(symbol: str, period: str, pipeline: MLOpsPipeline):
    logger.info(f'Training StockPredictor for {symbol}...')
    predictor = StockPredictor()
    df = predictor.fetch_data(symbol, period=period)
    predictor.fit(df=df)
    pipeline.log_training_run(model_name=f'stock_{symbol}', params={'symbol': symbol, 'period': period, 'model': 'XGB+LGBM'}, metrics=predictor.performance)
    pipeline.save_model(predictor, f'stock_{symbol}')
    logger.info(f'Stock model saved — MAPE: {predictor.performance['mape']:.2f}% | Direction: {predictor.performance['direction_accuracy']:.1f}%')
    return predictor

def main():
    parser = argparse.ArgumentParser(description='Train prediction models')
    parser.add_argument('--symbols', nargs='+', default=['NVDA', 'AAPL', 'MSFT'])
    parser.add_argument('--period', default='2y')
    parser.add_argument('--tune', action='store_true')
    args = parser.parse_args()
    symbols = [s.strip().upper() for raw in args.symbols for s in raw.split(',') if s.strip()]
    pipeline = MLOpsPipeline(model_dir='models', tracking_uri='mlruns')
    for symbol in symbols:
        try:
            train_sales(symbol, args.period, pipeline, tune=args.tune)
        except Exception as e:
            logger.error(f'Sales training failed for {symbol}: {e}')
        try:
            train_stock(symbol, args.period, pipeline)
        except Exception as e:
            logger.error(f'Stock training failed for {symbol}: {e}')
    if symbols:
        logger.info('Creating generic ensemble models...')
        try:
            generic_sales = SalesForecaster(tune=args.tune)
            generic_sales.fit(fetch(symbols[0], args.period))
            pipeline.save_model(generic_sales, 'sales_ensemble')
            logger.info('✅ sales_ensemble.pkl created')
        except Exception as e:
            logger.error(f'Failed to create sales_ensemble: {e}')
        try:
            generic_stock = StockPredictor()
            df = generic_stock.fetch_data(symbols[0], period=args.period)
            generic_stock.fit(df=df)
            pipeline.save_model(generic_stock, 'stock_ensemble')
            logger.info('✅ stock_ensemble.pkl created')
        except Exception as e:
            logger.error(f'Failed to create stock_ensemble: {e}')
    logger.info('Training complete. Models saved to ./models/')
if __name__ == '__main__':
    main()