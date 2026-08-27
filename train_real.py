import pandas as pd
import yfinance as yf
import os
symbol = 'AAPL'
ticker = yf.Ticker(symbol)
df = ticker.history(period='2y')
df.reset_index(inplace=True)
df.rename(columns={'Date': 'date', 'Close': 'sales'}, inplace=True)
df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
from app.models.sales_forecast import SalesForecaster
model = SalesForecaster()
model.fit(df[['date', 'sales']])