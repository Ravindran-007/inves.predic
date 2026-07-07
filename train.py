
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.getcwd())

os.makedirs('models', exist_ok=True)

np.random.seed(42)
dates = pd.date_range(start='2020-01-01', end='2025-12-31', freq='D')
trend = np.linspace(5000, 15000, len(dates))
seasonality = 2000 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365)
sales = trend + seasonality + np.random.randn(len(dates)) * 500
sales = np.maximum(sales, 100)

df = pd.DataFrame({'date': dates, 'sales': sales})

from app.models.sales_forecast import SalesForecaster

model = SalesForecaster()
model.fit(df)
