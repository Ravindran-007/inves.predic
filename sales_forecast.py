import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime, timedelta

def get_live_sales_data():
    rng = pd.date_range(datetime(2020, 1, 1), periods=60, freq='M')
    sales = np.random.randint(1000, 5000, size=len(rng))
    return pd.DataFrame({'ds': rng, 'y': sales})

def forecast_sales(periods=12):
    df = get_live_sales_data()
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq='M')
    forecast = model.predict(future)
    return (forecast, df)