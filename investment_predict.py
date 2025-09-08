# investment_predict.py
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def get_stock_data(ticker="AAPL", start="2020-01-01"):
    df = yf.download(ticker, start=start)
    df.reset_index(inplace=True)
    return df

def predict_investment(ticker="AAPL"):
    df = get_stock_data(ticker)

    df["Target"] = df["Close"].shift(-1)  # predict next day's close
    df = df.dropna()

    X = df[["Open", "High", "Low", "Close", "Volume"]]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = XGBRegressor(objective="reg:squarederror", n_estimators=100)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return preds, y_test.values, rmse, df
