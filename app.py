# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf
from prophet import Prophet
from strategies import (
    moving_average_strategy,
    bollinger_bands_strategy,
    rsi_strategy,
    backtest,
)

st.set_page_config(page_title="Investment & Sales Prediction Dashboard", layout="wide")
st.title("📊 Investment & Sales Prediction Dashboard")

# --- Sidebar Menu ---
option = st.sidebar.radio("Choose Prediction Type", ["Sales", "Investment"])

# ================== SALES FORECAST ==================
if option == "Sales":
    st.header("📉 Sales Forecast")
    ticker = st.text_input("Enter stock ticker (proxy for sales)", "AAPL")
    period = st.selectbox("Select history period", ["6mo", "1y", "2y", "5y"])
    forecast_days = st.slider("Forecast days", 30, 365, 90)
    
    data = yf.download(ticker, period=period, auto_adjust=False)
    
    # Handle potential MultiIndex columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns.values]
    
    if "Close" in data.columns:
        close = data["Close"].dropna()
    elif "Adj Close" in data.columns:
        close = data["Adj Close"].dropna()
    else:
        st.error("No valid price column found (Close/Adj Close).")
        st.stop()
    
    hist_df = pd.DataFrame({"ds": close.index, "y": close.to_numpy().ravel()})
    
    # Prophet forecast
    model = Prophet()
    model.fit(hist_df)
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    
    # Separate historical and future data for different colors
    historical_forecast = forecast[forecast["ds"] <= hist_df["ds"].max()]
    future_forecast = forecast[forecast["ds"] > hist_df["ds"].max()]
    
    # Create the plot
    fig1 = px.line(title=f"{ticker} Sales Forecast")
    
    # Add actual historical data (blue)
    fig1.add_scatter(
        x=hist_df["ds"], 
        y=hist_df["y"], 
        mode="lines+markers", 
        name="Actual Sales",
        line=dict(color="blue"),
        marker=dict(color="blue")
    )
    
    # Add historical forecast (light blue/gray for validation)
    fig1.add_scatter(
        x=historical_forecast["ds"], 
        y=historical_forecast["yhat"], 
        mode="lines", 
        name="Historical Fit",
        line=dict(color="lightblue", dash="dot")
    )
    
    # Add future predictions (orange/red)
    if not future_forecast.empty:
        fig1.add_scatter(
            x=future_forecast["ds"], 
            y=future_forecast["yhat"], 
            mode="lines", 
            name="Predicted Sales",
            line=dict(color="orange", width=3)
        )
        
        # Add confidence intervals for future predictions
        fig1.add_scatter(
            x=future_forecast["ds"],
            y=future_forecast["yhat_upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        )
        
        fig1.add_scatter(
            x=future_forecast["ds"],
            y=future_forecast["yhat_lower"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(255, 165, 0, 0.2)",
            name="Prediction Confidence",
            hoverinfo="skip"
        )
    
    # Update layout
    fig1.update_layout(
        xaxis_title="Date",
        yaxis_title="Price/Sales Value",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig1, use_container_width=True)

# ================== INVESTMENT STRATEGY ==================
elif option == "Investment":
    st.header("📈 Stock Market Strategy Backtest")
    ticker = st.text_input("Enter stock symbol", "AAPL")
    strategy = st.selectbox("Select Trading Strategy", ["Moving Average", "Bollinger Bands", "RSI"])
    period = st.selectbox("Select history period", ["6mo", "1y", "2y", "5y"])
    
    df = yf.download(ticker, period=period, auto_adjust=False)
    
    # Handle potential MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns.values]
    
    # Determine which price column to use and ensure it's named "Close"
    if "Close" in df.columns:
        price_col = "Close"
    elif "Adj Close" in df.columns:
        price_col = "Adj Close"
        # Rename to "Close" for consistency with strategies
        df = df.rename(columns={"Adj Close": "Close"})
    else:
        st.error("No valid price column found (Close/Adj Close).")
        st.stop()
    
    # Clean the data
    df = df.dropna()
    
    if df.empty:
        st.error("No data available after cleaning. Please try a different ticker or period.")
        st.stop()
    
    # Apply chosen strategy
    if strategy == "Moving Average":
        df = moving_average_strategy(df.copy())
    elif strategy == "Bollinger Bands":
        df = bollinger_bands_strategy(df.copy())
    elif strategy == "RSI":
        df = rsi_strategy(df.copy())
    
    # Backtest
    equity, net, stats = backtest(df["Close"], df["Signal"])
    
    # Plot equity curve
    equity_df = pd.DataFrame({
        "Date": equity.index,
        "Equity": equity.values
    })
    
    fig2 = px.line(equity_df, x="Date", y="Equity", title=f"{ticker} Strategy Equity Curve")
    st.plotly_chart(fig2, use_container_width=True)
    
    # Show stats
    st.subheader("📌 Strategy Statistics")
    st.json(stats)