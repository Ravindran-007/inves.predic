# strategies.py
import pandas as pd
import numpy as np

# --- Helper: ensure valid close price ---
def get_close(df):
    if "Close" in df.columns:
        return df["Close"].copy()
    elif "Adj Close" in df.columns:
        return df["Adj Close"].copy()
    else:
        raise KeyError("No valid price column found (Close/Adj Close).")


# --- Moving Average Strategy ---
def moving_average_strategy(df, short_window=20, long_window=50):
    close = get_close(df)

    df["SMA_Short"] = close.rolling(window=short_window, min_periods=1).mean()
    df["SMA_Long"] = close.rolling(window=long_window, min_periods=1).mean()

    df["Signal"] = 0
    df.loc[df["SMA_Short"] > df["SMA_Long"], "Signal"] = 1
    df.loc[df["SMA_Short"] < df["SMA_Long"], "Signal"] = -1

    return df


# --- Bollinger Bands Strategy ---
def bollinger_bands_strategy(df, window=20, num_std=2):
    close = get_close(df)

    rolling_mean = close.rolling(window=window, min_periods=1).mean()
    rolling_std = close.rolling(window=window, min_periods=1).std()

    df["Bollinger_Upper"] = rolling_mean + (rolling_std * num_std)
    df["Bollinger_Lower"] = rolling_mean - (rolling_std * num_std)

    df["Signal"] = 0
    df.loc[close < df["Bollinger_Lower"], "Signal"] = 1   # Buy
    df.loc[close > df["Bollinger_Upper"], "Signal"] = -1  # Sell

    return df


# --- RSI Strategy ---
def rsi_strategy(df, window=14):
    close = get_close(df)

    delta = close.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["Signal"] = 0
    df.loc[df["RSI"] < 30, "Signal"] = 1   # Buy
    df.loc[df["RSI"] > 70, "Signal"] = -1  # Sell

    return df


# --- Backtest ---
def backtest(prices, signals, initial_capital=10000):
    """
    Simple backtest:
    - Buy when Signal = 1
    - Sell when Signal = -1
    - Hold otherwise
    """
    prices = prices.ffill().bfill()
    signals = signals.fillna(0)

    positions = signals.shift(1).fillna(0)  # lag to avoid lookahead
    returns = prices.pct_change().fillna(0)

    equity = (1 + positions * returns).cumprod() * initial_capital
    net_profit = equity.iloc[-1] - initial_capital

    stats = {
        "Initial Capital": initial_capital,
        "Final Equity": round(float(equity.iloc[-1]), 2),
        "Net Profit": round(float(net_profit), 2),
        "Return (%)": round((net_profit / initial_capital) * 100, 2),
        "Max Equity": round(float(equity.max()), 2),
        "Min Equity": round(float(equity.min()), 2),
    }

    return equity, net_profit, stats
# strategies.py
import pandas as pd
import numpy as np

# --- Helper: ensure valid close price ---
def get_close(df):
    if "Close" in df.columns:
        return df["Close"].copy()
    elif "Adj Close" in df.columns:
        return df["Adj Close"].copy()
    else:
        raise KeyError("No valid price column found (Close/Adj Close).")


# --- Moving Average Strategy ---
def moving_average_strategy(df, short_window=20, long_window=50):
    close = get_close(df)

    df["SMA_Short"] = close.rolling(window=short_window, min_periods=1).mean()
    df["SMA_Long"] = close.rolling(window=long_window, min_periods=1).mean()

    df["Signal"] = 0
    df.loc[df["SMA_Short"] > df["SMA_Long"], "Signal"] = 1
    df.loc[df["SMA_Short"] < df["SMA_Long"], "Signal"] = -1

    return df


# --- Bollinger Bands Strategy ---
def bollinger_bands_strategy(df, window=20, num_std=2):
    close = get_close(df)

    rolling_mean = close.rolling(window=window, min_periods=1).mean()
    rolling_std = close.rolling(window=window, min_periods=1).std()

    df["Bollinger_Upper"] = rolling_mean + (rolling_std * num_std)
    df["Bollinger_Lower"] = rolling_mean - (rolling_std * num_std)

    df["Signal"] = 0
    df.loc[close < df["Bollinger_Lower"], "Signal"] = 1   # Buy
    df.loc[close > df["Bollinger_Upper"], "Signal"] = -1  # Sell

    return df


# --- RSI Strategy ---
def rsi_strategy(df, window=14):
    close = get_close(df)

    delta = close.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["Signal"] = 0
    df.loc[df["RSI"] < 30, "Signal"] = 1   # Buy
    df.loc[df["RSI"] > 70, "Signal"] = -1  # Sell

    return df


# --- Backtest ---
def backtest(prices, signals, initial_capital=10000):
    """
    Simple backtest:
    - Buy when Signal = 1
    - Sell when Signal = -1
    - Hold otherwise
    """
    prices = prices.ffill().bfill()
    signals = signals.fillna(0)

    positions = signals.shift(1).fillna(0)  # lag to avoid lookahead
    returns = prices.pct_change().fillna(0)

    equity = (1 + positions * returns).cumprod() * initial_capital
    net_profit = equity.iloc[-1] - initial_capital

    stats = {
        "Initial Capital": initial_capital,
        "Final Equity": round(float(equity.iloc[-1]), 2),
        "Net Profit": round(float(net_profit), 2),
        "Return (%)": round((net_profit / initial_capital) * 100, 2),
        "Max Equity": round(float(equity.max()), 2),
        "Min Equity": round(float(equity.min()), 2),
    }

    return equity, net_profit, stats
