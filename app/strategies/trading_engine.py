import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal
from ..risk.risk_manager import RiskManager

def _rsi(close: pd.Series, window: int=14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    return 100 - 100 / (1 + gain / (loss + 1e-10))

def _macd(close: pd.Series):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return (macd, signal)

def _bollinger(close: pd.Series, window: int=20, n_std: float=2.0):
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    return (mid + n_std * std, mid - n_std * std)

def rsi_signals(close: pd.Series, oversold: float=30, overbought: float=70) -> pd.Series:
    rsi = _rsi(close)
    sig = pd.Series(0, index=close.index)
    sig[rsi < oversold] = 1
    sig[rsi > overbought] = -1
    return sig

def macd_signals(close: pd.Series) -> pd.Series:
    macd, signal = _macd(close)
    sig = pd.Series(0, index=close.index)
    sig[(macd > signal) & (macd.shift(1) <= signal.shift(1))] = 1
    sig[(macd < signal) & (macd.shift(1) >= signal.shift(1))] = -1
    return sig

def bollinger_signals(close: pd.Series) -> pd.Series:
    upper, lower = _bollinger(close)
    sig = pd.Series(0, index=close.index)
    sig[close < lower] = 1
    sig[close > upper] = -1
    return sig

def ml_adaptive_signals(close: pd.Series, up_probs: pd.Series, buy_thresh: float=0.6, sell_thresh: float=0.4) -> pd.Series:
    sig = pd.Series(0, index=close.index)
    sig[up_probs >= buy_thresh] = 1
    sig[up_probs <= sell_thresh] = -1
    return sig

@dataclass
class BacktestResult:
    strategy: str
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    equity_curve: pd.Series
    signals: pd.Series

class TradingEngine:
    STRATEGIES = ('rsi', 'macd', 'bollinger', 'ml_adaptive', 'ensemble')

    def __init__(self, initial_capital: float=100000.0, stop_loss_pct: float=0.05, trailing_stop_pct: float=0.05):
        self.initial_capital = initial_capital
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.risk = RiskManager(initial_capital)

    def _generate_signals(self, close: pd.Series, strategy: str, up_probs: pd.Series | None=None) -> pd.Series:
        if strategy == 'rsi':
            return rsi_signals(close)
        if strategy == 'macd':
            return macd_signals(close)
        if strategy == 'bollinger':
            return bollinger_signals(close)
        if strategy == 'ml_adaptive':
            if up_probs is None:
                raise ValueError('up_probs required for ml_adaptive strategy')
            return ml_adaptive_signals(close, up_probs)
        if strategy == 'ensemble':
            combined = rsi_signals(close) + macd_signals(close) + bollinger_signals(close)
            sig = pd.Series(0, index=close.index)
            sig[combined >= 2] = 1
            sig[combined <= -2] = -1
            return sig
        raise ValueError(f'Unknown strategy: {strategy}. Choose from {self.STRATEGIES}')

    def _apply_stops(self, close: pd.Series, signals: pd.Series) -> pd.Series:
        signals = signals.copy()
        in_position = False
        entry_price = 0.0
        highest = 0.0
        for i, (price, sig) in enumerate(zip(close, signals)):
            if in_position:
                trailing_stop = highest * (1 - self.trailing_stop_pct)
                hard_stop = entry_price * (1 - self.stop_loss_pct)
                if price <= max(hard_stop, trailing_stop):
                    signals.iloc[i] = -1
                    in_position = False
                else:
                    highest = max(highest, price)
            if sig == 1 and (not in_position):
                in_position = True
                entry_price = price
                highest = price
            elif sig == -1 and in_position:
                in_position = False
        return signals

    def backtest(self, close: pd.Series, strategy: Literal['rsi', 'macd', 'bollinger', 'ml_adaptive', 'ensemble']='ensemble', up_probs: pd.Series | None=None) -> BacktestResult:
        raw_signals = self._generate_signals(close, strategy, up_probs)
        signals = self._apply_stops(close, raw_signals)
        positions = signals.shift(1).fillna(0)
        daily_ret = close.pct_change().fillna(0)
        strat_ret = positions * daily_ret
        equity = (1 + strat_ret).cumprod() * self.initial_capital
        ann_ret = self.risk.annualized_return(equity.values)
        sharpe = self.risk.sharpe_ratio(strat_ret.values)
        mdd = self.risk.max_drawdown(equity.values)
        trade_returns, in_pos, ep = ([], False, 0.0)
        for price, sig in zip(close, signals):
            if sig == 1 and (not in_pos):
                in_pos = True
                ep = price
            elif sig == -1 and in_pos:
                trade_returns.append((price - ep) / ep)
                in_pos = False
        if in_pos:
            trade_returns.append((close.iloc[-1] - ep) / ep)
        win_rate = float(np.mean([r > 0 for r in trade_returns])) if trade_returns else 0.0
        return BacktestResult(strategy=strategy, annualized_return=round(ann_ret * 100, 2), sharpe_ratio=round(sharpe, 4), max_drawdown=round(mdd * 100, 2), total_trades=len(trade_returns), win_rate=round(win_rate * 100, 2), equity_curve=equity, signals=signals)

    def compare_strategies(self, close: pd.Series, up_probs: pd.Series | None=None) -> pd.DataFrame:
        rows = []
        for strat in self.STRATEGIES:
            try:
                r = self.backtest(close, strat, up_probs)
                rows.append({'strategy': r.strategy, 'annualized_return_%': r.annualized_return, 'sharpe_ratio': r.sharpe_ratio, 'max_drawdown_%': r.max_drawdown, 'total_trades': r.total_trades, 'win_rate_%': r.win_rate})
            except Exception:
                pass
        return pd.DataFrame(rows).sort_values('sharpe_ratio', ascending=False)