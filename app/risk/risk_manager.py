import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Position:
    symbol: str
    entry_price: float
    shares: float
    stop_loss: float
    trailing_stop_pct: float = 0.05
    highest_price: float = field(init=False)

    def __post_init__(self):
        self.highest_price = self.entry_price

    def update_trailing_stop(self, current_price: float) -> float:
        if current_price > self.highest_price:
            self.highest_price = current_price
        return self.highest_price * (1 - self.trailing_stop_pct)

    def is_stopped_out(self, current_price: float) -> bool:
        trailing = self.update_trailing_stop(current_price)
        return current_price <= self.stop_loss or current_price <= trailing

class RiskManager:

    def __init__(self, portfolio_value: float=100000.0, max_risk_pct: float=0.02):
        self.portfolio_value = portfolio_value
        self.max_risk_pct = max_risk_pct
        self.positions: dict[str, Position] = {}

    def historical_var(self, returns: np.ndarray, confidence: float=0.95) -> float:
        return float(-np.percentile(returns, (1 - confidence) * 100))

    def parametric_var(self, returns: np.ndarray, confidence: float=0.95) -> float:
        from scipy.stats import norm
        mu, sigma = (returns.mean(), returns.std())
        return float(-(mu + norm.ppf(1 - confidence) * sigma))

    def var_report(self, returns: np.ndarray) -> dict:
        r = np.asarray(returns)
        return {'var_95_hist': round(self.historical_var(r, 0.95) * self.portfolio_value, 2), 'var_99_hist': round(self.historical_var(r, 0.99) * self.portfolio_value, 2), 'var_95_param': round(self.parametric_var(r, 0.95) * self.portfolio_value, 2), 'var_99_param': round(self.parametric_var(r, 0.99) * self.portfolio_value, 2)}

    def kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float, half_kelly: bool=True) -> float:
        if avg_loss == 0:
            return 0.0
        ratio = avg_win / avg_loss
        f = (ratio * win_rate - (1 - win_rate)) / ratio
        f = max(0.0, f)
        if half_kelly:
            f /= 2
        return min(f, self.max_risk_pct * 5)

    def position_size(self, entry_price: float, stop_price: float, kelly_f: Optional[float]=None) -> float:
        risk_amount = self.portfolio_value * (kelly_f or self.max_risk_pct)
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share == 0:
            return 0.0
        return round(risk_amount / risk_per_share, 4)

    def open_position(self, symbol: str, entry_price: float, shares: float, stop_loss_pct: float=0.05, trailing_stop_pct: float=0.05) -> Position:
        stop = entry_price * (1 - stop_loss_pct)
        pos = Position(symbol, entry_price, shares, stop, trailing_stop_pct)
        self.positions[symbol] = pos
        return pos

    def check_stops(self, prices: dict[str, float]) -> list[str]:
        triggered = [sym for sym, pos in self.positions.items() if pos.is_stopped_out(prices.get(sym, pos.entry_price))]
        for sym in triggered:
            del self.positions[sym]
        return triggered

    def max_drawdown(self, equity: np.ndarray) -> float:
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / (peak + 1e-10)
        return float(-dd.min())

    def sharpe_ratio(self, returns: np.ndarray, risk_free: float=0.05) -> float:
        excess = returns - risk_free / 252
        return float(np.mean(excess) / (np.std(excess) + 1e-10) * np.sqrt(252))

    def annualized_return(self, equity: np.ndarray, trading_days: int=252) -> float:
        if len(equity) < 2 or equity[0] == 0:
            return 0.0
        total = equity[-1] / equity[0]
        years = len(equity) / trading_days
        return float(total ** (1 / years) - 1)

    def full_report(self, returns: np.ndarray, equity: np.ndarray) -> dict:
        r = np.asarray(returns)
        eq = np.asarray(equity)
        return {**self.var_report(r), 'sharpe_ratio': round(self.sharpe_ratio(r), 4), 'max_drawdown_pct': round(self.max_drawdown(eq) * 100, 2), 'annualized_return': round(self.annualized_return(eq) * 100, 2), 'open_positions': len(self.positions)}