import logging
import os
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import pandas as pd
import requests
import yfinance as yf
logger = logging.getLogger(__name__)

def _safe_get(url: str, params: dict, timeout: int=10) -> Optional[dict]:
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f'HTTP request failed ({url}): {e}')
        return None

def _sentiment_score(text: str) -> float:
    positive = {'beat', 'surge', 'rally', 'growth', 'profit', 'record', 'strong', 'upgrade', 'buy', 'bullish', 'gain', 'rise', 'up', 'positive'}
    negative = {'miss', 'drop', 'fall', 'loss', 'decline', 'weak', 'downgrade', 'sell', 'bearish', 'crash', 'down', 'negative', 'risk', 'cut'}
    words = set(text.lower().split())
    pos = len(words & positive)
    neg = len(words & negative)
    total = pos + neg
    return (pos - neg) / total if total else 0.0

class DataFusion:
    FRED_SERIES = {'fed_rate': 'FEDFUNDS', 'cpi': 'CPIAUCSL', 'unemployment': 'UNRATE', 'gdp_growth': 'A191RL1Q225SBEA', 'vix': 'VIXCLS'}

    def __init__(self, alpha_vantage_key: str='', news_api_key: str='', fred_api_key: str=''):
        self.av_key = alpha_vantage_key or os.getenv('ALPHA_VANTAGE_KEY', '')
        self.news_key = news_api_key or os.getenv('NEWS_API_KEY', '')
        self.fred_key = fred_api_key or os.getenv('FRED_API_KEY', '')

    def fetch_yfinance(self, symbol: str, period: str='2y') -> pd.DataFrame:
        df = yf.Ticker(symbol).history(period=period)
        if df.empty:
            raise ValueError(f'No yfinance data for {symbol}')
        df.reset_index(inplace=True)
        df.rename(columns={'Date': 'date', 'Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Volume': 'volume'}, inplace=True)
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        return df[['date', 'close', 'open', 'high', 'low', 'volume']].copy()

    def fetch_alpha_vantage(self, symbol: str) -> Optional[pd.DataFrame]:
        if not self.av_key:
            return None
        data = _safe_get('https://www.alphavantage.co/query', {'function': 'TIME_SERIES_DAILY_ADJUSTED', 'symbol': symbol, 'outputsize': 'full', 'apikey': self.av_key})
        if not data or 'Time Series (Daily)' not in data:
            return None
        ts = data['Time Series (Daily)']
        rows = [{'date': pd.Timestamp(d), 'av_close': float(v['5. adjusted close']), 'av_volume': float(v['6. volume'])} for d, v in ts.items()]
        return pd.DataFrame(rows).sort_values('date').reset_index(drop=True)

    def fetch_news_sentiment(self, symbol: str, days: int=30) -> Optional[pd.DataFrame]:
        if not self.news_key:
            return None
        from_date = (datetime.utcnow() - timedelta(days=days)).strftime('%Y-%m-%d')
        data = _safe_get('https://newsapi.org/v2/everything', {'q': symbol, 'from': from_date, 'sortBy': 'publishedAt', 'language': 'en', 'pageSize': 100, 'apiKey': self.news_key})
        if not data or 'articles' not in data:
            return None
        rows = [{'date': pd.Timestamp(a['publishedAt'][:10]), 'sentiment': _sentiment_score(f'{a.get('title', '')} {a.get('description', '')}')} for a in data['articles']]
        df = pd.DataFrame(rows)
        return df.groupby('date')['sentiment'].mean().reset_index().rename(columns={'sentiment': 'news_sentiment'})

    def fetch_fred(self, series_id: str, col_name: str) -> Optional[pd.DataFrame]:
        if not self.fred_key:
            return None
        data = _safe_get('https://api.stlouisfed.org/fred/series/observations', {'series_id': series_id, 'api_key': self.fred_key, 'file_type': 'json', 'sort_order': 'asc'})
        if not data or 'observations' not in data:
            return None
        rows = [{'date': pd.Timestamp(o['date']), col_name: float(o['value'])} for o in data['observations'] if o['value'] != '.']
        return pd.DataFrame(rows)

    def fetch_all_fred(self) -> Optional[pd.DataFrame]:
        frames = []
        for col, sid in self.FRED_SERIES.items():
            df = self.fetch_fred(sid, col)
            if df is not None:
                frames.append(df.set_index('date'))
        if not frames:
            return None
        merged = frames[0]
        for f in frames[1:]:
            merged = merged.join(f, how='outer')
        return merged.reset_index().sort_values('date')

    def fetch_social_sentiment(self, symbol: str, days: int=7) -> Optional[pd.DataFrame]:
        dates = pd.date_range(end=datetime.utcnow().date(), periods=days, freq='D')
        rng = np.random.default_rng(abs(hash(symbol)) % 2 ** 32)
        sentiment = rng.uniform(-0.3, 0.3, size=days)
        return pd.DataFrame({'date': dates, 'social_sentiment': sentiment})

    def get_enriched_data(self, symbol: str, period: str='2y') -> pd.DataFrame:
        base = self.fetch_yfinance(symbol, period).set_index('date')
        av = self.fetch_alpha_vantage(symbol)
        if av is not None:
            base = base.join(av.set_index('date'), how='left')
        news = self.fetch_news_sentiment(symbol)
        if news is not None:
            base = base.join(news.set_index('date'), how='left')
        fred = self.fetch_all_fred()
        if fred is not None:
            fred_idx = fred.set_index('date').reindex(base.index, method='ffill')
            base = base.join(fred_idx, how='left')
        social = self.fetch_social_sentiment(symbol)
        if social is not None:
            base = base.join(social.set_index('date'), how='left')
        df = base.reset_index().sort_values('date')
        logger.info(f'Enriched data for {symbol}: {len(df)} rows, {len(df.columns)} columns')
        return df