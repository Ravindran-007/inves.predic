import json
import time
import hashlib
import functools
import logging
from typing import Any, Callable
import pandas as pd
logger = logging.getLogger(__name__)

def cache_key(*args, **kwargs) -> str:
    raw = json.dumps({'args': args, 'kwargs': kwargs}, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()

def serialize_df(df: pd.DataFrame) -> str:
    return df.to_json(orient='records', date_format='iso')

def deserialize_df(data: str) -> pd.DataFrame:
    return pd.read_json(data, orient='records')

def timer(fn: Callable) -> Callable:

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(f'{fn.__qualname__} completed in {elapsed_ms:.1f}ms')
        return result
    return wrapper

def safe_float(value: Any, default: float=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default