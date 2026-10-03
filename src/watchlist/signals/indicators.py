"""Vectorized indicators. Inputs are wide frames (dates x tickers) or Series.

Every function only looks backward (rolling windows, shifts >= 0), so values at
date t never use data after t. The backtester relies on that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(df: pd.DataFrame, n: int, min_periods: int | None = None) -> pd.DataFrame:
    return df.rolling(n, min_periods=min_periods or n).mean()


def pct_return(close: pd.DataFrame, n: int) -> pd.DataFrame:
    return close / close.shift(n) - 1


def rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    # No losses in the window -> RSI 100.
    return out.where(avg_loss != 0, 100.0).where(avg_gain.notna())


def true_range(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    prev = close.shift(1)
    a = (high - low).abs()
    b = (high - prev).abs()
    c = (low - prev).abs()
    return np.maximum(np.maximum(a, b.fillna(a)), c.fillna(a))


def atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    return true_range(high, low, close).rolling(n, min_periods=n).mean()


def up_down_volume_ratio(close: pd.DataFrame, volume: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    """Volume on up days / volume on down days over n days. > 1 suggests accumulation."""
    change = close.diff()
    up_vol = volume.where(change > 0, 0.0).rolling(n, min_periods=n // 2).sum()
    down_vol = volume.where(change < 0, 0.0).rolling(n, min_periods=n // 2).sum()
    ratio = up_vol / down_vol.replace(0, np.nan)
    return ratio.clip(upper=5.0).where(up_vol.notna())
