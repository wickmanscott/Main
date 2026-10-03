"""Raw features for every ticker on every date, computed once from a PricePanel."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.prices import PricePanel
from . import indicators as ind


@dataclass
class FeatureSet:
    raw: dict[str, pd.DataFrame]

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.raw[name]

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.raw["close"].index

    @property
    def tickers(self) -> list[str]:
        return list(self.raw["close"].columns)

    def row(self, date) -> pd.DataFrame:
        """All raw features for one date: tickers x features."""
        date = pd.Timestamp(date)
        return pd.DataFrame({k: v.loc[date] for k, v in self.raw.items()})


def compute_features(panel: PricePanel, benchmark: pd.Series | None = None) -> FeatureSet:
    # float32 halves memory: ~40 frames x 3000 tickers x 1500 days adds up fast.
    c, h, l, v = (getattr(panel, k).astype("float32") for k in ("close", "high", "low", "volume"))
    f: dict[str, pd.DataFrame] = {"close": c}

    # Momentum
    r21, r63, r126, r189, r252 = (ind.pct_return(c, n) for n in (21, 63, 126, 189, 252))
    f["ret_1w"] = ind.pct_return(c, 5)
    f["ret_1m"], f["ret_3m"], f["ret_6m"] = r21, r63, r126
    # IBD-style relative strength: the most recent quarter counts double.
    f["rs_raw"] = 0.4 * r63 + 0.2 * r126.fillna(r63) + 0.2 * r189.fillna(r126).fillna(r63) + 0.2 * r252.fillna(r189).fillna(r126).fillna(r63)
    f["mom_12_1"] = c.shift(21) / c.shift(252) - 1
    f["mom_12_1"] = f["mom_12_1"].fillna(c.shift(21) / c.shift(126) - 1)  # younger stocks: 6-1
    f["accel"] = r21 - r63 / 3

    # 52-week range
    high_52w = h.rolling(252, min_periods=126).max()
    low_52w = l.rolling(252, min_periods=126).min()
    f["high_52w"], f["low_52w"] = high_52w, low_52w
    f["prox_52w_high"] = c / high_52w

    # Trend template (Minervini stage-2 checklist), fraction of 8 conditions met.
    sma20, sma50, sma150 = ind.sma(c, 20), ind.sma(c, 50), ind.sma(c, 150, min_periods=120)
    sma200 = ind.sma(c, 200, min_periods=150)
    f["sma20"], f["sma50"], f["sma150"], f["sma200"] = sma20, sma50, sma150, sma200
    checks = [
        c > sma50,
        c > sma150,
        c > sma200,
        sma50 > sma150,
        sma150 > sma200,
        sma200 > sma200.shift(21),
        c >= 1.3 * low_52w,
        c >= 0.75 * high_52w,
    ]
    tt = sum(chk.astype(float) for chk in checks) / len(checks)
    f["trend_template"] = tt.where(sma200.notna() & high_52w.notna())

    # Oscillators / volume / volatility
    f["rsi"] = ind.rsi(c, 14)
    f["rsi_sweet"] = (1 - (f["rsi"] - 65).abs() / 30).clip(0, 1)
    avg_vol50 = v.rolling(50, min_periods=30).mean()
    f["rel_volume"] = v.rolling(5, min_periods=3).mean() / avg_vol50
    f["ud_volume"] = ind.up_down_volume_ratio(c, v, 50)
    atr10, atr14, atr50 = (ind.atr(h, l, c, n) for n in (10, 14, 50))
    f["atr14"] = atr14
    f["vcp"] = atr10 / atr50
    f["tightness"] = -f["vcp"]
    pivot = h.rolling(20, min_periods=20).max().shift(1)
    f["pivot_20d"] = pivot
    daily_breakout = ((c > pivot) & (v > 1.3 * avg_vol50)).astype(float)
    f["breakout"] = daily_breakout.rolling(5, min_periods=1).max().where(pivot.notna())
    f["extension"] = c / sma20 - 1
    f["dollar_volume"] = (c * v).rolling(20, min_periods=15).mean()
    f["support_10d"] = l.rolling(10, min_periods=5).min()

    # Relative strength line vs the benchmark making a new high.
    if benchmark is not None:
        bench = benchmark.reindex(c.index).ffill()
        rs_line = c.div(bench, axis=0)
        f["rs_line_high"] = (rs_line >= 0.98 * rs_line.rolling(252, min_periods=126).max()).astype(float).where(rs_line.notna())
        f["excess_3m"] = r63.sub(bench / bench.shift(63) - 1, axis=0)
    else:
        f["rs_line_high"] = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
        f["excess_3m"] = pd.DataFrame(np.nan, index=c.index, columns=c.columns)

    return FeatureSet(raw=f)


def forward_returns(panel: PricePanel, horizon: int) -> pd.DataFrame:
    """Return earned by a listener who buys at the next session's open after date t and
    sells at the close `horizon` sessions later: close[t+h] / open[t+1] - 1.

    Indexed by the signal date t. NaN at the end of the sample.
    """
    entry = panel.open.shift(-1)
    exit_ = panel.close.shift(-horizon)
    return exit_ / entry - 1
