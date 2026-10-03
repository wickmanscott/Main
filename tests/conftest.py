"""Synthetic market data with a planted, persistent drift so momentum has real signal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from watchlist.config import load_config
from watchlist.data.prices import PricePanel


def make_panel(n_tickers: int = 60, n_days: int = 900, seed: int = 0, persistence: float = 0.995):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    # Slowly-wandering expected return per ticker: trends persist, which is what momentum exploits.
    drift = np.zeros((n_days, n_tickers))
    drift[0] = rng.normal(0, 0.0015, n_tickers)
    for t in range(1, n_days):
        drift[t] = persistence * drift[t - 1] + rng.normal(0, 0.00015, n_tickers)
    rets = drift + rng.normal(0, 0.02, (n_days, n_tickers))
    close = 20 * np.exp(np.cumsum(rets, axis=0))
    open_ = close * np.exp(rng.normal(0, 0.004, close.shape))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.01, close.shape)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.01, close.shape)))
    base_vol = rng.uniform(2e5, 5e6, n_tickers)
    volume = base_vol * np.exp(rng.normal(0, 0.3, close.shape)) * (1 + 3 * np.clip(rets, 0, None) / 0.02 * 0.2)
    mk = lambda a: pd.DataFrame(a, index=dates, columns=tickers)
    panel = PricePanel(open=mk(open_), high=mk(high), low=mk(low), close=mk(close), volume=mk(volume))
    bench = pd.Series(400 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n_days))), index=dates, name="SPY")
    bench_panel = PricePanel(
        open=bench.to_frame() * 0.999, high=bench.to_frame() * 1.005, low=bench.to_frame() * 0.995,
        close=bench.to_frame(), volume=bench.to_frame() * 0 + 1e8,
    )
    return panel, bench_panel


@pytest.fixture(scope="session")
def synthetic():
    return make_panel()


@pytest.fixture()
def cfg(tmp_path):
    c = load_config()
    c["universe"]["min_price"] = 1.0
    c["universe"]["min_avg_dollar_volume"] = 1e5
    c["publish"].update(
        episodes_dir=str(tmp_path / "episodes"),
        site_dir=str(tmp_path / "docs"),
        ledger_path=str(tmp_path / "ledger" / "picks.csv"),
    )
    c["social"]["history_dir"] = str(tmp_path / "social")
    return c
