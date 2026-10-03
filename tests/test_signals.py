import numpy as np
import pandas as pd

from watchlist.signals import indicators as ind
from watchlist.signals.features import compute_features, forward_returns

from conftest import make_panel


def test_rsi_bounds_and_monotonic_series():
    up = pd.DataFrame({"A": np.arange(1, 60, dtype=float)})
    assert ind.rsi(up)["A"].iloc[-1] == 100.0
    down = pd.DataFrame({"A": np.arange(60, 1, -1, dtype=float)})
    assert ind.rsi(down)["A"].iloc[-1] < 1.0


def test_features_never_look_ahead():
    """Changing the future must not change any feature value today."""
    panel, bench = make_panel(n_tickers=10, n_days=400, seed=3)
    cut = panel.dates[300]
    fs_full = compute_features(panel, bench.close["SPY"])

    shocked = panel.asof(panel.dates[-1])
    for f in ("open", "high", "low", "close", "volume"):
        df = getattr(shocked, f).copy()
        df.loc[df.index > cut] *= 3.0
        setattr(shocked, f, df)
    fs_shocked = compute_features(shocked, bench.close["SPY"])

    for name in fs_full.raw:
        a = fs_full[name].loc[:cut]
        b = fs_shocked[name].loc[:cut]
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-5, obj=name)


def test_forward_returns_enter_next_open():
    panel, _ = make_panel(n_tickers=3, n_days=50, seed=1)
    fwd = forward_returns(panel, 5)
    d = panel.dates[10]
    expected = panel.close["T000"].iloc[15] / panel.open["T000"].iloc[11] - 1
    assert np.isclose(fwd.at[d, "T000"], expected)
    assert fwd.iloc[-1].isna().all()


def test_trend_template_full_marks_for_steady_uptrend():
    dates = pd.bdate_range("2022-01-03", periods=320)
    close = pd.DataFrame({"UP": np.linspace(10, 40, 320)}, index=dates)
    from watchlist.data.prices import PricePanel

    p = PricePanel(open=close * 0.995, high=close * 1.01, low=close * 0.99, close=close, volume=close * 0 + 1e6)
    fs = compute_features(p)
    assert fs["trend_template"]["UP"].iloc[-1] == 1.0
