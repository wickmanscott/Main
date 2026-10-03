import json

import pandas as pd

from watchlist.backtest.audit import audit, load_archive
from watchlist.data.universe import liquid_tickers, parse_nasdaq_listed, parse_other_listed

from conftest import make_panel

NASDAQ = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N
NIO|NIO Inc. - American Depositary Shares, each representing one Class A ordinary share|Q|N|N|100|N|N
QQQ|Invesco QQQ Trust, Series 1|G|N|N|100|Y|N
ZXZZT|NASDAQ TEST STOCK|G|Y|N|100|N|N
ABCDW|Some SPAC Corp - Warrant|G|N|N|100|N|N
ABCDU|Some SPAC Corp - Units|G|N|N|100|N|N
File Creation Time: 1002202618:01|||||||"""

OTHER = """ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
BRK.B|Berkshire Hathaway Inc. Class B|N|BRK.B|N|100|N|BRK=B
F|Ford Motor Company Common Stock|N|F|N|100|N|F
F$B|Ford Motor Co 6.20% Notes due 2059|N|F$B|N|100|N|F-B
SPY|SPDR S&P 500 ETF Trust|P|SPY|Y|100|N|SPY
File Creation Time: 1002202618:01||||||||"""


def test_symbol_directory_parsing():
    assert parse_nasdaq_listed(NASDAQ) == ["AAPL", "NIO"]
    assert parse_other_listed(OTHER) == ["BRK-B", "F"]


def test_liquidity_filter_orders_by_dollar_volume():
    panel, _ = make_panel(n_tickers=20, n_days=60, seed=2)
    out = liquid_tickers(panel, min_price=1.0, min_dollar_volume=0, max_n=5)
    dv = (panel.close * panel.volume).rolling(20).mean().iloc[-1]
    assert out == list(dv.sort_values(ascending=False).index[:5])


def test_archive_is_well_formed():
    a = load_archive()
    assert len(a["episodes"]) >= 11
    assert all(len(e["tickers"]) >= 5 for e in a["episodes"])
    assert all(pd.Timestamp(e["date"]) < pd.Timestamp("2021-01-01") for e in a["episodes"])


def test_audit_runs_on_synthetic_prices():
    panel, bench = make_panel(n_tickers=6, n_days=600, seed=4)
    names = panel.tickers
    d = panel.dates[400].strftime("%Y-%m-%d")
    archive = {"episodes": [{"date": d, "tickers": names[:5]}], "symbol_map": {names[0]: ["GONE", names[0]]}}
    df, summary = audit(archive, panel, bench)
    assert summary["with_data"] == 5
    assert set(summary["1w"]) >= {"hit_rate", "beat_spy_rate", "avg_excess_vs_spy"}
    assert "traits_winners_vs_losers_1m" in summary
