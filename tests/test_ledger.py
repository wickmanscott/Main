import numpy as np
import pandas as pd

from watchlist.data.prices import PricePanel
from watchlist.ledger import Ledger, scorecard, track_record
from watchlist.scoring import Pick


def _pick(t, slot="momentum"):
    return Pick(ticker=t, slot=slot, composite=0.8, composite_rank=1, sleeper=None, social_score=0.0, reasons=[])


def test_streaks_count_consecutive_episodes(tmp_path):
    led = Ledger(tmp_path / "picks.csv")
    led.append("2026-09-13", [_pick("AAA"), _pick("BBB")], {})
    led.append("2026-09-20", [_pick("AAA"), _pick("CCC")], {})
    led.append("2026-09-27", [_pick("AAA"), _pick("CCC"), _pick("DDD")], {})
    assert led.streaks() == {"AAA": 3, "CCC": 2, "DDD": 1}
    assert led.streaks(before="2026-09-27") == {"AAA": 2, "CCC": 1}
    # Re-running an episode replaces it rather than duplicating.
    led.append("2026-09-27", [_pick("AAA")], {})
    assert len(Ledger(tmp_path / "picks.csv").df) == 5


def test_scorecard_buys_next_open(tmp_path):
    dates = pd.bdate_range("2026-09-28", periods=6)  # Mon..Mon
    close = pd.DataFrame({"AAA": [10, 11, 12, 13, 14, 15.0]}, index=dates)
    opn = pd.DataFrame({"AAA": [9.5, 10, 11, 12, 13, 14.0]}, index=dates)
    panel = PricePanel(open=opn, high=close, low=close, close=close, volume=close * 0 + 1)
    spy = PricePanel(open=opn.rename(columns={"AAA": "SPY"}) * 0 + 100, high=close.rename(columns={"AAA": "SPY"}),
                     low=close.rename(columns={"AAA": "SPY"}), close=close.rename(columns={"AAA": "SPY"}) * 0 + 101,
                     volume=close.rename(columns={"AAA": "SPY"}))
    led = Ledger(tmp_path / "picks.csv")
    led.append("2026-09-27", [_pick("AAA")], {"AAA": 9.0})  # Sunday episode
    sc = scorecard(led, panel, spy)
    assert np.isclose(sc["picks"][0]["ret"], 15 / 9.5 - 1, atol=1e-4)  # Monday open -> latest close
    assert np.isclose(sc["benchmark_ret"], 0.01)
    assert sc["beat_benchmark"] == 1 and sc["best"] == "AAA"
    tr = track_record(led, panel, spy, sessions=5)
    assert tr["picks_scored"] == 1 and np.isclose(tr["avg_ret"], 14 / 9.5 - 1, atol=1e-4)
