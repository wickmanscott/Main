import numpy as np
import pandas as pd

from watchlist.data.prices import PricePanel
from watchlist.data.social import SocialSnapshot, social_frame
from watchlist.scoring import Model, select_picks
from watchlist.signals.features import compute_features


def _trend_panel(n_up=8, n_flat=40, n_days=400, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    cols, data = [], []
    for i in range(n_up):
        cols.append(f"UP{i}")
        data.append(20 * np.exp(np.cumsum(rng.normal(0.003, 0.012, n_days))))
    for i in range(n_flat):
        cols.append(f"FL{i}")
        data.append(20 * np.exp(np.cumsum(rng.normal(-0.0005, 0.015, n_days))))
    close = pd.DataFrame(np.array(data).T, index=dates, columns=cols)
    vol = pd.DataFrame(rng.uniform(5e5, 2e6, close.shape), index=dates, columns=cols)
    return PricePanel(open=close * 0.998, high=close * 1.01, low=close * 0.99, close=close, volume=vol)


def _xs(cfg, panel, social=None):
    fs = compute_features(panel)
    m = Model.build(fs, cfg["weights"], 1.0, 1e5)
    return m, m.cross_section(fs.dates[-1], social=social, social_weight=cfg["social_weight"] if social is not None else 0)


def test_uptrending_stocks_rank_on_top(cfg):
    # The composite blends trend strength with setup quality, so a random walk on a hot
    # streak can outscore a true uptrend with a sloppy setup. The uptrends should still dominate.
    m, xs = _xs(cfg, _trend_panel())
    ups = [t for t in xs.index if t.startswith("UP")]
    assert sum(t.startswith("UP") for t in xs.index[:10]) >= 5
    assert all(xs.index.get_loc(t) < 15 for t in ups)
    assert xs.index[0].startswith("UP")


def test_select_picks_fills_slots_without_duplicates(cfg):
    m, xs = _xs(cfg, _trend_panel())
    picks = select_picks(xs, cfg, m.weights)
    assert len(picks) == 5
    assert len({p.ticker for p in picks}) == 5
    assert all(p.reasons for p in picks)
    assert [p.slot for p in picks][:3] == ["momentum"] * 3


def test_repeat_cap_and_sector_cap(cfg):
    m, xs = _xs(cfg, _trend_panel())
    leader = xs.index[0]
    picks = select_picks(xs, cfg, m.weights, streaks={leader: cfg["selection"]["max_repeat_weeks"]})
    assert leader not in {p.ticker for p in picks}

    sectors = {t: "Tech" for t in xs.index}
    picks = select_picks(xs, cfg, m.weights, sectors=sectors)
    assert len(picks) == cfg["selection"]["max_per_sector"]


def test_extended_names_are_skipped(cfg):
    m, xs = _xs(cfg, _trend_panel())
    xs = xs.copy()
    leader = xs.index[0]
    xs.loc[leader, "extension"] = 0.6
    picks = select_picks(xs, cfg, m.weights)
    assert leader not in {p.ticker for p in picks if p.slot == "momentum"}


def test_social_slot_uses_buzz(cfg):
    panel = _trend_panel()
    m, xs0 = _xs(cfg, panel)
    # A mid-ranked name (not a top-3 leader even after the social boost) that passes the gates.
    sel = cfg["selection"]
    buzz = next(t for t in xs0.index[10:] if xs0.loc[t, "composite"] >= xs0["composite"].median()
                and xs0.loc[t, "trend_template"] >= sel["min_trend_template"] and xs0.loc[t, "extension"] <= sel["max_extension"])
    snap = SocialSnapshot("2026-10-02T00:00:00", reddit={
        buzz: {"rank": 3, "rank_24h_ago": 40, "mentions": 400, "mentions_24h_ago": 50, "upvotes": 900},
        "FL0": {"rank": 50, "rank_24h_ago": 45, "mentions": 20, "mentions_24h_ago": 25, "upvotes": 10},
    })
    social = social_frame(snap, panel.tickers)
    assert social.loc[buzz, "social_score"] > social.loc["FL0", "social_score"]
    _, xs = _xs(cfg, panel, social)
    picks = select_picks(xs, cfg, m.weights)
    social_picks = [p for p in picks if p.slot == "social"]
    assert [p.ticker for p in social_picks] == [buzz]
    assert any("Reddit" in r for r in social_picks[0].reasons)
