import numpy as np

from watchlist.backtest.engine import prepare, rebalance_dates, run
from watchlist.backtest.optimize import optimize
from watchlist.report import backtest_markdown

from conftest import make_panel


def _small_opt(cfg):
    cfg["backtest"]["optimize"].update(train_weeks=40, test_weeks=13, candidates=25)
    return cfg


def test_backtest_finds_planted_momentum(cfg, synthetic):
    panel, bench = synthetic
    prep = prepare(panel, bench, cfg)
    res = run(prep, cfg)
    s = res.summary
    assert s["periods"] > 100
    assert s["picks"] == 5 * s["periods"]
    assert s["mean_ic"] > 0.02 and s["ic_tstat"] > 2
    assert "equity" in s


def test_no_signal_in_pure_noise(cfg):
    """With no persistent drift, the model must not find an edge (guards against look-ahead)."""
    panel, bench = make_panel(n_tickers=60, n_days=700, seed=11, persistence=0.0)
    prep = prepare(panel, bench, cfg)
    s = run(prep, cfg).summary
    assert abs(s["mean_ic"]) < 0.03
    assert abs(s["ic_tstat"]) < 3


def test_weekly_rebalance_dates_are_week_ends(synthetic):
    panel, _ = synthetic
    d = rebalance_dates(panel.dates, 5)
    assert all(x.weekday() == 4 for x in d[:-1])  # synthetic calendar has no holidays


def test_optimizer_walk_forward(cfg, synthetic):
    panel, bench = synthetic
    cfg = _small_opt(cfg)
    prep = prepare(panel, bench, cfg)
    opt = optimize(prep, cfg)
    assert np.isclose(sum(opt.weights.values()), 1.0, atol=1e-3)
    assert len(opt.folds) >= 3
    assert opt.oos_tuned["periods"] > 0 and opt.oos_default["periods"] > 0
    md = backtest_markdown({5: run(prep, cfg)}, opt, cfg)
    assert "out of sample" in md and "Survivorship bias" in md
