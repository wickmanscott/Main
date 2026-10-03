"""Walk-forward backtest of the exact live pick logic.

At each weekly rebalance date t the model sees only data up to t (features are
backward-looking and the cross-section is sliced at t). Picks are bought at the
next session's open and held `horizon` sessions, the way a listener would trade
after a weekend episode.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.prices import PricePanel
from ..scoring import Model, select_picks
from ..signals.features import FeatureSet, compute_features, forward_returns

PERIODS_PER_YEAR = 252


def rebalance_dates(dates: pd.DatetimeIndex, every: int = 5) -> pd.DatetimeIndex:
    """Last trading day of each week when every == 5, otherwise every n-th day."""
    if every == 5:
        s = dates.to_series()
        return pd.DatetimeIndex(s.groupby(dates.to_period("W-FRI")).max().values)
    return dates[::every]


@dataclass
class Prepared:
    panel: PricePanel
    bench: pd.Series                     # benchmark close
    features: FeatureSet
    model: Model
    dates: pd.DatetimeIndex              # rebalance dates with enough data
    fwd: dict[int, pd.DataFrame]         # horizon -> forward returns (signal date index)
    bench_fwd: dict[int, pd.Series]


def prepare(panel: PricePanel, bench_panel: PricePanel, cfg: dict, weights: dict | None = None,
            start=None, min_names: int = 20) -> Prepared:
    ucfg, bcfg = cfg["universe"], cfg["backtest"]
    bench_sym = bench_panel.tickers[0]
    bench_close = bench_panel.close[bench_sym]
    fs = compute_features(panel, bench_close)
    model = Model.build(fs, weights or cfg["weights"], ucfg["min_price"], ucfg["min_avg_dollar_volume"])
    horizons = sorted(set(bcfg["horizons"]) | {bcfg["primary_horizon"]})
    fwd = {h: forward_returns(panel, h) for h in horizons}
    bench_fwd = {h: forward_returns(bench_panel, h)[bench_sym] for h in horizons}

    dates = rebalance_dates(fs.dates, bcfg.get("rebalance_every", 5))
    enough = model.eligible.sum(axis=1)
    dates = dates[enough.reindex(dates).fillna(0).values >= min_names]
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    return Prepared(panel, bench_close, fs, model, dates, fwd, bench_fwd)


@dataclass
class BacktestResult:
    horizon: int
    picks: pd.DataFrame            # one row per pick per period
    periods: pd.DataFrame          # one row per period
    summary: dict = field(default_factory=dict)


def _spearman(a: pd.Series, b: pd.Series) -> float:
    ok = a.notna() & b.notna()
    if ok.sum() < 10:
        return np.nan
    return float(a[ok].rank().corr(b[ok].rank()))


def run(prep: Prepared, cfg: dict, horizon: int | None = None, weights: dict | None = None,
        dates: pd.DatetimeIndex | None = None) -> BacktestResult:
    """Replay the weekly episode selection over history and score every pick."""
    bcfg = cfg["backtest"]
    h = horizon or bcfg["primary_horizon"]
    cost = bcfg.get("cost_bps", 0) / 1e4
    model = prep.model
    if weights is not None and weights != model.weights:
        model = Model(prep.features, model.eligible, model.scores,
                      _composite(model, weights), model.sleeper, dict(weights))
    dates = prep.dates if dates is None else dates
    fwd, bfwd = prep.fwd[h], prep.bench_fwd[h]

    pick_rows, period_rows = [], []
    streaks: dict[str, int] = {}
    for d in dates:
        if pd.isna(bfwd.get(d, np.nan)):
            continue  # not enough future data to score this period
        xs = model.cross_section(d)
        picks = select_picks(xs, cfg, model.weights, streaks=streaks)
        f = fwd.loc[d].reindex(xs.index)
        b = float(bfwd.loc[d])
        rets = []
        for p in picks:
            r = f.get(p.ticker, np.nan)
            if pd.isna(r):
                continue
            r_net = float(r) - cost
            rets.append(r_net)
            pick_rows.append({"date": d, "ticker": p.ticker, "slot": p.slot, "composite": p.composite,
                              "ret": r_net, "bench": b, "excess": r_net - b})
        n = len(xs)
        q = max(1, n // 5)
        ranked = f.loc[xs.index]  # xs is sorted by composite, descending
        period_rows.append({
            "date": d,
            "n_universe": n,
            "pick_ret": np.mean(rets) if rets else np.nan,
            "bench_ret": b,
            "universe_ret": float(f.mean()),
            "ic": _spearman(xs["composite"], f),
            "top_q": float(ranked.iloc[:q].mean()),
            "bottom_q": float(ranked.iloc[-q:].mean()),
        })
        new = {p.ticker for p in picks}
        streaks = {t: streaks.get(t, 0) + 1 for t in new}

    picks_df = pd.DataFrame(pick_rows)
    periods_df = pd.DataFrame(period_rows).set_index("date") if period_rows else pd.DataFrame()
    res = BacktestResult(h, picks_df, periods_df)
    res.summary = summarize(res, non_overlapping=(h <= bcfg.get("rebalance_every", 5)))
    return res


def _composite(model: Model, weights: dict) -> pd.DataFrame:
    from ..scoring import composite

    return composite(model.scores, weights)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return np.nan
    return float((equity / equity.cummax() - 1).min())


def summarize(res: BacktestResult, non_overlapping: bool = True) -> dict:
    p, per = res.picks, res.periods
    if p.empty or per.empty:
        return {"periods": 0}
    ic = per["ic"].dropna()
    out = {
        "horizon_days": res.horizon,
        "periods": int(len(per)),
        "start": str(per.index.min().date()),
        "end": str(per.index.max().date()),
        "picks": int(len(p)),
        "hit_rate": float((p["ret"] > 0).mean()),
        "beat_benchmark_rate": float((p["excess"] > 0).mean()),
        "avg_pick_return": float(p["ret"].mean()),
        "median_pick_return": float(p["ret"].median()),
        "avg_excess_vs_benchmark": float(p["excess"].mean()),
        "avg_excess_vs_universe": float((per["pick_ret"] - per["universe_ret"]).mean()),
        "mean_ic": float(ic.mean()) if len(ic) else np.nan,
        "ic_tstat": float(ic.mean() / ic.std(ddof=1) * np.sqrt(len(ic))) if len(ic) > 2 and ic.std() > 0 else np.nan,
        "ic_positive_rate": float((ic > 0).mean()) if len(ic) else np.nan,
        "top_minus_bottom_quintile": float((per["top_q"] - per["bottom_q"]).mean()),
        "best_pick": _extreme(p, "max"),
        "worst_pick": _extreme(p, "min"),
        "by_slot": {
            slot: {"picks": int(len(g)), "hit_rate": float((g["ret"] > 0).mean()),
                   "avg_excess_vs_benchmark": float(g["excess"].mean())}
            for slot, g in p.groupby("slot")
        },
    }
    if non_overlapping:
        ppy = PERIODS_PER_YEAR / res.horizon
        strat = per["pick_ret"].fillna(0.0)
        bench = per["bench_ret"].fillna(0.0)
        eq, beq = (1 + strat).cumprod(), (1 + bench).cumprod()
        years = len(per) / ppy
        out["equity"] = {
            "strategy_total_return": float(eq.iloc[-1] - 1),
            "benchmark_total_return": float(beq.iloc[-1] - 1),
            "strategy_cagr": float(eq.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan,
            "benchmark_cagr": float(beq.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan,
            "strategy_sharpe": float(strat.mean() / strat.std(ddof=1) * np.sqrt(ppy)) if strat.std() > 0 else np.nan,
            "benchmark_sharpe": float(bench.mean() / bench.std(ddof=1) * np.sqrt(ppy)) if bench.std() > 0 else np.nan,
            "strategy_max_drawdown": _max_drawdown(eq),
            "benchmark_max_drawdown": _max_drawdown(beq),
        }
        res.periods["equity"] = eq.values
        res.periods["bench_equity"] = beq.values
    return out


def _extreme(p: pd.DataFrame, how: str) -> dict:
    row = p.loc[p["ret"].idxmax() if how == "max" else p["ret"].idxmin()]
    return {"date": str(pd.Timestamp(row["date"]).date()), "ticker": row["ticker"], "ret": float(row["ret"])}


def prepare_from_scratch(panel: PricePanel, bench_panel: PricePanel, cfg: dict) -> Prepared:
    """Convenience wrapper used by the CLI."""
    years = cfg["backtest"].get("years", 5)
    start = panel.dates[-1] - pd.DateOffset(years=years)
    return prepare(panel, bench_panel, cfg, start=start)
