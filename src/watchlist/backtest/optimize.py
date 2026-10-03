"""Walk-forward weight tuning.

Random-search feature weights on a training window, then score the winner on
the *next* window it never saw. Roll forward and repeat. Only the stitched
out-of-sample results count as evidence; the final live weights are the average
of each fold's winner, which is steadier than trusting a single fold.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..scoring import PRICE_FEATURES
from .engine import Prepared, run


@dataclass
class FastArrays:
    dates: pd.DatetimeIndex
    X: np.ndarray          # [D, N, F] feature scores, NaN when ineligible
    gate: np.ndarray       # [D, N] passes selection gates
    fwd: np.ndarray        # [D, N] forward returns
    bench: np.ndarray      # [D]
    fwd_rank: np.ndarray   # [D, N] percentile rank of forward return among valid names


def build_arrays(prep: Prepared, cfg: dict, horizon: int) -> FastArrays:
    sel = cfg["selection"]
    fs, model = prep.features, prep.model
    bfwd = prep.bench_fwd[horizon].reindex(prep.dates)
    dates = prep.dates[bfwd.notna().values]
    X = np.stack([model.scores[k].reindex(dates).to_numpy(float) for k in PRICE_FEATURES], axis=-1)
    elig = model.eligible.reindex(dates).to_numpy(bool)
    gate = (
        elig
        & (fs["trend_template"].reindex(dates).to_numpy(float) >= sel.get("min_trend_template", 0.0))
        & (fs["extension"].reindex(dates).to_numpy(float) <= sel.get("max_extension", np.inf))
    )
    fwd_df = prep.fwd[horizon].reindex(dates).where(model.eligible.reindex(dates))
    return FastArrays(
        dates=dates,
        X=np.where(elig[..., None], X, np.nan),
        gate=gate,
        fwd=fwd_df.to_numpy(float),
        bench=bfwd.reindex(dates).to_numpy(float),
        fwd_rank=fwd_df.rank(axis=1, pct=True).to_numpy(float),
    )


def _composite(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    comp = np.nansum(X * w, axis=-1) / w.sum()
    return np.where(np.isnan(X[..., 0]), np.nan, comp)


def evaluate_fast(arr: FastArrays, w: np.ndarray, idx: np.ndarray, n_picks: int, cost: float) -> dict:
    """IC series and top-n excess return series for weights w over periods idx."""
    # Periods with no valid names produce all-NaN rows; those warnings are expected noise.
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _evaluate_fast(arr, w, idx, n_picks, cost)


def _evaluate_fast(arr: FastArrays, w: np.ndarray, idx: np.ndarray, n_picks: int, cost: float) -> dict:
    comp = _composite(arr.X[idx], w)
    comp_rank = pd.DataFrame(comp).rank(axis=1, pct=True).to_numpy()
    fr = arr.fwd_rank[idx]
    ok = ~np.isnan(comp_rank) & ~np.isnan(fr)
    a = np.where(ok, comp_rank, np.nan)
    b = np.where(ok, fr, np.nan)
    a = a - np.nanmean(a, axis=1, keepdims=True)
    b = b - np.nanmean(b, axis=1, keepdims=True)
    ic = np.nansum(a * b, axis=1) / np.sqrt(np.nansum(a * a, axis=1) * np.nansum(b * b, axis=1))

    gated = np.where(arr.gate[idx] & ~np.isnan(comp), comp, -np.inf)
    k = min(n_picks, gated.shape[1])
    top = np.argpartition(-gated, k - 1, axis=1)[:, :k]
    fwd = np.take_along_axis(arr.fwd[idx], top, axis=1)
    valid = np.take_along_axis(gated, top, axis=1) > -np.inf
    fwd = np.where(valid & ~np.isnan(fwd), fwd, np.nan)
    pick_ret = np.nanmean(fwd, axis=1) - cost
    excess = pick_ret - arr.bench[idx]
    return {"ic": ic, "excess": excess}


def _tstat(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return 0.0
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x)))


def objective(stats: dict, kind: str) -> float:
    if kind == "ic":
        return _tstat(stats["ic"])
    if kind == "excess":
        return _tstat(stats["excess"])
    return 0.5 * _tstat(stats["ic"]) + 0.5 * _tstat(stats["excess"])


@dataclass
class OptimizeResult:
    weights: dict[str, float]
    folds: pd.DataFrame
    oos_tuned: dict
    oos_default: dict


def optimize(prep: Prepared, cfg: dict, horizon: int | None = None) -> OptimizeResult:
    bcfg, ocfg = cfg["backtest"], cfg["backtest"]["optimize"]
    h = horizon or bcfg["primary_horizon"]
    cost = bcfg.get("cost_bps", 0) / 1e4
    n_picks = cfg["selection"]["slots"].get("momentum", cfg["selection"]["picks"])
    arr = build_arrays(prep, cfg, h)
    D = len(arr.dates)
    train, test = ocfg["train_weeks"], ocfg["test_weeks"]
    if D < train + test:
        # Not enough history for full folds: shrink proportionally.
        train, test = max(int(D * 0.7), 10), max(D - int(D * 0.7), 5)
    rng = np.random.default_rng(ocfg.get("seed", 7))
    default = np.array([cfg["weights"].get(k, 0.0) for k in PRICE_FEATURES])
    default = default / default.sum()
    cands = np.vstack([default, rng.dirichlet(np.ones(len(PRICE_FEATURES)), ocfg["candidates"])])

    rows, winners, oos_idx = [], [], []
    start = 0
    while start + train + test <= D:
        tr = np.arange(start, start + train)
        te = np.arange(start + train, start + train + test)
        scores = [objective(evaluate_fast(arr, w, tr, n_picks, cost), ocfg["objective"]) for w in cands]
        best = cands[int(np.argmax(scores))]
        winners.append(best)
        oos_idx.append((te, best))
        s_best = evaluate_fast(arr, best, te, n_picks, cost)
        s_def = evaluate_fast(arr, default, te, n_picks, cost)
        rows.append({
            "train_start": arr.dates[tr[0]].date(), "test_start": arr.dates[te[0]].date(),
            "test_end": arr.dates[te[-1]].date(), "train_objective": float(max(scores)),
            "test_ic_tuned": float(np.nanmean(s_best["ic"])), "test_ic_default": float(np.nanmean(s_def["ic"])),
            "test_excess_tuned": float(np.nanmean(s_best["excess"])), "test_excess_default": float(np.nanmean(s_def["excess"])),
        })
        start += test

    if not winners:
        raise ValueError(f"not enough rebalance periods ({D}) for walk-forward optimization")

    final = np.mean(winners, axis=0)
    final = final / final.sum()
    weights = {k: round(float(v), 4) for k, v in zip(PRICE_FEATURES, final)}

    # Stitched out-of-sample: each test window uses the weights chosen before it (true walk-forward).
    tuned_parts = [run(prep, cfg, h, weights=dict(zip(PRICE_FEATURES, w)), dates=arr.dates[te]) for te, w in oos_idx]
    all_te = np.concatenate([te for te, _ in oos_idx])
    default_res = run(prep, cfg, h, weights=dict(zip(PRICE_FEATURES, default)), dates=arr.dates[all_te])
    return OptimizeResult(weights, pd.DataFrame(rows), _stitch(tuned_parts, cfg, h), default_res.summary)


def _stitch(parts, cfg, h) -> dict:
    from .engine import BacktestResult, summarize

    picks = pd.concat([p.picks for p in parts if not p.picks.empty], ignore_index=True)
    periods = pd.concat([p.periods for p in parts if not p.periods.empty])
    res = BacktestResult(h, picks, periods)
    return summarize(res, non_overlapping=(h <= cfg["backtest"].get("rebalance_every", 5)))
