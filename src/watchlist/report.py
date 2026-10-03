"""Markdown/JSON reports for backtests, plus the tuned-weights file."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT

REPORTS = ROOT / "reports"


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v)

    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.2%}"


def _rate(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.1%}"


def summary_block(title: str, s: dict) -> list[str]:
    if not s or not s.get("periods"):
        return [f"### {title}", "", "Not enough data.", ""]
    lines = [
        f"### {title}",
        "",
        f"{s['periods']} weekly periods, {s['start']} to {s['end']}, {s['picks']} picks, {s['horizon_days']}-day hold.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Hit rate (pick went up) | {_rate(s['hit_rate'])} |",
        f"| Beat the benchmark | {_rate(s['beat_benchmark_rate'])} |",
        f"| Avg pick return | {_pct(s['avg_pick_return'])} |",
        f"| Median pick return | {_pct(s['median_pick_return'])} |",
        f"| Avg excess vs benchmark | {_pct(s['avg_excess_vs_benchmark'])} |",
        f"| Avg excess vs universe | {_pct(s['avg_excess_vs_universe'])} |",
        f"| Mean rank IC | {s['mean_ic']:.3f} (t = {s['ic_tstat']:.1f}, positive {_rate(s['ic_positive_rate'])} of weeks) |",
        f"| Top minus bottom quintile | {_pct(s['top_minus_bottom_quintile'])} |",
    ]
    eq = s.get("equity")
    if eq:
        lines += [
            f"| Equal-weight picks, total return | {_pct(eq['strategy_total_return'])} (benchmark {_pct(eq['benchmark_total_return'])}) |",
            f"| CAGR | {_pct(eq['strategy_cagr'])} (benchmark {_pct(eq['benchmark_cagr'])}) |",
            f"| Sharpe | {eq['strategy_sharpe']:.2f} (benchmark {eq['benchmark_sharpe']:.2f}) |",
            f"| Max drawdown | {_pct(eq['strategy_max_drawdown'])} (benchmark {_pct(eq['benchmark_max_drawdown'])}) |",
        ]
    if s.get("by_slot"):
        lines += ["", "| Slot | Picks | Hit rate | Avg excess |", "|---|---|---|---|"]
        for slot, v in s["by_slot"].items():
            lines.append(f"| {slot} | {v['picks']} | {_rate(v['hit_rate'])} | {_pct(v['avg_excess_vs_benchmark'])} |")
    return lines + [""]


CAVEATS = """## Read this before trusting the numbers

- **Survivorship bias.** The universe is today's listed stocks. Companies that were delisted
  or went bankrupt during the test aren't in it, which flatters absolute returns. Rank IC
  and excess vs the universe are less affected than raw returns.
- **Social buzz isn't in the backtest.** Free sources have no history. Every live run saves
  a snapshot to `data/social_history/`; once there are enough weeks, social can be tested too.
- **Costs.** Each pick is charged the configured round-trip cost. Small caps can cost more.
- **The out-of-sample section is the honest one.** In-sample numbers use weights that may
  have been tuned on the same data.
"""


def backtest_markdown(results: dict, opt=None, cfg: dict | None = None) -> str:
    lines = [f"# Backtest report ({date.today().isoformat()})", ""]
    if cfg:
        lines += [f"Weights: `{cfg.get('weights_source', 'settings.yaml')}`. "
                  f"Universe: {cfg['universe']['source']}, price >= ${cfg['universe']['min_price']}, "
                  f"20-day $ volume >= ${cfg['universe']['min_avg_dollar_volume']:,.0f}. "
                  f"Cost {cfg['backtest']['cost_bps']} bps per pick.", ""]
    lines += ["## Current weights, full period", ""]
    for h, res in sorted(results.items()):
        lines += summary_block(f"{h}-day hold", res.summary)
    if opt is not None:
        lines += ["## Walk-forward tuning (out of sample)", "",
                  "Each test window uses weights picked only from data before it.", ""]
        lines += summary_block("Tuned weights, out of sample", opt.oos_tuned)
        lines += summary_block("Default weights, same windows", opt.oos_default)
        lines += ["### Folds", "", md_table(opt.folds, "{:.4f}"), "",
                  "### New weights (average of fold winners)", "",
                  md_table(pd.DataFrame([opt.weights]), "{:.3f}"), ""]
    lines.append(CAVEATS)
    return "\n".join(lines)


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def save_backtest(results: dict, opt=None, cfg: dict | None = None, out_dir: Path = REPORTS,
                  write_weights: bool = False) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = backtest_markdown(results, opt, cfg)
    (out_dir / "backtest_latest.md").write_text(md)
    payload = {
        "generated": date.today().isoformat(),
        "results": {str(h): r.summary for h, r in results.items()},
        "walk_forward": None if opt is None else {
            "weights": opt.weights, "oos_tuned": opt.oos_tuned, "oos_default": opt.oos_default,
        },
    }
    (out_dir / "backtest_latest.json").write_text(json.dumps(_jsonable(payload), indent=1, default=str))
    for h, r in results.items():
        if "equity" in r.periods:
            r.periods[["pick_ret", "bench_ret", "equity", "bench_equity", "ic"]].to_csv(out_dir / f"equity_{h}d.csv")
    if write_weights and opt is not None:
        better = _oos_improved(opt)
        weights_path = ROOT / "config" / "weights.json"
        if better:
            weights_path.write_text(json.dumps(_jsonable({
                "generated": date.today().isoformat(),
                "weights": opt.weights,
                "oos_tuned": opt.oos_tuned,
                "oos_default": opt.oos_default,
            }), indent=1))
    return out_dir / "backtest_latest.md"


def _oos_improved(opt) -> bool:
    """Only adopt tuned weights if they beat the defaults out of sample on IC or excess return."""
    t, d = opt.oos_tuned, opt.oos_default
    if not t.get("periods") or not d.get("periods"):
        return False
    return (t["ic_tstat"] or 0) >= (d["ic_tstat"] or 0) or (t["avg_excess_vs_benchmark"] or 0) > (d["avg_excess_vs_benchmark"] or 0)


def load_model_card() -> dict | None:
    """Latest backtest headline numbers, for the episode dossier."""
    p = REPORTS / "backtest_latest.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    wf = data.get("walk_forward") or {}
    s = wf.get("oos_tuned") or next(iter(data.get("results", {}).values()), {})
    if not s:
        return None
    return {
        "as_of": data.get("generated"),
        "out_of_sample": bool(wf),
        "periods": s.get("periods"),
        "hit_rate": s.get("hit_rate"),
        "beat_benchmark_rate": s.get("beat_benchmark_rate"),
        "avg_excess_vs_benchmark": s.get("avg_excess_vs_benchmark"),
    }
