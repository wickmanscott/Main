"""Audit the original 2020 episodes: how did Scott's picks actually do, and what did
the winners look like on the day they were picked?

This is a small, hand-picked sample from one unusual market (summer 2020), so treat
the trait comparison as a sanity check for the model, not as proof.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ARCHIVE
from ..data.prices import PricePanel, download_yfinance
from ..report import md_table
from ..signals.features import compute_features

TRAITS = ["excess_3m", "trend_template", "prox_52w_high", "rel_volume", "rsi", "extension", "ret_1m"]
HORIZONS = {"1w": 5, "1m": 21, "3m": 63}


def load_archive(path: str | Path = ARCHIVE) -> dict:
    return json.loads(Path(path).read_text())


def _first_with_data(panel: PricePanel, candidates: list[str], date: pd.Timestamp) -> str | None:
    for sym in candidates:
        if sym in panel.close.columns:
            s = panel.close[sym]
            if s.loc[date - pd.Timedelta(days=10): date + pd.Timedelta(days=10)].notna().any():
                return sym
    return None


def audit(archive: dict, panel: PricePanel, bench: PricePanel) -> tuple[pd.DataFrame, dict]:
    bsym = bench.tickers[0]
    fs = compute_features(panel, bench.close[bsym])
    smap = {k: v for k, v in archive.get("symbol_map", {}).items() if not k.startswith("_")}
    rows = []
    for ep in archive["episodes"]:
        d = pd.Timestamp(ep["date"])
        after = panel.dates[panel.dates > d]
        if len(after) == 0:
            continue
        t1 = after[0]  # listeners buy the next session's open
        signal_day = panel.dates[panel.dates <= d][-1]
        for tk in ep["tickers"]:
            sym = _first_with_data(panel, smap.get(tk, [tk]), d)
            row = {"episode": ep["date"], "ticker": tk, "symbol_used": sym}
            if sym is None:
                row["status"] = "no price data (delisted or renamed)"
                rows.append(row)
                continue
            entry = panel.open.at[t1, sym]
            bentry = bench.open.at[t1, bsym]
            i1 = panel.dates.get_loc(t1)
            for label, h in HORIZONS.items():
                j = i1 + h - 1
                if j < len(panel.dates):
                    ex = panel.close.iat[j, panel.close.columns.get_loc(sym)]
                    bex = bench.close[bsym].iat[j]
                    row[f"ret_{label}"] = ex / entry - 1 if entry else np.nan
                    row[f"spy_{label}"] = bex / bentry - 1
            for trait in TRAITS:
                row[trait] = float(fs[trait].at[signal_day, sym]) if sym in fs[trait].columns else np.nan
            row["status"] = "ok"
            rows.append(row)
    df = pd.DataFrame(rows)
    ok = df[df["status"] == "ok"].copy()
    summary: dict = {"episodes": len(archive["episodes"]), "picks": len(df), "with_data": len(ok),
                     "missing": sorted(df.loc[df["status"] != "ok", "ticker"].unique().tolist())}
    for label in HORIZONS:
        col, spy = f"ret_{label}", f"spy_{label}"
        if col in ok:
            sub = ok.dropna(subset=[col])
            summary[label] = {
                "hit_rate": float((sub[col] > 0).mean()),
                "beat_spy_rate": float((sub[col] > sub[spy]).mean()),
                "avg_return": float(sub[col].mean()),
                "median_return": float(sub[col].median()),
                "avg_excess_vs_spy": float((sub[col] - sub[spy]).mean()),
            }
    if "ret_1m" in ok:
        sub = ok.dropna(subset=["ret_1m"])
        winners = sub[sub["ret_1m"] > sub["spy_1m"]]
        losers = sub[sub["ret_1m"] <= sub["spy_1m"]]
        summary["traits_winners_vs_losers_1m"] = {
            t: {"winners": float(winners[t].median()), "losers": float(losers[t].median())}
            for t in TRAITS if t in sub
        }
    return df, summary


def run_audit(archive_path: str | Path = ARCHIVE) -> tuple[pd.DataFrame, dict]:
    archive = load_archive(archive_path)
    symbols = set()
    for ep in archive["episodes"]:
        for tk in ep["tickers"]:
            symbols.update(archive.get("symbol_map", {}).get(tk, [tk]))
    dates = [pd.Timestamp(e["date"]) for e in archive["episodes"]]
    start = min(dates) - pd.Timedelta(days=420)  # a year of warm-up for the features
    end = max(dates) + pd.Timedelta(days=140)
    panel = download_yfinance(sorted(symbols), start, end)
    bench = download_yfinance(["SPY"], start, end)
    return audit(archive, panel, bench)


def audit_markdown(df: pd.DataFrame, summary: dict) -> str:
    lines = ["# 2020 archive audit: The (Stock) Watchlist", "",
             f"{summary['with_data']} of {summary['picks']} picks across {summary['episodes']} episodes had price data. "
             "Entry is the next session's open after the episode date.", ""]
    if summary["missing"]:
        lines += [f"No data (delisted or renamed, a live example of survivorship bias): {', '.join(summary['missing'])}", ""]
    lines += ["| Horizon | Hit rate | Beat SPY | Avg return | Median | Avg excess vs SPY |", "|---|---|---|---|---|---|"]
    for label in HORIZONS:
        if label in summary:
            s = summary[label]
            lines.append(f"| {label} | {s['hit_rate']:.0%} | {s['beat_spy_rate']:.0%} | {s['avg_return']:+.1%} | "
                         f"{s['median_return']:+.1%} | {s['avg_excess_vs_spy']:+.1%} |")
    if "traits_winners_vs_losers_1m" in summary:
        lines += ["", "## What winners looked like on pick day (median, 1-month winners vs SPY)", "",
                  "| Trait | Winners | Losers |", "|---|---|---|"]
        for t, v in summary["traits_winners_vs_losers_1m"].items():
            lines.append(f"| {t} | {v['winners']:.3f} | {v['losers']:.3f} |")
    lines += ["", "## Every pick", "", md_table(df)]
    return "\n".join(lines) + "\n"
