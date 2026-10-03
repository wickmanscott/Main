"""Every published pick, so each episode can open with an honest scorecard and the
show builds a live, out-of-sample track record over time."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import resolve
from .data.prices import PricePanel

COLUMNS = ["episode_date", "ticker", "slot", "ref_close", "composite", "composite_rank"]


class Ledger:
    def __init__(self, path: str | Path):
        self.path = resolve(path)
        if self.path.exists():
            self.df = pd.read_csv(self.path, parse_dates=["episode_date"])
        else:
            self.df = pd.DataFrame(columns=COLUMNS)

    @property
    def tickers(self) -> list[str]:
        return sorted(self.df["ticker"].unique()) if len(self.df) else []

    def episodes(self) -> list[pd.Timestamp]:
        return sorted(pd.to_datetime(self.df["episode_date"]).unique()) if len(self.df) else []

    def last_episode(self, before=None) -> tuple[pd.Timestamp | None, pd.DataFrame]:
        eps = self.episodes()
        if before is not None:
            eps = [e for e in eps if e < pd.Timestamp(before)]
        if not eps:
            return None, self.df.iloc[0:0]
        last = eps[-1]
        return last, self.df[self.df["episode_date"] == last]

    def streaks(self, before=None) -> dict[str, int]:
        """ticker -> number of consecutive episodes (ending with the latest) it appeared in."""
        eps = self.episodes()
        if before is not None:
            eps = [e for e in eps if e < pd.Timestamp(before)]
        out: dict[str, int] = {}
        alive: set[str] | None = None
        for ep in reversed(eps):
            names = set(self.df.loc[self.df["episode_date"] == ep, "ticker"])
            alive = names if alive is None else alive & names
            if not alive:
                break
            for t in alive:
                out[t] = out.get(t, 0) + 1
        return out

    def append(self, episode_date, picks: list, ref_closes: dict[str, float]) -> None:
        d = pd.Timestamp(episode_date)
        self.df = self.df[self.df["episode_date"] != d]  # re-running an episode replaces it
        rows = pd.DataFrame([{
            "episode_date": d, "ticker": p.ticker, "slot": p.slot,
            "ref_close": round(float(ref_closes.get(p.ticker, np.nan)), 4),
            "composite": round(p.composite, 4), "composite_rank": p.composite_rank,
        } for p in picks], columns=COLUMNS)
        self.df = rows if self.df.empty else pd.concat([self.df, rows], ignore_index=True)
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        out = self.df.copy()
        out["episode_date"] = pd.to_datetime(out["episode_date"]).dt.strftime("%Y-%m-%d")
        out.sort_values(["episode_date", "slot", "composite_rank"]).to_csv(self.path, index=False)


def _window_return(panel: PricePanel, ticker: str, start: pd.Timestamp, sessions: int | None):
    """Buy the first open after `start`, sell at the close `sessions` later (or the latest close)."""
    if ticker not in panel.close.columns:
        return None
    after = panel.dates[panel.dates > start]
    if len(after) == 0:
        return None
    entry = panel.open.at[after[0], ticker]
    if sessions is None:
        closes = panel.close[ticker].loc[after[0]:].dropna()
        if closes.empty:
            return None
        exit_, exit_date = closes.iloc[-1], closes.index[-1]
    else:
        if len(after) < sessions:
            return None
        exit_date = after[sessions - 1]
        exit_ = panel.close.at[exit_date, ticker]
    if pd.isna(entry) or pd.isna(exit_) or entry == 0:
        return None
    return {"entry_date": after[0], "exit_date": exit_date, "ret": float(exit_ / entry - 1)}


def scorecard(ledger: Ledger, panel: PricePanel, bench: PricePanel, before=None) -> dict | None:
    """How last episode's picks have done since listeners could buy them, vs the benchmark."""
    ep, rows = ledger.last_episode(before)
    if ep is None:
        return None
    bsym = bench.tickers[0]
    picks = []
    for _, r in rows.iterrows():
        w = _window_return(panel, r["ticker"], ep, None)
        b = _window_return(bench, bsym, ep, None)
        if w is None or b is None:
            picks.append({"ticker": r["ticker"], "slot": r["slot"], "ret": None})
            continue
        picks.append({"ticker": r["ticker"], "slot": r["slot"], "ret": round(w["ret"], 4),
                      "benchmark_ret": round(b["ret"], 4), "beat_benchmark": w["ret"] > b["ret"]})
    scored = [p for p in picks if p["ret"] is not None]
    out = {"episode_date": ep.strftime("%Y-%m-%d"), "benchmark": bsym, "picks": picks}
    if scored:
        out["avg_ret"] = round(float(np.mean([p["ret"] for p in scored])), 4)
        out["benchmark_ret"] = scored[0]["benchmark_ret"]
        out["winners"] = sum(p["ret"] > 0 for p in scored)
        out["beat_benchmark"] = sum(p["beat_benchmark"] for p in scored)
        best = max(scored, key=lambda p: p["ret"])
        worst = min(scored, key=lambda p: p["ret"])
        out["best"], out["worst"] = best["ticker"], worst["ticker"]
    return out


def track_record(ledger: Ledger, panel: PricePanel, bench: PricePanel, sessions: int = 5) -> dict | None:
    """Live record across every episode: 1-week return of each pick from the next open."""
    bsym = bench.tickers[0]
    rows = []
    for _, r in ledger.df.iterrows():
        ep = pd.Timestamp(r["episode_date"])
        w = _window_return(panel, r["ticker"], ep, sessions)
        b = _window_return(bench, bsym, ep, sessions)
        if w and b:
            rows.append({"ret": w["ret"], "excess": w["ret"] - b["ret"], "slot": r["slot"]})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    return {
        "episodes": len(ledger.episodes()),
        "picks_scored": len(df),
        "hold_sessions": sessions,
        "hit_rate": round(float((df["ret"] > 0).mean()), 3),
        "beat_benchmark_rate": round(float((df["excess"] > 0).mean()), 3),
        "avg_ret": round(float(df["ret"].mean()), 4),
        "avg_excess": round(float(df["excess"].mean()), 4),
    }
