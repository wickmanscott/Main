"""Social buzz: Reddit mention counts (ApeWisdom) and StockTwits trending.

Neither source offers free history, so every run saves a snapshot to
data/social_history/. After a few months those snapshots let the backtester
measure whether social acceleration actually predicts returns.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..config import resolve

log = logging.getLogger(__name__)

APEWISDOM_URL = "https://apewisdom.io/api/v1.0/filter/all-stocks/page/{page}"
STOCKTWITS_TRENDING_URL = "https://api.stocktwits.com/api/2/trending/symbols.json"
_HEADERS = {"User-Agent": "stock-watchlist-agent"}


@dataclass
class SocialSnapshot:
    taken_at: str
    reddit: dict[str, dict] = field(default_factory=dict)
    stocktwits_trending: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"taken_at": self.taken_at, "reddit": self.reddit, "stocktwits_trending": self.stocktwits_trending}

    @classmethod
    def from_json(cls, data: dict) -> "SocialSnapshot":
        return cls(data["taken_at"], data.get("reddit", {}), data.get("stocktwits_trending", []))


def fetch_apewisdom(pages: int = 3, timeout: float = 20) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for page in range(1, pages + 1):
        resp = requests.get(APEWISDOM_URL.format(page=page), timeout=timeout, headers=_HEADERS)
        resp.raise_for_status()
        data = resp.json()
        for row in data.get("results", []):
            ticker = str(row.get("ticker", "")).upper().replace(".", "-")
            if not ticker:
                continue
            out[ticker] = {
                "rank": _num(row.get("rank")),
                "rank_24h_ago": _num(row.get("rank_24h_ago")),
                "mentions": _num(row.get("mentions")),
                "mentions_24h_ago": _num(row.get("mentions_24h_ago")),
                "upvotes": _num(row.get("upvotes")),
                "name": row.get("name"),
            }
        if page >= int(data.get("pages", page)):
            break
    return out


def fetch_stocktwits_trending(timeout: float = 20) -> list[str]:
    resp = requests.get(STOCKTWITS_TRENDING_URL, timeout=timeout, headers=_HEADERS)
    resp.raise_for_status()
    return [s["symbol"].upper() for s in resp.json().get("symbols", []) if s.get("symbol")]


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def take_snapshot(cfg: dict, save: bool = True) -> SocialSnapshot:
    scfg = cfg.get("social", {})
    snap = SocialSnapshot(taken_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    if not scfg.get("enabled", True):
        return snap
    try:
        snap.reddit = fetch_apewisdom(scfg.get("apewisdom_pages", 3))
    except Exception as exc:  # social data is a bonus signal; never fail the episode over it
        log.warning("ApeWisdom fetch failed: %s", exc)
    if scfg.get("stocktwits", True):
        try:
            snap.stocktwits_trending = fetch_stocktwits_trending()
        except Exception as exc:
            log.warning("StockTwits fetch failed: %s", exc)
    if save and (snap.reddit or snap.stocktwits_trending):
        hist = resolve(scfg.get("history_dir", "data/social_history"))
        hist.mkdir(parents=True, exist_ok=True)
        (hist / f"{snap.taken_at[:10]}.json").write_text(json.dumps(snap.to_json(), indent=1, sort_keys=True))
    return snap


def social_frame(snap: SocialSnapshot, universe: list[str]) -> pd.DataFrame:
    """Per-ticker social metrics plus two scores in [0, 1]:

    social_score: buzz that is accelerating (mention velocity + rank jump + raw volume of talk)
    attention:    how much people are already talking about it (used to find "under the radar")
    """
    df = pd.DataFrame(index=pd.Index(universe, name="ticker"))
    reddit = pd.DataFrame.from_dict(snap.reddit, orient="index") if snap.reddit else pd.DataFrame()
    for col in ("mentions", "mentions_24h_ago", "rank", "rank_24h_ago", "upvotes"):
        df[col] = reddit[col].reindex(df.index).astype(float) if col in reddit else np.nan
    df["stocktwits_trending"] = df.index.isin(snap.stocktwits_trending)

    m, m0 = df["mentions"], df["mentions_24h_ago"]
    df["mention_velocity"] = (m - m0) / np.maximum(m0.fillna(0), 5.0)
    df["rank_change"] = df["rank_24h_ago"] - df["rank"]  # positive = climbing the leaderboard

    has = m.notna()
    score = pd.Series(0.0, index=df.index)
    if has.any():
        vel = df.loc[has, "mention_velocity"].rank(pct=True)
        vol = np.log1p(m[has]).rank(pct=True)
        jump = df.loc[has, "rank_change"].fillna(0).rank(pct=True)
        score[has] = 0.5 * vel + 0.3 * vol + 0.2 * jump
    score[df["stocktwits_trending"]] = np.maximum(score[df["stocktwits_trending"]], 0.5) + 0.1
    df["social_score"] = score.clip(0, 1)

    attention = pd.Series(0.0, index=df.index)
    if has.any():
        attention[has] = np.log1p(m[has]) / np.log1p(m[has].max())
    attention[df["stocktwits_trending"]] = np.maximum(attention[df["stocktwits_trending"]], 0.6)
    df["attention"] = attention.clip(0, 1)
    return df


def load_history(cfg: dict) -> dict[pd.Timestamp, SocialSnapshot]:
    hist = resolve(cfg.get("social", {}).get("history_dir", "data/social_history"))
    out = {}
    for f in sorted(Path(hist).glob("*.json")):
        out[pd.Timestamp(f.stem)] = SocialSnapshot.from_json(json.loads(f.read_text()))
    return out
