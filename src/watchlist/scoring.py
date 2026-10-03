"""Turn raw features into scores, rank the universe, and pick the episode's five stocks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .signals.features import FeatureSet

# Ranked against the rest of the universe each day (percentile, 0..1).
PCT_FEATURES = ["rs", "mom_12_1", "prox_52w_high", "accel", "ud_volume", "rel_volume", "tightness"]
# Already on a 0..1 scale.
RAW_FEATURES = ["trend_template", "rsi_sweet", "breakout"]
PRICE_FEATURES = PCT_FEATURES + RAW_FEATURES

_SOURCE = {"rs": "rs_raw"}  # score name -> raw feature name when they differ


def eligibility(fs: FeatureSet, min_price: float, min_dollar_volume: float) -> pd.DataFrame:
    return (
        (fs["close"] >= min_price)
        & (fs["dollar_volume"] >= min_dollar_volume)
        & fs["trend_template"].notna()
        & fs["rs_raw"].notna()
    )


def score_frames(fs: FeatureSet, eligible: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """0..1 score per feature, per date, per ticker. Percentiles use only eligible names that day."""
    scores = {}
    for name in PCT_FEATURES:
        raw = fs[_SOURCE.get(name, name)].where(eligible)
        scores[name] = raw.rank(axis=1, pct=True)
    for name in RAW_FEATURES:
        scores[name] = fs[name].where(eligible).fillna(0.0).where(eligible)
    # Attention proxy for "under the radar": how heavily traded the name is vs the universe.
    scores["dollar_volume"] = fs["dollar_volume"].where(eligible).rank(axis=1, pct=True)
    return scores


def composite(scores: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    total = sum(weights.get(k, 0.0) for k in PRICE_FEATURES)
    if total <= 0:
        raise ValueError("feature weights must sum to a positive number")
    out = None
    for k in PRICE_FEATURES:
        w = weights.get(k, 0.0)
        if w:
            term = scores[k].fillna(0.0) * w
            out = term if out is None else out + term
    return (out / total).where(scores["rs"].notna())


def sleeper_score(trend_template, prox_52w_high, extension, s_rs, s_ud_volume, s_tightness, rs_line_high, attention):
    """Under-the-radar score: a strong, tight, accumulating setup that few people are watching.

    Works on frames (backtest) or Series (one day's cross-section). NaN where the setup fails the gates.
    """
    gate = (trend_template >= 0.75) & (prox_52w_high >= 0.85) & (extension <= 0.15) & (s_rs >= 0.70)
    raw = 0.30 * s_rs + 0.20 * s_ud_volume + 0.20 * s_tightness + 0.10 * rs_line_high.fillna(0.0) + 0.20 * (1 - attention)
    return raw.where(gate)


def sleeper(fs: FeatureSet, scores: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # Backtestable attention proxy: dollar-volume percentile. Live runs blend in social mentions.
    return sleeper_score(
        fs["trend_template"], fs["prox_52w_high"], fs["extension"], scores["rs"],
        scores["ud_volume"], scores["tightness"], fs["rs_line_high"], scores["dollar_volume"],
    )


@dataclass
class Model:
    """Scores for all dates. Build once, then slice cross-sections for any date."""

    features: FeatureSet
    eligible: pd.DataFrame
    scores: dict[str, pd.DataFrame]
    composite: pd.DataFrame
    sleeper: pd.DataFrame
    weights: dict[str, float] = field(default_factory=dict)

    @classmethod
    def build(cls, fs: FeatureSet, weights: dict[str, float], min_price: float, min_dollar_volume: float) -> "Model":
        elig = eligibility(fs, min_price, min_dollar_volume)
        sc = score_frames(fs, elig)
        return cls(fs, elig, sc, composite(sc, weights), sleeper(fs, sc), dict(weights))

    def cross_section(self, date, social: pd.DataFrame | None = None, social_weight: float = 0.0) -> pd.DataFrame:
        date = pd.Timestamp(date)
        xs = self.features.row(date)
        xs["eligible"] = self.eligible.loc[date].astype(bool)
        for k, df in self.scores.items():
            xs[f"s_{k}"] = df.loc[date]
        xs["rs_rating"] = (xs["s_rs"] * 99).clip(1, 99).round()
        xs["composite"] = self.composite.loc[date]
        xs["sleeper"] = self.sleeper.loc[date]
        if social is not None:
            soc = social.reindex(xs.index)
            for col in soc.columns:
                xs[f"social_{col}" if not col.startswith("social_") else col] = soc[col]
            xs["social_score"] = xs["social_score"].fillna(0.0)
            xs["social_stocktwits_trending"] = xs["social_stocktwits_trending"].fillna(False).astype(bool)
            if social_weight:
                total = sum(self.weights.get(k, 0.0) for k in PRICE_FEATURES)
                xs["composite"] = (xs["composite"] * total + social_weight * xs["social_score"]) / (total + social_weight)
            # Live "under the radar": blend trading attention with social attention.
            att = 0.5 * xs["s_dollar_volume"] + 0.5 * xs["social_attention"].fillna(0.0)
            xs["sleeper"] = sleeper_score(
                xs["trend_template"], xs["prox_52w_high"], xs["extension"], xs["s_rs"],
                xs["s_ud_volume"], xs["s_tightness"], xs["rs_line_high"], att,
            )
        else:
            xs["social_score"] = 0.0
        xs["composite_rank"] = xs["composite"].rank(ascending=False, method="first")
        return xs[xs["eligible"]].sort_values("composite", ascending=False)


# ----------------------------------------------------------------------------- selection

@dataclass
class Pick:
    ticker: str
    slot: str
    composite: float
    composite_rank: int
    sleeper: float | None
    social_score: float
    reasons: list[str]
    weeks_on_list: int = 1

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def select_picks(
    xs: pd.DataFrame,
    cfg: dict,
    weights: dict[str, float],
    streaks: dict[str, int] | None = None,
    sectors: dict[str, str] | None = None,
) -> list[Pick]:
    """Fill the episode's slots: momentum leaders, a social-buzz name, and a sleeper.

    streaks: ticker -> consecutive previous episodes it appeared in.
    sectors: ticker -> sector, enables the max-per-sector rule when known.
    """
    sel = cfg["selection"]
    slots = sel.get("slots", {"momentum": sel.get("picks", 5)})
    n_total = sel.get("picks", 5)
    streaks = streaks or {}
    sectors = sectors or {}

    base = xs[
        xs["eligible"]
        & (xs["trend_template"] >= sel.get("min_trend_template", 0.0))
        & (xs["extension"] <= sel.get("max_extension", np.inf))
    ]
    chosen: list[Pick] = []

    def allowed(t: str) -> bool:
        if any(p.ticker == t for p in chosen):
            return False
        streak = streaks.get(t, 0)
        if streak and (not sel.get("allow_repeat", True) or streak >= sel.get("max_repeat_weeks", 3)):
            return False
        sec = sectors.get(t)
        if sec and sum(sectors.get(p.ticker) == sec for p in chosen) >= sel.get("max_per_sector", 99):
            return False
        return True

    def take(cands: pd.DataFrame, slot: str, n: int) -> None:
        added = 0
        for t, row in cands.iterrows():
            if added >= n or len(chosen) >= n_total:
                return
            if allowed(t):
                chosen.append(_make_pick(t, row, slot, weights, streaks))
                added += 1

    take(base.sort_values("composite", ascending=False), "momentum", slots.get("momentum", 0))

    if slots.get("social") and "social_score" in base and base["social_score"].gt(0).any():
        median = base["composite"].median()
        social = base[(base["social_score"] > 0) & (base["composite"] >= median)]
        take(social.sort_values("social_score", ascending=False), "social", slots["social"])

    if slots.get("sleeper"):
        sleepers = xs[xs["eligible"] & xs["sleeper"].notna()]
        take(sleepers.sort_values("sleeper", ascending=False), "sleeper", slots["sleeper"])

    # Any slot that couldn't be filled goes to the next-best momentum name.
    take(base.sort_values("composite", ascending=False), "momentum", n_total - len(chosen))

    order = {"momentum": 0, "social": 1, "sleeper": 2}
    return sorted(chosen, key=lambda p: (order.get(p.slot, 9), p.composite_rank))


def _make_pick(t: str, row: pd.Series, slot: str, weights: dict, streaks: dict) -> Pick:
    return Pick(
        ticker=t,
        slot=slot,
        composite=float(row["composite"]),
        composite_rank=int(row["composite_rank"]),
        sleeper=None if pd.isna(row.get("sleeper")) else float(row["sleeper"]),
        social_score=float(row.get("social_score", 0.0) or 0.0),
        reasons=explain(row, weights, slot),
        weeks_on_list=streaks.get(t, 0) + 1,
    )


def _fmt_pct(x: float) -> str:
    return f"{x * 100:+.0f}%"


def explain(row: pd.Series, weights: dict[str, float], slot: str = "momentum", top: int = 3) -> list[str]:
    """Plain-English reasons, from the features contributing most to this stock's score."""
    text = {
        "rs": lambda r: f"Relative strength rating {r['rs_rating']:.0f} (beating {r['rs_rating']:.0f}% of the market)",
        "mom_12_1": lambda r: f"{_fmt_pct(r['mom_12_1'])} over the past year, excluding the last month",
        "prox_52w_high": lambda r: f"Trading {max(0.0, 1 - r['prox_52w_high']) * 100:.1f}% below its 52-week high",
        "accel": lambda r: f"Momentum accelerating: {_fmt_pct(r['ret_1m'])} in the last month vs {_fmt_pct(r['ret_3m'])} over three",
        "trend_template": lambda r: f"Passes {round(r['trend_template'] * 8)}/8 stage-2 uptrend checks",
        "rsi_sweet": lambda r: f"RSI {r['rsi']:.0f}: strong without being overbought",
        "ud_volume": lambda r: f"Up-day volume is {r['ud_volume']:.1f}x down-day volume over 50 days (accumulation)",
        "rel_volume": lambda r: f"Volume running {r['rel_volume']:.1f}x its 50-day average",
        "tightness": lambda r: f"Volatility contracting: 10-day range is {r['vcp'] * 100:.0f}% of the 50-day range",
        "breakout": lambda r: "Broke out to a 20-day high on heavy volume this week",
    }
    contrib = {}
    for k in PRICE_FEATURES:
        s = row.get(f"s_{k}")
        if pd.notna(s) and weights.get(k):
            contrib[k] = weights[k] * s
    if row.get("breakout", 0) < 1:
        contrib.pop("breakout", None)
    reasons = []
    for k in sorted(contrib, key=contrib.get, reverse=True)[:top]:
        try:
            reasons.append(text[k](row))
        except (KeyError, TypeError, ValueError):
            continue
    if slot == "social" or row.get("social_score", 0) > 0.5:
        m, v = row.get("social_mentions"), row.get("social_mention_velocity")
        if pd.notna(m) and pd.notna(v):
            reasons.insert(0, f"Reddit buzz: {m:.0f} mentions in 24h ({_fmt_pct(v)} vs the prior day)")
        if row.get("social_stocktwits_trending"):
            reasons.insert(0, "Trending on StockTwits")
    if slot == "sleeper":
        reasons.insert(0, "Strong setup with low attention: lighter trading volume and little social chatter")
    return reasons
