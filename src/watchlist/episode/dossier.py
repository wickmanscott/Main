"""The research dossier: every fact the script writer is allowed to use, in one JSON object."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..data.prices import PricePanel
from ..scoring import Pick

DISCLAIMER = ("I'm not a financial advisor and this isn't financial advice. "
              "Always do your own DD before putting money into any stock.")


def ordinal(n: int) -> str:
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def week_of(d: date) -> date:
    """Monday of the trading week the episode is for (Sunday episodes cover the next week)."""
    return d + timedelta(days=(7 - d.weekday()) % 7)


def episode_title(d: date, tickers: list[str]) -> str:
    """Same format as the original show: 'Week of August 3rd, 2020. $GBTC $MARA ...'"""
    w = week_of(d)
    return f"Week of {w.strftime('%B')} {ordinal(w.day)}, {w.year}. " + " ".join(f"${t}" for t in tickers)


def _clean(v, nd: int = 4):
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if math.isnan(v) or math.isinf(v) else round(float(v), nd)
    return v


def _round_price(x) -> float | None:
    x = _clean(x)
    if x is None:
        return None
    return round(x, 2) if x >= 1 else round(x, 4)


def market_pulse(index_panel: PricePanel, xs: pd.DataFrame) -> dict:
    pulse = {}
    for sym in index_panel.tickers:
        c = index_panel.close[sym].dropna()
        if len(c) < 210:
            continue
        pulse[sym] = {
            "close": _round_price(c.iloc[-1]),
            "ret_1w": _clean(c.iloc[-1] / c.iloc[-6] - 1),
            "ret_1m": _clean(c.iloc[-1] / c.iloc[-22] - 1),
            "above_50d": bool(c.iloc[-1] > c.rolling(50).mean().iloc[-1]),
            "above_200d": bool(c.iloc[-1] > c.rolling(200).mean().iloc[-1]),
        }
    if len(xs):
        pulse["breadth"] = {
            "universe_size": int(len(xs)),
            "pct_above_50d": _clean((xs["close"] > xs["sma50"]).mean(), 3),
            "pct_stage2_uptrend": _clean((xs["trend_template"] >= 0.75).mean(), 3),
            "pct_within_5pct_of_52w_high": _clean((xs["prox_52w_high"] >= 0.95).mean(), 3),
        }
    return pulse


def levels(row: pd.Series) -> dict:
    close, atr = row["close"], row.get("atr14")
    sma50 = row.get("sma50")
    stop_atr = close - 2.5 * atr if pd.notna(atr) else np.nan
    invalidation = np.nanmax([sma50 if pd.notna(sma50) else np.nan, stop_atr]) if (pd.notna(sma50) or pd.notna(stop_atr)) else np.nan
    pivot = row.get("pivot_20d")
    return {
        "price": _round_price(close),
        "pivot_20d_high": _round_price(pivot),
        "above_pivot": bool(pd.notna(pivot) and close > pivot),
        "sma20": _round_price(row.get("sma20")),
        "sma50": _round_price(sma50),
        "high_52w": _round_price(row.get("high_52w")),
        "support_10d_low": _round_price(row.get("support_10d")),
        "invalidation": _round_price(invalidation),
        "avg_daily_range_pct": _clean(atr / close if pd.notna(atr) and close else None, 3),
    }


def risk_flags(row: pd.Series, profile: dict, earnings: str | None, asof: date) -> list[str]:
    flags = []
    ext = row.get("extension")
    if pd.notna(ext) and ext > 0.15:
        flags.append(f"Extended: {ext:.0%} above its 20-day average, pullbacks are common from here")
    if pd.notna(row.get("rsi")) and row["rsi"] > 75:
        flags.append(f"Overbought: RSI {row['rsi']:.0f}")
    if earnings:
        days = (pd.Timestamp(earnings).date() - asof).days
        if 0 <= days <= 21:
            flags.append(f"Earnings on {earnings}: expect a big move either way")
    mc = profile.get("market_cap")
    if mc and mc < 2e9:
        flags.append(f"Small cap (about ${mc / 1e6:,.0f} million): moves fast both directions")
    if row["close"] < 10:
        flags.append("Low-priced stock: bigger percentage swings")
    atr_pct = row.get("atr14", np.nan) / row["close"] if row["close"] else np.nan
    if pd.notna(atr_pct) and atr_pct > 0.06:
        flags.append(f"Volatile: average daily range about {atr_pct:.0%}")
    if row.get("social_score", 0) > 0.7:
        flags.append("Heavy social hype: crowded trades can reverse hard")
    spf = profile.get("short_pct_float")
    if spf and spf > 0.15:
        flags.append(f"High short interest ({spf:.0%} of float): squeezes cut both ways")
    return flags or ["General market risk: momentum names fall hardest when the market turns"]


def pick_dossier(pick: Pick, row: pd.Series, profile: dict, news: list[dict], earnings: str | None, asof: date) -> dict:
    social = {}
    if pd.notna(row.get("social_mentions")):
        social = {
            "reddit_mentions_24h": _clean(row.get("social_mentions"), 0),
            "reddit_mentions_prior_24h": _clean(row.get("social_mentions_24h_ago"), 0),
            "mention_change": _clean(row.get("social_mention_velocity"), 3),
            "reddit_rank": _clean(row.get("social_rank"), 0),
        }
    if row.get("social_stocktwits_trending"):
        social["stocktwits_trending"] = True
    return {
        "ticker": pick.ticker,
        "ticker_spoken": "-".join(c for c in pick.ticker if c.isalnum()),
        "slot": pick.slot,
        "weeks_on_list": pick.weeks_on_list,
        "name": profile.get("name") or pick.ticker,
        "sector": profile.get("sector"),
        "industry": profile.get("industry"),
        "market_cap": _clean(profile.get("market_cap"), 0),
        "business_summary": profile.get("summary"),
        "reasons": pick.reasons,
        "scores": {
            "composite": _clean(pick.composite, 3),
            "composite_rank": pick.composite_rank,
            "rs_rating": _clean(row.get("rs_rating"), 0),
            "sleeper": _clean(pick.sleeper, 3),
            "social": _clean(pick.social_score, 3),
        },
        "performance": {
            "ret_1w": _clean(row.get("ret_1w"), 3),
            "ret_1m": _clean(row.get("ret_1m"), 3),
            "ret_3m": _clean(row.get("ret_3m"), 3),
            "ret_6m": _clean(row.get("ret_6m"), 3),
        },
        "technicals": {
            "trend_checks_passed": int(round(row["trend_template"] * 8)) if pd.notna(row.get("trend_template")) else None,
            "rsi": _clean(row.get("rsi"), 1),
            "volume_vs_50d_avg": _clean(row.get("rel_volume"), 2),
            "up_down_volume_ratio": _clean(row.get("ud_volume"), 2),
            "pct_below_52w_high": _clean(1 - row["prox_52w_high"], 3) if pd.notna(row.get("prox_52w_high")) else None,
            "rs_line_at_high": bool(row.get("rs_line_high") == 1),
            "recent_breakout": bool(row.get("breakout") == 1),
        },
        "levels": levels(row),
        "social": social or None,
        "next_earnings": earnings,
        "news": news,
        "risk_flags": risk_flags(row, profile, earnings, asof),
    }


def build_dossier(cfg: dict, asof: date, picks: list[dict], pulse: dict, scorecard: dict | None,
                  track: dict | None, model_card: dict | None) -> dict:
    show = cfg["show"]
    return {
        "show": {"title": show["title"], "host": show["host"], "handle": show["handle"], "tagline": show["tagline"]},
        "data_as_of": asof.isoformat(),
        "week_of": week_of(asof).isoformat(),
        "title": episode_title(asof, [p["ticker"] for p in picks]),
        "market_pulse": pulse,
        "last_week_scorecard": scorecard,
        "live_track_record": track,
        "model_backtest": model_card,
        "picks": picks,
        "disclaimer": DISCLAIMER,
    }
