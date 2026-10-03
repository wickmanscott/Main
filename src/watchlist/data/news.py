"""Company profile, news headlines and earnings dates for the dossier (Yahoo Finance)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

log = logging.getLogger(__name__)


def _ticker(symbol: str):
    import yfinance as yf

    return yf.Ticker(symbol)


def fetch_profile(symbol: str) -> dict:
    out = {"ticker": symbol}
    try:
        info = _ticker(symbol).info or {}
    except Exception as exc:
        log.warning("profile %s failed: %s", symbol, exc)
        return out
    out.update(
        name=info.get("longName") or info.get("shortName") or symbol,
        sector=info.get("sector"),
        industry=info.get("industry"),
        market_cap=info.get("marketCap"),
        summary=(info.get("longBusinessSummary") or "")[:600] or None,
        short_pct_float=info.get("shortPercentOfFloat"),
        float_shares=info.get("floatShares"),
        analyst_count=info.get("numberOfAnalystOpinions"),
    )
    return out


def fetch_next_earnings(symbol: str) -> str | None:
    try:
        cal = _ticker(symbol).calendar
    except Exception:
        return None
    dates = None
    if isinstance(cal, dict):
        dates = cal.get("Earnings Date")
    elif isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
        dates = list(cal.loc["Earnings Date"].values)
    if not dates:
        return None
    if not isinstance(dates, (list, tuple)):
        dates = [dates]
    today = pd.Timestamp.now().normalize()
    future = sorted(pd.Timestamp(d) for d in dates if pd.Timestamp(d) >= today)
    return future[0].strftime("%Y-%m-%d") if future else None


def _parse_news_item(item: dict) -> dict | None:
    # yfinance >= 0.2.50 nests fields under "content"; older versions are flat.
    c = item.get("content", item)
    title = c.get("title")
    if not title:
        return None
    provider = c.get("provider")
    publisher = provider.get("displayName") if isinstance(provider, dict) else c.get("publisher")
    url = (c.get("canonicalUrl") or {}).get("url") if isinstance(c.get("canonicalUrl"), dict) else c.get("link")
    published = c.get("pubDate")
    if not published and c.get("providerPublishTime"):
        published = datetime.fromtimestamp(c["providerPublishTime"], tz=timezone.utc).isoformat()
    return {"title": title, "publisher": publisher, "url": url, "published": published,
            "summary": (c.get("summary") or "")[:300] or None}


def fetch_news(symbol: str, limit: int = 5, max_age_days: int = 10) -> list[dict]:
    try:
        raw = _ticker(symbol).news or []
    except Exception as exc:
        log.warning("news %s failed: %s", symbol, exc)
        return []
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=max_age_days)
    items = []
    for item in raw:
        parsed = _parse_news_item(item)
        if not parsed:
            continue
        if parsed["published"]:
            try:
                if pd.Timestamp(parsed["published"]).tz_convert("UTC") < cutoff:
                    continue
            except (ValueError, TypeError):
                pass
        items.append(parsed)
        if len(items) >= limit:
            break
    return items
