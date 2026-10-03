"""Build the stock universe: listed US common stocks, filtered for price and liquidity."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
import requests

from ..config import resolve
from .prices import PricePanel

log = logging.getLogger(__name__)

NASDAQ_LISTED = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

# Security names that aren't common stock. ADRs ("American Depositary Shares") stay in: NIO is one.
_EXCLUDE_NAME = re.compile(
    r"%|\b(?:warrants?|units?|rights?|preferred|notes? due|debentures?|"
    r"depositary shares, each representing a fractional)\b",
    re.IGNORECASE,
)


def _parse_pipe_file(text: str) -> pd.DataFrame:
    lines = [ln for ln in text.strip().splitlines() if ln and not ln.startswith("File Creation Time")]
    rows = [ln.split("|") for ln in lines]
    return pd.DataFrame(rows[1:], columns=rows[0])


def parse_nasdaq_listed(text: str) -> list[str]:
    df = _parse_pipe_file(text)
    df = df[(df["Test Issue"] == "N") & (df["ETF"] == "N")]
    df = df[~df["Security Name"].str.contains(_EXCLUDE_NAME, na=False)]
    return [s for s in df["Symbol"] if s.isalpha()]


def parse_other_listed(text: str) -> list[str]:
    df = _parse_pipe_file(text)
    df = df[(df["Test Issue"] == "N") & (df["ETF"] == "N")]
    df = df[~df["Security Name"].str.contains(_EXCLUDE_NAME, na=False)]
    out = []
    for sym in df["ACT Symbol"]:
        if "$" in sym:  # preferred series
            continue
        out.append(sym.replace(".", "-"))  # BRK.B -> BRK-B (Yahoo format)
    return out


def fetch_listed_symbols(timeout: float = 30) -> list[str]:
    symbols: list[str] = []
    for url, parser in ((NASDAQ_LISTED, parse_nasdaq_listed), (OTHER_LISTED, parse_other_listed)):
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "stock-watchlist-agent"})
        resp.raise_for_status()
        symbols.extend(parser(resp.text))
    return sorted(set(symbols))


def base_universe(cfg: dict) -> list[str]:
    ucfg = cfg["universe"]
    if ucfg.get("source") == "file":
        path = resolve(ucfg["file"])
        symbols = [ln.strip().upper() for ln in Path(path).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    else:
        symbols = fetch_listed_symbols()
    symbols = set(symbols) | set(ucfg.get("extra", []))
    symbols -= set(ucfg.get("exclude", []))
    symbols.discard(ucfg.get("benchmark", "SPY"))
    return sorted(symbols)


def liquid_tickers(panel: PricePanel, min_price: float, min_dollar_volume: float, max_n: int | None) -> list[str]:
    """Tickers passing price/liquidity filters on the last date, most liquid first."""
    close = panel.close.ffill().iloc[-1]
    dollar_vol = (panel.close * panel.volume).rolling(20, min_periods=15).mean().iloc[-1]
    ok = (close >= min_price) & (dollar_vol >= min_dollar_volume)
    ranked = dollar_vol[ok].sort_values(ascending=False)
    if max_n:
        ranked = ranked.head(max_n)
    return list(ranked.index)
