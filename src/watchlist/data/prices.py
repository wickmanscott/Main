"""Daily OHLCV price panels: wide DataFrames (dates x tickers), cached to parquet."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

log = logging.getLogger(__name__)

FIELDS = ("open", "high", "low", "close", "volume")


@dataclass
class PricePanel:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame

    @property
    def tickers(self) -> list[str]:
        return list(self.close.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    def frames(self) -> dict[str, pd.DataFrame]:
        return {f: getattr(self, f) for f in FIELDS}

    def asof(self, date) -> "PricePanel":
        """Only rows on or before `date`, so nothing downstream can see the future."""
        date = pd.Timestamp(date)
        return PricePanel(**{f: df.loc[:date] for f, df in self.frames().items()})

    def select(self, tickers: Iterable[str]) -> "PricePanel":
        cols = [t for t in tickers if t in self.close.columns]
        return PricePanel(**{f: df[cols] for f, df in self.frames().items()})

    def series(self, ticker: str) -> pd.DataFrame:
        """One ticker as a long OHLCV frame."""
        return pd.DataFrame({f: getattr(self, f)[ticker] for f in FIELDS}).dropna(how="all")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for f, df in self.frames().items():
            df.to_parquet(path / f"{f}.parquet")

    @classmethod
    def load(cls, path: str | Path) -> "PricePanel":
        path = Path(path)
        return cls(**{f: pd.read_parquet(path / f"{f}.parquet") for f in FIELDS})

    @classmethod
    def from_frames(cls, frames: dict[str, pd.DataFrame]) -> "PricePanel":
        close = frames["close"].sort_index()
        aligned = {f: frames[f].reindex(index=close.index, columns=close.columns) for f in FIELDS}
        return cls(**aligned)

    @classmethod
    def concat(cls, panels: list["PricePanel"]) -> "PricePanel":
        frames = {}
        for f in FIELDS:
            parts = [getattr(p, f) for p in panels if not getattr(p, f).empty]
            frames[f] = pd.concat(parts, axis=1) if parts else pd.DataFrame()
            frames[f] = frames[f].loc[:, ~frames[f].columns.duplicated()]
        return cls.from_frames(frames)


def _yf_frame_to_panel(raw: pd.DataFrame, tickers: list[str]) -> PricePanel:
    """Normalize yfinance's download() output (MultiIndex or flat columns) to a PricePanel."""
    if raw is None or raw.empty:
        empty = pd.DataFrame(columns=tickers, dtype=float)
        return PricePanel(**{f: empty.copy() for f in FIELDS})
    frames = {}
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = {str(c).lower() for c in raw.columns.get_level_values(0)}
        field_level = 0 if "close" in level0 else 1
        for f in FIELDS:
            sub = raw.xs(f.capitalize(), axis=1, level=field_level)
            frames[f] = sub.astype(float)
    else:  # single ticker, flat columns
        for f in FIELDS:
            frames[f] = raw[[f.capitalize()]].rename(columns={f.capitalize(): tickers[0]}).astype(float)
    for f in frames:
        frames[f].index = pd.DatetimeIndex(frames[f].index).tz_localize(None).normalize()
    return PricePanel.from_frames(frames)


def download_yfinance(
    tickers: Iterable[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
    chunk_size: int = 200,
    pause: float = 1.0,
) -> PricePanel:
    """Download split/dividend-adjusted daily bars from Yahoo Finance in chunks."""
    import yfinance as yf

    tickers = sorted(set(tickers))
    panels = []
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        log.info("Downloading prices %d-%d of %d", i + 1, i + len(chunk), len(tickers))
        for attempt in range(3):  # Yahoo rate-limits bursts from cloud IPs; back off and retry
            try:
                raw = yf.download(
                    chunk,
                    start=pd.Timestamp(start).strftime("%Y-%m-%d"),
                    end=None if end is None else pd.Timestamp(end).strftime("%Y-%m-%d"),
                    auto_adjust=True,
                    group_by="column",
                    threads=True,
                    progress=False,
                )
                break
            except Exception as exc:
                if attempt == 2:
                    raise
                log.warning("price download failed (%s), retrying in %ds", exc, 20 * (attempt + 1))
                time.sleep(20 * (attempt + 1))
        panels.append(_yf_frame_to_panel(raw, chunk))
        if pause and i + chunk_size < len(tickers):
            time.sleep(pause)
    panel = PricePanel.concat(panels)
    # Drop tickers with no data at all (delisted / bad symbols).
    keep = panel.close.columns[panel.close.notna().any()]
    return panel.select(keep)


def cache_key(tickers: Iterable[str], start, end=None) -> str:
    h = hashlib.sha1(",".join(sorted(set(tickers))).encode()).hexdigest()[:10]
    end_s = "latest" if end is None else pd.Timestamp(end).strftime("%Y%m%d")
    return f"{pd.Timestamp(start).strftime('%Y%m%d')}_{end_s}_{h}"


def load_or_download(
    tickers: Iterable[str],
    start,
    cache_dir: str | Path,
    end=None,
    max_age_hours: float = 12,
) -> PricePanel:
    tickers = sorted(set(tickers))
    path = Path(cache_dir) / "prices" / cache_key(tickers, start, end)
    marker = path / "close.parquet"
    if marker.exists() and (time.time() - marker.stat().st_mtime) < max_age_hours * 3600:
        log.info("Using cached prices at %s", path)
        return PricePanel.load(path)
    panel = download_yfinance(tickers, start, end)
    panel.save(path)
    return panel


def load_csv_dir(path: str | Path) -> PricePanel:
    """Load one CSV per ticker (Date,Open,High,Low,Close,Volume) — for offline use or other vendors."""
    frames: dict[str, dict[str, pd.Series]] = {f: {} for f in FIELDS}
    for csv in sorted(Path(path).glob("*.csv")):
        df = pd.read_csv(csv, parse_dates=[0], index_col=0)
        df.columns = [c.lower().replace("adj close", "adj_close") for c in df.columns]
        for f in FIELDS:
            frames[f][csv.stem.upper()] = df[f]
    return PricePanel.from_frames({f: pd.DataFrame(v) for f, v in frames.items()})
