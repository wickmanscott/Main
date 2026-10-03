"""End-to-end: market data -> scores -> picks -> dossier -> script -> audio -> feed."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from email.utils import format_datetime
from pathlib import Path

import pandas as pd

from .config import resolve
from .data import news as news_mod
from .data.prices import PricePanel, load_or_download
from .data.social import SocialSnapshot, social_frame, take_snapshot
from .data.universe import base_universe, liquid_tickers
from .episode import feed
from .episode.dossier import build_dossier, market_pulse, pick_dossier
from .episode.script import Script, write_script
from .episode.tts import synthesize
from .ledger import Ledger, scorecard, track_record
from .report import load_model_card
from .scoring import Model, Pick, select_picks
from .signals.features import compute_features

log = logging.getLogger(__name__)
INDEXES = ["QQQ", "IWM"]


def load_market(cfg: dict, extra: list[str] | tuple = (), lookback_days: int | None = None
                ) -> tuple[PricePanel, PricePanel, list[str]]:
    """Two-stage download: a quick 45-day pull to find liquid names, then full history for those.

    Returns (panel, index_panel, liquid). `panel` also holds `extra` (past picks, for the scorecard);
    `liquid` is the universe that gets scored.
    """
    ucfg, dcfg = cfg["universe"], cfg["data"]
    cache = resolve(dcfg["cache_dir"])
    today = pd.Timestamp.today().normalize()
    symbols = base_universe(cfg)
    quick = load_or_download(symbols, today - pd.Timedelta(days=45), cache)
    liquid = liquid_tickers(quick, ucfg["min_price"], ucfg["min_avg_dollar_volume"], ucfg.get("max_tickers"))
    wanted = sorted(set(liquid) | set(extra))
    start = today - pd.Timedelta(days=lookback_days or dcfg["lookback_days"])
    panel = load_or_download(wanted, start, cache)
    index_panel = load_or_download([ucfg["benchmark"], *INDEXES], start, cache)
    index_panel = index_panel.select([ucfg["benchmark"], *INDEXES])
    return panel, index_panel, [t for t in liquid if t in panel.close.columns]


@dataclass
class Scan:
    asof: pd.Timestamp
    model: Model
    xs: pd.DataFrame
    picks: list[Pick]
    profiles: dict[str, dict] = field(default_factory=dict)


def scan(cfg: dict, panel: PricePanel, index_panel: PricePanel, snapshot: SocialSnapshot | None,
         ledger: Ledger | None = None, fetch_profiles: bool = True, universe: list[str] | None = None,
         episode_date: date | None = None) -> Scan:
    ucfg = cfg["universe"]
    bench = index_panel.close[ucfg["benchmark"]]
    scoring_panel = panel.select(universe) if universe else panel
    fs = compute_features(scoring_panel, bench)
    model = Model.build(fs, cfg["weights"], ucfg["min_price"], ucfg["min_avg_dollar_volume"])
    asof = fs.dates[-1]
    social = social_frame(snapshot, scoring_panel.tickers) if snapshot and (snapshot.reddit or snapshot.stocktwits_trending) else None
    xs = model.cross_section(asof, social=social, social_weight=cfg.get("social_weight", 0.0) if social is not None else 0.0)

    # Profiles (sector, market cap) only for plausible picks: keeps API calls to ~25 a week.
    shortlist = list(xs.index[:15])
    if "sleeper" in xs:
        shortlist += list(xs["sleeper"].dropna().sort_values(ascending=False).index[:5])
    if social is not None:
        shortlist += list(xs["social_score"].sort_values(ascending=False).index[:5])
    profiles = {t: news_mod.fetch_profile(t) for t in dict.fromkeys(shortlist)} if fetch_profiles else {}
    sectors = {t: p.get("sector") for t, p in profiles.items() if p.get("sector")}
    # Episodes before this one (re-running an episode shouldn't count itself as a repeat).
    streaks = ledger.streaks(before=pd.Timestamp(episode_date or date.today())) if ledger else {}
    picks = select_picks(xs, cfg, model.weights, streaks=streaks, sectors=sectors)
    return Scan(asof, model, xs, picks, profiles)


@dataclass
class EpisodeOutput:
    episode_date: date
    directory: Path
    dossier: dict
    script: Script
    audio: dict | None


def run_episode(cfg: dict, *, episode_date: date | None = None, audio: bool = True, publish: bool = True,
                panel: PricePanel | None = None, index_panel: PricePanel | None = None,
                snapshot: SocialSnapshot | None = None, fetch_details: bool = True, client=None) -> EpisodeOutput:
    episode_date = episode_date or date.today()
    ledger = Ledger(cfg["publish"]["ledger_path"])
    universe = None
    if panel is None or index_panel is None:
        panel, index_panel, universe = load_market(cfg, extra=ledger.tickers)
    if snapshot is None and fetch_details:
        snapshot = take_snapshot(cfg)

    result = scan(cfg, panel, index_panel, snapshot, ledger, fetch_profiles=fetch_details, universe=universe,
                  episode_date=episode_date)
    bench_panel = index_panel.select([cfg["universe"]["benchmark"]])
    card = scorecard(ledger, panel, bench_panel, before=pd.Timestamp(episode_date))
    track = track_record(ledger, panel, bench_panel)

    pick_docs = []
    for p in result.picks:
        profile = result.profiles.get(p.ticker) or (news_mod.fetch_profile(p.ticker) if fetch_details else {})
        headlines = news_mod.fetch_news(p.ticker) if fetch_details else []
        earnings = news_mod.fetch_next_earnings(p.ticker) if fetch_details else None
        pick_docs.append(pick_dossier(p, result.xs.loc[p.ticker], profile, headlines, earnings, episode_date))

    dossier = build_dossier(cfg, episode_date, pick_docs, market_pulse(index_panel, result.xs), card, track, load_model_card())
    dossier["data_as_of"] = result.asof.date().isoformat()
    script = write_script(dossier, cfg, client=client)

    out_dir = resolve(cfg["publish"]["episodes_dir"]) / episode_date.isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dossier.json").write_text(json.dumps(dossier, indent=1, default=str))
    (out_dir / "script.txt").write_text(script.script + "\n")
    (out_dir / "show_notes.md").write_text(script.show_notes + "\n")
    (out_dir / "meta.json").write_text(json.dumps({
        "title": script.title, "script_source": script.source, "words": script.words,
        "validation_issues": script.issues, "weights_source": cfg.get("weights_source"),
        "picks": [p.to_dict() for p in result.picks],
    }, indent=1, default=str))

    audio_info = None
    if audio:
        fname = f"watchlist-{episode_date.isoformat()}.mp3"
        try:
            audio_info = synthesize(script.script, out_dir / fname, cfg)
        except Exception as exc:
            log.error("Text-to-speech failed, publishing without audio: %s", exc)
        if audio_info:
            tag = f"episode-{episode_date.isoformat()}"
            audio_info["file"], audio_info["tag"] = fname, tag
            audio_info["url"] = cfg["publish"]["audio_url_template"].format(tag=tag, file=fname)

    if publish:
        ref = result.xs["close"].to_dict()
        ledger.append(episode_date, result.picks, ref)
        entry = {
            "date": episode_date.isoformat(),
            "title": script.title,
            "guid": f"stock-watchlist-{episode_date.isoformat()}",
            "pub_date": format_datetime(datetime.now(timezone.utc)),
            "summary": f"5 stocks for your watchlist: {', '.join(p.ticker for p in result.picks)}. Not financial advice.",
            "notes_html": feed.markdown_to_html(script.show_notes),
            "tickers": [p.ticker for p in result.picks],
            "script_source": script.source,
            "audio_url": audio_info["url"] if audio_info else None,
            "audio_bytes": audio_info["bytes"] if audio_info else None,
            "duration_sec": audio_info["duration_sec"] if audio_info else None,
        }
        items = feed.upsert(cfg, entry)
        feed.publish_site(cfg, items, track_record(ledger, panel, bench_panel))
    return EpisodeOutput(episode_date, out_dir, dossier, script, audio_info)
