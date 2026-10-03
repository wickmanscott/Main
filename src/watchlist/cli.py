"""Command line: `watchlist scan | episode | backtest | audit-archive | scorecard | feed`."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date

import pandas as pd

from .config import ROOT, load_config


def _cfg(args) -> dict:
    overrides = {}
    if getattr(args, "max_tickers", None):
        overrides.setdefault("universe", {})["max_tickers"] = args.max_tickers
    if getattr(args, "universe_file", None):
        overrides.setdefault("universe", {}).update(source="file", file=args.universe_file)
    if getattr(args, "tts", None):
        overrides.setdefault("tts", {})["provider"] = args.tts
    return load_config(args.config, overrides)


def cmd_scan(args) -> None:
    from .data.social import take_snapshot
    from .ledger import Ledger
    from .pipeline import load_market, scan

    cfg = _cfg(args)
    ledger = Ledger(cfg["publish"]["ledger_path"])
    panel, index_panel, universe = load_market(cfg)
    snap = take_snapshot(cfg, save=False)
    res = scan(cfg, panel, index_panel, snap, ledger, fetch_profiles=not args.fast, universe=universe)
    cols = ["composite", "rs_rating", "trend_template", "prox_52w_high", "ret_1m", "rel_volume", "extension", "sleeper", "social_score"]
    print(f"Data as of {res.asof.date()} | {len(res.xs)} eligible stocks | weights: {cfg['weights_source']}\n")
    print(res.xs[cols].head(args.top).round(3).to_string())
    print("\nThis week's picks:")
    for p in res.picks:
        print(f"  {p.ticker:6s} {p.slot:9s} composite {p.composite:.3f} (#{p.composite_rank})")
        for r in p.reasons:
            print(f"           - {r}")


def cmd_episode(args) -> None:
    from .pipeline import run_episode

    cfg = _cfg(args)
    out = run_episode(cfg, episode_date=date.fromisoformat(args.date) if args.date else None,
                      audio=not args.no_audio, publish=not args.no_publish)
    print(f"Episode written to {out.directory}")
    print(f"Title: {out.script.title}")
    print(f"Script: {out.script.words} words via {out.script.source}" + (f" (issues: {out.script.issues})" if out.script.issues else ""))
    if out.audio:
        print(f"Audio: {out.audio['path']} ({out.audio['duration_sec'] // 60}m{out.audio['duration_sec'] % 60:02d}s)")
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:  # hand file paths to the next workflow steps
        with open(gh_out, "a") as fh:
            fh.write(f"date={out.episode_date.isoformat()}\n")
            fh.write(f"title={out.script.title}\n")
            fh.write(f"notes_path={out.directory / 'show_notes.md'}\n")
            if out.audio:
                fh.write(f"audio_path={out.audio['path']}\naudio_tag={out.audio['tag']}\n")


def cmd_backtest(args) -> None:
    from .backtest.engine import prepare, run
    from .backtest.optimize import optimize
    from .data.prices import load_or_download
    from .data.universe import base_universe, liquid_tickers
    from .report import save_backtest

    cfg = _cfg(args)
    if args.years:
        cfg["backtest"]["years"] = args.years
    years = cfg["backtest"]["years"]
    cache = ROOT / cfg["data"]["cache_dir"]
    today = pd.Timestamp.today().normalize()
    start = today - pd.DateOffset(years=years) - pd.Timedelta(days=400)  # + warm-up for 52-week features
    u = cfg["universe"]
    quick = load_or_download(base_universe(cfg), today - pd.Timedelta(days=45), cache)
    tickers = liquid_tickers(quick, u["min_price"], u["min_avg_dollar_volume"], u.get("max_tickers"))
    panel = load_or_download(tickers, start, cache, max_age_hours=72)
    bench = load_or_download([u["benchmark"]], start, cache, max_age_hours=72)
    prep = prepare(panel, bench, cfg, start=today - pd.DateOffset(years=years))
    print(f"Backtesting {len(panel.tickers)} stocks over {len(prep.dates)} weekly periods...")
    results = {h: run(prep, cfg, h) for h in cfg["backtest"]["horizons"]}
    opt = optimize(prep, cfg) if args.optimize else None
    path = save_backtest(results, opt, cfg, write_weights=args.optimize)
    print(path.read_text())


def cmd_audit(args) -> None:
    from .backtest.audit import audit_markdown, run_audit

    df, summary = run_audit()
    md = audit_markdown(df, summary)
    out = ROOT / "reports" / "archive_audit_2020.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(md)
    print(md)


def cmd_scorecard(args) -> None:
    from .ledger import Ledger, scorecard, track_record
    from .data.prices import load_or_download

    cfg = _cfg(args)
    ledger = Ledger(cfg["publish"]["ledger_path"])
    if not ledger.tickers:
        print("No published picks yet.")
        return
    start = min(ledger.episodes()) - pd.Timedelta(days=10)
    cache = ROOT / cfg["data"]["cache_dir"]
    panel = load_or_download(ledger.tickers, start, cache)
    bench = load_or_download([cfg["universe"]["benchmark"]], start, cache)
    print(json.dumps({"last_episode": scorecard(ledger, panel, bench), "track_record": track_record(ledger, panel, bench)},
                     indent=1, default=str))


def cmd_feed(args) -> None:
    from .episode import feed

    cfg = _cfg(args)
    path = feed.publish_site(cfg, feed.load_index(cfg))
    print(f"Rebuilt {path}/feed.xml and index.html")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="watchlist", description="The (Stock) Watchlist research agent")
    ap.add_argument("--config", default=None, help="path to settings.yaml")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--max-tickers", type=int, help="cap the universe (faster runs)")
        p.add_argument("--universe-file", help="score only the tickers in this file")

    p = sub.add_parser("scan", help="rank the universe and show this week's picks")
    common(p)
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--fast", action="store_true", help="skip company profile lookups")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("episode", help="research, write, voice and publish this week's episode")
    common(p)
    p.add_argument("--date", help="episode date (YYYY-MM-DD), default today")
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--no-publish", action="store_true", help="write files but don't touch the ledger or feed")
    p.add_argument("--tts", choices=["edge", "elevenlabs", "none"])
    p.set_defaults(func=cmd_episode)

    p = sub.add_parser("backtest", help="walk-forward backtest; --optimize re-tunes the weights")
    common(p)
    p.add_argument("--years", type=int)
    p.add_argument("--optimize", action="store_true")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("audit-archive", help="score the original 2020 episodes' picks")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("scorecard", help="live track record of published picks")
    p.set_defaults(func=cmd_scorecard)

    p = sub.add_parser("feed", help="rebuild docs/feed.xml and docs/index.html")
    p.set_defaults(func=cmd_feed)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
