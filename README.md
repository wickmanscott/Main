# The (Stock) Watchlist: research agent

This project automates **The (Stock) Watchlist**, Scott Wickman's podcast
(@scottiewick, 2020). Every week it scans the US market, scores stocks on momentum,
technicals and social buzz, picks **5 stocks to add to your watchlist**, writes the
episode in the show's original style with Claude, voices it, and publishes it to a
podcast RSS feed.

The scoring is backtested walk-forward, so you can measure how often its picks would
have worked before trusting them, and every published pick is scored the following week.

> Not financial advice. This is a research and content tool. Do your own DD.

## What it learned from the original show

From the 2020 archive (11 of the 17 episodes were recoverable, see `data/archive/episodes_2020.json`):

- **Format:** 5 tickers per episode, kept brief. Titles were just the date and the cashtags:
  `Week of August 3rd, 2020. $GBTC $MARA $RIOT $AAPL $MSFT`.
- **What got picked:** names riding a story with momentum behind it: SPACs (SPAQ, FVAC, LCA),
  EVs (TSLA, NKLA, NIO, BLNK), vaccines (MRNA, NVAX), sports betting (DKNG, PENN), and
  crypto proxies (MARA, RIOT, GBTC).
- **Hot names repeated.** DKNG and SPAQ each appeared 6 times in two weeks.
- **Mix:** one or two mega-cap anchors plus smaller names most people hadn't heard of yet.

These became the agent's rules. `prompts/style_guide.md` holds the voice and episode
structure. `config/settings.yaml` sets the 5-slot mix (3 momentum leaders, 1 social-buzz
name, 1 under-the-radar sleeper) and lets a winner stay on the list for up to 3 weeks.

## How a pick gets made

```
symbol directory (~6,000 US stocks)
  -> liquidity filter (price >= $2, $5M+/day)      -> ~3,000 stocks
  -> features for every stock, every day           (only past data; tested for look-ahead)
  -> composite score = weighted percentile ranks   (weights tuned by walk-forward backtest)
  -> 5 slots: 3 momentum + 1 social + 1 sleeper    (gates: uptrend, not overextended, sector cap)
  -> dossier: levels, news, earnings, risks        -> Claude writes the script -> TTS -> RSS
```

| Signal | What it measures | Why |
|---|---|---|
| Relative strength (IBD-style) | 3/6/9/12-month return, most recent quarter weighted double | Momentum is one of the best-documented return anomalies |
| 12-1 momentum | 12-month return excluding the last month | Classic academic momentum factor; skipping the last month avoids short-term reversal |
| 52-week-high proximity | Price / 52-week high | George & Hwang (2004): stocks near highs keep outperforming |
| Acceleration | 1-month pace vs 3-month pace | Catches names that are speeding up |
| Trend template | Minervini's 8 stage-2 checks (price above rising 50/150/200-day, etc.) | Filters for real uptrends |
| RSI sweet spot | RSI near 65 scores highest | Strong but not blown out |
| Up/down volume | Volume on up days vs down days, 50 days | Accumulation |
| Relative volume | 5-day vs 50-day volume | Fresh interest |
| Tightness | 10-day ATR / 50-day ATR | Volatility contraction before breakouts |
| Breakout | New 20-day high on 1.3x+ volume this week | Fresh entry trigger |
| Social buzz (live) | Reddit mention acceleration (ApeWisdom), StockTwits trending | The 2020 show was driven by retail buzz |
| Sleeper score | Strong RS + accumulation + tight base + **low attention** | "Flying under the radar" with room to run |

## Historical accuracy

```bash
watchlist backtest --years 5             # replay the exact weekly pick logic
watchlist backtest --years 5 --optimize  # also re-tune the weights, walk-forward
watchlist audit-archive                  # how the original 2020 picks actually did
```

- **Point-in-time.** On each historical Friday the model sees only data up to that day.
  Picks are bought at Monday's open and held 5, 10 or 21 trading days, after a 10 bps cost.
  A test changes future prices and checks that no past score moves. Another runs the model
  on pure noise and checks that it finds no edge.
- **Metrics:** hit rate, beat-SPY rate, average excess return, rank IC (does a higher score
  mean a higher return across the whole universe?), top-minus-bottom quintile spread,
  equity curve, Sharpe and max drawdown. Results are also split by slot.
- **Walk-forward tuning.** Weights are random-searched on 2 years of data, tested on the
  next 6 months, then the window rolls forward. Only that out-of-sample record counts. The
  new weights replace the defaults (`config/weights.json`) only if they beat them out of
  sample. A monthly GitHub Action reruns this.
- **Live record.** `data/ledger/picks.csv` logs every published pick. Each episode opens
  with last week's scorecard, and the site shows the running hit rate vs SPY.

**Caveats, stated plainly:** today's stock list leaves out companies that went bust
(survivorship bias), which flatters absolute returns. Rank IC is less affected. Free social
data has no history, so social buzz can't be backtested yet. Every run saves a snapshot to
`data/social_history/`, and after a few months of snapshots it can be tested. No backtest
guarantees future results.

## Setup

1. **Install** (Python 3.10+):
   ```bash
   pip install -e ".[dev]"
   pytest
   ```
2. **Add repository secrets** (Settings → Secrets and variables → Actions):
   - `ANTHROPIC_API_KEY`: Claude writes the scripts. Without it, a plainer template script is used.
   - Optional `ELEVENLABS_API_KEY` + `ELEVENLABS_VOICE_ID`: to have **your own cloned
     voice** read the episodes. Clone your voice from your old episodes on ElevenLabs, then
     set `tts.provider: elevenlabs` in `config/settings.yaml`. The default is a free
     Microsoft neural voice (edge-tts).
3. **Turn on GitHub Pages:** Settings → Pages → deploy from the default branch, folder `/docs`.
   Your feed will be at `https://wickmanscott.github.io/main/feed.xml`. Update
   `publish.site_url` and `publish.audio_url_template` if the repo or account name changes.
4. **Add cover art:** set `show.artwork_url` (a square image, 1400 to 3000 px). Apple
   requires it.
5. **First run:** Actions → *Weekly episode* → Run workflow. After that it runs every
   Sunday. Each run commits the episode files, ledger and feed, and attaches the MP3 to a
   GitHub Release.
6. **Submit the feed** once to Apple Podcasts Connect and Spotify for Creators. New episodes
   then appear automatically.

Run the backtest once early (Actions → *Monthly backtest and re-tune* → Run workflow) so the
first episode uses tuned weights and the reports exist.

## Commands

```bash
watchlist scan --top 30            # ranked universe and this week's 5 picks, no publishing
watchlist episode --no-publish     # full episode into episodes/<date>/ without touching the feed
watchlist episode                  # research, write, voice, publish
watchlist backtest --optimize      # walk-forward backtest and weight tuning -> reports/
watchlist audit-archive            # 2020 picks: 1w/1m/3m returns vs SPY, traits of winners
watchlist scorecard                # live track record of published picks
watchlist feed                     # rebuild docs/feed.xml and docs/index.html
```

`--max-tickers 500` makes any run faster. `--universe-file my_list.txt` scores your own list.

## Layout

```
config/settings.yaml        all knobs: universe, weights, slots, backtest, LLM, TTS, publishing
config/weights.json         tuned weights (written by the backtest when they beat the defaults OOS)
prompts/style_guide.md      the show's voice and structure, given to Claude every episode
data/archive/               recovered 2020 episodes, themes, renamed-ticker map
src/watchlist/
  data/                     prices (Yahoo, cached parquet), universe, social, news/profiles
  signals/                  indicators + per-day feature panels
  scoring.py                percentile scores, composite, sleeper, 5-slot selection
  backtest/                 walk-forward engine, optimizer, 2020 archive audit
  episode/                  dossier, Claude script writer, TTS, RSS/site
  ledger.py                 published picks, scorecard, live track record
  pipeline.py, cli.py
episodes/<date>/            dossier.json, script.txt, show_notes.md, meta.json (+ mp3 in Releases)
docs/                       feed.xml + index.html (GitHub Pages)
reports/                    backtest_latest.md/json, equity curves, archive audit
```

## Data sources

All free, no keys: Yahoo Finance via `yfinance` (prices, profiles, news, earnings dates),
the NASDAQ Trader symbol directory (universe), ApeWisdom (Reddit mentions), and StockTwits
(trending). Yahoo sometimes rate-limits cloud IPs. Downloads retry, but if it keeps failing,
`data/prices.py` has `load_csv_dir()` for plugging in another vendor's CSVs.
