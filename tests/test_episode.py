import json
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from types import SimpleNamespace

import pandas as pd

from watchlist.episode import feed
from watchlist.episode.dossier import episode_title, ordinal, week_of
from watchlist.episode.script import template_script, validate, write_script
from watchlist.ledger import Ledger
from watchlist.pipeline import run_episode

from conftest import make_panel


class FakeClient:
    """Stands in for anthropic.Anthropic(): records requests, returns canned scripts."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **kwargs):
        self.requests.append(kwargs)
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        text, stop = resp if isinstance(resp, tuple) else (resp, "end_turn")
        msg = SimpleNamespace(stop_reason=stop, content=[SimpleNamespace(type="text", text=text)])

        class Ctx:
            def __enter__(self_inner):
                return SimpleNamespace(get_final_message=lambda: msg)

            def __exit__(self_inner, *a):
                return False

        return Ctx()


def _good_script(dossier, words=1000):
    parts = [f"Next up, {p['name']}, ticker {p['ticker_spoken']}. " for p in dossier["picks"]]
    filler = "This setup has been acting really well and I'm watching it closely. " * (words // 12)
    body = " ".join(parts) + filler + "I'm not a financial advisor, this isn't financial advice, always do your own DD."
    return json.dumps({"title": "x", "script": body, "show_notes_markdown": "# notes\n- $A: reason"})


def test_title_format_matches_original_show():
    assert ordinal(1) == "1st" and ordinal(2) == "2nd" and ordinal(3) == "3rd" and ordinal(11) == "11th" and ordinal(23) == "23rd"
    assert week_of(date(2026, 10, 4)) == date(2026, 10, 5)   # Sunday episode -> next Monday
    assert week_of(date(2026, 10, 5)) == date(2026, 10, 5)
    assert episode_title(date(2026, 10, 4), ["NVDA", "PLTR"]) == "Week of October 5th, 2026. $NVDA $PLTR"


def _run(cfg, client, when=date(2026, 10, 4), n_days=700):
    panel, bench = make_panel(n_tickers=50, n_days=n_days, seed=5)
    index_panel = bench  # SPY only is fine for tests
    cfg["universe"]["benchmark"] = "SPY"
    return run_episode(cfg, episode_date=when, audio=False, publish=True, panel=panel,
                       index_panel=index_panel, snapshot=None, fetch_details=False, client=client)


def test_episode_end_to_end_with_claude(cfg):
    client = FakeClient([])
    client.responses = [None]  # filled below once we know the dossier

    def stream(**kwargs):
        dossier = json.loads(kwargs["messages"][0]["content"].split("<dossier>\n")[1].split("\n</dossier>")[0])
        client.responses = [_good_script(dossier)]
        return FakeClient._stream(client, **kwargs)

    client.beta.messages.stream = stream
    out = _run(cfg, client)
    assert out.script.source == "claude", out.script.issues
    req = client.requests[0]
    assert req["model"] == "claude-opus-5-5"
    assert req["fallbacks"] == "default"
    assert req["output_config"]["format"]["type"] == "json_schema"
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert out.script.title.startswith("Week of October 5th, 2026.")
    for f in ("dossier.json", "script.txt", "show_notes.md", "meta.json"):
        assert (out.directory / f).exists()
    dossier = json.loads((out.directory / "dossier.json").read_text())
    assert len(dossier["picks"]) == 5
    assert all(p["levels"]["price"] for p in dossier["picks"])
    assert len(Ledger(cfg["publish"]["ledger_path"]).df) == 5
    site = feed.load_index(cfg)
    assert site[0]["date"] == "2026-10-04"


def test_falls_back_to_template_when_api_fails(cfg):
    out = _run(cfg, FakeClient([RuntimeError("no credentials"), RuntimeError("no credentials")]))
    assert out.script.source == "template"
    assert not [i for i in out.script.issues if "never mentions" in i or "disclaimer" in i]


def test_refusal_and_bad_drafts_fall_back(cfg):
    out = _run(cfg, FakeClient([(json.dumps({"title": "", "script": "", "show_notes_markdown": ""}), "refusal")]))
    assert out.script.source == "template"
    bad = json.dumps({"title": "t", "script": "Only talking about one stock. " * 100, "show_notes_markdown": "n"})
    client = FakeClient([bad, bad])
    out = _run(cfg, client)
    assert out.script.source == "template"
    assert "Fix all of them" in client.requests[1]["messages"][0]["content"]


def test_second_episode_has_scorecard_and_feed(cfg):
    fail = lambda: FakeClient([RuntimeError("x")] * 2)
    _run(cfg, fail(), when=date(2026, 9, 27), n_days=690)
    out = _run(cfg, fail(), when=date(2026, 10, 4), n_days=700)
    # Synthetic calendar dates differ from real ones; scorecard uses whatever prices follow the episode date.
    ledger = Ledger(cfg["publish"]["ledger_path"])
    assert len(ledger.episodes()) == 2
    assert out.dossier["last_week_scorecard"] is not None


def test_feed_xml_is_valid_podcast_rss(cfg):
    entry = {
        "date": "2026-10-04", "title": "Week of October 5th, 2026. $A $B", "guid": "g1",
        "pub_date": "Sun, 04 Oct 2026 22:00:00 +0000", "summary": "s", "notes_html": "<p>n</p>",
        "audio_url": "https://example.com/a.mp3", "audio_bytes": 1234, "duration_sec": 420,
    }
    xml = feed.build_feed(cfg, [entry, {**entry, "date": "2026-09-27", "guid": "g0", "audio_url": None}])
    root = ET.fromstring(xml.split("\n", 1)[1])
    items = root.findall("./channel/item")
    assert len(items) == 1  # episodes without audio are site-only
    enc = items[0].find("enclosure")
    assert enc.get("type") == "audio/mpeg" and enc.get("length") == "1234"
    assert root.find("./channel/{http://www.itunes.com/dtds/podcast-1.0.dtd}author").text == "Scott Wickman"


def test_markdown_to_html():
    html = feed.markdown_to_html("# Title\n\nHello **world**\n- one\n- two\n\n_not advice_")
    assert "<h1>Title</h1>" in html and "<strong>world</strong>" in html
    assert html.count("<li>") == 2 and "<em>not advice</em>" in html


def test_speech_normalizer():
    from watchlist.episode.tts import normalize_for_speech

    out = normalize_for_speech("ticker N-V-D-A near $142.50, up 12% in 2026. Russell 2000. Follow @scottiewick")
    assert "N V D A" in out and "142.50 dollars" in out and "12 percent" in out
    assert "twenty twenty-six" in out and "Russell 2000" in out and "Scottie Wick" in out
