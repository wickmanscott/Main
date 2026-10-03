"""Podcast RSS feed (Apple/Spotify compatible) and a one-page site, served from docs/ by GitHub Pages."""

from __future__ import annotations

import html
import json
import re
import xml.etree.ElementTree as ET
from email.utils import format_datetime
from datetime import datetime, timezone
from pathlib import Path

from ..config import resolve

ITUNES = "http://www.itunes.com/dtds/podcast-1.0.dtd"
CONTENT = "http://purl.org/rss/1.0/modules/content/"
ATOM = "http://www.w3.org/2005/Atom"
ET.register_namespace("itunes", ITUNES)
ET.register_namespace("content", CONTENT)
ET.register_namespace("atom", ATOM)


def index_path(cfg: dict) -> Path:
    return resolve(cfg["publish"]["episodes_dir"]) / "index.json"


def load_index(cfg: dict) -> list[dict]:
    p = index_path(cfg)
    return json.loads(p.read_text()) if p.exists() else []


def save_index(cfg: dict, items: list[dict]) -> None:
    p = index_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    items = sorted(items, key=lambda e: e["date"], reverse=True)
    p.write_text(json.dumps(items, indent=1))


def upsert(cfg: dict, entry: dict) -> list[dict]:
    items = [e for e in load_index(cfg) if e["date"] != entry["date"]]
    items.append(entry)
    save_index(cfg, items)
    return load_index(cfg)


def markdown_to_html(md: str) -> str:
    """Just enough Markdown for show notes: headings, bullets, bold, italics, paragraphs."""
    out, in_list = [], False
    for line in md.splitlines():
        s = html.escape(line.strip())
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<em>\1</em>", s)
        if s.startswith(("- ", "* ")):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{s[2:]}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        if s.startswith("#"):
            level = min(len(s) - len(s.lstrip("#")), 4)
            out.append(f"<h{level}>{s.lstrip('#').strip()}</h{level}>")
        elif s:
            out.append(f"<p>{s}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def _sub(parent, tag, text=None, **attrs):
    el = ET.SubElement(parent, tag, attrs)
    if text is not None:
        el.text = text
    return el


def build_feed(cfg: dict, items: list[dict]) -> str:
    show, pub = cfg["show"], cfg["publish"]
    site = pub["site_url"].rstrip("/")
    rss = ET.Element("rss", {"version": "2.0"})
    ch = ET.SubElement(rss, "channel")
    _sub(ch, "title", show["title"])
    _sub(ch, "link", site)
    _sub(ch, "description", show["description"].strip())
    _sub(ch, "language", show.get("language", "en-us"))
    _sub(ch, "lastBuildDate", format_datetime(datetime.now(timezone.utc)))
    ET.SubElement(ch, f"{{{ATOM}}}link", {"href": f"{site}/feed.xml", "rel": "self", "type": "application/rss+xml"})
    _sub(ch, f"{{{ITUNES}}}author", show["host"])
    _sub(ch, f"{{{ITUNES}}}summary", show["description"].strip())
    _sub(ch, f"{{{ITUNES}}}explicit", "true" if show.get("explicit") else "false")
    _sub(ch, f"{{{ITUNES}}}type", "episodic")
    owner = ET.SubElement(ch, f"{{{ITUNES}}}owner")
    _sub(owner, f"{{{ITUNES}}}name", show["host"])
    if show.get("email"):
        _sub(owner, f"{{{ITUNES}}}email", show["email"])
    if show.get("artwork_url"):
        ET.SubElement(ch, f"{{{ITUNES}}}image", {"href": show["artwork_url"]})
        img = ET.SubElement(ch, "image")
        _sub(img, "url", show["artwork_url"])
        _sub(img, "title", show["title"])
        _sub(img, "link", site)
    cat = ET.SubElement(ch, f"{{{ITUNES}}}category", {"text": show.get("category", "Business")})
    if show.get("subcategory"):
        ET.SubElement(cat, f"{{{ITUNES}}}category", {"text": show["subcategory"]})

    for e in sorted(items, key=lambda x: x["date"], reverse=True):
        if not e.get("audio_url"):
            continue  # podcast apps need an enclosure; text-only episodes stay on the site
        it = ET.SubElement(ch, "item")
        _sub(it, "title", e["title"])
        _sub(it, "guid", e["guid"], isPermaLink="false")
        _sub(it, "pubDate", e["pub_date"])
        _sub(it, "link", f"{site}/#{e['date']}")
        _sub(it, "description", e["summary"])
        _sub(it, f"{{{CONTENT}}}encoded", e["notes_html"])
        ET.SubElement(it, "enclosure", {"url": e["audio_url"], "length": str(e["audio_bytes"]), "type": "audio/mpeg"})
        _sub(it, f"{{{ITUNES}}}duration", str(e.get("duration_sec", 0)))
        _sub(it, f"{{{ITUNES}}}explicit", "true" if show.get("explicit") else "false")
        _sub(it, f"{{{ITUNES}}}episodeType", "full")
    ET.indent(rss)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(rss, encoding="unicode") + "\n"


def build_site(cfg: dict, items: list[dict], track: dict | None = None) -> str:
    show = cfg["show"]
    site = cfg["publish"]["site_url"].rstrip("/")
    rec = ""
    if track:
        rec = (f"<p class='rec'>Live track record: {track['picks_scored']} picks over {track['episodes']} episodes. "
               f"{track['hit_rate']:.0%} went up and {track['beat_benchmark_rate']:.0%} beat SPY over "
               f"{track['hold_sessions']} trading days (avg excess {track['avg_excess']:+.2%}).</p>")
    eps = []
    for e in sorted(items, key=lambda x: x["date"], reverse=True):
        audio = f"<audio controls preload='none' src='{html.escape(e['audio_url'])}'></audio>" if e.get("audio_url") else ""
        eps.append(f"<article id='{e['date']}'>{audio}{e['notes_html']}</article>")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(show['title'])}</title>
<link rel="alternate" type="application/rss+xml" title="{html.escape(show['title'])}" href="{site}/feed.xml">
<style>
:root {{ --bg:#fff; --fg:#111; --muted:#666; --line:#e5e5e5; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#111; --fg:#eee; --muted:#999; --line:#333; }} }}
body {{ background:var(--bg); color:var(--fg); font:16px/1.55 system-ui,sans-serif; max-width:760px; margin:0 auto; padding:24px 16px; }}
header p, .rec {{ color:var(--muted); }} article {{ border-top:1px solid var(--line); padding:16px 0; }}
audio {{ width:100%; margin:8px 0; }} a {{ color:inherit; }}
</style></head><body>
<header><h1>{html.escape(show['title'])}</h1><p>{html.escape(show['tagline'])} Hosted by {html.escape(show['host'])} ({html.escape(show['handle'])}).</p>
<p><a href="{site}/feed.xml">RSS feed</a></p>{rec}</header>
{''.join(eps) or '<p>No episodes yet.</p>'}
</body></html>
"""


def publish_site(cfg: dict, items: list[dict], track: dict | None = None) -> Path:
    site_dir = resolve(cfg["publish"]["site_dir"])
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "feed.xml").write_text(build_feed(cfg, items))
    (site_dir / "index.html").write_text(build_site(cfg, items, track))
    (site_dir / ".nojekyll").write_text("")
    return site_dir
