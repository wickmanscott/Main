"""Write the episode script in Scott's style with Claude, grounded only in the dossier."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from ..config import STYLE_GUIDE

log = logging.getLogger(__name__)

SCRIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "script": {"type": "string"},
        "show_notes_markdown": {"type": "string"},
    },
    "required": ["title", "script", "show_notes_markdown"],
    "additionalProperties": False,
}


@dataclass
class Script:
    title: str
    script: str
    show_notes: str
    source: str  # "claude" or "template"
    issues: list[str] = field(default_factory=list)

    @property
    def words(self) -> int:
        return len(self.script.split())


def _mentions(text: str, pick: dict) -> bool:
    t = text.lower()
    spoken = pick["ticker_spoken"].lower()
    name = (pick.get("name") or "").lower()
    first_word = re.split(r"[\s,.]", name)[0] if name else ""
    return (spoken in t or re.search(rf"\b{re.escape(pick['ticker'].lower())}\b", t) is not None
            or (len(first_word) > 3 and first_word in t))


def validate(script: Script, dossier: dict, target_words: tuple[int, int]) -> list[str]:
    issues = []
    for p in dossier["picks"]:
        if not _mentions(script.script, p):
            issues.append(f"script never mentions {p['ticker']} ({p['name']})")
    low = script.script.lower()
    if "financial advi" not in low or not re.search(r"\bdd\b|due diligence|own research", low):
        issues.append("missing the disclaimer (not a financial advisor, do your own DD)")
    lo, hi = target_words
    if script.words < lo * 0.7:
        issues.append(f"too short: {script.words} words (target {lo}-{hi})")
    if script.words > hi * 1.4:
        issues.append(f"too long: {script.words} words (target {lo}-{hi})")
    if re.search(r"(^|\s)[#*]{1,3}\s|\$[A-Z]{1,5}\b", script.script):
        issues.append("script contains markdown or cashtags; it must be plain spoken text")
    return issues


def _user_prompt(dossier: dict, target_words: tuple[int, int], issues: list[str] | None) -> str:
    msg = (
        "Here is this week's research dossier. Write the episode.\n\n"
        f"Target length: {target_words[0]}-{target_words[1]} words for the spoken script.\n"
        f"Use exactly this title: {dossier['title']}\n\n"
        "<dossier>\n" + json.dumps(dossier, indent=1, default=str) + "\n</dossier>"
    )
    if issues:
        msg += ("\n\nA previous draft had these problems. Fix all of them:\n- " + "\n- ".join(issues))
    return msg


def write_with_claude(dossier: dict, cfg: dict, client=None, issues: list[str] | None = None) -> Script:
    import anthropic

    llm = cfg["llm"]
    client = client or anthropic.Anthropic()
    target = tuple(llm.get("target_words", (900, 1400)))
    with client.beta.messages.stream(
        model=llm.get("model", "claude-opus-5-5"),
        max_tokens=llm.get("max_tokens", 32000),
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        thinking={"type": "adaptive"},
        output_config={"effort": llm.get("effort", "high"), "format": {"type": "json_schema", "schema": SCRIPT_SCHEMA}},
        # The style guide never changes between runs, so it's cached; the dossier comes after it.
        system=[{"type": "text", "text": STYLE_GUIDE.read_text(), "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": _user_prompt(dossier, target, issues)}],
    ) as stream:
        msg = stream.get_final_message()

    if msg.stop_reason == "refusal":
        raise RuntimeError(f"Claude declined to write the script: {getattr(msg, 'stop_details', None)}")
    if msg.stop_reason == "max_tokens":
        raise RuntimeError("script hit max_tokens before finishing")
    text = next(b.text for b in msg.content if b.type == "text")
    data = json.loads(text)
    return Script(data["title"].strip() or dossier["title"], data["script"].strip(),
                  data["show_notes_markdown"].strip(), "claude")


def write_script(dossier: dict, cfg: dict, client=None, max_attempts: int = 2) -> Script:
    """Claude first; if it's unavailable or keeps failing validation, fall back to the template."""
    target = tuple(cfg["llm"].get("target_words", (900, 1400)))
    issues: list[str] | None = None
    last: Script | None = None
    for attempt in range(max_attempts):
        try:
            last = write_with_claude(dossier, cfg, client=client, issues=issues)
        except Exception as exc:
            log.warning("Claude script attempt %d failed: %s", attempt + 1, exc)
            break
        issues = validate(last, dossier, target)
        if not issues:
            last.title = dossier["title"]  # keep the show's exact title format
            return last
        log.warning("Script attempt %d had issues: %s", attempt + 1, issues)
    if last is not None and last.source == "claude" and not any("never mentions" in i or "disclaimer" in i for i in issues or []):
        last.title, last.issues = dossier["title"], issues or []
        return last  # minor issues only (e.g. length): still better than the template
    tmpl = template_script(dossier)
    tmpl.issues = validate(tmpl, dossier, target)
    return tmpl


# ----------------------------------------------------------------------------- template fallback

def _say_pct(x: float | None) -> str:
    if x is None:
        return ""
    word = "up" if x >= 0 else "down"
    return f"{word} about {abs(x) * 100:.0f} percent"


def _say_price(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:,.2f} dollars" if x < 100 else f"{x:,.0f} dollars"


def template_script(d: dict) -> Script:
    """Deterministic script so an episode still ships if the API is down."""
    show = d["show"]
    w = d["week_of"]
    lines = [
        f"What's up everybody, welcome back to {show['title']}. I'm {show['host'].split()[0]}, "
        f"and here are five stocks to put on your watchlist for the week of {_spoken_date(w)}."
    ]
    spy = d["market_pulse"].get("SPY")
    if spy:
        trend = "above" if spy["above_50d"] else "below"
        lines.append(f"Quick market check. The S and P 500 was {_say_pct(spy['ret_1w'])} last week and is trading "
                     f"{trend} its 50-day moving average.")
    br = d["market_pulse"].get("breadth")
    if br:
        lines.append(f"About {br['pct_above_50d'] * 100:.0f} percent of the stocks I track are above their 50-day, "
                     f"so {'momentum is working' if br['pct_above_50d'] > 0.55 else 'be picky right now'}.")
    sc = d.get("last_week_scorecard")
    if sc and sc.get("avg_ret") is not None:
        lines.append(f"Scorecard from last episode: the picks averaged {_say_pct(sc['avg_ret'])} versus "
                     f"{_say_pct(sc['benchmark_ret'])} for the S and P. {sc['beat_benchmark']} of "
                     f"{len([p for p in sc['picks'] if p['ret'] is not None])} beat the market. "
                     f"Best was {sc['best']}, worst was {sc['worst']}.")
    intro = {"momentum": "a momentum leader", "social": "the one everybody's talking about",
             "sleeper": "the under-the-radar one"}
    for i, p in enumerate(d["picks"], 1):
        lv, perf = p["levels"], p["performance"]
        name = p["name"]
        repeat = " It's back on the list again this week." if p["weeks_on_list"] > 1 else ""
        reasons = " ".join(r.rstrip(".") + "." for r in p["reasons"][:3])
        part = [f"Number {i}, {intro.get(p['slot'], 'a momentum leader')}: {name}, ticker {p['ticker_spoken']}.{repeat}"]
        if perf.get("ret_1m") is not None:
            part.append(f"It's {_say_pct(perf['ret_1m'])} over the last month.")
        part.append(f"Why it's on the list: {reasons}")
        if lv["above_pivot"]:
            part.append(f"It just pushed above its 20-day high near {_say_price(lv['pivot_20d_high'])}.")
        elif lv["pivot_20d_high"]:
            part.append(f"The breakout level I'm watching is around {_say_price(lv['pivot_20d_high'])}.")
        if lv["invalidation"]:
            part.append(f"The setup breaks if it loses about {_say_price(lv['invalidation'])}.")
        part.append(f"The risk: {p['risk_flags'][0].rstrip('.')}.")
        lines.append(" ".join(part))
    spoken = ", ".join(p["ticker_spoken"] for p in d["picks"])
    lines.append(f"That's the list: {spoken}. As always, {d['disclaimer']} "
                 f"Follow me on Twitter {show['handle'].replace('@', 'at ')}. See you next week.")
    notes = [f"# {d['title']}", "", show["tagline"], ""]
    for p in d["picks"]:
        lv = p["levels"]
        notes.append(f"- **${p['ticker']}** ({p['slot']}): {p['reasons'][0] if p['reasons'] else ''}. "
                     f"Watching {lv['pivot_20d_high']} / invalidation {lv['invalidation']}.")
    notes += ["", f"_{d['disclaimer']}_"]
    return Script(d["title"], "\n\n".join(lines), "\n".join(notes), "template")


def _spoken_date(iso: str) -> str:
    from datetime import date

    from .dossier import ordinal

    dt = date.fromisoformat(iso)
    return f"{dt.strftime('%B')} {ordinal(dt.day)}"
