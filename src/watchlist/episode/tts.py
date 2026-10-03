"""Text-to-speech: free Microsoft neural voices via edge-tts, or your own cloned voice on ElevenLabs."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import requests

log = logging.getLogger(__name__)

ELEVENLABS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=mp3_44100_128"
BITRATE = {"edge": 48_000, "elevenlabs": 128_000}  # bits/s of each provider's MP3 output


def _chunks(text: str, limit: int) -> list[str]:
    """Split on paragraph boundaries so each request stays under the provider's limit."""
    out, cur = [], ""
    for para in text.split("\n\n"):
        if cur and len(cur) + len(para) + 2 > limit:
            out.append(cur)
            cur = para
        else:
            cur = f"{cur}\n\n{para}" if cur else para
    if cur:
        out.append(cur)
    return out


def _edge(text: str, out: Path, voice: str, rate: str) -> None:
    import edge_tts

    async def go():
        await edge_tts.Communicate(text, voice, rate=rate).save(str(out))

    asyncio.run(go())


def _elevenlabs(text: str, out: Path, model: str) -> None:
    key, voice = os.environ.get("ELEVENLABS_API_KEY"), os.environ.get("ELEVENLABS_VOICE_ID")
    if not key or not voice:
        raise RuntimeError("set ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID to use ElevenLabs")
    with open(out, "wb") as fh:
        for chunk in _chunks(text, 4500):
            resp = requests.post(
                ELEVENLABS_URL.format(voice_id=voice),
                headers={"xi-api-key": key, "Content-Type": "application/json"},
                json={"text": chunk, "model_id": model},
                timeout=300,
            )
            resp.raise_for_status()
            fh.write(resp.content)  # MP3 frames concatenate cleanly


def synthesize(text: str, out: str | Path, cfg: dict) -> dict | None:
    """Render `text` to an MP3. Returns {path, bytes, duration_sec, provider} or None if disabled."""
    tcfg = cfg.get("tts", {})
    provider = tcfg.get("provider", "edge")
    if provider == "none":
        return None
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if provider == "elevenlabs":
        _elevenlabs(text, out, tcfg.get("elevenlabs_model", "eleven_multilingual_v2"))
    elif provider == "edge":
        _edge(text, out, tcfg.get("edge_voice", "en-US-AndrewNeural"), tcfg.get("edge_rate", "+0%"))
    else:
        raise ValueError(f"unknown tts provider: {provider}")
    size = out.stat().st_size
    return {"path": str(out), "bytes": size, "duration_sec": int(size * 8 / BITRATE[provider]), "provider": provider}
