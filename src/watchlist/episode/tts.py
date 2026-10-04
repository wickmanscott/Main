"""Text-to-speech providers.

kokoro      free, offline neural voice (Kokoro-82M, ONNX); model files download once from GitHub
edge        free Microsoft neural voices via edge-tts (needs Microsoft's speech endpoint)
elevenlabs  your own cloned voice (paid API)
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
from pathlib import Path

import requests

log = logging.getLogger(__name__)

ELEVENLABS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=mp3_44100_128"
BITRATE = {"edge": 48_000, "elevenlabs": 128_000, "kokoro": 96_000}  # bits/s of each provider's MP3 output
KOKORO_FILES = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"


_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def _two_digits(n: int) -> str:
    if n < 20:
        return _ONES[n]
    return _TENS[n // 10] + ("" if n % 10 == 0 else "-" + _ONES[n % 10])


def normalize_for_speech(text: str) -> str:
    """Fix things neural TTS voices misread: spelled tickers, $/%, years, the show's handle."""
    text = re.sub(r"\b[A-Z0-9](?:-[A-Z0-9])+\b", lambda m: m.group(0).replace("-", " "), text)  # N-V-D-A -> N V D A
    text = re.sub(r"@?\bscottiewick\b", "Scottie Wick", text, flags=re.IGNORECASE)
    text = re.sub(r"\$(\d[\d,]*(?:\.\d+)?)\s*(billion|million|trillion)?",
                  lambda m: f"{m.group(1)} {m.group(2) + ' ' if m.group(2) else ''}dollars", text)
    text = re.sub(r"(\d)\s*%", r"\1 percent", text)
    # 2010-2099 read as years ("twenty twenty-six"), but leave "Russell 2000" alone.
    text = re.sub(r"\b20([1-9]\d)\b", lambda m: "twenty " + _two_digits(int(m.group(1))), text)
    return text


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


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s", url)
    with requests.get(url, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as fh:
            for block in resp.iter_content(1 << 20):
                fh.write(block)
    tmp.rename(dest)
    return dest


def _kokoro(text: str, out: Path, voice: str, speed: float, model_dir: Path, pause_sec: float = 0.45) -> float:
    """Render paragraph by paragraph with a short pause between them; encode to MP3 with ffmpeg."""
    import numpy as np
    import soundfile as sf
    from kokoro_onnx import Kokoro

    model = _download(KOKORO_FILES + "kokoro-v1.0.onnx", model_dir / "kokoro-v1.0.onnx")
    voices = _download(KOKORO_FILES + "voices-v1.0.bin", model_dir / "voices-v1.0.bin")
    engine = Kokoro(str(model), str(voices))
    parts, sr = [], 24_000
    for para in (p.strip() for p in text.split("\n\n")):
        if not para:
            continue
        samples, sr = engine.create(para, voice=voice, speed=speed, lang="en-us")
        parts += [samples, np.zeros(int(sr * pause_sec), dtype=samples.dtype)]
    audio = np.concatenate(parts) if parts else np.zeros(sr, dtype="float32")
    wav = out.with_suffix(".wav")
    sf.write(wav, audio, sr)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-codec:a", "libmp3lame",
                    "-b:a", "96k", str(out)], check=True)
    wav.unlink()
    return len(audio) / sr


def synthesize(text: str, out: str | Path, cfg: dict) -> dict | None:
    """Render `text` to an MP3. Returns {path, bytes, duration_sec, provider} or None if disabled."""
    tcfg = cfg.get("tts", {})
    provider = tcfg.get("provider", "edge")
    if provider == "none":
        return None
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = normalize_for_speech(text)
    duration = None
    if provider == "kokoro":
        from ..config import resolve

        duration = _kokoro(text, out, tcfg.get("kokoro_voice", "am_michael"), float(tcfg.get("kokoro_speed", 1.05)),
                           resolve(tcfg.get("kokoro_model_dir", "data/cache/tts")))
    elif provider == "elevenlabs":
        _elevenlabs(text, out, tcfg.get("elevenlabs_model", "eleven_multilingual_v2"))
    elif provider == "edge":
        _edge(text, out, tcfg.get("edge_voice", "en-US-AndrewNeural"), tcfg.get("edge_rate", "+0%"))
    else:
        raise ValueError(f"unknown tts provider: {provider}")
    size = out.stat().st_size
    seconds = duration if duration is not None else size * 8 / BITRATE[provider]
    return {"path": str(out), "bytes": size, "duration_sec": int(seconds), "provider": provider}
