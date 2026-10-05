#!/usr/bin/env python3
"""
Kokoro TTS HTTP server — OpenAI-compatible /v1/audio/speech endpoint.
ADR: kokoro-tts-openai-server — replaces Supertonic at port 8779 for OpenClaw TTS.

Wraps Kokoro KPipeline for on-device TTS on Apple Silicon (MPS).
Mirrors the OpenAI TTS API subset that OpenClaw uses.

Environment variables:
  KOKORO_SERVER_PORT   — port to listen on (default: 8779)
  KOKORO_SERVER_HOST   — bind address (default: 127.0.0.1)
  KOKORO_DEFAULT_VOICE — default voice (default: af_heart)
  KOKORO_DEFAULT_LANG  — default language code (default: en-us)
  KOKORO_DEFAULT_SPEED — speech speed multiplier (default: 1.0)
  KOKORO_PLAY_LOCAL    — if "1", play via afplay after synthesis
"""

import http.server
import io
import json
import os
import subprocess
import tempfile
import threading
import time

import numpy as np
import soundfile as sf
import torch

from kokoro import KPipeline

# Apple Silicon MPS fallback for ops not yet on Metal
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

PORT = int(os.environ.get("KOKORO_SERVER_PORT", "8779"))
HOST = os.environ.get("KOKORO_SERVER_HOST", "127.0.0.1")
DEFAULT_VOICE = os.environ.get("KOKORO_DEFAULT_VOICE", "af_heart")
DEFAULT_LANG = os.environ.get("KOKORO_DEFAULT_LANG", "en-us")
DEFAULT_SPEED = float(os.environ.get("KOKORO_DEFAULT_SPEED", "1.0"))
PLAY_LOCAL = os.environ.get("KOKORO_PLAY_LOCAL", "0") == "1"

SAMPLE_RATE = 24000


def _best_device() -> str:
    # KOKORO_DEVICE overrides auto-detection.
    # Default: cpu — MPS has torch.istft tensor-resize bug (corrupts multi-chunk audio).
    # Set KOKORO_DEVICE=mps only after PyTorch fixes istftnet.py:97 resize warning.
    override = os.environ.get("KOKORO_DEVICE", "cpu").strip().lower()
    if override in ("cpu", "mps", "cuda"):
        return override
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = _best_device()

# Pipeline cache (lang → KPipeline), guarded by lock
_pipelines: dict[str, KPipeline] = {}
_pipeline_lock = threading.Lock()
# Serialise synthesis so concurrent requests don't corrupt pipeline state
_synthesis_lock = threading.Lock()
# Serialise playback so concurrent requests don't overlap audio output
_playback_lock = threading.Lock()


def get_pipeline(lang: str) -> KPipeline:
    with _pipeline_lock:
        if lang not in _pipelines:
            print(f"[kokoro-tts] Loading pipeline lang={lang} on {DEVICE}…", flush=True)
            _pipelines[lang] = KPipeline(
                lang_code=lang, repo_id="hexgrad/Kokoro-82M", device=DEVICE
            )
            print(f"[kokoro-tts] Pipeline ready lang={lang}", flush=True)
        return _pipelines[lang]


def synthesize(text: str, voice: str, lang: str, speed: float) -> np.ndarray:
    """Synthesize text → numpy float32 audio array (24 kHz mono)."""
    with _synthesis_lock:
        pipeline = get_pipeline(lang)
        chunks = []
        for result in pipeline(text, voice=voice, speed=speed):
            if result.audio is not None:
                chunks.append(result.audio.numpy())
    if not chunks:
        raise RuntimeError("Kokoro produced no audio")
    return np.concatenate(chunks)


def to_wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV")
    return buf.getvalue()


_playback_queue_depth = 0
_PLAYBACK_MAX_QUEUE = 2  # drop audio if more than 2 items queued

def play_locally(wav_bytes: bytes) -> None:
    """Play via afplay on macOS (blocking — serialised by _playback_lock).
    Drops audio if queue backs up to prevent CoreAudio contention that freezes UI."""
    global _playback_queue_depth
    if not PLAY_LOCAL:
        return

    _playback_queue_depth += 1
    if _playback_queue_depth > _PLAYBACK_MAX_QUEUE:
        _playback_queue_depth -= 1
        print(f"[kokoro-tts] dropping audio — queue depth {_playback_queue_depth + 1} exceeds max {_PLAYBACK_MAX_QUEUE}", flush=True)
        return

    try:
        with _playback_lock:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(wav_bytes)
                tmp = f.name
            try:
                subprocess.run(["afplay", tmp], check=False, timeout=30)
            except subprocess.TimeoutExpired:
                print("[kokoro-tts] afplay timeout — killing stuck playback", flush=True)
            finally:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
    finally:
        _playback_queue_depth -= 1


def _ffmpeg_convert(wav_bytes: bytes, out_ext: str, codec_args: list[str]) -> bytes | None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        src = f.name
    dst = src.replace(".wav", out_ext)
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", src] + codec_args + [dst],
            capture_output=True,
            check=False,  # returncode checked below; non-zero → return None
        )
        if result.returncode != 0:
            return None
        with open(dst, "rb") as f:
            return f.read()
    except FileNotFoundError:
        return None  # ffmpeg not installed
    finally:
        for p in (src, dst):
            try:
                os.unlink(p)
            except OSError:
                pass


def to_mp3(wav_bytes: bytes) -> bytes | None:
    return _ffmpeg_convert(wav_bytes, ".mp3", ["-codec:a", "libmp3lame", "-q:a", "2"])


def to_opus(wav_bytes: bytes) -> bytes | None:
    return _ffmpeg_convert(wav_bytes, ".opus", ["-codec:a", "libopus", "-b:a", "64k"])


# ── HTTP handler ──────────────────────────────────────────────────────────────

class KokoroHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # type: ignore[override]
        print(f"[kokoro-tts] {fmt % args}", flush=True)

    def _send_json(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, status: int, message: str) -> None:
        self._send_json(status, {"error": {"message": message, "type": "invalid_request_error"}})

    def do_GET(self) -> None:  # type: ignore[override]
        if self.path in ("/", "/health"):
            self._send_json(200, {
                "status": "ok",
                "provider": "kokoro-tts",
                "device": DEVICE,
                "default_voice": DEFAULT_VOICE,
                "default_lang": DEFAULT_LANG,
            })
        elif self.path == "/v1/models":
            self._send_json(200, {
                "object": "list",
                "data": [{"id": "kokoro-82m", "object": "model", "owned_by": "kokoro"}],
            })
        else:
            self._send_error(404, f"Not found: {self.path}")

    def do_POST(self) -> None:  # type: ignore[override]
        if self.path != "/v1/audio/speech":
            self._send_error(404, f"Not found: {self.path}")
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            self._send_error(400, "Invalid JSON body")
            return

        text = str(body.get("input", "")).strip()
        if not text:
            self._send_error(400, "input must be a non-empty string")
            return

        voice = str(body.get("voice", DEFAULT_VOICE))
        lang = str(body.get("language", DEFAULT_LANG))
        speed = float(body.get("speed", DEFAULT_SPEED))
        fmt = str(body.get("response_format", "wav")).lower()

        t0 = time.monotonic()
        try:
            audio = synthesize(text, voice, lang, speed)
        except (RuntimeError, ValueError, OSError) as exc:
            print(f"[kokoro-tts] synthesis error: {exc}", flush=True)
            self._send_error(500, f"Synthesis failed: {exc}")
            return

        wav_bytes = to_wav_bytes(audio)
        play_locally(wav_bytes)  # ← afplay side-effect, blocks until playback finishes

        # Format negotiation
        audio_bytes: bytes = wav_bytes
        content_type = "audio/wav"
        if fmt == "mp3":
            converted = to_mp3(wav_bytes)
            if converted:
                audio_bytes, content_type = converted, "audio/mpeg"
        elif fmt == "opus":
            converted = to_opus(wav_bytes)
            if converted:
                audio_bytes, content_type = converted, "audio/opus"
        elif fmt == "pcm":
            audio_bytes = wav_bytes[44:]  # strip 44-byte WAV header → raw int16 PCM
            content_type = "audio/pcm"

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(audio_bytes)))
        self.send_header("X-Voice", voice)
        self.send_header("X-Duration-Ms", str(elapsed_ms))
        self.end_headers()
        self.wfile.write(audio_bytes)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    # Pre-warm pipeline so first request doesn't block
    print(f"[kokoro-tts] Pre-warming pipeline lang={DEFAULT_LANG} on {DEVICE}…", flush=True)
    get_pipeline(DEFAULT_LANG)

    server = http.server.ThreadingHTTPServer((HOST, PORT), KokoroHandler)
    print(f"[kokoro-tts] Listening on http://{HOST}:{PORT}", flush=True)
    print(f"[kokoro-tts] Voice={DEFAULT_VOICE}  Lang={DEFAULT_LANG}  Speed={DEFAULT_SPEED}", flush=True)
    print(f"[kokoro-tts] Local afplay: {'enabled' if PLAY_LOCAL else 'disabled'}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
