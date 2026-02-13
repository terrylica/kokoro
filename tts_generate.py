"""
Kokoro TTS CLI — generate WAV from text.

Called directly by the telegram-sync bot (no HTTP server needed).
Supports chunked mode for streaming playback of long text.

Usage:
  # Single WAV output:
  .venv/bin/python tts_generate.py --text "Hello" --voice af_heart --lang en-us --speed 1.0 --output /tmp/tts.wav

  # Chunked streaming (prints one WAV path per line as each chunk is ready):
  .venv/bin/python tts_generate.py --text "Long text..." --voice af_heart --lang en-us --speed 1.0 --output /tmp/tts.wav --chunk

Exit codes:
  0  Success (WAV written to --output)
  1  Error (message on stderr)
"""

import argparse
import os
import re
import sys
import time

import numpy as np
import soundfile as sf
import torch

from kokoro import KPipeline

# Enable MPS fallback for ops not yet implemented on Metal
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _best_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = _best_device()

_pipelines: dict[str, KPipeline] = {}


def get_pipeline(lang: str) -> KPipeline:
    if lang not in _pipelines:
        _pipelines[lang] = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M", device=DEVICE)
    return _pipelines[lang]


def sanitize_text(text: str) -> str:
    """Remove surrogates and non-printable chars that crash spaCy."""
    # Encode with surrogateescape then decode with replace to strip surrogates
    text = text.encode("utf-8", errors="surrogateescape").decode("utf-8", errors="replace")
    # Remove replacement chars and other control chars (keep newlines/tabs)
    text = re.sub(r"[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text


def chunk_text(text: str, max_chars: int = 500) -> list[str]:
    """Split text into chunks using a hierarchical boundary strategy.

    Priority: paragraph borders → sentence borders → character fallback.
    Each chunk ≤ max_chars unless a single sentence exceeds the limit.
    """
    # Level 1: Split on paragraph borders (double newline or single newline)
    paragraphs = re.split(r"\n\s*\n|\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        # If paragraph fits in current chunk, append it
        if current and len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}"
            continue

        # Flush current chunk if non-empty
        if current:
            chunks.append(current)
            current = ""

        # If paragraph fits within max_chars, use it directly
        if len(para) <= max_chars:
            current = para
            continue

        # Level 2: Paragraph too long — split on sentence boundaries
        sentences = re.split(r"(?<=[.!?;:])\s+", para)
        for sentence in sentences:
            if not sentence.strip():
                continue

            if current and len(current) + len(sentence) + 1 > max_chars:
                chunks.append(current)
                current = ""

            if len(sentence) <= max_chars:
                current = f"{current} {sentence}" if current else sentence
                continue

            # Level 3: Sentence too long — split on character boundary at word breaks
            if current:
                chunks.append(current)
                current = ""
            words = sentence.split()
            for word in words:
                if current and len(current) + len(word) + 1 > max_chars:
                    chunks.append(current)
                    current = word
                else:
                    current = f"{current} {word}" if current else word

    if current.strip():
        chunks.append(current.strip())
    return chunks if chunks else [text]


def generate_wav(pipeline: KPipeline, text: str, voice: str, speed: float, output: str) -> bool:
    """Generate a single WAV file. Returns True on success."""
    samples_list = []
    for result in pipeline(text, voice=voice, speed=speed):
        if result.audio is not None:
            samples_list.append(result.audio.numpy())
    if not samples_list:
        return False
    audio = np.concatenate(samples_list)
    sf.write(output, audio, 24000, format="WAV")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Kokoro TTS generate WAV")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--voice", default="af_heart", help="Voice name")
    parser.add_argument("--lang", default="en-us", help="Language code")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed")
    parser.add_argument("--output", required=True, help="Output WAV path")
    parser.add_argument("--chunk", action="store_true",
                        help="Chunked streaming: print WAV path per chunk as ready")
    args = parser.parse_args()

    text = sanitize_text(args.text)
    if not text.strip():
        print("Empty text", file=sys.stderr)
        sys.exit(1)

    t0 = time.monotonic()
    pipeline = get_pipeline(args.lang)

    if args.chunk:
        # Chunked mode: split text, generate each chunk as a separate WAV
        # Print each path to stdout as it's ready for immediate playback
        chunks = chunk_text(text)
        base, ext = os.path.splitext(args.output)
        for i, chunk in enumerate(chunks):
            chunk_path = f"{base}-{i:03d}{ext}" if len(chunks) > 1 else args.output
            if generate_wav(pipeline, chunk, args.voice, args.speed, chunk_path):
                # Flush immediately so caller can start playing this chunk
                print(chunk_path, flush=True)
            else:
                print(f"Warning: chunk {i} produced no audio", file=sys.stderr)

        gen_ms = int((time.monotonic() - t0) * 1000)
        print(f"DONE {gen_ms}", flush=True)
    else:
        # Single WAV mode (original behavior)
        if not generate_wav(pipeline, text, args.voice, args.speed, args.output):
            print("No audio generated", file=sys.stderr)
            sys.exit(1)

        gen_ms = int((time.monotonic() - t0) * 1000)
        print(f"{gen_ms}", flush=True)


if __name__ == "__main__":
    main()
