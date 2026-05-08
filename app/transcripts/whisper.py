"""
Whisper transcription wrapper.

Accepts raw audio bytes, writes to a temp file (Whisper requires a path),
transcribes, and returns the transcript text.

Model is loaded lazily and cached in-process so it is only loaded once
per application lifetime.
"""
from __future__ import annotations

import logging
import os
import tempfile
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# Whisper model size. Can be overridden by env var WHISPER_MODEL.
# Options: tiny, base, small, medium, large
_WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")


@lru_cache(maxsize=1)
def _get_whisper_model():
    """Load and cache the Whisper model (first call may take a moment)."""
    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError(
            "openai-whisper is required. pip install openai-whisper"
        ) from exc

    logger.info("Loading Whisper model: %s", _WHISPER_MODEL_SIZE)
    model = whisper.load_model(_WHISPER_MODEL_SIZE)
    logger.info("Whisper model loaded.")
    return model


def transcribe_audio(audio_bytes: bytes, filename: str) -> str:
    """
    Transcribe audio from raw bytes.

    Writes to a named temp file because Whisper's API requires a file path.
    The temp file is deleted after transcription.

    Returns the transcript as a plain string.
    """
    suffix = Path(filename).suffix or ".wav"
    model = _get_whisper_model()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        logger.info("Transcribing %s (%d bytes)…", filename, len(audio_bytes))
        result = model.transcribe(tmp_path, fp16=False, language=None)
        transcript: str = result["text"].strip()
        logger.info("Transcription complete: %d characters", len(transcript))
        return transcript
    finally:
        os.unlink(tmp_path)
