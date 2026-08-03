"""
Media I/O utilities.

Currently contains a single responsibility:
- extract_audio(video_file) -> path/to/mono_16k.wav

OCR-based name extraction was removed: the pipeline now identifies real
speaker names via SpeechBrain biometric matching in speaker_mapper_voice.py
(see seed_data.json for the enrolled voice database).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import uuid
from typing import Optional

logger = logging.getLogger(__name__)


def extract_audio(video_file: str, output_audio_file: Optional[str] = None) -> str:
    """
    Extract a mono 16 kHz WAV from a video file using ffmpeg.

    Args:
        video_file: Path to the input video (mp4/mkv/webm/etc).
        output_audio_file: Optional output WAV path. If None, a uuid-named file
            is created in the current working directory.

    Returns:
        Path to the extracted WAV.
    """
    ffmpeg_bin = os.getenv("FFMPEG_PATH", "ffmpeg")
    if not shutil.which(ffmpeg_bin) and not os.path.exists(ffmpeg_bin):
        raise RuntimeError(f"ffmpeg not found at '{ffmpeg_bin}' or in PATH")
    if not os.path.exists(video_file):
        raise FileNotFoundError(f"Video file not found: {video_file}")

    if output_audio_file is None:
        output_audio_file = f"audio_{uuid.uuid4().hex}.wav"

    command = [
        ffmpeg_bin, "-y", "-i", video_file,
        "-ac", "1",          # mono
        "-ar", "16000",      # 16 kHz
        "-vn",               # drop video
        "-c:a", "pcm_s16le", # 16-bit PCM
        output_audio_file,
    ]

    logger.info(f"FFmpeg: {video_file} -> {output_audio_file}")
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg audio extraction failed: {e.stderr}") from e

    size_mb = os.path.getsize(output_audio_file) / (1024 * 1024)
    logger.info(f"FFmpeg done: {output_audio_file} ({size_mb:.1f} MB)")
    return output_audio_file
