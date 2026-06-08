"""
Whisper-based audio transcription.

Streams the input WAV from disk in 5-minute chunks instead of loading the full
file into RAM. This is the critical fix for long meetings: a 2-hour mono-16kHz
WAV is ~220 MB raw and ~880 MB as float64 if fully loaded -- streaming keeps
the per-chunk footprint to ~36 MB regardless of total length.

Overlap: each chunk overlaps the previous by OVERLAP_SECONDS (30s) so that
words at chunk boundaries are not cut. Segments from the overlap zone at the
start of each chunk (except the first) are discarded — only new content past
the midpoint of the overlap is kept, preventing duplicate segments.

Public API:
    transcribe_audio_chunked(audio_path) -> List[Dict]
    unload_model()     # free Whisper VRAM before running Pyannote
"""

from __future__ import annotations

import gc
import logging
import os
from typing import Dict, List, Optional

import soundfile as sf
import torch
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


IS_RUNPOD = os.getenv("RUNPOD_POD_ID") is not None

if IS_RUNPOD:
    # Persistent volume on RunPod for Whisper weights between cold starts
    os.environ["HF_HOME"] = "/runpod-volume/huggingface"
    os.environ["TORCH_HOME"] = "/runpod-volume/torch"
    logger.info("RunPod environment detected - using persistent volume for model cache")


# =============================
# MODEL & PERFORMANCE CONFIG
# =============================

MODEL_NAME = "large-v2"
CHUNK_SIZE_SECONDS = 300        # 5 minutes per Whisper call
OVERLAP_SECONDS = 30            # overlap between consecutive chunks to avoid boundary word cuts
BEAM_SIZE = 5
TEMPERATURE = 0.0
TARGET_SAMPLE_RATE = 16000      # Whisper expects mono 16kHz

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"


# =============================
# MODEL SINGLETON (lazy-loaded so unload/reload is cheap)
# =============================

_MODEL: Optional[WhisperModel] = None


def _get_model() -> WhisperModel:
    global _MODEL
    if _MODEL is None:
        logger.info(f"Loading Whisper '{MODEL_NAME}' on {DEVICE} ({COMPUTE_TYPE})...")
        _MODEL = WhisperModel(
            MODEL_NAME,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
        )
        logger.info("Whisper loaded.")
    return _MODEL


def unload_model() -> None:
    """Free the Whisper model from RAM/VRAM. Safe to call multiple times."""
    global _MODEL
    if _MODEL is None:
        return
    logger.info("Unloading Whisper model...")
    try:
        del _MODEL
    finally:
        _MODEL = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Whisper unloaded.")


# =============================
# PUBLIC API
# =============================

def transcribe_audio_chunked(audio_path: str) -> List[Dict]:
    """
    Transcribe an audio file using chunked Faster-Whisper streamed from disk.

    Consecutive chunks overlap by OVERLAP_SECONDS (30s). For each chunk after
    the first, segments whose start time falls before (chunk_start + OVERLAP_SECONDS/2)
    are dropped — they were already captured by the previous chunk. This eliminates
    duplicate segments while ensuring boundary words are not silently cut.

    Args:
        audio_path: Path to WAV audio (mono 16kHz recommended).

    Returns:
        List of dicts: [{"start": float, "end": float, "text": str}, ...]
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = _get_model()

    results: List[Dict] = []
    chunk_index = 0

    with sf.SoundFile(audio_path, mode="r") as src:
        sample_rate = src.samplerate
        channels = src.channels
        total_frames = len(src)
        total_duration = total_frames / float(sample_rate)
        frames_per_chunk = int(CHUNK_SIZE_SECONDS * sample_rate)
        overlap_frames = int(OVERLAP_SECONDS * sample_rate)
        # Each chunk advances by (chunk_size - overlap) frames so consecutive
        # chunks share OVERLAP_SECONDS of audio at their boundary.
        step_frames = frames_per_chunk - overlap_frames

        logger.info(
            f"Transcribing {audio_path}: duration={total_duration:.1f}s, "
            f"sr={sample_rate}Hz, channels={channels}, "
            f"chunk_size={CHUNK_SIZE_SECONDS}s, overlap={OVERLAP_SECONDS}s, "
            f"step={CHUNK_SIZE_SECONDS - OVERLAP_SECONDS}s"
        )

        if sample_rate != TARGET_SAMPLE_RATE:
            logger.warning(
                f"Sample rate is {sample_rate}Hz, expected {TARGET_SAMPLE_RATE}Hz. "
                "Quality may degrade. Re-encode the audio with ffmpeg -ar 16000 if possible."
            )

        offset_frames = 0
        while offset_frames < total_frames:
            src.seek(offset_frames)
            chunk = src.read(frames_per_chunk, dtype="float32", always_2d=False)

            # Down-mix to mono if needed
            if chunk.ndim > 1:
                chunk = chunk.mean(axis=1)

            chunk_offset_seconds = offset_frames / float(sample_rate)
            chunk_index += 1
            chunk_duration = len(chunk) / float(sample_rate)

            # Segments before this cutoff belong to the previous chunk's territory.
            # We keep only segments that START at or after this boundary.
            # For the first chunk there is no previous chunk, so cutoff = 0.
            dedup_cutoff = (
                chunk_offset_seconds + OVERLAP_SECONDS / 2.0
                if chunk_index > 1
                else 0.0
            )

            logger.info(
                f"  Chunk {chunk_index}: start={chunk_offset_seconds:.1f}s "
                f"duration={chunk_duration:.1f}s samples={len(chunk)} "
                f"dedup_cutoff={dedup_cutoff:.1f}s"
            )

            try:
                segments, _ = model.transcribe(
                    chunk,
                    beam_size=BEAM_SIZE,
                    temperature=TEMPERATURE,
                    vad_filter=False,
                    condition_on_previous_text=False,   # chunks are independent
                    no_speech_threshold=0.6,
                    log_prob_threshold=-1.0,
                )
                new_segs = 0
                for seg in segments:
                    abs_start = round(float(seg.start) + chunk_offset_seconds, 2)
                    abs_end   = round(float(seg.end)   + chunk_offset_seconds, 2)
                    # Drop overlap-zone duplicates from the start of this chunk
                    if abs_start < dedup_cutoff:
                        continue
                    results.append({
                        "start": abs_start,
                        "end":   abs_end,
                        "text":  seg.text.strip(),
                    })
                    new_segs += 1
                logger.info(f"  Chunk {chunk_index}: {new_segs} new segments kept")
            finally:
                del chunk
                gc.collect()

            offset_frames += step_frames

    logger.info(f"Transcription complete: {len(results)} segments across {chunk_index} chunks")
    return results
