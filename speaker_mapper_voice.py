"""
Biometric speaker mapping using SpeechBrain ECAPA-TDNN embeddings.

Takes a diarized transcript (segments with SPEAKER_XX labels) plus the source
audio and returns a mapping {SPEAKER_XX: real_name} for every speaker that
passes the dual-threshold confidence gate.

The dual-threshold gating is taken from meet-voice-identity-main/test_chunking.py:
    Gate A (clean audio):       score >= 0.70 AND gap >= 0.10
    Gate B (acoustic mismatch): score >= 0.50 AND gap >= 0.20

Speakers that fail both gates are left as their original SPEAKER_XX ID.

The seed database is a JSON file produced by
meet-voice-identity-main/seed_data.py and looks like:
    [
        {
            "name": "Ketan Patil",
            "source_file": "ketan_eng_clean.wav",
            "embedding": [192 floats],
            "model_version": "speechbrain/spkrec-ecapa-voxceleb"
        },
        ...
    ]
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio

logger = logging.getLogger(__name__)


# =========================================
# Thresholds (from meet-voice-identity-main/test_chunking.py)
# =========================================
HIGH_SCORE_THRESHOLD = 0.70
HIGH_SCORE_GAP = 0.10
MID_SCORE_THRESHOLD = 0.50
MID_SCORE_GAP = 0.20

SAMPLE_RATE = 16000
MAX_CHUNKS_PER_SPEAKER = 5
TARGET_CHUNK_SECONDS = 4.0
MIN_USABLE_SEGMENT_SECONDS = 1.5

# Lazy-loaded speaker recognition engine
_ENGINE = None


def _get_engine():
    """Load SpeechBrain ECAPA-TDNN on first use (CPU, serverless-friendly)."""
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    from speechbrain.inference.speaker import SpeakerRecognition

    logger.info("Loading SpeechBrain ECAPA-TDNN (one-time)...")
    _ENGINE = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.getenv(
            "SPEECHBRAIN_CACHE",
            "/runpod-volume/speechbrain/spkrec-ecapa-voxceleb"
            if os.getenv("RUNPOD_POD_ID")
            else "pretrained_models/spkrec-ecapa-voxceleb",
        ),
        run_opts={"device": "cpu"},
    )
    logger.info("SpeechBrain loaded.")
    return _ENGINE


def _load_seed_db(seed_path: str) -> List[Dict]:
    if not os.path.exists(seed_path):
        raise FileNotFoundError(f"Seed DB not found: {seed_path}")
    with open(seed_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Seed DB at {seed_path} is empty or malformed")
    return data


def _pick_chunks_for_speaker(
    segments: List[Dict], speaker_id: str
) -> List[Tuple[float, float]]:
    """
    Pick up to MAX_CHUNKS_PER_SPEAKER (start, end) windows of about
    TARGET_CHUNK_SECONDS for the given speaker_id from a diarized transcript.

    Picks the LONGEST segments first since they yield the cleanest embeddings.
    Drops any segment shorter than MIN_USABLE_SEGMENT_SECONDS.
    """
    candidates: List[Tuple[float, float]] = []
    for seg in segments:
        if seg.get("speaker") != speaker_id:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        dur = end - start
        if dur < MIN_USABLE_SEGMENT_SECONDS:
            continue
        # Trim long segments to TARGET_CHUNK_SECONDS from the middle
        if dur > TARGET_CHUNK_SECONDS:
            mid = (start + end) / 2
            half = TARGET_CHUNK_SECONDS / 2
            candidates.append((max(start, mid - half), min(end, mid + half)))
        else:
            candidates.append((start, end))

    # Longest first
    candidates.sort(key=lambda x: x[1] - x[0], reverse=True)
    return candidates[:MAX_CHUNKS_PER_SPEAKER]


def _embed_chunks(
    audio_path: str, chunks: List[Tuple[float, float]]
) -> Optional[torch.Tensor]:
    """
    Read chunks from audio_path with torchaudio, encode each via SpeechBrain,
    and return the averaged 192-d embedding tensor. Returns None if no usable
    chunks could be loaded.
    """
    if not chunks:
        return None

    engine = _get_engine()
    vectors = []

    for start_s, end_s in chunks:
        frame_offset = int(start_s * SAMPLE_RATE)
        num_frames = int((end_s - start_s) * SAMPLE_RATE)
        if num_frames <= 0:
            continue
        try:
            signal, fs = torchaudio.load(
                audio_path,
                frame_offset=frame_offset,
                num_frames=num_frames,
            )
        except Exception as e:
            logger.warning(f"torchaudio.load failed for {start_s}-{end_s}: {e}")
            continue

        if fs != SAMPLE_RATE:
            signal = torchaudio.functional.resample(signal, fs, SAMPLE_RATE)
        if signal.shape[0] > 1:
            signal = signal.mean(dim=0, keepdim=True)

        try:
            vec = engine.encode_batch(signal).squeeze()
            vectors.append(vec)
        except Exception as e:
            logger.warning(f"encode_batch failed: {e}")
            continue

    if not vectors:
        return None
    return torch.mean(torch.stack(vectors), dim=0)


def _match_speaker(
    speaker_vec: torch.Tensor, seed_db: List[Dict]
) -> Tuple[str, float, float, List[Dict]]:
    """
    Compare a speaker embedding against the seed DB. Returns
    (best_name, best_score, gap_to_next_distinct, top3_results).
    """
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    results = []
    for person in seed_db:
        enrolled = torch.tensor(person["embedding"])
        score = float(cos(speaker_vec, enrolled).item())
        results.append({"name": person["name"], "score": score})

    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[0]

    # Gap vs the next person whose name differs from the top match
    next_best = next((r for r in results[1:] if r["name"] != top["name"]), None)
    gap = top["score"] - next_best["score"] if next_best else 1.0

    return top["name"], top["score"], gap, results[:3]


def map_speakers(
    audio_path: str,
    diarized_segments: List[Dict],
    seed_db_path: str,
) -> Dict[str, str]:
    """
    Build a mapping {SPEAKER_XX: real_name} for all speakers that pass the
    dual-threshold gate. Unmatched speakers do not appear in the mapping
    (callers should fall back to the original SPEAKER_XX label).

    Args:
        audio_path: Path to the source WAV (mono 16kHz recommended).
        diarized_segments: Output of diarize.assign_speakers_to_transcript().
        seed_db_path: Path to seed_data.json (the enrolled voice DB).

    Returns:
        Dict mapping speaker_id -> real_name for matched speakers only.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    seed_db = _load_seed_db(seed_db_path)
    unique_speakers = sorted({seg.get("speaker") for seg in diarized_segments if seg.get("speaker")})
    logger.info(f"VoiceMapper: analyzing {len(unique_speakers)} speakers against {len(seed_db)} enrolled profiles")

    mapping: Dict[str, str] = {}

    for speaker_id in unique_speakers:
        if not speaker_id or speaker_id == "UNKNOWN_SPEAKER":
            continue

        chunks = _pick_chunks_for_speaker(diarized_segments, speaker_id)
        if not chunks:
            logger.info(f"VoiceMapper: {speaker_id} -> no usable chunks (>{MIN_USABLE_SEGMENT_SECONDS}s)")
            continue

        speaker_vec = _embed_chunks(audio_path, chunks)
        if speaker_vec is None:
            logger.info(f"VoiceMapper: {speaker_id} -> embedding failed")
            continue

        name, score, gap, top3 = _match_speaker(speaker_vec, seed_db)
        top3_str = ", ".join(f"{r['name']}({r['score']:.2f})" for r in top3)
        logger.info(
            f"VoiceMapper: {speaker_id} top3=[{top3_str}] best={name} score={score:.3f} gap={gap:.3f}"
        )

        gate_a = score >= HIGH_SCORE_THRESHOLD and gap >= HIGH_SCORE_GAP
        gate_b = score >= MID_SCORE_THRESHOLD and gap >= MID_SCORE_GAP

        if gate_a:
            logger.info(f"VoiceMapper: {speaker_id} -> {name} (Gate A)")
            mapping[speaker_id] = name
        elif gate_b:
            logger.info(f"VoiceMapper: {speaker_id} -> {name} (Gate B)")
            mapping[speaker_id] = name
        else:
            logger.info(f"VoiceMapper: {speaker_id} REJECTED (below gates)")

    logger.info(f"VoiceMapper: mapped {len(mapping)}/{len(unique_speakers)} speakers")
    return mapping


def unload():
    """Release SpeechBrain model. Useful between runs in long-lived workers."""
    global _ENGINE
    _ENGINE = None
    import gc
    gc.collect()
