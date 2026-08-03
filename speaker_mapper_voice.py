"""
Production-Grade Biometric Speaker Mapping Engine.

Features:
- SpeechBrain ECAPA-TDNN & modular embedding extraction
- Multi-sample dynamic profile centroid matching
- Pre-match Agglomerative Speaker Fragment Merging (speaker_merger.py)
- Dynamic Adaptive Threshold Engine (adaptive_threshold.py)
- Continuous Explainable Confidence Engine (confidence_engine.py)
- Tiered Fallback Unknown Speaker Resolver (unknown_resolver.py)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Tuple, Any

import soundfile as sf
try:
    import torch
    import torchaudio
except ImportError:
    torch = None
    torchaudio = None

from adaptive_threshold import AdaptiveThresholdEngine, ThresholdConfig
from cache_resolver import CacheResolver
from confidence_engine import ConfidenceEngine
from speaker_merger import SpeakerFragmentMerger, _cosine_sim
from unknown_resolver import UnknownSpeakerResolver

logger = logging.getLogger(__name__)

# Fallback default thresholds (overridden by AdaptiveThresholdEngine)
HIGH_SCORE_THRESHOLD = float(os.getenv("HIGH_SCORE_THRESHOLD", "0.68"))
HIGH_SCORE_GAP = float(os.getenv("HIGH_SCORE_GAP", "0.08"))
MID_SCORE_THRESHOLD = float(os.getenv("MID_SCORE_THRESHOLD", "0.48"))
MID_SCORE_GAP = float(os.getenv("MID_SCORE_GAP", "0.15"))
CLUSTER_MERGE_THRESHOLD = float(os.getenv("CLUSTER_MERGE_THRESHOLD", "0.78"))

SAMPLE_RATE = 16000
MAX_CHUNKS_PER_SPEAKER = 5
TARGET_CHUNK_SECONDS = 4.0
MIN_USABLE_SEGMENT_SECONDS = 1.5

_ENGINE = None


def _get_engine():
    """Load SpeechBrain ECAPA-TDNN on first use."""
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    from speechbrain.inference.speaker import SpeakerRecognition

    savedir = CacheResolver.get_speechbrain_cache(workdir=os.path.dirname(os.path.abspath(__file__)))
    strategy = CacheResolver.resolve_speechbrain_strategy_object()
    logger.info(f"Loading SpeechBrain ECAPA-TDNN from cache dir: {savedir} (strategy={strategy})")
    _ENGINE = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=savedir,
        local_strategy=strategy,
        run_opts={"device": "cpu"},
    )
    logger.info("SpeechBrain loaded successfully.")
    return _ENGINE


def _load_seed_db(seed_path: str) -> List[Dict]:
    if not os.path.exists(seed_path):
        raise FileNotFoundError(f"Seed DB not found: {seed_path}")
    with open(seed_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Seed DB at {seed_path} is empty or malformed")
    return data


def _calculate_snr(signal: torch.Tensor) -> float:
    """Calculate Signal-to-Noise Ratio (SNR in dB) of audio signal tensor."""
    rms = float(torch.sqrt(torch.mean(signal ** 2)).item()) + 1e-8
    noise_floor = float(torch.std(signal[: int(len(signal) * 0.1)]).item()) + 1e-8
    snr = 20.0 * math.log10(rms / noise_floor)
    return max(0.0, min(50.0, snr))


import math


def _pick_chunks_for_speaker(
    segments: List[Dict], speaker_id: str
) -> List[Tuple[float, float]]:
    """Pick up to MAX_CHUNKS_PER_SPEAKER longest segments for embedding extraction."""
    candidates: List[Tuple[float, float]] = []
    for seg in segments:
        if seg.get("speaker") != speaker_id:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        dur = end - start
        if dur < MIN_USABLE_SEGMENT_SECONDS:
            continue
        if dur > TARGET_CHUNK_SECONDS:
            mid = (start + end) / 2
            half = TARGET_CHUNK_SECONDS / 2
            candidates.append((max(start, mid - half), min(end, mid + half)))
        else:
            candidates.append((start, end))

    candidates.sort(key=lambda x: x[1] - x[0], reverse=True)
    return candidates[:MAX_CHUNKS_PER_SPEAKER]


def _embed_chunks(
    audio_path: str, chunks: List[Tuple[float, float]]
) -> Tuple[Optional[torch.Tensor], float]:
    """
    Read audio chunks, extract SpeechBrain embeddings, and aggregate via RMS weighting.
    Returns (averaged_embedding_tensor, estimated_snr_db).
    """
    if not chunks:
        return None, 0.0

    engine = _get_engine()
    vectors = []
    weights = []
    snrs = []

    for start_s, end_s in chunks:
        frame_offset = int(start_s * SAMPLE_RATE)
        num_frames = int((end_s - start_s) * SAMPLE_RATE)
        if num_frames <= 0:
            continue
        try:
            data, fs = sf.read(
                audio_path,
                start=frame_offset,
                stop=frame_offset + num_frames,
                dtype="float32",
                always_2d=True,
            )
            signal = torch.from_numpy(data.T)
        except Exception as e:
            logger.warning(f"sf.read failed for {start_s}-{end_s}: {e}")
            continue

        if fs != SAMPLE_RATE:
            signal = torchaudio.functional.resample(signal, fs, SAMPLE_RATE)
        if signal.shape[0] > 1:
            signal = signal.mean(dim=0, keepdim=True)

        try:
            vec = engine.encode_batch(signal).squeeze()
            rms = float(torch.sqrt(torch.mean(signal ** 2)).item()) + 1e-6
            snr = _calculate_snr(signal.squeeze())
            vectors.append(vec)
            weights.append(rms)
            snrs.append(snr)
        except Exception as e:
            logger.warning(f"encode_batch failed: {e}")
            continue

    if not vectors:
        return None, 0.0

    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    norm_weights = weights_tensor / (torch.sum(weights_tensor) + 1e-8)
    stacked = torch.stack(vectors)
    centroid = torch.sum(stacked * norm_weights.unsqueeze(1), dim=0)
    normalized_centroid = torch.nn.functional.normalize(centroid, p=2, dim=0)
    avg_snr = float(sum(snrs) / len(snrs)) if snrs else 20.0

    return normalized_centroid, avg_snr


def _match_speaker(
    speaker_vec: torch.Tensor, seed_db: List[Dict]
) -> Tuple[str, float, float, List[Dict[str, Any]]]:
    """
    Compare speaker vector against enrolled profiles (supporting multi-sample centroids).
    Returns (best_name, best_score, gap_to_next_distinct, top_results).
    """
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    results = []

    for person in seed_db:
        name = person.get("name", "Unknown")
        emp_id = person.get("employee_id") or person.get("employeeId")

        # Multi-sample centroid support
        if "centroid_vector" in person and isinstance(person["centroid_vector"], list):
            enrolled_vec = torch.tensor(person["centroid_vector"]) if torch is not None else person["centroid_vector"]
        elif "embedding" in person and isinstance(person["embedding"], list):
            enrolled_vec = torch.tensor(person["embedding"]) if torch is not None else person["embedding"]
        else:
            continue

        score = _cosine_sim(speaker_vec, enrolled_vec)
        results.append({"name": name, "employee_id": emp_id, "score": score})

    if not results:
        return "Unknown", 0.0, 0.0, []

    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[0]

    next_best = next((r for r in results[1:] if r["name"] != top["name"]), None)
    gap = top["score"] - next_best["score"] if next_best else HIGH_SCORE_GAP

    return top["name"], top["score"], gap, results[:5]


def map_speakers(
    audio_path: str,
    diarized_segments: List[Dict],
    seed_db_path: str,
    enrolled_profiles: Optional[List[Dict]] = None,
    expected_roster: Optional[List[str]] = None,
    meeting_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Build a mapping {SPEAKER_XX: real_name} for all speakers using the production pipeline.
    """
    res = map_speakers_explainable(
        audio_path=audio_path,
        diarized_segments=diarized_segments,
        seed_db_path=seed_db_path,
        enrolled_profiles=enrolled_profiles,
        expected_roster=expected_roster,
        meeting_metadata=meeting_metadata,
    )
    return res["speaker_mapping"]


def map_speakers_explainable(
    audio_path: str,
    diarized_segments: List[Dict],
    seed_db_path: str,
    enrolled_profiles: Optional[List[Dict]] = None,
    expected_roster: Optional[List[str]] = None,
    meeting_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute full biometric speaker mapping pipeline returning both speaker_mapping and explainability diagnostics.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if enrolled_profiles and isinstance(enrolled_profiles, list) and len(enrolled_profiles) > 0:
        seed_db = enrolled_profiles
        logger.info(f"VoiceMapper: Loaded {len(seed_db)} dynamic enrolled profiles")
    else:
        seed_db = _load_seed_db(seed_db_path)

    raw_speakers = sorted({seg.get("speaker") for seg in diarized_segments if seg.get("speaker")})
    logger.info(f"VoiceMapper: Extracting vectors for {len(raw_speakers)} raw speaker IDs")

    # Step 1: Extract initial speaker vectors
    raw_speaker_vectors: Dict[str, torch.Tensor] = {}
    speaker_snrs: Dict[str, float] = {}

    for speaker_id in raw_speakers:
        if not speaker_id or speaker_id == "UNKNOWN_SPEAKER":
            continue
        chunks = _pick_chunks_for_speaker(diarized_segments, speaker_id)
        if not chunks:
            continue
        vec, snr = _embed_chunks(audio_path, chunks)
        if vec is not None:
            raw_speaker_vectors[speaker_id] = vec
            speaker_snrs[speaker_id] = snr

    # Step 2: Speaker Fragment Merge Engine
    merger = SpeakerFragmentMerger(merge_threshold=CLUSTER_MERGE_THRESHOLD)
    consolidated_segments, speaker_vectors, canonical_map = merger.merge_fragments(
        diarized_segments, raw_speaker_vectors
    )

    # Step 3: Adaptive Threshold Engine
    audio_duration = max((float(s.get("end", 0.0)) for s in diarized_segments), default=600.0)
    avg_meeting_snr = float(sum(speaker_snrs.values()) / len(speaker_snrs)) if speaker_snrs else 20.0

    threshold_engine = AdaptiveThresholdEngine(
        base_high_threshold=HIGH_SCORE_THRESHOLD,
        base_high_gap=HIGH_SCORE_GAP,
        base_mid_threshold=MID_SCORE_THRESHOLD,
        base_mid_gap=MID_SCORE_GAP,
    )
    thresholds = threshold_engine.compute_thresholds(
        snr_db=avg_meeting_snr,
        num_speakers=len(speaker_vectors),
        duration_seconds=audio_duration,
    )

    # Step 4 & 5: Matching, Confidence, and Tiered Unknown Resolution
    confidence_engine = ConfidenceEngine()
    unknown_resolver = UnknownSpeakerResolver()

    mapping: Dict[str, str] = {}
    diagnostics: List[Dict[str, Any]] = []
    assigned_names: Dict[str, str] = {}

    for speaker_id, speaker_vec in speaker_vectors.items():
        name, score, gap, top_candidates = _match_speaker(speaker_vec, seed_db)

        gate_a = score >= thresholds.high_threshold and gap >= thresholds.high_gap
        gate_b = score >= thresholds.mid_threshold and gap >= thresholds.mid_gap
        gate_passed = "Gate A" if gate_a else ("Gate B" if gate_b else None)

        spk_dur = sum(
            float(s.get("end", 0)) - float(s.get("start", 0))
            for s in consolidated_segments
            if s.get("speaker") == speaker_id
        )

        match_diag = confidence_engine.evaluate_match(
            speaker_id=speaker_id,
            best_name=name,
            best_score=score,
            gap=gap,
            top_candidates=top_candidates,
            gate_passed=gate_passed,
            high_thresh=thresholds.high_threshold,
            high_gap=thresholds.high_gap,
            mid_thresh=thresholds.mid_threshold,
            mid_gap=thresholds.mid_gap,
            snr_db=speaker_snrs.get(speaker_id, avg_meeting_snr),
            speaking_duration_sec=spk_dur,
        )

        if match_diag["is_matched"]:
            matched_name = match_diag["matched_identity"]
            if matched_name not in assigned_names:
                mapping[speaker_id] = matched_name
                assigned_names[matched_name] = speaker_id
            else:
                # Duplicate name already assigned -> Fallback to Tiered Unknown Resolver
                res = unknown_resolver.resolve_unmatched_speaker(
                    speaker_id=speaker_id,
                    speaker_vec=speaker_vec,
                    top_candidates=top_candidates,
                    expected_roster=expected_roster,
                    meeting_metadata=meeting_metadata,
                )
                mapping[speaker_id] = res["resolved_name"]
                match_diag["matched_identity"] = res["resolved_name"]
                match_diag["resolution_tier"] = res["resolution_tier"]
        else:
            # Failed primary gates -> Fallback to Unknown Resolver
            res = unknown_resolver.resolve_unmatched_speaker(
                speaker_id=speaker_id,
                speaker_vec=speaker_vec,
                top_candidates=top_candidates,
                expected_roster=expected_roster,
                meeting_metadata=meeting_metadata,
            )
            mapping[speaker_id] = res["resolved_name"]
            match_diag["matched_identity"] = res["resolved_name"]
            match_diag["resolution_tier"] = res["resolution_tier"]

        diagnostics.append(match_diag)

    # Propagate canonical mapping for raw alias IDs
    full_speaker_mapping: Dict[str, str] = {}
    for raw_id, canonical_id in canonical_map.items():
        if canonical_id in mapping:
            full_speaker_mapping[raw_id] = mapping[canonical_id]
        else:
            full_speaker_mapping[raw_id] = raw_id

    logger.info(
        f"VoiceMapper: Pipeline completed for {len(full_speaker_mapping)} speaker labels. "
        f"Mapped identities: {full_speaker_mapping}"
    )

    return {
        "speaker_mapping": full_speaker_mapping,
        "consolidated_segments": consolidated_segments,
        "diagnostics": diagnostics,
        "adaptive_thresholds": thresholds._asdict(),
        "canonical_alias_map": canonical_map,
    }


def unload():
    """Release SpeechBrain model from RAM/VRAM."""
    global _ENGINE
    _ENGINE = None
    import gc
    gc.collect()
