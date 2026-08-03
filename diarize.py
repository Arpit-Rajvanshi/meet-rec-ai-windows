"""
Speaker diarization with Pyannote 3.1.

Public API:
    run_pyannote(audio_path, num_speakers=None, min_speakers=None, max_speakers=None)
    assign_speakers_to_transcript(whisper_segments, pyannote_segments)

The run_pyannote call frees the Whisper model (if loaded by transcription.py)
BEFORE loading the Pyannote pipeline. This prevents holding both ~3GB Whisper
weights and ~1GB Pyannote weights on the GPU simultaneously, which was a
contributing factor to OOM/long-runtime failures on long audio.
"""

from __future__ import annotations

import gc
import logging
import os
from typing import Dict, List, Optional

import soundfile as sf
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline

logger = logging.getLogger(__name__)
load_dotenv()


IS_RUNPOD = os.getenv("RUNPOD_POD_ID") is not None
if IS_RUNPOD:
    os.environ["HF_HOME"] = "/runpod-volume/huggingface"
    os.environ["TORCH_HOME"] = "/runpod-volume/torch"
    logger.info("RunPod environment detected - using persistent volume for model cache")

HF_TOKEN = os.getenv("HF_TOKEN")


def _free_whisper_before_pyannote() -> None:
    """
    Best-effort: ask transcription.py to release Whisper before we load Pyannote.
    Silent if transcription is not imported.
    """
    try:
        import transcription
        transcription.unload_model()
    except Exception as e:
        logger.debug(f"Could not unload Whisper before Pyannote (non-fatal): {e}")


def _load_pipeline(token: str) -> Pipeline:
    """Try the three known auth methods across pyannote versions."""
    try:
        return Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=token,
        )
    except TypeError:
        pass
    try:
        return Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token,
        )
    except TypeError:
        pass
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    return Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")


def run_pyannote(
    audio_path: str,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> List[Dict]:
    """
    Run Pyannote 3.1 diarization on a WAV file.

    Returns a list of {"start", "end", "speaker"} dicts. Speakers are anonymous
    SPEAKER_00, SPEAKER_01, ... labels; resolve them later with biometric mapping.
    """
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN is missing! Pyannote 3.1 requires a Hugging Face token.")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if num_speakers:
        logger.info(f"Pyannote: constraining to exactly {num_speakers} speaker(s)")
    elif min_speakers or max_speakers:
        logger.info(f"Pyannote: speaker range min={min_speakers} max={max_speakers}")
    else:
        logger.info("Pyannote: auto-detecting speakers")

    # Free Whisper VRAM before we allocate Pyannote's
    _free_whisper_before_pyannote()

    pipeline = None
    try:
        pipeline = _load_pipeline(HF_TOKEN)
        if pipeline is None:
            raise RuntimeError("Failed to load Pyannote pipeline")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)

        data, sample_rate = sf.read(audio_path, dtype="float32")
        waveform = torch.from_numpy(data)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.transpose(0, 1)

        audio_dict = {"waveform": waveform, "sample_rate": sample_rate}

        diarization = pipeline(
            audio_dict,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        annotation = getattr(diarization, "speaker_diarization", diarization)

        speaker_segments: List[Dict] = []
        unique_speakers = set()
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            speaker_segments.append({
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
            })
            unique_speakers.add(str(speaker))

        logger.info(
            f"Pyannote done: {len(speaker_segments)} segments, "
            f"{len(unique_speakers)} unique speakers"
        )
        if num_speakers and len(unique_speakers) != num_speakers:
            logger.warning(
                f"Expected {num_speakers} speakers, detected {len(unique_speakers)}"
            )
        return speaker_segments

    except Exception as e:
        logger.error(f"Pyannote failed: {e}", exc_info=True)
        raise

    finally:
        # Release Pyannote VRAM/RAM as soon as we're done
        logger.info("Unloading Pyannote pipeline...")
        if pipeline is not None:
            try:
                del pipeline
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def assign_speakers_to_transcript(
    whisper_segments: List[Dict],
    pyannote_segments: List[Dict],
) -> List[Dict]:
    """
    Merge Whisper transcription with Pyannote speaker labels by max-overlap.

    For each Whisper segment, find the Pyannote segment with the most
    time-overlap and tag the Whisper segment with that speaker.
    """
    final_transcript: List[Dict] = []

    for text_seg in whisper_segments:
        try:
            tStart = float(text_seg["start"])
            tEnd = float(text_seg["end"])
        except (ValueError, TypeError, KeyError):
            tStart = float(text_seg.get("start", 0))
            tEnd = float(text_seg.get("end", tStart + 3))

        # Score each Pyannote speaker by overlap duration with this text segment
        speaker_counts: Dict[str, float] = {}
        for pa_seg in pyannote_segments:
            pa_start = float(pa_seg["start"])
            pa_end = float(pa_seg["end"])
            overlap = max(0.0, min(tEnd, pa_end) - max(tStart, pa_start))
            if overlap > 0:
                spk = pa_seg["speaker"]
                speaker_counts[spk] = speaker_counts.get(spk, 0.0) + overlap

        best_speaker = (
            max(speaker_counts, key=speaker_counts.get)
            if speaker_counts
            else "UNKNOWN_SPEAKER"
        )

        final_transcript.append({
            "start": tStart,
            "end": tEnd,
            "speaker": best_speaker,
            "text": text_seg.get("text", ""),
        })

    logger.info(f"Merged {len(final_transcript)} transcript segments with speakers")
    return final_transcript
