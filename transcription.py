from typing import List, Dict
import os
import json
import subprocess
import torch
import soundfile as sf
from faster_whisper import WhisperModel


IS_RUNPOD = os.getenv('RUNPOD_POD_ID') is not None

if IS_RUNPOD:
    # Use RunPod volume for model cache to persist between cold starts
    os.environ['HF_HOME'] = '/runpod-volume/huggingface'
    os.environ['TORCH_HOME'] = '/runpod-volume/torch'
    print("🔧 RunPod environment detected - using persistent volume for model cache")
    
# =============================
# MODEL & PERFORMANCE CONFIG
# =============================

MODEL_NAME = "large-v2"          # High accuracy, production grade
CHUNK_SIZE_SECONDS = 300         # 5 minutes (safe for long meetings)
BEAM_SIZE = 5                    # Accuracy vs speed tradeoff
TEMPERATURE = 0.0                # Deterministic output

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

# -----------------------------
# LOAD  MODEL once at module scope
# -----------------------------

MODEL = WhisperModel(
    MODEL_NAME,
    device=DEVICE,
    compute_type=COMPUTE_TYPE
)

# =============================
# PUBLIC API (ENTRYPOINT)
# =============================

def transcribe_audio_chunked(audio_path: str) -> List[Dict]:
    """
    Transcribe an audio file using chunked Faster-Whisper.

    Args:
        audio_path (str): Path to WAV audio (mono, 16kHz preferred)

    Returns:
        List[Dict]: [
            {
              "start": float,
              "end": float,
              "text": str
            }
        ]
    """

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # -----------------------------
    # LOAD AUDIO
    # -----------------------------
    audio, sample_rate = sf.read(audio_path)
    total_duration = len(audio) / sample_rate

    # -----------------------------
    # USE MODEL 
    # -----------------------------
    model = MODEL

    results: List[Dict] = []
    offset = 0.0

    # -----------------------------
    # CHUNKED TRANSCRIPTION
    # -----------------------------
    while offset < total_duration:
        end_offset = min(offset + CHUNK_SIZE_SECONDS, total_duration)

        start_sample = int(offset * sample_rate)
        end_sample = int(end_offset * sample_rate)
        chunk_audio = audio[start_sample:end_sample]

        segments, _ = model.transcribe(
            chunk_audio,
            beam_size=BEAM_SIZE,
            temperature=TEMPERATURE,
            vad_filter=False,
            condition_on_previous_text=True,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0
        )

        for seg in segments:
            results.append({
                "start": round(seg.start + offset, 2),
                "end": round(seg.end + offset, 2),
                "text": seg.text.strip()
            })

        offset += CHUNK_SIZE_SECONDS

    return results