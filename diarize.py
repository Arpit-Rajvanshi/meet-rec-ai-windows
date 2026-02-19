"""
Diarization Module - MINIMAL GPU ONLY (for RunPod)

ONLY contains GPU functions:
- run_pyannote() - Speaker diarization
- assign_speakers_to_transcript() - Merge with transcription

NO utilities, NO concatenation, NO formatting.
All post-processing happens on your server.
"""

import os
import torch
import gc
import logging
from pyannote.audio import Pipeline
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()
logger = logging.getLogger(__name__)

IS_RUNPOD = os.getenv('RUNPOD_POD_ID') is not None

if IS_RUNPOD:
    os.environ['HF_HOME'] = '/runpod-volume/huggingface'
    os.environ['TORCH_HOME'] = '/runpod-volume/torch'
    print("🔧 RunPod environment detected - using persistent volume for model cache")

HF_TOKEN = os.getenv("HF_TOKEN")

# =============================
# GPU FUNCTION 1: PYANNOTE DIARIZATION
# =============================

def run_pyannote(audio_path, num_speakers=None, min_speakers=None, max_speakers=None):
    """
    Runs Pyannote speaker diarization on GPU.
    Returns segments with FLOAT timestamps and SPEAKER_XX labels.
    
    Args:
        audio_path: Path to audio file
        num_speakers: Exact number of speakers (optional)
        min_speakers: Minimum speakers (optional)
        max_speakers: Maximum speakers (optional)
        
    Returns:
        List of dicts: [{"start": 0.0, "end": 5.2, "speaker": "SPEAKER_00"}]
    """
    print(f"🗣️  Loading Pyannote Diarization...")
    
    if num_speakers:
        print(f"   🎯 Constraining to exactly {num_speakers} speaker(s)")
    elif min_speakers or max_speakers:
        constraint_msg = []
        if min_speakers:
            constraint_msg.append(f"min={min_speakers}")
        if max_speakers:
            constraint_msg.append(f"max={max_speakers}")
        print(f"   🎯 Speaker range: {', '.join(constraint_msg)}")
    
    if not HF_TOKEN:
        raise ValueError("❌ HF_TOKEN is missing! Pyannote requires Hugging Face token.")

    pipeline = None
    try:
        # Try different authentication methods (different pyannote versions)
        pipeline = None
        auth_error = None
        
        # Try method 1: token parameter (newest versions)
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=HF_TOKEN
            )
        except TypeError as e1:
            auth_error = e1
            # Try method 2: use_auth_token parameter (older versions)
            try:
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=HF_TOKEN
                )
            except TypeError as e2:
                # Try method 3: environment variable (fallback)
                os.environ['HUGGING_FACE_HUB_TOKEN'] = HF_TOKEN
                try:
                    pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1"
                    )
                except Exception as e3:
                    raise Exception(f"All authentication methods failed. Errors: token={e1}, use_auth_token={e2}, env={e3}")
        
        if pipeline is None:
            raise Exception("Failed to load Pyannote pipeline")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)

        # Run inference
        diarization = pipeline(
            audio_path, 
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
        
        speaker_segments = []
        unique_speakers = set()
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker)
            })
            unique_speakers.add(str(speaker))
        
        print(f"✅ Pyannote finished. Found {len(speaker_segments)} segments across {len(unique_speakers)} speaker(s).")
        
        if num_speakers and len(unique_speakers) != num_speakers:
            print(f"   ⚠️  Warning: Expected {num_speakers} speakers but detected {len(unique_speakers)}")
        
        return speaker_segments

    except Exception as e:
        print(f"❌ Pyannote failed: {e}")
        logger.error(f"Pyannote error details: {e}", exc_info=True)
        return []
    
    finally:
        # Free GPU memory
        print("🗑️  Unloading Pyannote...")
        if pipeline is not None:
            del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================
# GPU FUNCTION 2: MERGE TRANSCRIPTION + DIARIZATION
# =============================

def assign_speakers_to_transcript(whisper_segments, pyannote_segments):
    """
    Merges Whisper transcription with Pyannote speaker labels.
    
    Args:
        whisper_segments: List from Whisper [{"start": 0.0, "end": 5.2, "text": "..."}]
        pyannote_segments: List from Pyannote [{"start": 0.0, "end": 5.2, "speaker": "SPEAKER_00"}]
        
    Returns:
        Merged list: [{"start": 0.0, "end": 5.2, "speaker": "SPEAKER_00", "text": "..."}]
    """
    final_transcript = []
    
    for text_seg in whisper_segments:
        # Ensure timestamps are float
        try:
            tStart = float(text_seg['start'])
            tEnd = float(text_seg['end'])
        except (ValueError, TypeError, KeyError):
            tStart = float(text_seg.get('start', 0))
            tEnd = float(text_seg.get('end', tStart + 3))
        
        # Find best matching speaker based on overlap
        speaker_counts = {}
        for pa_seg in pyannote_segments:
            pa_start = float(pa_seg['start'])
            pa_end = float(pa_seg['end'])
            
            overlap_start = max(tStart, pa_start)
            overlap_end = min(tEnd, pa_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            if overlap_duration > 0:
                spk = pa_seg['speaker']
                speaker_counts[spk] = speaker_counts.get(spk, 0) + overlap_duration

        best_speaker = max(speaker_counts, key=speaker_counts.get) if speaker_counts else "UNKNOWN_SPEAKER"

        final_transcript.append({
            "start": tStart,
            "end": tEnd,
            "speaker": best_speaker,
            "text": text_seg.get('text', '')
        })
    
    print(f"✅ Merged {len(final_transcript)} transcript segments with speakers")
    return final_transcript