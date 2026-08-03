"""
End-to-End Real Meeting Pipeline Execution & Validation Script.
Executes actual production modules sequentially on meet-1.mp3.
"""

from __future__ import annotations

import os
import sys
import time
import json
import logging
import traceback
from pathlib import Path

# Add meet-rec-ai directory to python sys.path
WORKDIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(WORKDIR))

# Load .env file explicitly
from dotenv import load_dotenv
load_dotenv(WORKDIR / ".env")

# Ensure FFmpeg is available in environment
LOCAL_FFMPEG_PATH = r"D:\Tools\ffmpeg-8.1.2-essentials_build\bin\ffmpeg.exe"
if not os.path.exists(LOCAL_FFMPEG_PATH):
    import glob
    candidates = glob.glob(r"D:\Tools\**\ffmpeg.exe", recursive=True)
    if candidates:
        LOCAL_FFMPEG_PATH = candidates[0]

if "FFMPEG_PATH" not in os.environ and os.path.exists(LOCAL_FFMPEG_PATH):
    os.environ["FFMPEG_PATH"] = LOCAL_FFMPEG_PATH

# Configure logging to record warnings/errors
class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        if record.levelno >= logging.WARNING:
            self.records.append(self.format(record))

log_capture = ListHandler()
log_capture.setFormatter(logging.Formatter("%(levelname)s [%(name)s]: %(message)s"))
logging.getLogger().addHandler(log_capture)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("e2e_runner")


def run_e2e():
    total_start_time = time.time()
    warnings_list = []
    
    media_file = str(WORKDIR / "meet-1.mp3")
    wav_path = str(WORKDIR / "extracted_audio.wav")
    seed_db_path = str(WORKDIR / "seed_data.json")

    print("=" * 70)
    print("STARTING REAL MEETING END-TO-END PIPELINE VALIDATION")
    print(f"Input Media File: {media_file}")
    print(f"Audio Path: {wav_path}")
    print(f"Seed DB Path: {seed_db_path}")
    print("=" * 70)

    # Validate inputs
    if not os.path.exists(seed_db_path):
        raise FileNotFoundError(f"Seed DB missing: {seed_db_path}")

    # Stage 1: Audio Extraction
    t0 = time.time()
    print("\n--- STAGE 1: Audio Extraction (media_io.py) ---")
    if os.path.exists(wav_path):
        print(f"[REUSED] Existing extracted audio found: {wav_path} ({os.path.getsize(wav_path) / (1024*1024):.2f} MB)")
        extraction_time = round(time.time() - t0, 3)
    else:
        if not os.path.exists(media_file):
            raise FileNotFoundError(f"Input media file missing: {media_file}")
        import media_io
        media_io.extract_audio(media_file, wav_path)
        extraction_time = round(time.time() - t0, 3)
        print(f"Extraction completed in {extraction_time}s -> {wav_path}")

    # Import pipeline modules
    import transcription
    import diarize
    import speaker_mapper_voice
    import transcript_writer

    # Ensure HF_TOKEN is synced to diarize module
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN is missing from environment/env file!")
    diarize.HF_TOKEN = hf_token

    # Stage 2: Faster-Whisper Transcription
    t0 = time.time()
    whisper_json_path = WORKDIR / "whisper_segments.json"
    print("\n--- STAGE 2: Faster-Whisper Transcription (transcription.py) ---")
    if os.path.exists(whisper_json_path):
        print(f"[REUSED] Existing Whisper transcription segments found: {whisper_json_path}")
        with open(whisper_json_path, "r", encoding="utf-8") as f:
            whisper_segments = json.load(f)
        whisper_time = 0.0
    else:
        whisper_segments = transcription.transcribe_audio_chunked(wav_path)
        whisper_time = round(time.time() - t0, 3)
        with open(whisper_json_path, "w", encoding="utf-8") as f:
            json.dump(whisper_segments, f, indent=2)
    print(f"Whisper Transcription Completed in {whisper_time}s ({len(whisper_segments)} text segments)")

    # Stage 3: PyAnnote Diarization
    t0 = time.time()
    print("\n--- STAGE 3: PyAnnote Diarization (diarize.py) ---")
    pyannote_segments = diarize.run_pyannote(wav_path)
    diarization_time = round(time.time() - t0, 3)
    unique_pyannote_speakers = sorted(list({s["speaker"] for s in pyannote_segments}))
    print(f"PyAnnote Diarization Completed in {diarization_time}s ({len(pyannote_segments)} speaker turns, {len(unique_pyannote_speakers)} unique raw speakers: {unique_pyannote_speakers})")

    # Stage 4: Speaker Identification
    t0 = time.time()
    print("\n--- STAGE 4: Speaker Identification (speaker_mapper_voice.py) ---")
    merged_transcript = diarize.assign_speakers_to_transcript(whisper_segments, pyannote_segments)
    explainable_res = speaker_mapper_voice.map_speakers_explainable(
        audio_path=wav_path,
        diarized_segments=merged_transcript,
        seed_db_path=seed_db_path,
    )
    speaker_id_time = round(time.time() - t0, 3)
    print(f"Speaker Identification Completed in {speaker_id_time}s")

    speaker_mapping = explainable_res.get("speaker_mapping", {})
    consolidated_segments = explainable_res.get("consolidated_segments", merged_transcript)
    diagnostics = explainable_res.get("diagnostics", [])
    adaptive_thresholds = explainable_res.get("adaptive_thresholds", {})
    canonical_alias_map = explainable_res.get("canonical_alias_map", {})

    # Stage 5: Transcript Generation
    t0 = time.time()
    print("\n--- STAGE 5: Transcript Generation (transcript_writer.py) ---")
    transcript_text = transcript_writer.render(
        consolidated_segments, speaker_mapping=speaker_mapping, concatenate_runs=True
    )
    transcript_writer.write_file(str(WORKDIR / "transcript.txt"), transcript_text)

    # Save output artifacts
    with open(WORKDIR / "speaker_mapping.json", "w", encoding="utf-8") as f:
        json.dump(speaker_mapping, f, indent=2)

    with open(WORKDIR / "diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    total_processing_time = round(time.time() - total_start_time, 3)

    output_payload = {
        "status": "success",
        "speaker_mapping": speaker_mapping,
        "transcription": consolidated_segments,
        "transcript_text": transcript_text,
        "metadata": {
            "total_segments": len(consolidated_segments),
            "speakers_mapped": len(speaker_mapping),
            "unique_raw_speakers": len(unique_pyannote_speakers),
            "raw_speakers": unique_pyannote_speakers,
            "canonical_alias_map": canonical_alias_map,
            "timings": {
                "total_processing_time_sec": total_processing_time,
                "extraction_time_sec": extraction_time,
                "whisper_time_sec": whisper_time,
                "diarization_time_sec": diarization_time,
                "speaker_id_time_sec": speaker_id_time,
            },
            "diagnostics": diagnostics,
            "adaptive_thresholds": adaptive_thresholds,
        },
    }

    with open(WORKDIR / "output.json", "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    # Collect captured warnings
    warnings_list.extend(log_capture.records)

    # Extract metrics for final summary report
    unknown_speakers = [
        diag["speaker_id"] for diag in diagnostics if not diag.get("is_matched") or "Unknown" in diag.get("matched_identity", "")
    ]

    fragmentation_events = [
        f"{raw_id} -> {target_id}"
        for raw_id, target_id in canonical_alias_map.items()
        if raw_id != target_id
    ]

    # Save full summary report file for easy review
    summary_report = {
        "processing_time_sec": total_processing_time,
        "whisper_time_sec": whisper_time,
        "diarization_time_sec": diarization_time,
        "speaker_id_time_sec": speaker_id_time,
        "speakers_detected_raw": len(unique_pyannote_speakers),
        "final_speaker_mapping": speaker_mapping,
        "diagnostics": diagnostics,
        "unknown_speakers": unknown_speakers,
        "fragmentation_events": fragmentation_events,
        "warnings": warnings_list,
        "exceptions": [],
    }

    with open(WORKDIR / "e2e_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    print("\n" + "=" * 70)
    print("SUCCESS: ALL 5 STAGES COMPLETED CLEANLY")
    print("=" * 70)
    return summary_report


if __name__ == "__main__":
    try:
        run_e2e()
    except Exception as e:
        print("\n" + "!" * 70)
        print("PIPELINE STAGE FAILURE ENCOUNTERED")
        print("!" * 70)
        print(f"\n[ROOT CAUSE EXPLANATION]: {e}")
        print("\n[FULL TRACEBACK]:")
        traceback.print_exc()
        sys.exit(1)
