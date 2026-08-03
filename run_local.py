"""
Local Pipeline Runner for Speaker Identification Validation.
Orchestrates existing meet-rec-ai modules without modifying production code.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Inject local FFmpeg binary path before module imports
LOCAL_FFMPEG_PATH = r"D:\Tools\ffmpeg-8.1.2-essentials_build\bin\ffmpeg.exe"
if not os.path.exists(LOCAL_FFMPEG_PATH):
    # Fallback search under D:\Tools if path varies
    import glob
    candidates = glob.glob(r"D:\Tools\**\ffmpeg.exe", recursive=True)
    if candidates:
        LOCAL_FFMPEG_PATH = candidates[0]

if "FFMPEG_PATH" not in os.environ and os.path.exists(LOCAL_FFMPEG_PATH):
    os.environ["FFMPEG_PATH"] = LOCAL_FFMPEG_PATH

import diarize
import media_io
import speaker_mapper_voice
import transcript_writer
import transcription

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("run_local")


def run_local_pipeline(media_file: str, stage1_only: bool = False, stage2_only: bool = False):
    start_time = time.time()
    workdir = Path(os.path.dirname(os.path.abspath(__file__)))
    seed_db_path = str(workdir / "seed_data.json")

    # If media_file doesn't exist, check for local fallbacks (e.g. meet-1.mp3 or meet-1.mp4)
    if not os.path.exists(media_file):
        alt_file = str(workdir / media_file)
        if os.path.exists(alt_file):
            media_file = alt_file
        elif os.path.exists(str(workdir / "meet-1.mp3")):
            media_file = str(workdir / "meet-1.mp3")
        else:
            raise FileNotFoundError(f"Input file not found: {media_file}")

    if not os.path.exists(seed_db_path):
        raise FileNotFoundError(f"Seed DB not found: {seed_db_path}")

    # Stage 1: Audio Extraction
    print("=" * 60)
    print(f"Stage 1: Audio Extraction (media_io.py) -> {media_file}")
    print("=" * 60)
    wav_path = str(workdir / "extracted_audio.wav")
    media_io.extract_audio(media_file, wav_path)
    audio_size_mb = os.path.getsize(wav_path) / (1024 * 1024)
    print(f"Stage 1 Completed: Audio extracted to {wav_path} ({audio_size_mb:.2f} MB)\n")

    if stage1_only:
        print("=" * 60)
        print("STOPPING AFTER STAGE 1 (Audio Extraction Verified)")
        print("=" * 60)
        return

    # Stage 2: Whisper Transcription
    print("=" * 60)
    print("Stage 2: Whisper Transcription (transcription.py)")
    print("=" * 60)
    whisper_segments = transcription.transcribe_audio_chunked(wav_path)
    print(f"Stage 2 Completed: Whisper ({len(whisper_segments)} text segments)\n")

    if stage2_only:
        print("=" * 60)
        print("STOPPING AFTER STAGE 2 (Whisper Transcription Verified)")
        print("=" * 60)
        return

    # Stage 3: PyAnnote Diarization
    print("=" * 60)
    print("Stage 3: PyAnnote Diarization (diarize.py)")
    print("=" * 60)
    if not os.getenv("HF_TOKEN"):
        raise ValueError("HF_TOKEN environment variable is missing! PyAnnote requires HF_TOKEN.")
    pyannote_segments = diarize.run_pyannote(wav_path)
    print(f"Stage 3 Completed: PyAnnote ({len(pyannote_segments)} speaker turns)\n")

    # Stage 4: Merge & Speaker Mapping
    print("=" * 60)
    print("Stage 4: Speaker Mapping (speaker_mapper_voice.py)")
    print("=" * 60)
    merged = diarize.assign_speakers_to_transcript(whisper_segments, pyannote_segments)
    explainable_res = speaker_mapper_voice.map_speakers_explainable(
        audio_path=wav_path,
        diarized_segments=merged,
        seed_db_path=seed_db_path,
    )
    speaker_mapping = explainable_res.get("speaker_mapping", {})
    consolidated = explainable_res.get("consolidated_segments", merged)
    diagnostics = explainable_res.get("diagnostics", [])
    adaptive_thresholds = explainable_res.get("adaptive_thresholds", {})
    print("Stage 4 Completed: Speaker Mapping\n")

    # Stage 5: Transcript Generation & Output Exports
    print("=" * 60)
    print("Stage 5: Transcript Generation (transcript_writer.py)")
    print("=" * 60)
    transcript_text = transcript_writer.render(
        consolidated, speaker_mapping=speaker_mapping, concatenate_runs=True
    )
    transcript_writer.write_file(str(workdir / "transcript.txt"), transcript_text)

    # Save output files
    with open(workdir / "speaker_mapping.json", "w", encoding="utf-8") as f:
        json.dump(speaker_mapping, f, indent=2)

    with open(workdir / "diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    output_payload = {
        "status": "success",
        "speaker_mapping": speaker_mapping,
        "transcription": consolidated,
        "transcript_text": transcript_text,
        "metadata": {
            "total_segments": len(consolidated),
            "speakers_mapped": len(speaker_mapping),
            "processing_time_seconds": round(time.time() - start_time, 2),
            "diagnostics": diagnostics,
            "adaptive_thresholds": adaptive_thresholds,
        },
    }
    with open(workdir / "output.json", "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print("Stage 5 Completed: Transcript Generation\n")
    print("=" * 60)
    print("ALL STAGES COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local Pipeline Runner")
    parser.add_argument("--video", type=str, default="meet-1.mp4", help="Input video file")
    parser.add_argument("--stage1-only", action="store_true", help="Stop after Stage 1 audio extraction")
    parser.add_argument("--stage2-only", action="store_true", help="Stop after Stage 2 Whisper transcription")
    args = parser.parse_args()

    try:
        run_local_pipeline(args.video, stage1_only=args.stage1_only, stage2_only=args.stage2_only)
    except Exception as e:
        print(f"\n[ERROR] PIPELINE FAILURE AT STAGE: {e}")
        logger.exception("Pipeline failed")
        sys.exit(1)
