"""
RunPod handler - FULL pipeline.

Single GPU container that owns:
  1. Resolve input  (video_url | audio_url | audio_base64)
  2. (Optional) Download video, extract audio with ffmpeg
  3. (Optional) Upload audio to R2 for traceability / hand-off
  4. Whisper large-v2 transcription (streamed in 5-minute chunks)
  5. Pyannote 3.1 diarization (Whisper unloaded first)
  6. Merge transcript with speaker labels
  7. SpeechBrain ECAPA-TDNN biometric speaker mapping
  8. Render transcript.txt
  9. Upload transcript.txt to R2 (optional)
 10. POST transcript to meet-insights service (INSIGHTS_ENDPOINT_URL)

Input shape (event['input']):
{
    # ONE of these three is required:
    "video_url":   "https://...",       # we'll ffmpeg-extract audio
    "audio_url":   "https://...",       # we'll use it directly
    "audio_base64":"...",                # legacy / small files only

    # Optional:
    "meeting_id":   "abc-defg-hij",     # used in R2 keys; default = uuid
    "min_speakers": 2,
    "max_speakers": 10,
    "num_speakers": null,
    "upload_audio_to_r2":      true,    # default true when audio came from ffmpeg
    "upload_transcript_to_r2": true,    # default true
    "return_transcript_inline": true,   # also return the full transcript text in response
}

Output:
{
    "status": "success",
    "meeting_id": "...",
    "transcript_text":     "[00:00:01] Ketan Patil: hello...",  # if return_transcript_inline
    "transcript_url":      "https://...presigned...",
    "audio_url":           "https://...presigned...",
    "speaker_mapping":     {"SPEAKER_00": "Ketan Patil", ...},
    "transcript_segments": [{start, end, speaker, text}, ...],
    "metadata": {
        "total_segments":       N,
        "speakers_detected":    K,
        "speakers_mapped":      M,
        "audio_duration_seconds": float,
        "processing_time_seconds": float
    },
    "insights": {                        # present when INSIGHTS_ENDPOINT_URL is set
        "summary": "...",
        "action_items": [...],
        "insights": {...},
        "stub": false
    }
}
"""

from __future__ import annotations

import base64
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests
import runpod

import diarize
import media_io
import speaker_mapper_voice
import transcript_writer
import transcription

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s",
)
logger = logging.getLogger(__name__)


SEED_DB_PATH = os.getenv("SEED_DB_PATH", "/app/seed_data.json")
R2_AUDIO_PREFIX = os.getenv("R2_AUDIO_PREFIX", "audios")
R2_TRANSCRIPT_PREFIX = os.getenv("R2_TRANSCRIPT_PREFIX", "transcripts")
PRESIGN_EXPIRY_SECONDS = int(os.getenv("R2_PRESIGN_EXPIRY", "604800"))  # 7 days
INSIGHTS_ENDPOINT_URL = os.getenv("INSIGHTS_ENDPOINT_URL", "").rstrip("/")
INSIGHTS_TIMEOUT_SECONDS = int(os.getenv("INSIGHTS_TIMEOUT", "120"))


# =========================================
# Helpers
# =========================================

def _err(message: str, error_type: str = "PipelineError", **extra) -> Dict[str, Any]:
    logger.error(f"{error_type}: {message}")
    return {"status": "error", "message": message, "error_type": error_type, **extra}


def _download_url_to(local_path: str, url: str, label: str) -> str:
    logger.info(f"Downloading {label} from URL -> {local_path}")
    with requests.get(url, stream=True, timeout=(30, 600)) as resp:
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    logger.info(f"Downloaded {label}: {size_mb:.1f} MB")
    return local_path


def _decode_base64_to(local_path: str, b64: str) -> str:
    data = base64.b64decode(b64)
    with open(local_path, "wb") as f:
        f.write(data)
    size_mb = len(data) / (1024 * 1024)
    logger.info(f"Decoded base64 audio: {size_mb:.1f} MB -> {local_path}")
    return local_path


def _r2_available() -> bool:
    return bool(
        os.getenv("R2_ACCESS_KEY_ID")
        and os.getenv("R2_SECRET_ACCESS_KEY")
        and os.getenv("R2_ENDPOINT_URL")
    )


def _safe_filename_stem(url_or_id: str) -> str:
    """Pick a filesystem-safe stem from a URL or id."""
    try:
        path = urlparse(url_or_id).path
        stem = Path(path).stem if path else url_or_id
    except Exception:
        stem = url_or_id
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:64]
    return stem or "input"


# =========================================
# Stage handlers
# =========================================

def _resolve_audio(
    input_data: Dict[str, Any],
    workdir: Path,
    meeting_id: str,
) -> Dict[str, Any]:
    """
    Materialize a local mono-16k WAV from whichever input was provided.

    Returns a dict with keys: audio_path, came_from ('video'|'audio_url'|'base64')
    """
    video_url = input_data.get("video_url")
    audio_url = input_data.get("audio_url")
    audio_b64 = input_data.get("audio_base64")

    audio_path = workdir / "audio.wav"

    if video_url:
        logger.info("Input: video_url")
        video_local = workdir / f"video_{_safe_filename_stem(video_url)}.bin"
        _download_url_to(str(video_local), video_url, "video")
        media_io.extract_audio(str(video_local), str(audio_path))
        try:
            os.remove(video_local)
        except OSError:
            pass
        return {"audio_path": str(audio_path), "came_from": "video"}

    if audio_url:
        logger.info("Input: audio_url")
        # Use the URL'd file directly if it's already WAV; otherwise re-encode
        url_stem = Path(urlparse(audio_url).path).suffix.lower()
        if url_stem == ".wav":
            _download_url_to(str(audio_path), audio_url, "audio")
        else:
            raw_local = workdir / f"audio_raw{url_stem or '.bin'}"
            _download_url_to(str(raw_local), audio_url, "audio")
            media_io.extract_audio(str(raw_local), str(audio_path))
            try:
                os.remove(raw_local)
            except OSError:
                pass
        return {"audio_path": str(audio_path), "came_from": "audio_url"}

    if audio_b64:
        logger.info("Input: audio_base64 (legacy path)")
        _decode_base64_to(str(audio_path), audio_b64)
        return {"audio_path": str(audio_path), "came_from": "base64"}

    raise ValueError("One of 'video_url', 'audio_url', or 'audio_base64' is required")


# =========================================
# Insights helper
# =========================================

def _call_insights(transcript_text: str, meeting_id: str) -> Optional[Dict[str, Any]]:
    """
    POST the finished transcript to meet-insights and return the parsed JSON.
    Non-fatal: logs warning and returns None on any failure.
    Only fires when INSIGHTS_ENDPOINT_URL is set.
    """
    if not INSIGHTS_ENDPOINT_URL:
        logger.info("INSIGHTS_ENDPOINT_URL not set — skipping insights call")
        return None

    url = f"{INSIGHTS_ENDPOINT_URL}/analyze"
    payload = {"transcript_text": transcript_text, "meeting_id": meeting_id}
    try:
        logger.info(f"Calling meet-insights: {url}")
        resp = requests.post(url, json=payload, timeout=INSIGHTS_TIMEOUT_SECONDS)
        resp.raise_for_status()
        result = resp.json()
        logger.info(
            f"meet-insights responded: stub={result.get('stub')} "
            f"summary_len={len(result.get('summary',''))}"
        )
        return result
    except Exception as e:
        logger.warning(f"meet-insights call failed (non-fatal): {e}")
        return None


# =========================================
# Main handler
# =========================================

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    start_time = time.time()
    audio_r2_key: Optional[str] = None
    transcript_r2_key: Optional[str] = None

    try:
        input_data = event.get("input", {}) or {}
        meeting_id = input_data.get("meeting_id") or uuid.uuid4().hex[:12]
        min_speakers = input_data.get("min_speakers")
        max_speakers = input_data.get("max_speakers")
        num_speakers = input_data.get("num_speakers")

        upload_audio_to_r2_flag = input_data.get(
            "upload_audio_to_r2",
            True,  # default on; harmless if R2 isn't configured (we'll skip)
        )
        upload_transcript_to_r2_flag = input_data.get(
            "upload_transcript_to_r2", True
        )
        return_inline = input_data.get("return_transcript_inline", True)

        logger.info("=" * 70)
        logger.info(f"FULL PIPELINE | meeting_id={meeting_id}")
        logger.info("=" * 70)

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)

            # ---- Stage 1: resolve audio ----
            audio_info = _resolve_audio(input_data, workdir, meeting_id)
            audio_path = audio_info["audio_path"]
            came_from = audio_info["came_from"]

            # ---- Stage 2: optional R2 upload of audio ----
            audio_presigned_url: Optional[str] = None
            if upload_audio_to_r2_flag:
                if not _r2_available():
                    logger.warning("R2 envs not set; skipping audio upload")
                else:
                    try:
                        import r2_client
                        audio_r2_key = f"{R2_AUDIO_PREFIX}/{meeting_id}.wav"
                        r2_client.upload_file(audio_path, audio_r2_key, content_type="audio/wav")
                        audio_presigned_url = r2_client.generate_presigned_url(
                            audio_r2_key, expires_in=PRESIGN_EXPIRY_SECONDS
                        )
                    except Exception as e:
                        logger.warning(f"R2 audio upload failed (non-fatal): {e}")
                        audio_r2_key = None

            # ---- Stage 3: transcribe (Whisper) ----
            logger.info("STAGE 3: Whisper transcription")
            whisper_segments = transcription.transcribe_audio_chunked(audio_path)
            logger.info(f"Whisper produced {len(whisper_segments)} segments")

            # ---- Stage 4: diarize (Pyannote) ----
            logger.info("STAGE 4: Pyannote diarization")
            pyannote_segments = diarize.run_pyannote(
                audio_path,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            logger.info(f"Pyannote produced {len(pyannote_segments)} speaker segments")

            # ---- Stage 5: merge ----
            logger.info("STAGE 5: Merge transcript with speakers")
            merged = diarize.assign_speakers_to_transcript(
                whisper_segments, pyannote_segments
            )

            # ---- Stage 6: biometric speaker mapping ----
            logger.info("STAGE 6: Biometric speaker mapping (SpeechBrain)")
            speaker_mapping: Dict[str, str] = {}
            if not os.path.exists(SEED_DB_PATH):
                logger.warning(
                    f"Seed DB not found at {SEED_DB_PATH}; skipping voice mapping. "
                    "Speakers will remain as SPEAKER_XX."
                )
            else:
                try:
                    speaker_mapping = speaker_mapper_voice.map_speakers(
                        audio_path=audio_path,
                        diarized_segments=merged,
                        seed_db_path=SEED_DB_PATH,
                    )
                except Exception as e:
                    logger.warning(f"Voice mapping failed (non-fatal): {e}", exc_info=True)
                finally:
                    speaker_mapper_voice.unload()

            # ---- Stage 7: render transcript.txt ----
            logger.info("STAGE 7: Render transcript.txt")
            transcript_text = transcript_writer.render(
                merged, speaker_mapping=speaker_mapping, concatenate_runs=True
            )

            transcript_local_path = workdir / "transcript.txt"
            transcript_writer.write_file(str(transcript_local_path), transcript_text)

            # ---- Stage 8: upload transcript.txt ----
            transcript_presigned_url: Optional[str] = None
            if upload_transcript_to_r2_flag:
                if not _r2_available():
                    logger.warning("R2 envs not set; skipping transcript upload")
                else:
                    try:
                        import r2_client
                        transcript_r2_key = f"{R2_TRANSCRIPT_PREFIX}/{meeting_id}.txt"
                        r2_client.upload_text(transcript_text, transcript_r2_key)
                        transcript_presigned_url = r2_client.generate_presigned_url(
                            transcript_r2_key, expires_in=PRESIGN_EXPIRY_SECONDS
                        )
                    except Exception as e:
                        logger.warning(f"R2 transcript upload failed (non-fatal): {e}")
                        transcript_r2_key = None

            # ---- Done ----
            unique_speakers = sorted({s["speaker"] for s in merged if s.get("speaker")})
            audio_duration = max((s["end"] for s in merged), default=0.0)
            processing_time = time.time() - start_time

            logger.info("=" * 70)
            logger.info(
                f"DONE | segments={len(merged)} speakers={len(unique_speakers)} "
                f"mapped={len(speaker_mapping)} duration={audio_duration:.1f}s "
                f"elapsed={processing_time:.1f}s"
            )
            logger.info("=" * 70)

            response: Dict[str, Any] = {
                "status": "success",
                "meeting_id": meeting_id,
                "input_kind": came_from,
                "speaker_mapping": speaker_mapping,
                "transcript_segments": merged,
                "metadata": {
                    "total_segments": len(merged),
                    "speakers_detected": len(unique_speakers),
                    "speakers_mapped": len(speaker_mapping),
                    "speaker_ids": unique_speakers,
                    "audio_duration_seconds": round(audio_duration, 2),
                    "processing_time_seconds": round(processing_time, 2),
                },
            }
            if return_inline:
                response["transcript_text"] = transcript_text
            if audio_presigned_url:
                response["audio_url"] = audio_presigned_url
                response["audio_r2_key"] = audio_r2_key
            if transcript_presigned_url:
                response["transcript_url"] = transcript_presigned_url
                response["transcript_r2_key"] = transcript_r2_key

            # ---- Stage 10: call meet-insights ----
            insights_result = _call_insights(transcript_text, meeting_id)
            if insights_result:
                response["insights"] = insights_result

            return response

    except FileNotFoundError as e:
        return _err(str(e), "FileNotFoundError")
    except ValueError as e:
        return _err(str(e), "ValidationError")
    except requests.HTTPError as e:
        return _err(f"Download failed: {e}", "DownloadError")
    except Exception as e:
        logger.exception("Pipeline failed")
        return _err(str(e), type(e).__name__)


# =========================================
# RunPod entrypoint
# =========================================
if __name__ == "__main__":
    logger.info("Starting RunPod GPU handler (full pipeline)")
    runpod.serverless.start({"handler": handler})
