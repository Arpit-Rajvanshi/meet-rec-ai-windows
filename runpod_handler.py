"""
RunPod handler - FULL pipeline.

Single GPU container that owns:
  1. Resolve input  (r2_key | video_url | audio_url | audio_base64)
  2. (Optional) Download video, extract audio with ffmpeg
  3. (Optional) Upload audio to R2 for traceability / hand-off
  4. Whisper large-v2 transcription (streamed in 5-minute chunks w/ overlap)
  5. Pyannote 3.1 diarization (Whisper unloaded first)
  6. Merge transcript with speaker labels
  7. SpeechBrain ECAPA-TDNN biometric speaker mapping
  8. Render transcript.txt
  9. Upload transcript.txt to R2 (optional)
 10. POST transcript to meet-insights service (INSIGHTS_ENDPOINT_URL)

Input shape (event['input']):
{
    # ONE of these is required:
    "r2_key":      "recordings/18689.mkv",  # fetched directly from R2 via boto3
    "video_url":   "https://...",            # streamed download, then ffmpeg-extract
    "audio_url":   "https://...",            # streamed download, re-encode if not WAV
    "audio_base64":"...",                    # legacy / small files only

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
from typing import Any, Dict, Generator, Optional
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

# How often (seconds) to emit a heartbeat yield so RunPod doesn't kill idle-looking jobs
HEARTBEAT_INTERVAL = 30


# =========================================
# Helpers
# =========================================

def _err(message: str, error_type: str = "PipelineError", **extra) -> Dict[str, Any]:
    logger.error(f"{error_type}: {message}")
    return {"status": "error", "message": message, "error_type": error_type, **extra}


def _download_url_to(local_path: str, url: str, label: str, retries: int = 3) -> str:
    """
    Stream-download a URL to a local file. Never loads the full content into RAM.
    Retries up to `retries` times with exponential backoff (2s, 4s, 8s).
    Chunk size is 8 MB to efficiently handle 10 GB+ files.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 2):
        try:
            logger.info(f"Downloading {label} (attempt {attempt}/{retries + 1}) -> {local_path}")
            with requests.get(url, stream=True, timeout=(30, 900)) as resp:
                resp.raise_for_status()
                bytes_written = 0
                with open(local_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            bytes_written += len(chunk)
            size_mb = bytes_written / (1024 * 1024)
            logger.info(f"Downloaded {label}: {size_mb:.1f} MB")
            return local_path
        except Exception as e:
            last_exc = e
            if attempt > retries:
                break
            wait = 2 ** attempt
            logger.warning(
                f"Download attempt {attempt} failed for {label}: {e}. "
                f"Retrying in {wait}s..."
            )
            time.sleep(wait)
    raise RuntimeError(f"Download failed after {retries + 1} attempts for {label}: {last_exc}") from last_exc


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

    Priority: r2_key > video_url > audio_url > audio_base64

    Returns a dict with keys: audio_path, came_from
    """
    r2_key    = input_data.get("r2_key")
    video_url = input_data.get("video_url")
    audio_url = input_data.get("audio_url")
    audio_b64 = input_data.get("audio_base64")

    audio_path = workdir / "audio.wav"

    if r2_key:
        logger.info(f"Input: r2_key={r2_key}")
        if not _r2_available():
            raise ValueError(
                "r2_key provided but R2 credentials (R2_ACCESS_KEY_ID, "
                "R2_SECRET_ACCESS_KEY, R2_ENDPOINT_URL) are not set in env"
            )
        import r2_client
        # Preserve original extension so ffmpeg can detect the container format
        ext = Path(r2_key).suffix or ".mkv"
        raw_local = workdir / f"r2_{_safe_filename_stem(r2_key)}{ext}"
        logger.info(f"Downloading r2_key={r2_key} via R2 API -> {raw_local}")
        r2_client.download_file(r2_key, str(raw_local))
        size_mb = os.path.getsize(str(raw_local)) / (1024 * 1024)
        logger.info(f"R2 download complete: {size_mb:.1f} MB")
        media_io.extract_audio(str(raw_local), str(audio_path))
        try:
            os.remove(raw_local)
        except OSError:
            pass
        return {"audio_path": str(audio_path), "came_from": "r2_key"}

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
        url_ext = Path(urlparse(audio_url).path).suffix.lower()
        if url_ext == ".wav":
            _download_url_to(str(audio_path), audio_url, "audio")
        else:
            raw_local = workdir / f"audio_raw{url_ext or '.bin'}"
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

    raise ValueError(
        "One of 'r2_key', 'video_url', 'audio_url', or 'audio_base64' is required"
    )


# =========================================
# Insights helper
# =========================================

def _call_insights(transcript_text: str, meeting_id: str) -> Optional[Dict[str, Any]]:
    """
    TODO: Deprecated / unreferenced legacy helper.
    POST the finished transcript to meet-insights and return the parsed JSON.
    Non-fatal: logs warning and returns None on any failure.
    Only fires when INSIGHTS_ENDPOINT_URL is set (defaults to empty in production).
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
            f"summary_len={len(result.get('summary', ''))}"
        )
        return result
    except Exception as e:
        logger.warning(f"meet-insights call failed (non-fatal): {e}")
        return None


# =========================================
# Heartbeat helper
# =========================================

class _Heartbeat:
    """
    Tracks when we last yielded a progress update. Call .ping() inside
    long-running stages; it returns a heartbeat dict if HEARTBEAT_INTERVAL
    seconds have elapsed since the last yield, or None if it's too soon.
    """
    def __init__(self):
        self._last = time.time()

    def ping(self, stage: str, detail: str = "") -> Optional[Dict[str, Any]]:
        now = time.time()
        if now - self._last >= HEARTBEAT_INTERVAL:
            self._last = now
            msg = f"[heartbeat] stage={stage}" + (f" {detail}" if detail else "")
            logger.info(msg)
            return {"status": "processing", "stage": stage, "detail": detail}
        return None


# =========================================
# Main handler (generator — yields heartbeats, final result last)
# =========================================

def handler(job: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """
    RunPod generator handler. Yields intermediate progress dicts (heartbeats)
    so RunPod does not kill the job during long stages, then yields the final
    result as the last item. RunPod uses the last yielded value as the job output.
    """
    start_time = time.time()
    hb = _Heartbeat()
    audio_r2_key: Optional[str] = None
    transcript_r2_key: Optional[str] = None

    try:
        input_data = job.get("input", {}) or {}
        meeting_id = input_data.get("meeting_id") or uuid.uuid4().hex[:12]
        min_speakers = input_data.get("min_speakers")
        max_speakers = input_data.get("max_speakers")
        num_speakers = input_data.get("num_speakers")

        raw_attendees = input_data.get("attendee_ids") or input_data.get("attendees")
        if (
            num_speakers is None
            and min_speakers is None
            and max_speakers is None
            and isinstance(raw_attendees, list)
            and len(raw_attendees) == 1
        ):
            first = raw_attendees[0]
            first_id = first.get("id") if isinstance(first, dict) else first
            if isinstance(first_id, str) and first_id.strip():
                num_speakers = 1
                logger.info(
                    f"[{meeting_id}] Verified single-attendee recording ({first_id.strip()}) "
                    "-> constraining num_speakers=1"
                )

        upload_audio_to_r2_flag = input_data.get("upload_audio_to_r2", True)
        upload_transcript_to_r2_flag = input_data.get("upload_transcript_to_r2", True)
        return_inline = input_data.get("return_transcript_inline", True)

        logger.info("=" * 70)
        logger.info(f"FULL PIPELINE | meeting_id={meeting_id}")
        logger.info("=" * 70)

        yield {"status": "processing", "stage": "init", "meeting_id": meeting_id}

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)

            # ---- Stage 1: resolve audio ----
            logger.info(f"[{meeting_id}] STAGE 1: Resolve audio input")
            yield {"status": "processing", "stage": "downloading", "meeting_id": meeting_id}
            audio_info = _resolve_audio(input_data, workdir, meeting_id)
            audio_path = audio_info["audio_path"]
            came_from = audio_info["came_from"]
            audio_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            logger.info(f"[{meeting_id}] Audio ready: {audio_path} ({audio_size_mb:.1f} MB)")

            # ---- Stage 2: optional R2 upload of audio ----
            audio_presigned_url: Optional[str] = None
            if upload_audio_to_r2_flag:
                if not _r2_available():
                    logger.warning(f"[{meeting_id}] R2 envs not set; skipping audio upload")
                else:
                    try:
                        import r2_client
                        audio_r2_key = f"{R2_AUDIO_PREFIX}/{meeting_id}.wav"
                        logger.info(f"[{meeting_id}] STAGE 2: Uploading audio to R2 -> {audio_r2_key}")
                        yield {"status": "processing", "stage": "uploading_audio", "meeting_id": meeting_id}
                        r2_client.upload_file(audio_path, audio_r2_key, content_type="audio/wav")
                        audio_presigned_url = r2_client.generate_presigned_url(
                            audio_r2_key, expires_in=PRESIGN_EXPIRY_SECONDS
                        )
                    except Exception as e:
                        logger.warning(f"[{meeting_id}] R2 audio upload failed (non-fatal): {e}")
                        audio_r2_key = None

            # ---- Stage 3: transcribe (Whisper) ----
            logger.info(f"[{meeting_id}] STAGE 3: Whisper transcription")
            yield {"status": "processing", "stage": "transcribing", "meeting_id": meeting_id}
            stage3_start = time.time()

            # Transcription is the longest stage; emit heartbeats while it runs.
            # We run it in a thread so we can yield from the main generator thread.
            import threading
            whisper_result: Dict[str, Any] = {}
            whisper_exc: list = []

            def _run_whisper():
                try:
                    whisper_result["segments"] = transcription.transcribe_audio_chunked(audio_path)
                except Exception as e:
                    whisper_exc.append(e)

            t = threading.Thread(target=_run_whisper, daemon=True)
            t.start()
            while t.is_alive():
                t.join(timeout=HEARTBEAT_INTERVAL)
                if t.is_alive():
                    elapsed = time.time() - stage3_start
                    yield {
                        "status": "processing",
                        "stage": "transcribing",
                        "elapsed_s": round(elapsed, 0),
                        "meeting_id": meeting_id,
                    }

            if whisper_exc:
                raise whisper_exc[0]
            whisper_segments = whisper_result["segments"]
            logger.info(
                f"[{meeting_id}] Whisper done: {len(whisper_segments)} segments "
                f"in {time.time() - stage3_start:.1f}s"
            )

            # ---- Stage 4: diarize (Pyannote) ----
            logger.info(f"[{meeting_id}] STAGE 4: Pyannote diarization")
            yield {"status": "processing", "stage": "diarizing", "meeting_id": meeting_id}
            stage4_start = time.time()

            pyannote_result: Dict[str, Any] = {}
            pyannote_exc: list = []

            def _run_pyannote():
                try:
                    pyannote_result["segments"] = diarize.run_pyannote(
                        audio_path,
                        num_speakers=num_speakers,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                    )
                except Exception as e:
                    pyannote_exc.append(e)

            t2 = threading.Thread(target=_run_pyannote, daemon=True)
            t2.start()
            while t2.is_alive():
                t2.join(timeout=HEARTBEAT_INTERVAL)
                if t2.is_alive():
                    elapsed = time.time() - stage4_start
                    yield {
                        "status": "processing",
                        "stage": "diarizing",
                        "elapsed_s": round(elapsed, 0),
                        "meeting_id": meeting_id,
                    }

            if pyannote_exc:
                raise pyannote_exc[0]
            pyannote_segments = pyannote_result.get("segments")
            if not pyannote_segments:
                raise RuntimeError("Diarization failed: Pyannote produced zero speaker segments")
            logger.info(
                f"[{meeting_id}] Pyannote done: {len(pyannote_segments)} segments "
                f"in {time.time() - stage4_start:.1f}s"
            )

            # ---- Stage 5: merge ----
            logger.info(f"[{meeting_id}] STAGE 5: Merge transcript with speakers")
            yield {"status": "processing", "stage": "merging", "meeting_id": meeting_id}
            merged = diarize.assign_speakers_to_transcript(whisper_segments, pyannote_segments)

            # ---- Stage 6: biometric speaker mapping ----
            logger.info(f"[{meeting_id}] STAGE 6: Biometric speaker mapping (SpeechBrain & Explainable Engine)")
            yield {"status": "processing", "stage": "speaker_mapping", "meeting_id": meeting_id}
            speaker_mapping: Dict[str, str] = {}
            mapping_diagnostics: List[Dict[str, Any]] = []
            adaptive_thresholds: Dict[str, Any] = {}
            voice_mapping_status = "completed"
            voice_mapping_reason: Optional[str] = None

            enrolled_profiles = input_data.get("enrolled_profiles") or input_data.get("voice_profiles")
            expected_roster = input_data.get("expected_roster") or input_data.get("participants")
            meeting_metadata = input_data.get("metadata") or {}

            if not os.path.exists(SEED_DB_PATH) and not enrolled_profiles:
                voice_mapping_status = "skipped"
                voice_mapping_reason = f"Seed DB file not found at {SEED_DB_PATH} and no dynamic enrolled_profiles provided"
                logger.warning(
                    f"[{meeting_id}] Seed DB not found at {SEED_DB_PATH} and no enrolled_profiles in input; "
                    "skipping voice mapping. Speakers will remain as SPEAKER_XX."
                )
            else:
                try:
                    explainable_res = speaker_mapper_voice.map_speakers_explainable(
                        audio_path=audio_path,
                        diarized_segments=merged,
                        seed_db_path=SEED_DB_PATH,
                        enrolled_profiles=enrolled_profiles if isinstance(enrolled_profiles, list) else None,
                        expected_roster=expected_roster if isinstance(expected_roster, list) else None,
                        meeting_metadata=meeting_metadata if isinstance(meeting_metadata, dict) else None,
                    )
                    speaker_mapping = explainable_res.get("speaker_mapping", {})
                    mapping_diagnostics = explainable_res.get("diagnostics", [])
                    adaptive_thresholds = explainable_res.get("adaptive_thresholds", {})
                    if "consolidated_segments" in explainable_res:
                        merged = explainable_res["consolidated_segments"]

                    if not speaker_mapping:
                        voice_mapping_status = "unmapped"
                        voice_mapping_reason = "No speakers passed biometric confidence gates or database was empty"
                except Exception as e:
                    voice_mapping_status = "error"
                    voice_mapping_reason = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
                    logger.warning(f"[{meeting_id}] Voice mapping failed (non-fatal): {e}", exc_info=True)
                finally:
                    speaker_mapper_voice.unload()

            # ---- Stage 7: render transcript.txt ----
            logger.info(f"[{meeting_id}] STAGE 7: Render transcript.txt")
            transcript_text = transcript_writer.render(
                merged, speaker_mapping=speaker_mapping, concatenate_runs=True
            )
            transcript_local_path = workdir / "transcript.txt"
            transcript_writer.write_file(str(transcript_local_path), transcript_text)

            # ---- Stage 8: upload transcript.txt ----
            transcript_presigned_url: Optional[str] = None
            if upload_transcript_to_r2_flag:
                if not _r2_available():
                    logger.warning(f"[{meeting_id}] R2 envs not set; skipping transcript upload")
                else:
                    try:
                        import r2_client
                        transcript_r2_key = f"{R2_TRANSCRIPT_PREFIX}/{meeting_id}.txt"
                        logger.info(f"[{meeting_id}] STAGE 8: Uploading transcript -> {transcript_r2_key}")
                        yield {"status": "processing", "stage": "uploading_transcript", "meeting_id": meeting_id}
                        r2_client.upload_text(transcript_text, transcript_r2_key)
                        transcript_presigned_url = r2_client.generate_presigned_url(
                            transcript_r2_key, expires_in=PRESIGN_EXPIRY_SECONDS
                        )
                    except Exception as e:
                        logger.warning(f"[{meeting_id}] R2 transcript upload failed (non-fatal): {e}")
                        transcript_r2_key = None

            # ---- Done ----
            unique_speakers = sorted({s["speaker"] for s in merged if s.get("speaker")})
            audio_duration = max((s["end"] for s in merged), default=0.0)
            processing_time = time.time() - start_time

            logger.info("=" * 70)
            logger.info(
                f"[{meeting_id}] DONE | segments={len(merged)} "
                f"speakers={len(unique_speakers)} mapped={len(speaker_mapping)} "
                f"duration={audio_duration:.1f}s elapsed={processing_time:.1f}s"
            )
            logger.info("=" * 70)

            response: Dict[str, Any] = {
                "status": "success",
                "meeting_id": meeting_id,
                "input_kind": came_from,
                "speaker_mapping": speaker_mapping,
                "mapping_status": voice_mapping_status,
                "mapping_reason": voice_mapping_reason,
                "transcript_segments": merged,
                "transcription": merged,
                "metadata": {
                    "total_segments": len(merged),
                    "speakers_detected": len(unique_speakers),
                    "speakers_mapped": len(speaker_mapping),
                    "speaker_ids": unique_speakers,
                    "audio_duration_seconds": round(audio_duration, 2),
                    "processing_time_seconds": round(processing_time, 2),
                    "voice_mapping_status": voice_mapping_status,
                    "voice_mapping_reason": voice_mapping_reason,
                    "mapping_status": voice_mapping_status,
                    "mapping_reason": voice_mapping_reason,
                    "diagnostics": mapping_diagnostics,
                    "adaptive_thresholds": adaptive_thresholds,
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

            yield response

    except FileNotFoundError as e:
        yield _err(str(e), "FileNotFoundError")
    except ValueError as e:
        yield _err(str(e), "ValidationError")
    except requests.HTTPError as e:
        yield _err(f"Download failed: {e}", "DownloadError")
    except Exception as e:
        logger.exception("Pipeline failed")
        yield _err(str(e), type(e).__name__)


# =========================================
# RunPod entrypoint
# =========================================
if __name__ == "__main__":
    logger.info("Starting RunPod GPU handler (full pipeline)")
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": False})
