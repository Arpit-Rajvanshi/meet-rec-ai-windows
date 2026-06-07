"""
Render a diarized, speaker-mapped transcript into a plain transcript.txt.

Output format (IRIS-compatible):
    [HH:MM:SS] speaker_or_name: text
    [HH:MM:SS] speaker_or_name: text
    ...

Consecutive segments from the same speaker are concatenated to reduce noise
from Whisper's chunking. Empty/whitespace-only segments are skipped.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


def _fmt_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS, zero-padded."""
    if seconds is None or seconds < 0:
        seconds = 0
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _resolve_label(speaker_id: str, mapping: Dict[str, str]) -> str:
    """Return the real name if matched, otherwise the SPEAKER_XX id."""
    if not speaker_id:
        return "UNKNOWN_SPEAKER"
    return mapping.get(speaker_id, speaker_id)


def _concatenate_runs(
    segments: List[Dict], mapping: Dict[str, str]
) -> List[Dict]:
    """
    Merge consecutive segments that share the same resolved speaker label.
    Preserves first segment's start and last segment's end.
    """
    if not segments:
        return []

    merged: List[Dict] = []
    current = None
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        label = _resolve_label(seg.get("speaker", ""), mapping)
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))

        if current and current["label"] == label:
            current["end"] = end
            current["text"] = f"{current['text']} {text}".strip()
        else:
            if current:
                merged.append(current)
            current = {"label": label, "start": start, "end": end, "text": text}

    if current:
        merged.append(current)
    return merged


def render(
    segments: List[Dict],
    speaker_mapping: Optional[Dict[str, str]] = None,
    concatenate_runs: bool = True,
) -> str:
    """
    Render diarized segments into a transcript.txt string.

    Args:
        segments: list of dicts with keys {start, end, speaker, text}
        speaker_mapping: optional {SPEAKER_XX: real_name}; unmatched speakers keep their ID
        concatenate_runs: merge consecutive same-speaker segments (default True)

    Returns:
        UTF-8 string ready to write to disk or upload.
    """
    mapping = speaker_mapping or {}

    if concatenate_runs:
        rows = _concatenate_runs(segments, mapping)
    else:
        rows = []
        for seg in segments:
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            rows.append(
                {
                    "label": _resolve_label(seg.get("speaker", ""), mapping),
                    "start": float(seg.get("start", 0.0)),
                    "end": float(seg.get("end", 0.0)),
                    "text": text,
                }
            )

    lines = [f"[{_fmt_timestamp(r['start'])}] {r['label']}: {r['text']}" for r in rows]
    return "\n".join(lines) + ("\n" if lines else "")


def write_file(path: str, text: str) -> str:
    """Write transcript text to disk in UTF-8."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path
