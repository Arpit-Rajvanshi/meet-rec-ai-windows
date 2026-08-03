"""
Tiered Fallback Unknown Speaker Resolver Cascade.

Prevents silent failure or hallucination when an acoustic speaker fails primary biometric matching.
Executes a 4-tier fallback cascade:
  Tier 1: Expected Roster Match (relaxed threshold for expected meeting attendees)
  Tier 2: Full Profile Vector Search
  Tier 3: Contextual Metadata Inference (e.g. single participant meeting)
  Tier 4: Structured Safe Unknown Tag ("Unknown Speaker XX")
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)


class UnknownSpeakerResolver:
    """
    Tiered fallback cascade to safely handle unmatched acoustic speaker clusters.
    """

    def resolve_unmatched_speaker(
        self,
        speaker_id: str,
        speaker_vec: Optional[torch.Tensor],
        top_candidates: List[Dict[str, Any]],
        expected_roster: Optional[List[str]] = None,
        meeting_metadata: Optional[Dict[str, Any]] = None,
        roster_threshold: float = 0.45,
    ) -> Dict[str, Any]:
        """
        Execute resolution cascade for an unmatched speaker.
        """
        # Tier 1: Expected Roster Match
        if expected_roster and top_candidates:
            roster_set = {r.strip().lower() for r in expected_roster if r}
            for cand in top_candidates:
                cand_name = cand.get("name", "")
                cand_score = float(cand.get("score", 0.0))
                if cand_name.lower() in roster_set and cand_score >= roster_threshold:
                    logger.info(
                        f"UnknownResolver: Tier 1 (Roster Match) resolved {speaker_id} -> "
                        f"'{cand_name}' (score {cand_score:.3f} >= roster threshold {roster_threshold})"
                    )
                    return {
                        "resolved_name": cand_name,
                        "resolution_tier": "TIER_1_EXPECTED_ROSTER",
                        "confidence_score": round(cand_score, 3),
                        "explanation": f"Matched expected attendee '{cand_name}' via relaxed roster gate.",
                    }

        # Tier 2: Top Vector Candidate (if close to mid threshold)
        if top_candidates and len(top_candidates) > 0:
            top = top_candidates[0]
            top_name = top.get("name")
            top_score = float(top.get("score", 0.0))
            if top_score >= 0.42:
                logger.info(
                    f"UnknownResolver: Tier 2 (Vector Soft-Match) resolved {speaker_id} -> "
                    f"'{top_name}' (soft score {top_score:.3f})"
                )
                return {
                    "resolved_name": top_name,
                    "resolution_tier": "TIER_2_VECTOR_SOFT_MATCH",
                    "confidence_score": round(top_score, 3),
                    "explanation": f"Soft vector match to '{top_name}' with score {top_score:.3f}.",
                }

        # Tier 3: Contextual Metadata (Single participant meeting)
        if meeting_metadata and isinstance(meeting_metadata, dict):
            participants = meeting_metadata.get("expected_participants") or []
            if len(participants) == 1 and isinstance(participants[0], str):
                single_name = participants[0]
                logger.info(
                    f"UnknownResolver: Tier 3 (Single Speaker Context) resolved {speaker_id} -> '{single_name}'"
                )
                return {
                    "resolved_name": single_name,
                    "resolution_tier": "TIER_3_CONTEXTUAL_METADATA",
                    "confidence_score": 0.60,
                    "explanation": f"Inferred '{single_name}' from single-participant meeting metadata.",
                }

        # Tier 4: Safe Unknown Tag
        clean_num = speaker_id.replace("SPEAKER_", "").replace("SPEAKER", "")
        fallback_name = f"Unknown Speaker {clean_num.strip()}"
        logger.info(
            f"UnknownResolver: Tier 4 (Safe Unknown) assigning '{fallback_name}' to {speaker_id}"
        )
        return {
            "resolved_name": fallback_name,
            "resolution_tier": "TIER_4_SAFE_UNKNOWN",
            "confidence_score": 0.0,
            "explanation": "No biometric or roster match passed thresholds. Assigned structured unknown tag.",
        }
