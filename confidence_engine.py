"""
Explainable Biometric Confidence Engine.

Generates continuous confidence metrics, multi-candidate probability distributions,
detailed evidence objects, and human-readable diagnostic text for every speaker identification match.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


class ConfidenceEngine:
    """
    Computes explainable confidence metrics for biometric speaker identification.
    """

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def evaluate_match(
        self,
        speaker_id: str,
        best_name: str,
        best_score: float,
        gap: float,
        top_candidates: List[Dict[str, Any]],
        gate_passed: Optional[str],
        high_thresh: float,
        high_gap: float,
        mid_thresh: float,
        mid_gap: float,
        snr_db: float = 20.0,
        speaking_duration_sec: float = 0.0,
        sample_count: int = 1,
    ) -> Dict[str, Any]:
        """
        Build an explainable diagnostic match result.
        """
        # Continuous confidence calculation
        score_diff = best_score - mid_thresh
        gap_diff = gap - high_gap
        raw_confidence = self._sigmoid(8.0 * score_diff + 4.0 * gap_diff)

        # Categorize confidence level
        if gate_passed == "Gate A" or (gate_passed and raw_confidence >= 0.85):
            confidence_level = "HIGH"
        elif gate_passed == "Gate B" or (gate_passed and raw_confidence >= 0.60):
            confidence_level = "MEDIUM"
        elif raw_confidence >= 0.40:
            confidence_level = "LOW"
        else:
            confidence_level = "REJECTED"

        # Format alternative candidates (excluding top match)
        alternatives = []
        for cand in top_candidates:
            if cand.get("name") != best_name:
                alternatives.append({
                    "name": cand.get("name"),
                    "employee_id": cand.get("employee_id"),
                    "similarity": round(float(cand.get("score", 0.0)), 3),
                })

        runner_up_str = f" to runner-up {alternatives[0]['name']}" if alternatives else ""

        if gate_passed:
            reason = (
                f"Matched '{best_name}' via {gate_passed} with similarity {best_score:.3f} "
                f"and separation gap {gap:.3f}{runner_up_str}."
            )
        else:
            reason = (
                f"Rejected match for '{best_name}' (score {best_score:.3f}, gap {gap:.3f}) "
                f"which failed required dynamic thresholds (High: {high_thresh:.2f}, Mid: {mid_thresh:.2f})."
            )

        evidence = {
            "speaker_id": speaker_id,
            "snr_db": round(snr_db, 1),
            "speaking_duration_sec": round(speaking_duration_sec, 2),
            "analyzed_samples_count": sample_count,
            "gap_to_runner_up": round(gap, 3),
            "gate_passed": gate_passed or "NONE",
            "thresholds_used": {
                "high_threshold": high_thresh,
                "high_gap": high_gap,
                "mid_threshold": mid_thresh,
                "mid_gap": mid_gap,
            },
        }

        return {
            "speaker_id": speaker_id,
            "matched_identity": best_name if gate_passed else speaker_id,
            "is_matched": bool(gate_passed),
            "similarity_score": round(best_score, 4),
            "confidence_score": round(raw_confidence, 4),
            "confidence_level": confidence_level,
            "gate_passed": gate_passed,
            "evidence": evidence,
            "alternative_candidates": alternatives[:3],
            "explainability_reason": reason,
        }
