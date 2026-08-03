"""
Adaptive Threshold Engine for Biometric Voice Matching.

Replaces rigid, hardcoded cosine similarity constants with dynamic, acoustic-aware
gating thresholds computed from Signal-to-Noise Ratio (SNR), meeting duration,
speaker density, and intra-speaker vector variance.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, NamedTuple

logger = logging.getLogger(__name__)


class ThresholdConfig(NamedTuple):
    high_threshold: float
    high_gap: float
    mid_threshold: float
    mid_gap: float
    explanation: str


class AdaptiveThresholdEngine:
    """
    Computes context-aware gating thresholds based on audio quality metrics.
    """

    def __init__(
        self,
        base_high_threshold: float = 0.68,
        base_high_gap: float = 0.08,
        base_mid_threshold: float = 0.48,
        base_mid_gap: float = 0.15,
    ) -> None:
        self.base_high_threshold = base_high_threshold
        self.base_high_gap = base_high_gap
        self.base_mid_threshold = base_mid_threshold
        self.base_mid_gap = base_mid_gap

    def compute_thresholds(
        self,
        snr_db: float = 20.0,
        num_speakers: int = 2,
        duration_seconds: float = 600.0,
        embedding_variance: float = 0.0,
    ) -> ThresholdConfig:
        """
        Dynamically adjust cosine thresholds based on acoustic conditions.

        - Low SNR (<15dB): Relaxes score floor slightly (-0.03), increases gap requirement (+0.03).
        - High speaker density (>5 speakers): Increases score floor (+0.04) to avoid impostor matches.
        - Long duration (>60 mins): Slightly relaxes gap (-0.02) to handle natural voice fatigue.
        """
        high_thresh = self.base_high_threshold
        high_gap = self.base_high_gap
        mid_thresh = self.base_mid_threshold
        mid_gap = self.base_mid_gap

        adjustments = []

        # SNR Adjustment
        if snr_db < 15.0:
            snr_penalty = max(-0.05, (snr_db - 15.0) * 0.005)
            high_thresh += snr_penalty
            mid_thresh += snr_penalty
            high_gap += 0.02
            adjustments.append(f"Noisy audio (SNR={snr_db:.1f}dB): threshold {snr_penalty:+.3f}")

        # Speaker Density Adjustment
        if num_speakers >= 5:
            density_boost = min(0.06, (num_speakers - 4) * 0.015)
            high_thresh += density_boost
            mid_thresh += density_boost
            adjustments.append(f"High speaker count (N={num_speakers}): threshold {density_boost:+.3f}")

        # Duration Adjustment
        if duration_seconds > 3600.0:  # > 1 hour
            gap_relax = -0.02
            high_gap = max(0.04, high_gap + gap_relax)
            mid_gap = max(0.08, mid_gap + gap_relax)
            adjustments.append(f"Long meeting ({duration_seconds/60:.0f}m): gap requirement {gap_relax:+.3f}")

        # Variance Adjustment
        if embedding_variance > 0.1:
            high_thresh = max(0.55, high_thresh - 0.02)
            adjustments.append(f"High embedding variance ({embedding_variance:.2f}): relaxed floor -0.02")

        # Clamp bounds
        high_thresh = max(0.50, min(0.85, high_thresh))
        mid_thresh = max(0.40, min(0.70, mid_thresh))
        high_gap = max(0.04, min(0.20, high_gap))
        mid_gap = max(0.08, min(0.30, mid_gap))

        explanation_str = "; ".join(adjustments) if adjustments else "Standard acoustic conditions (baseline thresholds applied)"

        logger.info(
            f"AdaptiveThresholds: Gate A [score>={high_thresh:.3f}, gap>={high_gap:.3f}] | "
            f"Gate B [score>={mid_thresh:.3f}, gap>={mid_gap:.3f}] ({explanation_str})"
        )

        return ThresholdConfig(
            high_threshold=round(high_thresh, 4),
            high_gap=round(high_gap, 4),
            mid_threshold=round(mid_thresh, 4),
            mid_gap=round(mid_gap, 4),
            explanation=explanation_str,
        )
