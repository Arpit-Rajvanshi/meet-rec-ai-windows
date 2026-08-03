"""
Comprehensive Unit & Integration Test Suite for Production Speaker Identification Pipeline.

Tests:
1. SpeakerFragmentMerger (agglomerative turn consolidation)
2. AdaptiveThresholdEngine (SNR & speaker density dynamic thresholding)
3. ConfidenceEngine (continuous metrics, evidence generation, candidate distributions)
4. UnknownSpeakerResolver (4-tier resolution cascade)
5. Benchmark Suite (DER, JER, F1 metrics calculation)
"""

import os
import sys
import unittest
import math

try:
    import torch
except ImportError:
    torch = None

sys.path.insert(0, os.path.dirname(__file__))

import speaker_mapper_voice
from adaptive_threshold import AdaptiveThresholdEngine
from confidence_engine import ConfidenceEngine
from speaker_merger import SpeakerFragmentMerger
from unknown_resolver import UnknownSpeakerResolver
import benchmark_suite


def _create_vector(lst):
    if torch is not None:
        return torch.nn.functional.normalize(torch.tensor(lst), p=2, dim=0)
    norm = math.sqrt(sum(x * x for x in lst)) + 1e-8
    return [x / norm for x in lst]


class TestSpeakerIdentificationPipeline(unittest.TestCase):

    def test_speaker_fragment_merger(self):
        """Verify SpeakerFragmentMerger consolidates over-segmented speaker IDs."""
        merger = SpeakerFragmentMerger(merge_threshold=0.75)

        # Create 2 vectors for SPEAKER_00 and SPEAKER_01 that are nearly identical (sim ~0.98)
        v0 = _create_vector([1.0, 0.1] + [0.0] * 190)
        v1 = _create_vector([0.98, 0.12] + [0.0] * 190)

        speaker_vectors = {"SPEAKER_00": v0, "SPEAKER_01": v1}
        segments = [
            {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00", "text": "Hello"},
            {"start": 4.5, "end": 8.0, "speaker": "SPEAKER_01", "text": "Hi there"},
        ]

        updated_segments, consolidated_vecs, alias_map = merger.merge_fragments(
            segments, speaker_vectors
        )

        # Should consolidate SPEAKER_01 into SPEAKER_00
        self.assertEqual(len(consolidated_vecs), 1)
        self.assertIn("SPEAKER_00", consolidated_vecs)
        self.assertEqual(alias_map["SPEAKER_01"], "SPEAKER_00")
        self.assertEqual(updated_segments[1]["speaker"], "SPEAKER_00")

    def test_adaptive_threshold_engine(self):
        """Verify AdaptiveThresholdEngine adjusts thresholds based on acoustic metrics."""
        engine = AdaptiveThresholdEngine(base_high_threshold=0.68, base_high_gap=0.08)

        # Clean audio baseline
        clean = engine.compute_thresholds(snr_db=25.0, num_speakers=2, duration_seconds=300.0)
        self.assertEqual(clean.high_threshold, 0.68)

        # Noisy audio (SNR < 15dB) -> Should adjust threshold slightly
        noisy = engine.compute_thresholds(snr_db=10.0, num_speakers=2, duration_seconds=300.0)
        self.assertLess(noisy.high_threshold, clean.high_threshold)

        # High speaker count (>5) -> Should raise threshold to prevent impostor match
        crowded = engine.compute_thresholds(snr_db=25.0, num_speakers=8, duration_seconds=300.0)
        self.assertGreater(crowded.high_threshold, clean.high_threshold)

    def test_confidence_engine(self):
        """Verify ConfidenceEngine generates structured evidence and confidence metrics."""
        engine = ConfidenceEngine()
        top3 = [
            {"name": "Arpit Rajvanshi", "score": 0.92, "employee_id": "ATG-001"},
            {"name": "Kartik", "score": 0.45, "employee_id": "ATG-002"},
        ]

        diag = engine.evaluate_match(
            speaker_id="SPEAKER_00",
            best_name="Arpit Rajvanshi",
            best_score=0.92,
            gap=0.47,
            top_candidates=top3,
            gate_passed="Gate A",
            high_thresh=0.68,
            high_gap=0.08,
            mid_thresh=0.48,
            mid_gap=0.15,
            snr_db=22.0,
            speaking_duration_sec=45.0,
        )

        self.assertEqual(diag["matched_identity"], "Arpit Rajvanshi")
        self.assertTrue(diag["is_matched"])
        self.assertEqual(diag["confidence_level"], "HIGH")
        self.assertGreater(diag["confidence_score"], 0.80)
        self.assertIn("Gate A", diag["explainability_reason"])
        self.assertIn("evidence", diag)
        self.assertEqual(diag["evidence"]["snr_db"], 22.0)

    def test_unknown_speaker_resolver(self):
        """Verify UnknownSpeakerResolver executes 4-tier fallback cascade."""
        resolver = UnknownSpeakerResolver()
        top_candidates = [
            {"name": "Ketan Patil", "score": 0.44, "employee_id": "ATG-003"}
        ]

        # Tier 1 Roster Match test
        res_t1 = resolver.resolve_unmatched_speaker(
            speaker_id="SPEAKER_02",
            speaker_vec=None,
            top_candidates=top_candidates,
            expected_roster=["Ketan Patil", "Arpit Rajvanshi"],
            roster_threshold=0.40,
        )
        self.assertEqual(res_t1["resolved_name"], "Ketan Patil")
        self.assertEqual(res_t1["resolution_tier"], "TIER_1_EXPECTED_ROSTER")

        # Tier 4 Safe Unknown test
        res_t4 = resolver.resolve_unmatched_speaker(
            speaker_id="SPEAKER_05",
            speaker_vec=None,
            top_candidates=[],
            expected_roster=[],
        )
        self.assertEqual(res_t4["resolved_name"], "Unknown Speaker 05")
        self.assertEqual(res_t4["resolution_tier"], "TIER_4_SAFE_UNKNOWN")

    def test_benchmark_suite_metrics(self):
        """Verify benchmark metric calculation (DER, JER, Accuracy, F1)."""
        ref_turns = [
            {"start": 0.0, "end": 10.0, "speaker": "Alice"},
            {"start": 10.0, "end": 20.0, "speaker": "Bob"},
        ]
        hyp_turns = [
            {"start": 0.0, "end": 10.0, "speaker": "Alice"},
            {"start": 10.0, "end": 20.0, "speaker": "Bob"},
        ]

        der = benchmark_suite.calculate_der(ref_turns, hyp_turns)
        acc = benchmark_suite.calculate_accuracy_metrics(ref_turns, hyp_turns)

        self.assertEqual(der, 0.0)
        self.assertEqual(acc["accuracy"], 1.0)
        self.assertEqual(acc["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
