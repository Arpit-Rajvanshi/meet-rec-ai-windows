"""
Speaker Fragment Merger Engine.

Solves PyAnnote speaker over-segmentation by applying Agglomerative Hierarchical
Clustering (AHC) across speaker vector embeddings with temporal continuity weighting.

Combines acoustic similarity (cosine distance) with temporal turn proximity to merge
fragmented speaker IDs (e.g., SPEAKER_01 and SPEAKER_02 belonging to the same physical person)
prior to biometric enrollment matching.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple, Set

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)

DEFAULT_CLUSTER_MERGE_THRESHOLD = 0.78  # Cosine similarity floor for merging fragments
MAX_TEMPORAL_GAP_SECONDS = 5.0  # Max gap between speaker turns to apply temporal boost
TEMPORAL_BOOST_FACTOR = 0.05  # Cosine similarity bonus for temporally adjacent turns


def _cosine_sim(v1: Any, v2: Any) -> float:
    if torch is not None and isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        return float(cos(v1, v2).item())
    
    l1 = v1.tolist() if hasattr(v1, "tolist") else list(v1)
    l2 = v2.tolist() if hasattr(v2, "tolist") else list(v2)
    dot = sum(a * b for a, b in zip(l1, l2))
    norm1 = math.sqrt(sum(a * a for a in l1)) + 1e-8
    norm2 = math.sqrt(sum(b * b for b in l2)) + 1e-8
    return dot / (norm1 * norm2)



class SpeakerFragmentMerger:
    """
    Agglomerative speaker segment merger using embedding distance and temporal turn context.
    """

    def __init__(
        self,
        merge_threshold: float = DEFAULT_CLUSTER_MERGE_THRESHOLD,
        temporal_gap_sec: float = MAX_TEMPORAL_GAP_SECONDS,
        temporal_boost: float = TEMPORAL_BOOST_FACTOR,
    ) -> None:
        self.merge_threshold = merge_threshold
        self.temporal_gap_sec = temporal_gap_sec
        self.temporal_boost = temporal_boost

    def _compute_turn_gaps(
        self, segments: List[Dict], spk1: str, spk2: str
    ) -> float:
        """
        Find minimum temporal gap (in seconds) between any turn of spk1 and any turn of spk2.
        """
        spk1_turns = [(float(s["start"]), float(s["end"])) for s in segments if s.get("speaker") == spk1]
        spk2_turns = [(float(s["start"]), float(s["end"])) for s in segments if s.get("speaker") == spk2]

        if not spk1_turns or not spk2_turns:
            return float("inf")

        min_gap = float("inf")
        for s1, e1 in spk1_turns:
            for s2, e2 in spk2_turns:
                if e1 <= s2:
                    gap = s2 - e1
                elif e2 <= s1:
                    gap = s1 - e2
                else:
                    gap = 0.0  # Overlapping turns
                if gap < min_gap:
                    min_gap = gap
                    if min_gap == 0.0:
                        break
            if min_gap == 0.0:
                break
        return min_gap

    def merge_fragments(
        self,
        segments: List[Dict],
        speaker_vectors: Dict[str, torch.Tensor],
    ) -> Tuple[List[Dict], Dict[str, torch.Tensor], Dict[str, str]]:
        """
        Merge over-segmented speaker IDs in `segments` and `speaker_vectors`.

        Returns:
            Tuple of:
              - Updated segments list with canonical speaker labels.
              - Consolidated speaker_vectors dictionary mapping canonical speaker ID -> centroid vector.
              - Alias map mapping original speaker_id -> canonical speaker_id.
        """
        speakers = sorted(list(speaker_vectors.keys()))
        if len(speakers) <= 1:
            return segments, speaker_vectors, {s: s for s in speakers}

        canonical_map: Dict[str, str] = {s: s for s in speakers}

        # Build pairwise similarity matrix
        sim_matrix: Dict[Tuple[str, str], float] = {}
        for i in range(len(speakers)):
            for j in range(i + 1, len(speakers)):
                spk1, spk2 = speakers[i], speakers[j]
                v1, v2 = speaker_vectors[spk1], speaker_vectors[spk2]
                base_sim = _cosine_sim(v1, v2)

                # Calculate temporal turn proximity boost
                gap = self._compute_turn_gaps(segments, spk1, spk2)
                boost = 0.0
                if gap <= self.temporal_gap_sec:
                    boost = self.temporal_boost * (1.0 - (gap / self.temporal_gap_sec))

                effective_sim = min(1.0, base_sim + boost)
                sim_matrix[(spk1, spk2)] = effective_sim
                sim_matrix[(spk2, spk1)] = effective_sim

        # Iterative Agglomerative Merging (Single-pass greedy merge)
        merged = True
        while merged:
            merged = False
            best_pair: Optional[Tuple[str, str]] = None
            best_score = self.merge_threshold

            active_speakers = sorted(list(set(canonical_map.values())))
            for i in range(len(active_speakers)):
                for j in range(i + 1, len(active_speakers)):
                    s1, s2 = active_speakers[i], active_speakers[j]
                    score = sim_matrix.get((s1, s2), 0.0)
                    if score >= best_score:
                        best_score = score
                        best_pair = (s1, s2)

            if best_pair:
                s_keep, s_drop = best_pair[0], best_pair[1]
                logger.info(
                    f"SpeakerMerger: Consolidating fragment {s_drop} -> {s_keep} "
                    f"(effective similarity: {best_score:.3f} >= threshold {self.merge_threshold})"
                )

                # Re-map alias pointers
                for orig, curr in list(canonical_map.items()):
                    if curr == s_drop:
                        canonical_map[orig] = s_keep

                # Recompute combined centroid for s_keep
                v_keep = speaker_vectors[s_keep]
                v_drop = speaker_vectors[s_drop]
                if torch is not None and isinstance(v_keep, torch.Tensor):
                    combined_vec = (v_keep + v_drop) / 2.0
                    combined_vec = torch.nn.functional.normalize(combined_vec, p=2, dim=0)
                else:
                    l_k = v_keep.tolist() if hasattr(v_keep, "tolist") else list(v_keep)
                    l_d = v_drop.tolist() if hasattr(v_drop, "tolist") else list(v_drop)
                    mean_l = [(a + b) / 2.0 for a, b in zip(l_k, l_d)]
                    norm = math.sqrt(sum(x * x for x in mean_l)) + 1e-8
                    combined_vec = [x / norm for x in mean_l]

                speaker_vectors[s_keep] = combined_vec
                del speaker_vectors[s_drop]

                # Update sim matrix for s_keep vs remaining active speakers
                remaining = [s for s in active_speakers if s not in (s_keep, s_drop)]
                for r in remaining:
                    v_r = speaker_vectors[r]
                    new_sim = _cosine_sim(combined_vec, v_r)
                    sim_matrix[(s_keep, r)] = new_sim
                    sim_matrix[(r, s_keep)] = new_sim

                merged = True

        # Update segment speaker labels
        updated_segments: List[Dict] = []
        for seg in segments:
            seg_copy = dict(seg)
            curr_spk = seg_copy.get("speaker")
            if curr_spk in canonical_map:
                seg_copy["speaker"] = canonical_map[curr_spk]
            updated_segments.append(seg_copy)

        logger.info(
            f"SpeakerMerger: Reduced {len(speakers)} raw speaker IDs to "
            f"{len(speaker_vectors)} consolidated speaker cluster(s)"
        )
        return updated_segments, speaker_vectors, canonical_map
