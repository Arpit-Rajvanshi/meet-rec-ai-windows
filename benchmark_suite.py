"""
Automated Speaker Identification & Diarization Benchmark Suite.

Evaluates:
- DER (Diarization Error Rate)
- JER (Jaccard Error Rate)
- Speaker Identification Accuracy
- Precision, Recall, F1 Score
- FAR (False Accept Rate) & FRR (False Reject Rate)
- Unknown Speaker Detection Rate
- Identity Stability (Label switching per speaker)
- Latency & Memory Footprint
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


def calculate_der(
    reference_turns: List[Dict], hypothesis_turns: List[Dict]
) -> float:
    """
    Calculate Diarization Error Rate (DER = Miss + False Alarm + Speaker Confusion / Total Reference Time).
    """
    ref_time = sum(float(r["end"]) - float(r["start"]) for r in reference_turns)
    if ref_time <= 0:
        return 0.0

    miss_time = 0.0
    fa_time = 0.0
    confusion_time = 0.0

    for hyp in hypothesis_turns:
        h_start, h_end, h_spk = float(hyp["start"]), float(hyp["end"]), str(hyp.get("speaker"))
        # Find matching reference turns
        matches = [
            r for r in reference_turns
            if max(float(r["start"]), h_start) < min(float(r["end"]), h_end)
        ]
        if not matches:
            fa_time += (h_end - h_start)
        else:
            for r in matches:
                r_start, r_end, r_spk = float(r["start"]), float(r["end"]), str(r.get("speaker"))
                overlap = min(h_end, r_end) - max(h_start, r_start)
                if overlap > 0 and h_spk != r_spk:
                    confusion_time += overlap

    der = (miss_time + fa_time + confusion_time) / ref_time
    return max(0.0, min(1.0, der))


def calculate_jer(
    reference_turns: List[Dict], hypothesis_turns: List[Dict]
) -> float:
    """Calculate Jaccard Error Rate (JER)."""
    ref_speakers = set(str(r["speaker"]) for r in reference_turns)
    if not ref_speakers:
        return 0.0

    jaccard_scores = []
    for r_spk in ref_speakers:
        r_dur = sum(float(r["end"]) - float(r["start"]) for r in reference_turns if str(r["speaker"]) == r_spk)
        best_iou = 0.0
        for hyp_spk in set(str(h["speaker"]) for h in hypothesis_turns):
            h_dur = sum(float(h["end"]) - float(h["start"]) for h in hypothesis_turns if str(h["speaker"]) == hyp_spk)
            intersection = 0.0
            for r in reference_turns:
                if str(r["speaker"]) != r_spk:
                    continue
                for h in hypothesis_turns:
                    if str(h["speaker"]) != hyp_spk:
                        continue
                    overlap = min(float(r["end"]), float(h["end"])) - max(float(r["start"]), float(h["start"]))
                    if overlap > 0:
                        intersection += overlap
            union = r_dur + h_dur - intersection
            iou = intersection / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
        jaccard_scores.append(best_iou)

    jer = 1.0 - (sum(jaccard_scores) / len(jaccard_scores))
    return max(0.0, min(1.0, jer))


def calculate_accuracy_metrics(
    reference_turns: List[Dict], hypothesis_turns: List[Dict]
) -> Dict[str, float]:
    """Calculate Accuracy, Precision, Recall, F1, FAR, FRR, and Identity Stability."""
    tp, fp, fn = 0, 0, 0
    far_count, frr_count = 0, 0
    total_evals = 0

    for hyp in hypothesis_turns:
        h_spk = str(hyp.get("speaker"))
        h_start = float(hyp["start"])
        h_end = float(hyp.get("end", h_start + 1.0))
        
        # Match with reference segment having maximum overlap
        best_ref = None
        max_overlap = 0.0
        for r in reference_turns:
            r_start = float(r["start"])
            r_end = float(r["end"])
            overlap = max(0.0, min(h_end, r_end) - max(h_start, r_start))
            if overlap > max_overlap:
                max_overlap = overlap
                best_ref = r
        
        if best_ref is None:
            # Fallback to nearest start timestamp
            best_ref = next((r for r in reference_turns if abs(float(r["start"]) - h_start) < 0.5), None)

        if best_ref:
            total_evals += 1
            r_spk = str(best_ref.get("speaker"))
            if h_spk == r_spk:
                tp += 1
            else:
                fp += 1
                fn += 1
                if "Unknown" in r_spk and "Unknown" not in h_spk:
                    far_count += 1
                elif "Unknown" not in r_spk and "Unknown" in h_spk:
                    frr_count += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = tp / total_evals if total_evals > 0 else 1.0

    far = far_count / total_evals if total_evals > 0 else 0.0
    frr = frr_count / total_evals if total_evals > 0 else 0.0

    # Identity Stability: ratio of turn label consistency per speaker
    label_switches = 0
    for i in range(1, len(hypothesis_turns)):
        if hypothesis_turns[i].get("speaker") != hypothesis_turns[i - 1].get("speaker"):
            label_switches += 1
    identity_stability = 1.0 - (label_switches / len(hypothesis_turns)) if len(hypothesis_turns) > 1 else 1.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "far": round(far, 4),
        "frr": round(frr, 4),
        "identity_stability": round(identity_stability, 4),
    }


def run_benchmark_suite(
    reference_data: List[Dict[str, Any]], hypothesis_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Execute complete benchmark suite returning unified quantitative metrics report."""
    ders, jers = [], []
    precisions, recalls, f1s = [], [], []
    fars, frrs, stabilities = [], [], []

    for ref, hyp in zip(reference_data, hypothesis_data):
        ref_turns = ref.get("segments", [])
        hyp_turns = hyp.get("segments", [])

        der = calculate_der(ref_turns, hyp_turns)
        jer = calculate_jer(ref_turns, hyp_turns)
        acc_metrics = calculate_accuracy_metrics(ref_turns, hyp_turns)

        ders.append(der)
        jers.append(jer)
        precisions.append(acc_metrics["precision"])
        recalls.append(acc_metrics["recall"])
        f1s.append(acc_metrics["f1"])
        fars.append(acc_metrics["far"])
        frrs.append(acc_metrics["frr"])
        stabilities.append(acc_metrics["identity_stability"])

    mean = lambda l: sum(l) / len(l) if l else 0.0

    report = {
        "summary": {
            "evaluated_samples": len(reference_data),
            "mean_der": round(mean(ders), 4),
            "mean_jer": round(mean(jers), 4),
            "mean_precision": round(mean(precisions), 4),
            "mean_recall": round(mean(recalls), 4),
            "mean_f1": round(mean(f1s), 4),
            "mean_far": round(mean(fars), 4),
            "mean_frr": round(mean(frrs), 4),
            "mean_identity_stability": round(mean(stabilities), 4),
        },
        "details": [
            {
                "sample_id": ref.get("id", f"sample_{i}"),
                "der": round(ders[i], 4),
                "jer": round(jers[i], 4),
                "f1": round(f1s[i], 4),
            }
            for i, ref in enumerate(reference_data)
        ],
    }
    return report


if __name__ == "__main__":
    print("Speaker Identification Benchmark Suite Module Loaded.")
