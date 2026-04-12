"""Evaluation metrics (thesis §3.6).

Drift detector metrics:
    DetectorResult   — dataclass accumulating alarms per detector
    detection_latency, false_positive_rate, true_positive_rate

Anomaly detection:
    auroc_at_severity — AUROC per severity level (degradation curve)

XAI:
    ADA score lives in xai/gradcam.py (compute_ada).
    SHAP KS test lives in xai/shap_analysis.py (ks_test).
    LIME overlap lives in xai/lime_analysis.py (overlap_coefficient).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass
class DetectorResult:
    """Accumulates alarm events for one detector over the full stream."""
    name: str
    # Sample indices at which the detector fired
    alarm_indices: list[int] = field(default_factory=list)
    # For each alarm: was it a true positive? (set during evaluation)
    is_true_positive: list[bool] = field(default_factory=list)
    # Samples from drift onset to alarm (one per detected drift event)
    latencies: list[int] = field(default_factory=list)

    def record_alarm(self, sample_idx: int, is_tp: bool, latency: int | None = None):
        self.alarm_indices.append(sample_idx)
        self.is_true_positive.append(is_tp)
        if latency is not None:
            self.latencies.append(latency)

    @property
    def n_alarms(self) -> int:
        return len(self.alarm_indices)

    @property
    def false_positive_rate(self) -> float:
        if self.n_alarms == 0:
            return 0.0
        return sum(not tp for tp in self.is_true_positive) / self.n_alarms

    @property
    def true_positive_rate(self) -> float:
        if self.n_alarms == 0:
            return 0.0
        return sum(self.is_true_positive) / self.n_alarms

    @property
    def mean_detection_latency(self) -> float | None:
        return float(np.mean(self.latencies)) if self.latencies else None


def classify_alarms(
    alarm_indices: list[int],
    drift_onsets: list[int],
    tolerance: int,
) -> list[bool]:
    """Label each alarm as TP (True Positive) or FP.

    An alarm is a TP if it falls within [onset, onset + tolerance) for any
    known drift onset. Each onset can be claimed by at most one alarm.
    """
    claimed = set()
    labels = []
    for idx in alarm_indices:
        matched = False
        for onset in drift_onsets:
            if onset not in claimed and onset <= idx < onset + tolerance:
                claimed.add(onset)
                matched = True
                break
        labels.append(matched)
    return labels


def auroc_at_severity(
    labels_by_sev: dict[int, list[int]],
    scores_by_sev: dict[int, list[float]],
) -> dict[int, float]:
    """Compute AUROC separately for each severity level.

    scores: defective-class confidence (higher = more defective).
    Returns {severity: auroc} degradation curve.
    """
    result = {}
    for sev in sorted(labels_by_sev):
        y_true = labels_by_sev[sev]
        y_score = scores_by_sev[sev]
        try:
            result[sev] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            result[sev] = float("nan")
    return result


def print_detector_summary(results: list[DetectorResult]) -> None:
    header = f"{'Detector':<16} {'Alarms':>7} {'TPR':>8} {'FPR':>8} {'Mean latency':>14}"
    print(header)
    print("-" * len(header))
    for r in results:
        lat = f"{r.mean_detection_latency:.1f}" if r.mean_detection_latency is not None else "—"
        print(f"{r.name:<16} {r.n_alarms:>7} {r.true_positive_rate:>8.3f} "
              f"{r.false_positive_rate:>8.3f} {lat:>14}")
