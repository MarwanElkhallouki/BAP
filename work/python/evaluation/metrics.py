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
    # Severity recorded at alarm time (optional; kept for reporting fallback)
    alarm_severities: list[int | None] = field(default_factory=list)
    # Optional stream context for event-level and stream-normalized reporting
    stream_len: int | None = None
    drift_onsets: list[int] = field(default_factory=list)
    detection_tolerance: int | None = None
    drift_event_severity: dict[int, int] = field(default_factory=dict)

    def record_alarm(
        self,
        sample_idx: int,
        is_tp: bool,
        latency: int | None = None,
        severity: int | None = None,
    ):
        self.alarm_indices.append(sample_idx)
        self.is_true_positive.append(is_tp)
        self.alarm_severities.append(severity)
        if latency is not None:
            self.latencies.append(latency)

    def set_stream_context(
        self,
        *,
        stream_len: int | None = None,
        drift_onsets: list[int] | None = None,
        tolerance: int | None = None,
        drift_event_severity: dict[int, int] | None = None,
    ) -> None:
        """Attach optional context needed for event-level reporting."""
        if stream_len is not None:
            self.stream_len = stream_len
        if drift_onsets is not None:
            self.drift_onsets = list(sorted(drift_onsets))
        if tolerance is not None:
            self.detection_tolerance = tolerance
        if drift_event_severity is not None:
            self.drift_event_severity = dict(drift_event_severity)

    @property
    def n_alarms(self) -> int:
        return len(self.alarm_indices)

    @property
    def n_false_alarms(self) -> int:
        return sum(not tp for tp in self.is_true_positive)

    @property
    def false_positive_rate(self) -> float:
        if self.n_alarms == 0:
            return 0.0
        return self.n_false_alarms / self.n_alarms

    @property
    def true_positive_rate(self) -> float:
        if self.n_alarms == 0:
            return 0.0
        return sum(self.is_true_positive) / self.n_alarms

    @property
    def mean_detection_latency(self) -> float | None:
        return float(np.mean(self.latencies)) if self.latencies else None

    @property
    def false_alarms_per_1000_non_drift(self) -> float | None:
        if self.stream_len is None or self.detection_tolerance is None:
            return None
        non_drift = _count_non_drift_samples(
            self.stream_len, self.drift_onsets, self.detection_tolerance
        )
        if non_drift == 0:
            return 0.0
        return 1000.0 * self.n_false_alarms / non_drift

    @property
    def matched_drift_onsets(self) -> list[int]:
        if self.detection_tolerance is None:
            return []
        matches = _match_alarms_to_onsets(
            self.alarm_indices, self.drift_onsets, self.detection_tolerance
        )
        return [m for m in matches if m is not None]

    @property
    def n_detected_drift_events(self) -> int:
        return len(self.matched_drift_onsets)

    @property
    def n_total_drift_events(self) -> int:
        return len(self.drift_onsets)

    @property
    def missed_drift_onsets(self) -> list[int]:
        detected = set(self.matched_drift_onsets)
        return [o for o in self.drift_onsets if o not in detected]

    @property
    def n_missed_drift_events(self) -> int:
        return len(self.missed_drift_onsets)

    @property
    def latency_by_severity(self) -> dict[int, float]:
        if self.detection_tolerance is None:
            return {}
        matches = _match_alarms_to_onsets(
            self.alarm_indices, self.drift_onsets, self.detection_tolerance
        )
        by_sev: dict[int, list[int]] = {}
        for alarm_idx, onset, alarm_sev in zip(
            self.alarm_indices, matches, self.alarm_severities
        ):
            if onset is None:
                continue
            latency = alarm_idx - onset
            severity = self.drift_event_severity.get(onset, alarm_sev)
            if severity is None:
                continue
            by_sev.setdefault(severity, []).append(latency)
        return {sev: float(np.mean(vals)) for sev, vals in sorted(by_sev.items())}


def _match_alarms_to_onsets(
    alarm_indices: list[int],
    drift_onsets: list[int],
    tolerance: int,
) -> list[int | None]:
    """Match each alarm to at most one drift onset (and vice versa)."""
    claimed = set()
    matched_onsets: list[int | None] = []
    for idx in alarm_indices:
        matched_onset: int | None = None
        for onset in drift_onsets:
            if onset not in claimed and onset <= idx < onset + tolerance:
                claimed.add(onset)
                matched_onset = onset
                break
        matched_onsets.append(matched_onset)
    return matched_onsets


def _count_non_drift_samples(stream_len: int, drift_onsets: list[int], tolerance: int) -> int:
    """Count samples outside all drift-detection tolerance windows."""
    if stream_len <= 0:
        return 0
    drift_mask = np.zeros(stream_len, dtype=bool)
    for onset in drift_onsets:
        start = max(0, onset)
        end = min(stream_len, onset + tolerance)
        if start < end:
            drift_mask[start:end] = True
    return int(stream_len - np.count_nonzero(drift_mask))


def classify_alarms(
    alarm_indices: list[int],
    drift_onsets: list[int],
    tolerance: int,
) -> list[bool]:
    """Label each alarm as TP (True Positive) or FP.

    An alarm is a TP if it falls within [onset, onset + tolerance) for any
    known drift onset. Each onset can be claimed by at most one alarm.
    """
    return [matched is not None for matched in _match_alarms_to_onsets(
        alarm_indices, drift_onsets, tolerance
    )]


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
    header = (
        f"{'Detector':<16} {'Alarms':>7} {'TPR':>8} {'FPR':>8} "
        f"{'FA/1k ND':>10} {'Missed':>10} {'Mean latency':>14}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        lat = f"{r.mean_detection_latency:.1f}" if r.mean_detection_latency is not None else "—"
        fa_per_1k = (
            f"{r.false_alarms_per_1000_non_drift:.2f}"
            if r.false_alarms_per_1000_non_drift is not None
            else "—"
        )
        missed = (
            f"{r.n_missed_drift_events}/{r.n_total_drift_events}"
            if r.n_total_drift_events > 0
            else "—"
        )
        print(
            f"{r.name:<16} {r.n_alarms:>7} {r.true_positive_rate:>8.3f} "
            f"{r.false_positive_rate:>8.3f} {fa_per_1k:>10} {missed:>10} {lat:>14}"
        )
        sev_latency = r.latency_by_severity
        sev_text = (
            ", ".join(f"s{sev}={val:.1f}" for sev, val in sev_latency.items())
            if sev_latency
            else "—"
        )
        print(f"{'':<16} latency by severity: {sev_text}")
