"""Drift detector wrappers around River, with a unified interface.

Phase 5: ADWIN + DDM only (removed EDDM, MDDM, Naive baseline for speed).

Each detector exposes:
    .update(error: int) -> str   # "stable" | "warning" | "drift"
    .name: str
    .n_alarms: int

Error signal: 1 if the model prediction was wrong, 0 if correct.

Detector hyperparameters match the original publications:
    DDM  : warning=2.0, drift=3.0 (binomial model; instant computation)
    ADWIN: delta=0.002 (adaptive window; sub-linear in window size)
"""

from __future__ import annotations

from river import drift as river_drift
from river.drift import binary as river_drift_binary

from config import (
    ADWIN_DELTA,
    DDM_DRIFT_THRESHOLD,
    DDM_WARNING_THRESHOLD,
)

DriftState = str  # "stable" | "warning" | "drift"


class _RiverWrapper:
    """Wraps a River detector to expose a common interface."""

    def __init__(self, detector, name: str):
        self._det = detector
        self.name = name
        self.n_alarms = 0

    def update(self, error: int) -> DriftState:
        self._det.update(error)
        if self._det.drift_detected:
            self.n_alarms += 1
            return "drift"
        if getattr(self._det, "warning_detected", False):
            return "warning"
        return "stable"

    def reset(self) -> None:
        self._det = self._det.__class__()


def make_ddm() -> _RiverWrapper:
    det = river_drift_binary.DDM(
        warm_start=30,
        warning_threshold=DDM_WARNING_THRESHOLD,
        drift_threshold=DDM_DRIFT_THRESHOLD,
    )
    return _RiverWrapper(det, "DDM")


def make_adwin() -> _RiverWrapper:
    det = river_drift.ADWIN(delta=ADWIN_DELTA)
    return _RiverWrapper(det, "ADWIN")


def make_all_detectors(train_accuracy: float) -> list:
    """Return list of active drift detectors.
    
    Phase 5: Returns only ADWIN + DDM (fastest drift detection methods).
    """
    return [
        make_ddm(),
        make_adwin(),
    ]
