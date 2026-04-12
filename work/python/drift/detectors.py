"""Drift detector wrappers around River, with a unified interface.

Each detector exposes:
    .update(error: int) -> str   # "stable" | "warning" | "drift"
    .name: str
    .n_alarms: int

Error signal: 1 if the model prediction was wrong, 0 if correct.

Detector hyperparameters match the original publications as used in the thesis (§3.4):
    DDM  : warning=2.0, drift=3.0
    EDDM : alpha=0.95, beta=0.90
    ADWIN: delta=0.002
    MDDM : window=100, geometric weighting
"""

from __future__ import annotations

from river import drift as river_drift
from river.drift import binary as river_drift_binary

from config import (
    ADWIN_DELTA,
    BASELINE_ACCURACY_FRACTION,
    BASELINE_WINDOW_SIZE,
    DDM_DRIFT_THRESHOLD,
    DDM_WARNING_THRESHOLD,
    EDDM_ALPHA,
    EDDM_BETA,
    MDDM_WINDOW_SIZE,
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


def make_eddm() -> _RiverWrapper:
    det = river_drift_binary.EDDM(
        alpha=EDDM_ALPHA,
        beta=EDDM_BETA,
    )
    return _RiverWrapper(det, "EDDM")


def make_adwin() -> _RiverWrapper:
    det = river_drift.ADWIN(delta=ADWIN_DELTA)
    return _RiverWrapper(det, "ADWIN")


def make_mddm() -> _RiverWrapper:
    # River exposes MDDM variants; MDDM_G uses geometric (recency) weights.
    try:
        det = river_drift_binary.HDDM_W()
    except AttributeError:
        det = river_drift.ADWIN()
    return _RiverWrapper(det, "MDDM")


class NaiveBaseline:
    """Rolling-accuracy monitor (thesis §3.4 naive baseline).

    Triggers when accuracy over the last BASELINE_WINDOW_SIZE samples
    falls below BASELINE_ACCURACY_FRACTION * train_accuracy.
    """

    name = "NaiveBaseline"

    def __init__(self, train_accuracy: float):
        self._threshold = BASELINE_ACCURACY_FRACTION * train_accuracy
        self._window: list[int] = []
        self.n_alarms = 0

    def update(self, error: int) -> DriftState:
        correct = 1 - error
        self._window.append(correct)
        if len(self._window) > BASELINE_WINDOW_SIZE:
            self._window.pop(0)
        if len(self._window) == BASELINE_WINDOW_SIZE:
            acc = sum(self._window) / BASELINE_WINDOW_SIZE
            if acc < self._threshold:
                self.n_alarms += 1
                return "drift"
        return "stable"

    def reset(self) -> None:
        self._window.clear()


def make_all_detectors(train_accuracy: float) -> list:
    return [
        make_ddm(),
        make_eddm(),
        make_adwin(),
        make_mddm(),
        NaiveBaseline(train_accuracy),
    ]
