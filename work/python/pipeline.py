"""Core inference + drift-detection + XAI pipeline (thesis §3.1).

Flow per sample:
  image → ResNet-50 → prediction → error signal → active drift detectors
                                                   ↓ (on alarm)
                                        XAI analysis on pre/post-drift windows

Usage: instantiate Pipeline, then call .run(stream).
"""

from __future__ import annotations

from bisect import bisect_right
import logging
import random
from dataclasses import dataclass, field
from typing import Iterator

import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import (
    DETECTION_TOLERANCE_WINDOW,
    DISABLE_DETECTOR_AFTER_ALARM,
    MVTEC_ROOT,
    XAI_POST_DRIFT_WINDOW,
    XAI_PRE_DRIFT_WINDOW,
    XAI_SAMPLE_SIZE,
    SAVE_XAI_CHECKPOINTS,
    OUTPUT_DIR,
    XAI_SCALE_GRADCAM_LARGE_THRESHOLD,
    XAI_SCALE_LIME_LARGE_THRESHOLD,
    XAI_SCALE_GRADCAM_WEIGHT,
    XAI_SCALE_LIME_WEIGHT,
    XAI_SCALE_COMBINED_LARGE_THRESHOLD,
)
from data.mvtec import load_defect_mask
from drift.detectors import make_all_detectors
from evaluation.metrics import DetectorResult
from xai.gradcam import compute_gradcam, compute_ada, mean_heatmap, change_map
from xai.lime_analysis import compute_lime_top_k, overlap_coefficient
from checkpoint.xai_checkpoint import (
    XAICheckpoint,
    checkpoint_filename,
    minimal_scale_interpretation_dict,
    save_xai_checkpoint,
)

logger = logging.getLogger(__name__)


@dataclass
class AlarmEvent:
    detector_name: str
    sample_index:  int
    severity:      int
    drift_type: str = "unknown"                # Phase 7: "corruption", "defect", "geometric"
    scale: str = None                          # Phase 7: "small", "large", or None
    xai: dict = field(default_factory=dict)   # populated after XAI analysis
    inferred_scale: str | None = None         # inferred from XAI outputs
    scale_interpretation: dict = field(default_factory=dict)  # rationale and scores
    checkpoint_path: str = None                # Phase 4: path to saved checkpoint


@dataclass
class StreamRecord:
    """One sample in the rolling buffer."""
    image_np:   np.ndarray
    label:      int
    prediction: int
    score:      float          # defective-class confidence
    severity:   int
    path:       str
    drift_type: str = "unknown"      # Phase 7: from stream metadata
    scale: str = None                # Phase 7: from stream metadata


class Pipeline:
    def __init__(
        self,
        model:          torch.nn.Module,
        device:         torch.device,
        transform,
        train_accuracy: float,
        category:       str,
        corruption_type: str = None,  # Phase 1: from pre-computed dataset
    ):
        self.model    = model
        self.device   = device
        self.transform = transform
        self.category = category
        self.corruption_type = corruption_type or "unknown"

        self.detectors = make_all_detectors(train_accuracy)
        self.det_results: dict[str, DetectorResult] = {
            d.name: DetectorResult(name=d.name) for d in self.detectors
        }
        self.alarms: list[AlarmEvent] = []
        self._disabled_detectors: set[str] = set()   # detectors that have fired once and are done
        self._xai_last: dict[str, int] = {}           # used when DISABLE_DETECTOR_AFTER_ALARM=False

        # Rolling buffer (kept to avoid unbounded memory growth)
        self._buffer: list[StreamRecord] = []
        self._buffer_cap = XAI_PRE_DRIFT_WINDOW + XAI_POST_DRIFT_WINDOW + 50
        # Full-stream metric bookkeeping (independent from rolling XAI buffer)
        self._labels_by_sev: dict[int, list[int]] = {}
        self._scores_by_sev: dict[int, list[float]] = {}
        # Per-severity reservoir for ADA (keeps memory bounded while covering full stream)
        self._ada_reservoir_by_sev: dict[int, list[StreamRecord]] = {}
        self._ada_seen_defective_by_sev: dict[int, int] = {}

        # Phase 4: Checkpoint counter
        self._alarm_counter = 0

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _infer(self, img_np: np.ndarray) -> tuple[int, float]:
        """Inference step.

        Returns: (prediction, confidence)
        """
        self.model.eval()
        from PIL import Image as _PIL
        img_pil = _PIL.fromarray(img_np) if isinstance(img_np, np.ndarray) else img_np
        tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        probs  = F.softmax(self.model(tensor), dim=1).squeeze()
        pred   = int(probs.argmax().item())
        score  = float(probs[1].item())   # confidence for defective class

        return pred, score

    # ------------------------------------------------------------------
    # XAI (triggered on alarm)
    # ------------------------------------------------------------------
    def _xai_window(self, alarm_buf_idx: int) -> dict:
        """Phase 5: Compute XAI analysis (Grad-CAM + LIME only; removed SHAP for speed)."""
        pre_start = max(0, alarm_buf_idx - XAI_PRE_DRIFT_WINDOW)
        pre_recs  = self._buffer[pre_start:alarm_buf_idx]
        post_recs = self._buffer[alarm_buf_idx: alarm_buf_idx + XAI_POST_DRIFT_WINDOW]

        if not pre_recs or not post_recs:
            return {}

        def to_tensors(recs: list[StreamRecord]) -> torch.Tensor:
            from PIL import Image as _PIL
            return torch.stack([
                self.transform(_PIL.fromarray(r.image_np) if isinstance(r.image_np, np.ndarray) else r.image_np)
                for r in recs
            ]).to(self.device)

        # Limit to XAI_SAMPLE_SIZE for LIME
        pre_sample  = pre_recs[-XAI_SAMPLE_SIZE:]
        post_sample = post_recs[:XAI_SAMPLE_SIZE]
        pre_imgs    = [r.image_np for r in pre_sample]
        post_imgs   = [r.image_np for r in post_sample]

        # --- Grad-CAM (Phase 5: fast, no perturbations) ---
        pre_cam  = compute_gradcam(self.model, to_tensors(pre_sample),  1, self.device)
        torch.cuda.empty_cache()
        post_cam = compute_gradcam(self.model, to_tensors(post_sample), 1, self.device)
        torch.cuda.empty_cache()
        cam_change = change_map(mean_heatmap(pre_cam), mean_heatmap(post_cam))

        # --- LIME (Phase 5: reduce perturbations if needed) ---
        pre_lime  = compute_lime_top_k(
            self.model, pre_imgs, self.transform, self.device, n_samples=XAI_SAMPLE_SIZE
        )
        post_lime = compute_lime_top_k(
            self.model, post_imgs, self.transform, self.device, n_samples=XAI_SAMPLE_SIZE
        )
        lime_ovlp = overlap_coefficient(pre_lime, post_lime)

        return {
            "gradcam_change_map":       cam_change,
            "lime_overlap_coefficient": lime_ovlp,
        }

    def _infer_scale_from_xai(self, xai: dict) -> tuple[str | None, dict]:
        """Infer alarm scale ('small'/'large') from available XAI signals."""
        interpretation: dict = {"method": "rule-based-xai"}
        components: list[tuple[float, float, str]] = []  # (component_score, weight, reason)

        cam_change = xai.get("gradcam_change_map")
        if cam_change is not None:
            grad_mag = float(np.mean(np.abs(np.asarray(cam_change))))
            grad_threshold = float(XAI_SCALE_GRADCAM_LARGE_THRESHOLD)
            grad_score = min(1.0, grad_mag / grad_threshold) if grad_threshold > 0 else 0.0
            grad_weight = float(XAI_SCALE_GRADCAM_WEIGHT)
            if np.isfinite(grad_score) and np.isfinite(grad_weight) and grad_weight > 0:
                components.append((grad_score, grad_weight, "gradcam"))
            interpretation["gradcam_change_magnitude"] = grad_mag
            interpretation["gradcam_large_threshold"] = grad_threshold
            interpretation["gradcam_component_score"] = grad_score

        lime_overlap = xai.get("lime_overlap_coefficient")
        if lime_overlap is not None:
            lime_overlap = float(lime_overlap)
            lime_threshold = float(XAI_SCALE_LIME_LARGE_THRESHOLD)
            if lime_threshold > 0:
                lime_score = min(1.0, max(0.0, (lime_threshold - lime_overlap) / lime_threshold))
            else:
                lime_score = 0.0
            lime_weight = float(XAI_SCALE_LIME_WEIGHT)
            if np.isfinite(lime_score) and np.isfinite(lime_weight) and lime_weight > 0:
                components.append((lime_score, lime_weight, "lime"))
            interpretation["lime_overlap_coefficient"] = lime_overlap
            interpretation["lime_large_threshold"] = lime_threshold
            interpretation["lime_component_score"] = lime_score

        if not components:
            interpretation["status"] = "missing_xai_or_invalid_components"
            interpretation["rationale"] = "No valid Grad-CAM/LIME alarm-level components available."
            return None, interpretation

        weighted_sum = sum(score * weight for score, weight, _ in components)
        weight_total = sum(weight for _, weight, _ in components)
        if weight_total <= 0:
            interpretation["status"] = "invalid_weights"
            interpretation["rationale"] = "Combined component weights are non-positive."
            return None, interpretation
        combined_score = weighted_sum / weight_total
        interpretation["combined_large_score"] = float(combined_score)
        interpretation["combined_large_threshold"] = float(XAI_SCALE_COMBINED_LARGE_THRESHOLD)
        interpretation["used_components"] = [name for _, _, name in components]

        inferred_scale = (
            "large"
            if combined_score >= float(XAI_SCALE_COMBINED_LARGE_THRESHOLD)
            else "small"
        )
        interpretation["rationale"] = (
            f"Large score {combined_score:.3f} "
            f"{'>=' if inferred_scale == 'large' else '<'} "
            f"{float(XAI_SCALE_COMBINED_LARGE_THRESHOLD):.3f}"
        )
        return inferred_scale, interpretation

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(
        self,
        stream: Iterator[tuple[np.ndarray, int, int, str]],
        drift_onsets: list[int] | None = None,
        stream_len: int | None = None,
        global_pbar=None,
    ) -> tuple[list[AlarmEvent], dict[str, DetectorResult]]:
        """Process the image stream.

        stream:       yields (image_np, label, severity, path)
        drift_onsets: sample indices at which severity transitions occur
                      (used to label alarms as TP/FP and compute latency)
        stream_len:   total number of samples — enables accurate ETA in progress bar
        """
        drift_onsets = sorted(drift_onsets or [])
        drift_onset_set = set(drift_onsets)
        drift_event_severity: dict[int, int] = {}
        claimed_onsets_by_detector: dict[str, set[int]] = {
            det.name: set() for det in self.detectors
        }
        for result in self.det_results.values():
            result.set_stream_context(
                stream_len=stream_len,
                drift_onsets=drift_onsets,
                tolerance=DETECTION_TOLERANCE_WINDOW,
            )

        _t_infer = 0.0
        _n_infer = 0
        _t_xai   = 0.0
        _n_xai   = 0

        def _fmt(s: float) -> str:
            s = int(s)
            h, m = s // 3600, (s % 3600) // 60
            if h:   return f"{h}h {m:02d}m"
            if m:   return f"{m}m {s % 60:02d}s"
            return f"{s}s"

        pbar = tqdm(stream, desc=f"[{self.category}] streaming",
                    total=stream_len, position=1, leave=False)

        for sample_idx, sample_data in enumerate(pbar):
            # Phase 1/7: Stream now includes metadata (drift_type, scale)
            if len(sample_data) == 6:
                img_np, label, severity, path, drift_type, scale = sample_data
            else:
                # Backwards compatibility: old format (img_np, label, severity, path)
                img_np, label, severity, path = sample_data
                drift_type, scale = "unknown", None
            if sample_idx in drift_onset_set:
                drift_event_severity[sample_idx] = int(severity)
            
            _t0 = time.perf_counter()
            pred, score = self._infer(img_np)
            _t_infer += time.perf_counter() - _t0
            _n_infer += 1
            if global_pbar is not None:
                global_pbar.update(1)
            error = int(pred != label)

            # Phase 7: Include drift type and scale metadata
            rec = StreamRecord(
                img_np, label, pred, score, severity, path,
                drift_type=drift_type,
                scale=scale,
            )
            self._buffer.append(rec)
            if len(self._buffer) > self._buffer_cap:
                self._buffer.pop(0)
            self._labels_by_sev.setdefault(severity, []).append(label)
            self._scores_by_sev.setdefault(severity, []).append(score)
            if label == 1:
                seen = self._ada_seen_defective_by_sev.get(severity, 0) + 1
                self._ada_seen_defective_by_sev[severity] = seen
                reservoir = self._ada_reservoir_by_sev.setdefault(severity, [])
                if len(reservoir) < XAI_SAMPLE_SIZE:
                    reservoir.append(rec)
                else:
                    replace_at = random.randint(0, seen - 1)
                    if replace_at < XAI_SAMPLE_SIZE:
                        reservoir[replace_at] = rec

            buf_idx = len(self._buffer) - 1

            for det in self.detectors:
                if det.name in self._disabled_detectors:
                    continue
                state = det.update(error)
                if state == "drift":
                    logger.info(
                        "[%s] alarm at sample %d (severity=%d)", det.name, sample_idx, severity
                    )
                    # Determine if TP/FP and compute latency
                    onset_i = bisect_right(drift_onsets, sample_idx) - 1
                    onset = drift_onsets[onset_i] if onset_i >= 0 else None
                    claimed_onsets = claimed_onsets_by_detector[det.name]
                    is_tp = (
                        onset is not None
                        and onset not in claimed_onsets
                        and (sample_idx - onset) < DETECTION_TOLERANCE_WINDOW
                    )
                    if is_tp:
                        claimed_onsets.add(onset)
                    latency = (sample_idx - onset) if (onset is not None and is_tp) else None
                    self.det_results[det.name].record_alarm(
                        sample_idx, is_tp, latency, severity=severity
                    )

                    alarm = AlarmEvent(
                        detector_name=det.name,
                        sample_index=sample_idx,
                        severity=severity,
                        drift_type=drift_type,
                        scale=scale,
                    )
                    xai_ran_successfully = False
                    if DISABLE_DETECTOR_AFTER_ALARM:
                        if buf_idx >= XAI_PRE_DRIFT_WINDOW // 2:
                            try:
                                _t0_xai = time.perf_counter()
                                alarm.xai = self._xai_window(buf_idx)
                                xai_ran_successfully = True
                                _t_xai += time.perf_counter() - _t0_xai
                                _n_xai += 1
                            except Exception as exc:
                                logger.warning("XAI failed: %s", exc)
                        self._disabled_detectors.add(det.name)
                        logger.info("[%s] disabled after first alarm", det.name)
                    else:
                        last_xai = self._xai_last.get(det.name, -XAI_POST_DRIFT_WINDOW)
                        if buf_idx >= XAI_PRE_DRIFT_WINDOW // 2 and (sample_idx - last_xai) >= XAI_POST_DRIFT_WINDOW:
                            try:
                                _t0_xai = time.perf_counter()
                                alarm.xai = self._xai_window(buf_idx)
                                xai_ran_successfully = True
                                _t_xai += time.perf_counter() - _t0_xai
                                _n_xai += 1
                                self._xai_last[det.name] = sample_idx
                            except Exception as exc:
                                logger.warning("XAI failed: %s", exc)

                    alarm.inferred_scale, alarm.scale_interpretation = self._infer_scale_from_xai(alarm.xai)

                    if DISABLE_DETECTOR_AFTER_ALARM and SAVE_XAI_CHECKPOINTS and xai_ran_successfully:
                        try:
                            pre_recs = self._buffer[max(0, buf_idx - XAI_PRE_DRIFT_WINDOW):buf_idx]
                            post_recs = self._buffer[buf_idx:buf_idx + XAI_POST_DRIFT_WINDOW]

                            pre_samples = [{
                                "image_np": r.image_np, "prediction": r.prediction,
                                "confidence": r.score, "label": r.label, "path": r.path
                            } for r in pre_recs[-XAI_SAMPLE_SIZE:]]

                            post_samples = [{
                                "image_np": r.image_np, "prediction": r.prediction,
                                "confidence": r.score, "label": r.label, "path": r.path
                            } for r in post_recs[:XAI_SAMPLE_SIZE]]

                            ckpt = XAICheckpoint(
                                detector_name=det.name,
                                sample_index=sample_idx,
                                drift_type=drift_type,
                                scale=scale,
                                category=self.category,
                                corruption_type=self.corruption_type,
                                severity=severity,
                                model_state_dict=self.model.state_dict(),
                                pre_drift_samples=pre_samples,
                                post_drift_samples=post_samples,
                                inferred_scale=alarm.inferred_scale,
                                scale_interpretation=minimal_scale_interpretation_dict(
                                    alarm.scale_interpretation
                                ),
                            )

                            ckpt_name = checkpoint_filename(
                                self.category,
                                self.corruption_type,
                                det.name,
                                self._alarm_counter,
                            )
                            ckpt_path = OUTPUT_DIR / "xai_checkpoints" / ckpt_name
                            save_xai_checkpoint(ckpt, ckpt_path)
                            alarm.checkpoint_path = str(ckpt_path)
                            self._alarm_counter += 1
                        except Exception as exc:
                            logger.warning("Checkpoint save failed: %s", exc)

                    self.alarms.append(alarm)

            # --- progress bar postfix ---
            mean_infer = _t_infer / _n_infer
            mean_xai   = _t_xai / _n_xai if _n_xai else None
            if DISABLE_DETECTOR_AFTER_ALARM:
                n_done = len(self._disabled_detectors)
                n_left = len(self.detectors) - n_done
                postfix: dict = {"xai": f"{n_done}/{len(self.detectors)}"}
                if mean_xai is not None:
                    postfix["xai_avg"] = _fmt(mean_xai)
                    if stream_len:
                        eta = (stream_len - sample_idx - 1) * mean_infer + n_left * mean_xai
                        postfix["ETA"] = _fmt(eta)
                pbar.set_postfix(postfix, refresh=False)
            elif mean_xai is not None:
                pbar.set_postfix({"xai_calls": _n_xai, "xai_avg": _fmt(mean_xai)}, refresh=False)

        for result in self.det_results.values():
            result.set_stream_context(drift_event_severity=drift_event_severity)
        return self.alarms, self.det_results

    # ------------------------------------------------------------------
    # Post-hoc ADA curve (per-severity, requires MVTec masks)
    # ------------------------------------------------------------------
    def compute_ada_curve(self) -> dict[int, float]:
        """Mean ADA score at each severity over up to XAI_SAMPLE_SIZE defective images."""
        from xai.gradcam import compute_gradcam, compute_ada

        ada_curve: dict[int, float] = {}
        for sev, recs in self._ada_reservoir_by_sev.items():
            if not recs:
                continue
            chosen = list(recs)
            scores = []
            for rec in chosen:
                mask = load_defect_mask(MVTEC_ROOT, self.category, rec.path)
                if mask is None:
                    continue
                from PIL import Image as _PIL
                img_pil = _PIL.fromarray(rec.image_np) if isinstance(rec.image_np, np.ndarray) else rec.image_np
                tensor = self.transform(img_pil).unsqueeze(0)
                cam = compute_gradcam(self.model, tensor, target_class=1, device=self.device)
                scores.append(compute_ada(cam[0], mask))
            if scores:
                ada_curve[sev] = float(np.mean(scores))

        return ada_curve

    # ------------------------------------------------------------------
    # AUROC degradation curve
    # ------------------------------------------------------------------
    def compute_auroc_curve(self) -> dict[int, float]:
        from evaluation.metrics import auroc_at_severity
        return auroc_at_severity(self._labels_by_sev, self._scores_by_sev)
