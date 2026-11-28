"""Energy accounting on top of any latency predictor, from a measured power calibration.

The calibration format and how to measure one are in the simulator guide
(docs/docs/advanced_features/sglang_simulator.mdx).

Example simulator configuration:
    "predictor": {
        "name": "ml",
        "database_path": "latency_model.pkl",
        "energy": {"calibration_path": "energy_calibration.json"}
    }
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from typing import Any

import msgspec
from sglang_simulator.time_predictor.base import InferTimePredictor, ScheduleBatch
from sglang_simulator.time_predictor.ml_features import (
    ML_FEATURE_NAMES,
    extract_ml_features,
)

CALIBRATION_FORMAT_VERSION = 1
_LOG1P_PREFIX = "log1p_"
_FEATURE_INDEX = {name: idx for idx, name in enumerate(ML_FEATURE_NAMES)}
# Predictor time is measured as scheduler CPU time and advances the simulated clock,
# so a model that needs only these sums skips the 18-feature extractor.
_BATCH_SUMS = {
    "batch_size": lambda batch: batch.batch_size,
    "sum_extend": lambda batch: batch.num_context_tokens,
    "sum_past": lambda batch: batch.total_past_kv_length,
}


def _resolve_feature(name: str) -> tuple[str, bool] | None:
    """(ML feature name, apply log1p), or None if the name is unknown."""
    base = name.removeprefix(_LOG1P_PREFIX)
    if base != name and base in _FEATURE_INDEX:
        # log1p_sum_past resolves to log1p(sum_past), the same value as the native feature.
        return base, True
    if name in _FEATURE_INDEX:
        return name, False
    return None


def _require_non_negative(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")


class PhasePowerModel(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Average power in watts, linear in named batch features."""

    intercept_w: float
    # Keys are ML_FEATURE_NAMES entries, optionally prefixed with "log1p_".
    coefficients_w: dict[str, float] = {}

    def __post_init__(self):
        unknown = sorted(n for n in self.coefficients_w if _resolve_feature(n) is None)
        if unknown:
            raise ValueError(
                f"unknown energy features {unknown}; use ML_FEATURE_NAMES, "
                f"optionally prefixed with '{_LOG1P_PREFIX}'"
            )


class ActiveGapModel(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Power drawn at the start of a gap between steps, before the GPUs idle."""

    power_w: float
    max_s: float

    def __post_init__(self):
        _require_non_negative(value=self.power_w, name="active_gap.power_w")
        _require_non_negative(value=self.max_s, name="active_gap.max_s")


class EnergyCalibration(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Power of one model on one deployment; watts are totals over its devices."""

    format_version: int
    idle_power_w: float
    prefill: PhasePowerModel
    decode: PhasePowerModel
    power_limit_w: float | None = None
    # None charges every gap at idle_power_w.
    active_gap: ActiveGapModel | None = None
    # Free-form measurement record, such as device, model and driver.
    provenance: dict[str, Any] = {}

    def __post_init__(self):
        if self.format_version != CALIBRATION_FORMAT_VERSION:
            raise ValueError(
                f"unsupported format_version {self.format_version}; "
                f"expected {CALIBRATION_FORMAT_VERSION}"
            )
        _require_non_negative(value=self.idle_power_w, name="idle_power_w")
        if self.power_limit_w is not None:
            _require_non_negative(value=self.power_limit_w, name="power_limit_w")
            if self.power_limit_w < self.idle_power_w:
                raise ValueError("power_limit_w must not be below idle_power_w")
        if self.active_gap is not None:
            if self.active_gap.power_w < self.idle_power_w:
                raise ValueError("active_gap.power_w must not be below idle_power_w")
            if (
                self.power_limit_w is not None
                and self.active_gap.power_w > self.power_limit_w
            ):
                raise ValueError("active_gap.power_w must not exceed power_limit_w")

    @classmethod
    def from_file(cls, path: str) -> EnergyCalibration:
        # Same path handling as the ML predictor's database_path.
        with open(os.path.expandvars(os.path.expanduser(path)), "rb") as f:
            return msgspec.json.decode(f.read(), type=cls)


class _PhasePower:
    """A phase model with its feature lookups resolved once."""

    def __init__(self, *, model: PhasePowerModel, idle_w: float, limit_w: float | None):
        self._intercept_w = model.intercept_w
        self._terms = [
            (*_resolve_feature(name), weight)
            for name, weight in model.coefficients_w.items()
        ]
        self._needs_extractor = any(
            feature not in _BATCH_SUMS for feature, _, _ in self._terms
        )
        self._idle_w = idle_w
        self._limit_w = math.inf if limit_w is None else limit_w

    def power_w(self, batch: ScheduleBatch) -> float:
        if self._needs_extractor:
            features = extract_ml_features(
                [req.extend_length for req in batch.reqs],
                [req.past_kv_length for req in batch.reqs],
            )
            values = [features[_FEATURE_INDEX[f]] for f, _, _ in self._terms]
        else:
            values = [_BATCH_SUMS[f](batch) for f, _, _ in self._terms]
        power = self._intercept_w
        for value, (_, use_log1p, weight) in zip(values, self._terms, strict=True):
            power += weight * (math.log1p(value) if use_log1p else value)
        return min(max(power, self._idle_w), self._limit_w)


class EnergyPredictor(InferTimePredictor):
    """Wraps a latency predictor and charges each step and each gap between steps.

    Gaps are counted from the first step's start to the last step's end.
    """

    def __init__(
        self,
        *,
        inner: InferTimePredictor,
        calibration: EnergyCalibration,
        clock: Callable[[], float],
    ):
        super().__init__(inner.model, inner.hw, inner.config)
        self._inner = inner
        self._clock = clock
        self._idle_w = calibration.idle_power_w
        self._active_gap = calibration.active_gap
        self._prefill = _PhasePower(
            model=calibration.prefill,
            idle_w=calibration.idle_power_w,
            limit_w=calibration.power_limit_w,
        )
        self._decode = _PhasePower(
            model=calibration.decode,
            idle_w=calibration.idle_power_w,
            limit_w=calibration.power_limit_w,
        )
        self._clear()

    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        latency = self._inner.predict_infer_time(batch)
        # Negative latency is the inner predictor's error signal; it adds no energy.
        step_s = max(float(latency), 0.0)
        if batch.is_decode():
            self._decode_j += self._decode.power_w(batch) * step_s
        else:
            self._prefill_j += self._prefill.power_w(batch) * step_s
        self._busy_s += step_s

        start = self._clock()
        if self._last_end_s is not None and start > self._last_end_s:
            self._add_gap(start - self._last_end_s)
        end = start + step_s
        if self._last_end_s is None or end > self._last_end_s:
            self._last_end_s = end
        return latency

    def get_metrics(self) -> dict:
        active_gap_j = 0.0
        if self._active_gap is not None:
            active_gap_j = self._active_gap.power_w * self._active_gap_s
        idle_j = self._idle_w * self._idle_s
        busy_j = self._prefill_j + self._decode_j
        total_j = busy_j + active_gap_j + idle_j
        total_s = self._busy_s + self._active_gap_s + self._idle_s
        return {
            **self._inner.get_metrics(),
            "energy_total_j": total_j,
            "energy_busy_j": busy_j,
            "energy_active_gap_j": active_gap_j,
            "energy_idle_j": idle_j,
            "energy_prefill_j": self._prefill_j,
            "energy_decode_j": self._decode_j,
            "energy_busy_s": self._busy_s,
            "energy_active_gap_s": self._active_gap_s,
            "energy_idle_s": self._idle_s,
            "energy_avg_power_w": total_j / total_s if total_s else 0.0,
        }

    def reset_metrics(self) -> None:
        self._inner.reset_metrics()
        self._clear()

    def _add_gap(self, gap_s: float) -> None:
        active_s = 0.0
        if self._active_gap is not None:
            active_s = min(gap_s, self._active_gap.max_s)
        self._active_gap_s += active_s
        self._idle_s += gap_s - active_s

    def _clear(self) -> None:
        self._prefill_j = 0.0
        self._decode_j = 0.0
        self._busy_s = 0.0
        self._active_gap_s = 0.0
        self._idle_s = 0.0
        self._last_end_s: float | None = None


def calc_energy_per_token_metrics(metrics: dict) -> dict:
    if "energy_total_j" not in metrics or not metrics["total_output"]:
        return {}
    return {
        "energy_j_per_output_token": metrics["energy_total_j"] / metrics["total_output"]
    }
