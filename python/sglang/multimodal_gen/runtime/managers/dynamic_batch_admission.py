# SPDX-License-Identifier: Apache-2.0
"""Admission control for native diffusion dynamic batching."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass, field, fields, is_dataclass
from difflib import get_close_matches
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.pipelines_core import Req
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

logger = init_logger(__name__)

_BATCHING_RULE_KEYS = frozenset(
    {
        "model",
        "model_contains",
        "resolution",
        "device_memory_gb_min",
        "device_memory_gb_max",
        "offload",
        "max_batch_size",
        "max_cost",
        # Accepted for config provenance; ignored by admission.
        "calibration",
    }
)

_MEMORY_PROFILE_SCHEMA_VERSION = 1
_CACHE_ENV = "SGLANG_DIFFUSION_BATCH_MEMORY_CACHE"


@dataclass(frozen=True)
class MemoryAdmissionPolicy:
    """Internal policy for measured memory admission."""

    # Recent observations kept per profile.
    history_size: int = 32
    # Minimum interval between profile cache writes.
    save_interval_s: float = 60.0
    # Successes required near safe cost before clearing an OOM boundary.
    oom_clear_successes: int = 8
    # Minimum allocator slack.
    internal_reserve_min_mb: float = 512.0
    # Headroom used only by the cold-start rough estimate.
    rough_headroom_fraction: float = 0.10
    rough_headroom_min_mb: float = 512.0


_MEMORY_POLICY = MemoryAdmissionPolicy()


@dataclass(frozen=True)
class AdmissionLimit:
    """Batch size and cost caps after batching rule matching."""

    max_batch_size: int
    max_cost: float | None = None
    cap_reason: str | None = None

    def reject_reason(self, *, batch_size: int, batch_cost: float) -> str | None:
        if batch_size > self.max_batch_size:
            return self.cap_reason or f"config_cap:{self.max_batch_size}"
        if self.max_cost is not None and batch_cost > self.max_cost:
            return f"cost_budget:{batch_cost:.0f}>{self.max_cost:.0f}"
        return None

    def stop_reason_for_next_cost(self, next_batch_cost: float) -> str | None:
        if self.max_cost is not None and next_batch_cost > self.max_cost:
            return f"cost_budget_next:{next_batch_cost:.0f}>{self.max_cost:.0f}"
        return None


@dataclass(frozen=True)
class BatchingRule:
    """Batching config admission rule."""

    model: str | None = None
    model_contains: str | None = None
    resolution: str | None = None
    device_memory_gb_min: float | None = None
    device_memory_gb_max: float | None = None
    offload: bool | None = None
    max_batch_size: int = 1
    max_cost: float | None = None
    source: str = "user"

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source: str) -> BatchingRule:
        if not isinstance(data, dict):
            raise ValueError(
                f"batching config rule from {source} must be an object, "
                f"got {type(data).__name__}"
            )
        _validate_rule_keys(data, source=source)
        if "max_batch_size" not in data:
            raise ValueError("batching config rule requires max_batch_size")

        rule = cls(
            model=_optional_str(data.get("model")),
            model_contains=_optional_str(data.get("model_contains")),
            resolution=_optional_str(data.get("resolution")),
            device_memory_gb_min=_optional_float(data.get("device_memory_gb_min")),
            device_memory_gb_max=_optional_float(data.get("device_memory_gb_max")),
            offload=_optional_bool(data.get("offload")),
            max_batch_size=int(data["max_batch_size"]),
            max_cost=_optional_float(data.get("max_cost")),
            source=source,
        )
        rule.validate()
        return rule

    def validate(self) -> None:
        if self.model is not None and self.model_contains is not None:
            raise ValueError(
                "batching config rule cannot set both model and model_contains"
            )
        if self.model is None and self.model_contains is None:
            raise ValueError("batching config rule requires model or model_contains")
        if self.max_batch_size < 1:
            raise ValueError("batching config rule max_batch_size must be >= 1")
        if self.max_cost is not None and self.max_cost <= 0.0:
            raise ValueError("batching config rule max_cost must be > 0")
        if (
            self.device_memory_gb_min is not None
            and self.device_memory_gb_max is not None
            and self.device_memory_gb_min > self.device_memory_gb_max
        ):
            raise ValueError(
                "batching config rule device_memory_gb_min must be <= device_memory_gb_max"
            )

    def matches(
        self,
        *,
        model_path: str,
        resolution: str | None,
        device_memory_gb: float | None,
        offload: bool,
    ) -> bool:
        if self.model is not None and self.model != model_path:
            return False
        if self.model_contains is not None and self.model_contains not in model_path:
            return False
        if self.resolution not in (None, "*") and self.resolution != resolution:
            return False
        if self.offload is not None and self.offload != offload:
            return False
        if device_memory_gb is None:
            return True
        if (
            self.device_memory_gb_min is not None
            and device_memory_gb < self.device_memory_gb_min
        ):
            return False
        if (
            self.device_memory_gb_max is not None
            and device_memory_gb > self.device_memory_gb_max
        ):
            return False
        return True


@dataclass(frozen=True)
class MemoryObservation:
    batch_cost: float
    peak_memory_mb: float
    batch_size: int
    baseline_memory_mb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryObservation":
        return cls(
            batch_cost=float(data["batch_cost"]),
            peak_memory_mb=float(data["peak_memory_mb"]),
            batch_size=max(1, int(data.get("batch_size", 1))),
            baseline_memory_mb=_optional_float(data.get("baseline_memory_mb")),
        )


@dataclass
class MemoryProfile:
    """Memory observations for one runtime/request key.

    ``successes`` stores recent successful batch measurements. Predictions use
    a monotone upper envelope over those points, indexed by relative batch cost.
    ``ratio_residuals`` and ``absolute_residuals_mb`` keep recent extrapolation
    misses as adaptive headroom. ``oom_cost`` is the lowest observed cost that
    ran out of memory and is treated as a hard boundary until recovery succeeds.
    """

    successes: list[MemoryObservation] = field(default_factory=list)
    ratio_residuals: list[float] = field(default_factory=list)
    absolute_residuals_mb: list[float] = field(default_factory=list)
    oom_cost: float | None = None
    recovery_cost: float | None = None
    recovery_successes: int = 0

    _cache_dirty: bool = field(default=True, init=False, repr=False)
    _distinct_success_cost_count: int = field(default=0, init=False, repr=False)
    _max_success_batch_size: int = field(default=0, init=False, repr=False)
    _max_success_cost: float = field(default=0.0, init=False, repr=False)
    _residual_ratio: float = field(default=1.0, init=False, repr=False)
    _residual_abs_mb: float = field(default=0.0, init=False, repr=False)
    _points: list[tuple[float, float]] = field(
        default_factory=list, init=False, repr=False
    )

    def observe_success(
        self,
        batch_cost: float,
        peak_memory_mb: float,
        *,
        batch_size: int,
        baseline_memory_mb: float | None = None,
    ) -> None:
        """Record a successful batch and update prediction residuals."""
        if batch_cost <= 0.0 or peak_memory_mb <= 0.0:
            return

        previous_success_costs = self.get_num_success_costs()
        previous_max_cost = self.get_max_success_cost()
        predicted = self._estimate_base_peak_memory_mb(batch_cost)
        if (
            previous_success_costs >= 2
            and predicted is not None
            and predicted > 0.0
            and batch_cost > previous_max_cost
        ):
            self.ratio_residuals.append(max(1.0, peak_memory_mb / predicted))
            self.absolute_residuals_mb.append(max(0.0, peak_memory_mb - predicted))
            del self.ratio_residuals[: -_MEMORY_POLICY.history_size]
            del self.absolute_residuals_mb[: -_MEMORY_POLICY.history_size]

        self.successes.append(
            MemoryObservation(
                batch_cost=float(batch_cost),
                peak_memory_mb=float(peak_memory_mb),
                batch_size=max(1, int(batch_size)),
                baseline_memory_mb=_optional_float(baseline_memory_mb),
            )
        )
        del self.successes[: -_MEMORY_POLICY.history_size]
        self._update_oom_boundary_after_success(batch_cost)
        self._cache_dirty = True

    def observe_oom(self, batch_cost: float) -> None:
        if batch_cost <= 0.0:
            return
        if self.oom_cost is None or batch_cost < self.oom_cost:
            self.oom_cost = float(batch_cost)
            self.recovery_cost = self.get_max_success_cost()
            self.recovery_successes = 0
            self._cache_dirty = True

    def estimate_peak_memory_mb(self, batch_cost: float) -> float | None:
        predicted = self._estimate_base_peak_memory_mb(batch_cost)
        if predicted is None:
            return None
        if batch_cost <= self.get_max_success_cost():
            return predicted
        return max(
            predicted * self._get_residual_ratio(),
            predicted + self._get_residual_mb(),
        )

    def estimate_rough_peak_memory_mb(self, batch_cost: float) -> float | None:
        """Estimate cold-start memory before the measured fit is available."""
        if batch_cost <= 0.0:
            return None
        seeds = [
            obs
            for obs in self.successes
            if obs.batch_cost > 0.0
            and obs.peak_memory_mb > 0.0
            and obs.baseline_memory_mb is not None
        ]
        if not seeds:
            return None

        lower_or_equal = [obs for obs in seeds if obs.batch_cost <= batch_cost]
        seed = max(
            lower_or_equal or seeds,
            key=lambda obs: obs.batch_cost,
        )
        baseline_mb = min(
            max(0.0, float(seed.baseline_memory_mb or 0.0)),
            seed.peak_memory_mb,
        )
        variable_mb_per_cost = max(0.0, seed.peak_memory_mb - baseline_mb) / max(
            seed.batch_cost, 1e-9
        )
        extra_cost = max(0.0, batch_cost - seed.batch_cost)
        rough_headroom_mb = (
            max(
                _MEMORY_POLICY.rough_headroom_min_mb,
                variable_mb_per_cost
                * extra_cost
                * _MEMORY_POLICY.rough_headroom_fraction,
            )
            if extra_cost > 0.0
            else 0.0
        )
        return max(
            seed.peak_memory_mb,
            baseline_mb + variable_mb_per_cost * batch_cost + rough_headroom_mb,
        )

    def get_num_success_costs(self) -> int:
        self._update_cached_stats()
        return self._distinct_success_cost_count

    def get_max_success_batch_size(self) -> int:
        self._update_cached_stats()
        return self._max_success_batch_size

    def get_max_success_cost(self) -> float:
        self._update_cached_stats()
        return self._max_success_cost

    def to_dict(self) -> dict[str, Any]:
        return {
            "successes": [obs.to_dict() for obs in self.successes],
            "ratio_residuals": list(self.ratio_residuals),
            "absolute_residuals_mb": list(self.absolute_residuals_mb),
            "oom_cost": self.oom_cost,
            "recovery_cost": self.recovery_cost,
            "recovery_successes": self.recovery_successes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryProfile":
        profile = cls(
            successes=[
                MemoryObservation.from_dict(item)
                for item in data.get("successes", [])
                if isinstance(item, dict)
            ],
            oom_cost=_optional_float(data.get("oom_cost")),
            recovery_cost=_optional_float(data.get("recovery_cost")),
            recovery_successes=int(data.get("recovery_successes", 0)),
        )
        profile._recompute_residuals_from_successes()
        profile._cache_dirty = True
        return profile

    def _recompute_residuals_from_successes(self) -> None:
        """Rebuild adaptive residuals after loading cached observations."""
        replay = type(self)()
        for obs in self.successes:
            replay.observe_success(
                obs.batch_cost,
                obs.peak_memory_mb,
                batch_size=obs.batch_size,
                baseline_memory_mb=obs.baseline_memory_mb,
            )
        self.ratio_residuals = replay.ratio_residuals
        self.absolute_residuals_mb = replay.absolute_residuals_mb

    def _update_oom_boundary_after_success(self, batch_cost: float) -> None:
        if self.oom_cost is None or self.recovery_cost is None:
            return
        if batch_cost + 1e-9 < self.recovery_cost:
            return
        self.recovery_successes += 1
        if self.recovery_successes >= _MEMORY_POLICY.oom_clear_successes:
            self.oom_cost = None
            self.recovery_cost = None
            self.recovery_successes = 0

    def _estimate_base_peak_memory_mb(self, batch_cost: float) -> float | None:
        self._update_cached_stats()
        if not self._points:
            return None
        if len(self._points) == 1:
            return self._points[0][1]
        if batch_cost <= self._points[-1][0]:
            return _interpolate_peak_memory_mb(self._points, batch_cost)
        return _extrapolate_peak_memory_mb(self._points, batch_cost)

    def _get_residual_ratio(self) -> float:
        self._update_cached_stats()
        return self._residual_ratio

    def _get_residual_mb(self) -> float:
        self._update_cached_stats()
        return self._residual_abs_mb

    def _update_cached_stats(self) -> None:
        if not self._cache_dirty:
            return

        self._distinct_success_cost_count = len(
            {obs.batch_cost for obs in self.successes}
        )
        self._max_success_batch_size = max(
            (obs.batch_size for obs in self.successes), default=0
        )
        self._max_success_cost = max(
            (obs.batch_cost for obs in self.successes), default=0.0
        )
        self._residual_ratio = max(self.ratio_residuals, default=1.0)
        self._residual_abs_mb = max(self.absolute_residuals_mb, default=0.0)

        by_cost: dict[float, float] = {}
        for obs in self.successes:
            by_cost[obs.batch_cost] = max(
                by_cost.get(obs.batch_cost, 0.0), obs.peak_memory_mb
            )
        self._points = _get_monotone_upper_points(sorted(by_cost.items()))
        self._cache_dirty = False


class BatchAdmissionController:
    """Apply static and memory caps before adding requests to a batch."""

    def __init__(self, server_args: ServerArgs, gpu_id: int):
        self._mode = getattr(server_args, "batching_mode", "dynamic")
        self._user_max_batch_size = int(server_args.batching_max_size)
        self._model_path = server_args.model_path
        self._offload = bool(server_args.layerwise_offload_components)
        self._gpu_id = gpu_id
        self._device_name = _get_device_name(gpu_id)
        device_memory_mb = _get_device_total_memory_mb(gpu_id)
        self._device_memory_gb = (
            None if device_memory_mb is None else device_memory_mb / 1024.0
        )
        self._memory_reserve_fraction = float(
            server_args.batching_memory_reserve_fraction
        )
        self._rules = load_batching_config(server_args.batching_config)
        self._pipeline_config = server_args.pipeline_config
        self._limit_cache: dict[str | None, AdmissionLimit] = {}

        self._runtime_memory_key = self._build_runtime_memory_key(server_args)
        self._runtime_memory_hash = _stable_hash(self._runtime_memory_key)
        self._lora_revision = 0
        self._memory_profiles: dict[tuple[Any, ...], MemoryProfile] = {}
        self._memory_budget_mb: float | None = None
        self._memory_profile_revision = 0
        self._memory_profiles_dirty = False
        self._last_profile_save_s = 0.0
        self._memory_profile_cache_path = self._get_memory_profile_cache_path(
            getattr(server_args, "batching_memory_profile_cache", None)
        )
        self._load_memory_profile_cache()

        if self.enabled:
            logger.info(
                "Batch admission enabled: user_max=%d, device=%s %.1fGiB, rules=%d, memory_profiles=%d",
                self._user_max_batch_size,
                self._device_name or "unknown",
                self._device_memory_gb or 0.0,
                len(self._rules),
                len(self._memory_profiles),
            )

    @property
    def enabled(self) -> bool:
        return self._mode == "dynamic" and self._user_max_batch_size > 1

    def refresh_memory_budget(self) -> None:
        """Refresh the memory budget for one scheduler selection pass."""
        if not self.enabled:
            self._memory_budget_mb = None
            return
        self._memory_budget_mb = self._compute_memory_budget_mb()

    def get_reject_reason_for_candidate(
        self, current_reqs: list[Req], candidate_req: Req
    ) -> str | None:
        if not self.enabled:
            return None

        batch_size = len(current_reqs) + 1
        batch_cost = self.get_batch_cost(current_reqs) + self.get_request_cost(
            candidate_req
        )
        first_req = current_reqs[0] if current_reqs else candidate_req
        limit = self.get_admission_limit(first_req)
        reject_reason = limit.reject_reason(
            batch_size=batch_size,
            batch_cost=batch_cost,
        )
        if reject_reason is not None:
            return reject_reason
        return self._get_memory_reject_reason(
            batch_size=batch_size,
            batch_cost=batch_cost,
            memory_key=self._get_memory_key(first_req),
        )

    def is_batch_full(self, reqs: list[Req]) -> bool:
        """Return whether one more compatible request would exceed admission."""
        if not self.enabled or not reqs:
            return len(reqs) >= self._user_max_batch_size

        next_batch_size = len(reqs) + 1
        next_cost = self.get_batch_cost(reqs) + self.get_request_cost(reqs[0])
        limit = self.get_admission_limit(reqs[0])
        if next_batch_size > limit.max_batch_size:
            return True
        if limit.max_cost is not None and next_cost > limit.max_cost:
            return True
        return (
            self._get_memory_reject_reason(
                batch_size=next_batch_size,
                batch_cost=next_cost,
                memory_key=self._get_memory_key(reqs[0]),
                next_batch=True,
            )
            is not None
        )

    def get_batch_stop_reason(self, reqs: list[Req]) -> str | None:
        if not self.enabled or not reqs:
            return None

        next_batch_size = len(reqs) + 1
        next_cost = self.get_batch_cost(reqs) + self.get_request_cost(reqs[0])
        limit = self.get_admission_limit(reqs[0])
        if next_batch_size > limit.max_batch_size:
            return limit.cap_reason or f"config_cap:{limit.max_batch_size}"
        reason = limit.stop_reason_for_next_cost(next_cost)
        if reason is not None:
            return reason
        return self._get_memory_reject_reason(
            batch_size=next_batch_size,
            batch_cost=next_cost,
            memory_key=self._get_memory_key(reqs[0]),
            next_batch=True,
        )

    def get_max_admissible_batch_size(self, req: Req) -> int:
        """Return the largest homogeneous batch size currently admissible.

        This is used for metrics/capacity reporting, so it mirrors candidate
        admission instead of using a looser static cap. The binary search is
        cached on primitive inputs plus memory budget/profile revision because
        it can be called several times in one scheduler selection pass while the
        profile and budget are unchanged.
        """
        if not self.enabled:
            return 1

        limit = self.get_admission_limit(req)
        cost_per_req = self.get_request_cost(req)
        return self._compute_max_admissible_batch_size_cached(
            limit.max_batch_size,
            limit.max_cost,
            cost_per_req,
            self._get_memory_key(req),
            self._memory_budget_mb,
            self._memory_profile_revision,
        )

    @lru_cache(maxsize=1024)
    def _compute_max_admissible_batch_size_cached(
        self,
        max_batch_size: int,
        max_cost: float | None,
        cost_per_req: float,
        memory_key: tuple[Any, ...],
        memory_budget_mb: float | None,
        memory_profile_revision: int,
    ) -> int:
        del memory_budget_mb, memory_profile_revision
        low = 1
        high = max_batch_size
        while low < high:
            mid = (low + high + 1) // 2
            batch_cost = cost_per_req * mid
            if max_cost is not None and batch_cost > max_cost:
                high = mid - 1
            elif (
                self._get_memory_reject_reason(
                    batch_size=mid,
                    batch_cost=batch_cost,
                    memory_key=memory_key,
                )
                is not None
            ):
                high = mid - 1
            else:
                low = mid
        return low

    def observe_batch_result(self, reqs: list[Req], output_batch: OutputBatch) -> None:
        if not self.enabled or not reqs:
            return

        batch_cost = self.get_batch_cost(reqs)
        if batch_cost <= 0.0:
            return

        profile = self._memory_profiles.setdefault(
            self._get_memory_key(reqs[0]), MemoryProfile()
        )
        if output_batch.is_oom:
            profile.observe_oom(batch_cost)
            self._mark_memory_profiles_changed()
            self._save_memory_profile_cache_if_due()
            return

        if output_batch.error is not None:
            return

        if output_batch.peak_memory_mb > 0.0:
            profile.observe_success(
                batch_cost,
                output_batch.peak_memory_mb,
                batch_size=len(reqs),
                baseline_memory_mb=output_batch.pre_forward_reserved_memory_mb,
            )
            self._mark_memory_profiles_changed()
            self._save_memory_profile_cache_if_due()

    def update_lora_revision(self) -> None:
        """Bump the LoRA revision used in memory profile keys."""
        self._lora_revision += 1

    def flush_memory_profile_cache(self) -> None:
        if not self._memory_profiles_dirty:
            return
        self._save_memory_profile_cache()

    def _mark_memory_profiles_changed(self) -> None:
        self._memory_profiles_dirty = True
        self._memory_profile_revision += 1

    def _save_memory_profile_cache_if_due(self) -> None:
        if not self._memory_profiles_dirty:
            return
        now = time.monotonic()
        if now - self._last_profile_save_s >= _MEMORY_POLICY.save_interval_s:
            self._save_memory_profile_cache()

    def get_admission_limit(self, req: Req) -> AdmissionLimit:
        """Return the admission limit for a request shape."""
        cache_key = req.resolution_key
        cached = self._limit_cache.get(cache_key)
        if cached is not None:
            return cached

        rules = self._get_matching_rules(req)
        if not rules:
            limit = AdmissionLimit(max_batch_size=self._user_max_batch_size)
            self._limit_cache[cache_key] = limit
            return limit

        config_cap = min(rule.max_batch_size for rule in rules)
        max_batch_size = min(self._user_max_batch_size, config_cap)
        cap_reason = (
            f"config_cap:{max_batch_size}"
            if max_batch_size < self._user_max_batch_size
            else None
        )
        costs = [rule.max_cost for rule in rules if rule.max_cost is not None]
        limit = AdmissionLimit(
            max_batch_size=max(1, max_batch_size),
            max_cost=min(costs) if costs else None,
            cap_reason=cap_reason,
        )
        self._limit_cache[cache_key] = limit
        return limit

    def get_batch_cost(self, reqs: list[Req]) -> float:
        return sum(self.get_request_cost(req) for req in reqs)

    def get_request_cost(self, req: Req) -> float:
        cached = getattr(req, "_batch_admission_cost", None)
        if cached is not None:
            return float(cached)
        cost = float(self._pipeline_config.estimate_request_cost(req))
        req._batch_admission_cost = cost  # type: ignore[attr-defined]
        return cost

    def _get_memory_reject_reason(
        self,
        *,
        batch_size: int,
        batch_cost: float,
        memory_key: tuple[Any, ...],
        next_batch: bool = False,
    ) -> str | None:
        """Apply OOM, calibration, and budget gates."""
        profile = self._memory_profiles.get(memory_key)
        if profile is not None and profile.oom_cost is not None:
            if batch_cost + 1e-9 >= profile.oom_cost:
                return f"memory_oom:{batch_cost:.0f}>={profile.oom_cost:.0f}"

        rough_predicted_mb = self._estimate_rough_peak_memory_mb(profile, batch_cost)
        calibration_reject = self._get_memory_calibration_reject_reason(
            profile,
            batch_size=batch_size,
            batch_cost=batch_cost,
        )
        if calibration_reject is not None:
            return calibration_reject

        predicted_mb = None
        used_rough_prediction = False
        if profile is not None and profile.get_num_success_costs() >= 2:
            predicted_mb = profile.estimate_peak_memory_mb(batch_cost)
        elif rough_predicted_mb is not None:
            predicted_mb = rough_predicted_mb
            used_rough_prediction = True
        if predicted_mb is None or self._memory_budget_mb is None:
            return None
        if predicted_mb > self._memory_budget_mb:
            prefix = "memory_rough_budget" if used_rough_prediction else "memory_budget"
            suffix = "_next" if next_batch else ""
            return (
                f"{prefix}{suffix}:{predicted_mb:.0f}>{self._memory_budget_mb:.0f}MiB"
            )
        return None

    def _get_memory_calibration_reject_reason(
        self,
        profile: MemoryProfile | None,
        *,
        batch_size: int,
        batch_cost: float,
    ) -> str | None:
        """Gate new profiles through a geometric cold-start ramp."""
        if profile is None or not profile.successes:
            if batch_size > 1:
                return "memory_uncalibrated:allow=1"
            return None

        max_observed_batch = profile.get_max_success_batch_size()
        allowed_batch = max(1, max_observed_batch * 2)
        if batch_size > allowed_batch:
            return f"memory_uncalibrated:allow={allowed_batch}"

        if profile.get_num_success_costs() < 2 and batch_size > max(
            2, max_observed_batch
        ):
            return f"memory_uncalibrated:allow={max(2, max_observed_batch)}"

        return None

    def _estimate_rough_peak_memory_mb(
        self, profile: MemoryProfile | None, batch_cost: float
    ) -> float | None:
        if profile is None:
            return None
        return profile.estimate_rough_peak_memory_mb(batch_cost)

    def _get_memory_key(self, req: Req) -> tuple[Any, ...]:
        return (
            ("runtime", self._runtime_memory_key),
            ("lora_revision", self._lora_revision),
            ("request", self._get_request_memory_key(req)),
        )

    def _get_request_memory_key(self, req: Req) -> tuple[Any, ...]:
        cached = getattr(req, "_batch_request_memory_key", None)
        if cached is not None:
            return cached

        key = (
            ("resolution", req.resolution_key),
            ("width", getattr(req, "width", None)),
            ("height", getattr(req, "height", None)),
            ("num_frames", getattr(req, "num_frames", None)),
            ("num_outputs_per_prompt", getattr(req, "num_outputs_per_prompt", None)),
            ("data_type", _freeze_key_value(getattr(req, "data_type", None))),
            ("max_sequence_length", getattr(req, "max_sequence_length", None)),
            (
                "prompt_template",
                _freeze_key_value(getattr(req, "prompt_template", None)),
            ),
            (
                "do_classifier_free_guidance",
                getattr(req, "do_classifier_free_guidance", False),
            ),
            ("condition_image", getattr(req, "condition_image", None) is not None),
            ("image_path", bool(getattr(req, "image_path", None))),
            ("return_frames", getattr(req, "return_frames", None)),
            ("enable_upscaling", getattr(req, "enable_upscaling", None)),
            (
                "enable_frame_interpolation",
                getattr(req, "enable_frame_interpolation", None),
            ),
            ("sampling", _get_sampling_memory_key(req)),
            (
                "diffusers_kwargs",
                _freeze_key_value(
                    (getattr(req, "extra", None) or {}).get("diffusers_kwargs")
                ),
            ),
        )
        req._batch_request_memory_key = key  # type: ignore[attr-defined]
        return key

    def _build_runtime_memory_key(self, server_args: "ServerArgs") -> tuple[Any, ...]:
        return (
            (
                "model",
                (
                    ("model_path", server_args.model_path),
                    ("model_id", server_args.model_id),
                    ("pipeline_class_name", server_args.pipeline_class_name),
                    ("pipeline_config", _freeze_key_value(server_args.pipeline_config)),
                ),
            ),
            (
                "device",
                (
                    ("name", self._device_name),
                    ("memory_gb", self._device_memory_gb),
                ),
            ),
            (
                "runtime",
                (
                    ("backend", _freeze_key_value(server_args.backend)),
                    ("attention_backend", server_args.attention_backend),
                    (
                        "component_attention_backends",
                        _freeze_key_value(server_args.component_attention_backends),
                    ),
                    ("quantization", server_args.quantization),
                    (
                        "transformer_weights_path",
                        server_args.transformer_weights_path,
                    ),
                    ("enable_torch_compile", server_args.enable_torch_compile),
                ),
            ),
            (
                "parallelism",
                (
                    ("num_gpus", server_args.num_gpus),
                    ("tp_size", server_args.tp_size),
                    ("sp_degree", server_args.sp_degree),
                    ("ulysses_degree", server_args.ulysses_degree),
                    ("ring_degree", server_args.ring_degree),
                    ("dp_size", server_args.dp_size),
                    ("dp_degree", server_args.dp_degree),
                    ("enable_cfg_parallel", server_args.enable_cfg_parallel),
                    ("cfg_parallel_degree", server_args.cfg_parallel_degree),
                ),
            ),
            (
                "offload",
                (
                    ("dit_cpu_offload", server_args.dit_cpu_offload),
                    ("dit_layerwise_offload", server_args.dit_layerwise_offload),
                    (
                        "layerwise_offload_components",
                        _freeze_key_value(server_args.layerwise_offload_components),
                    ),
                    (
                        "dit_offload_prefetch_size",
                        server_args.dit_offload_prefetch_size,
                    ),
                    (
                        "text_encoder_cpu_offload",
                        server_args.text_encoder_cpu_offload,
                    ),
                    (
                        "image_encoder_cpu_offload",
                        server_args.image_encoder_cpu_offload,
                    ),
                    ("vae_cpu_offload", server_args.vae_cpu_offload),
                    ("use_fsdp_inference", server_args.use_fsdp_inference),
                ),
            ),
            (
                "lora",
                (
                    ("path", server_args.lora_path),
                    ("nickname", server_args.lora_nickname),
                    ("scale", server_args.lora_scale),
                    ("weight_name", server_args.lora_weight_name),
                ),
            ),
        )

    def _get_matching_rules(self, req: Req) -> list[BatchingRule]:
        return [
            rule
            for rule in self._rules
            if rule.matches(
                model_path=self._model_path,
                resolution=req.resolution_key,
                device_memory_gb=self._device_memory_gb,
                offload=self._offload,
            )
        ]

    def _compute_memory_budget_mb(self) -> float | None:
        if current_platform.is_cpu():
            return None
        total_mb = _get_device_total_memory_mb(self._gpu_id)
        available_mb = _get_available_device_memory_mb(self._gpu_id)
        reserved_mb = current_platform.get_process_reserved_memory_mb()
        if total_mb is None:
            return None

        internal_reserve_mb = max(
            _MEMORY_POLICY.internal_reserve_min_mb,
            total_mb * self._memory_reserve_fraction,
        )
        budget_mb = max(0.0, total_mb - internal_reserve_mb)
        if available_mb is not None and reserved_mb is not None:
            external_used_mb = max(0.0, total_mb - available_mb - reserved_mb)
            budget_mb = max(0.0, budget_mb - external_used_mb)
        return budget_mb

    def _get_memory_profile_cache_path(self, configured: str | None) -> str | None:
        path = configured or os.getenv(_CACHE_ENV)
        if path is None:
            path = os.path.join(
                os.path.expanduser("~"),
                ".cache",
                "sglang",
                "diffusion_batch_memory",
            )
        if str(path).lower() in ("", "none", "off", "false"):
            return None
        path = os.path.expanduser(path)
        if path.endswith(".json"):
            return path
        return os.path.join(path, f"{self._runtime_memory_hash}.json")

    def _load_memory_profile_cache(self) -> None:
        path = self._memory_profile_cache_path
        if path is None or not os.path.exists(path):
            return
        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            logger.warning("Failed to load memory profile cache %s: %s", path, exc)
            return

        if payload.get("schema_version") != _MEMORY_PROFILE_SCHEMA_VERSION:
            logger.warning(
                "Ignoring memory profile cache with unsupported schema: %s", path
            )
            return
        if payload.get("runtime_hash") != self._runtime_memory_hash:
            logger.warning(
                "Ignoring memory profile cache for different runtime: %s", path
            )
            return

        profiles = payload.get("profiles", [])
        for item in profiles:
            if not isinstance(item, dict):
                continue
            key = item.get("key")
            profile_data = item.get("profile")
            if key is None or not isinstance(profile_data, dict):
                continue
            self._memory_profiles[_unjsonable(key)] = MemoryProfile.from_dict(
                profile_data
            )
        observation_count = sum(
            len(profile.successes) for profile in self._memory_profiles.values()
        )
        logger.info(
            "Loaded memory profile cache: path=%s, profiles=%d, observations=%d",
            path,
            len(self._memory_profiles),
            observation_count,
        )

    def _save_memory_profile_cache(self) -> None:
        path = self._memory_profile_cache_path
        if path is None:
            self._memory_profiles_dirty = False
            return

        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "schema_version": _MEMORY_PROFILE_SCHEMA_VERSION,
            "runtime_hash": self._runtime_memory_hash,
            "runtime_key": _jsonable(self._runtime_memory_key),
            "profiles": [
                {
                    "key": _jsonable(key),
                    "profile": profile.to_dict(),
                }
                for key, profile in self._memory_profiles.items()
                if profile.successes or profile.oom_cost is not None
            ],
        }
        fd, tmp_path = tempfile.mkstemp(
            prefix=".memory_profiles_",
            suffix=".json",
            dir=os.path.dirname(path),
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
            self._memory_profiles_dirty = False
            self._last_profile_save_s = time.monotonic()
            logger.info(
                "Saved memory profile cache: path=%s, profiles=%d",
                path,
                len(payload["profiles"]),
            )
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


def load_batching_config(path: str | None) -> list[BatchingRule]:
    if path is None:
        return []

    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    source = os.path.abspath(path)
    entries = _config_entries(payload)
    rules = [BatchingRule.from_dict(entry, source=source) for entry in entries]
    if not rules:
        raise ValueError(f"batching config {source} does not contain any rules")
    return rules


def _config_entries(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and payload.get("schema_version") not in (None, 1):
        raise ValueError("batching config schema_version must be 1")
    if isinstance(payload, dict) and isinstance(payload.get("rules"), list):
        return payload["rules"]
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        entries: list[dict[str, Any]] = []
        for key, value in payload.items():
            if key == "schema_version" or not isinstance(value, dict):
                continue
            model, _sep, resolution = key.partition("|")
            entry = dict(value)
            if model:
                entry.setdefault("model", model)
            if resolution:
                entry.setdefault("resolution", resolution)
            entries.append(entry)
        return entries
    raise ValueError(
        "batching config must be a {'schema_version': 1, 'rules': [...]} object, "
        "a list of rules, or a mapping keyed by model|resolution"
    )


def _validate_rule_keys(data: dict[str, Any], *, source: str) -> None:
    unknown = sorted(set(data) - _BATCHING_RULE_KEYS)
    if not unknown:
        return

    hints = []
    for key in unknown:
        matches = get_close_matches(key, _BATCHING_RULE_KEYS, n=1)
        if matches:
            hints.append(f"{key!r} (did you mean {matches[0]!r}?)")
        else:
            hints.append(repr(key))
    raise ValueError(
        f"batching config rule from {source} contains unknown key(s): "
        f"{', '.join(hints)}"
    )


def _get_sampling_memory_key(req: Req) -> tuple[Any, ...] | None:
    """Reuse the scheduler's SamplingParams compatibility boundary.

    Dynamic batching only co-batches requests whose non-``batch_sig_exclude``
    sampling fields match. Reusing the same metadata keeps memory profile reuse
    conservative without maintaining a second include/exclude list here.
    """
    sp = getattr(req, "sampling_params", None)
    if sp is None:
        return None

    items = []
    for field_info in fields(SamplingParams):
        name = field_info.name
        if name.startswith("_") or field_info.metadata.get("batch_sig_exclude", False):
            continue
        items.append((name, _freeze_key_value(getattr(sp, name, None))))
    return tuple(items)


def _get_device_name(gpu_id: int) -> str | None:
    try:
        return current_platform.get_device_name(gpu_id)
    except Exception:
        return None


def _get_device_total_memory_mb(gpu_id: int) -> float | None:
    try:
        return current_platform.get_device_total_memory(gpu_id) / (1024**2)
    except Exception:
        return None


def _get_available_device_memory_mb(gpu_id: int) -> float | None:
    try:
        return (
            current_platform.get_available_gpu_memory(gpu_id, empty_cache=False)
            * 1024.0
        )
    except Exception:
        return None


def _freeze_key_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return tuple(
            (field_info.name, _freeze_key_value(getattr(value, field_info.name)))
            for field_info in fields(value)
        )
    if isinstance(value, dict):
        return tuple(
            (str(key), _freeze_key_value(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_key_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_key_value(item) for item in value))
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return ("tensor", tuple(value.shape), str(value.dtype))
    if callable(value):
        return (
            "callable",
            getattr(value, "__module__", None),
            getattr(value, "__qualname__", type(value).__qualname__),
        )
    return repr(value)


def _get_monotone_upper_points(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    monotone: list[tuple[float, float]] = []
    current_peak = 0.0
    for cost, peak in points:
        current_peak = max(current_peak, peak)
        monotone.append((cost, current_peak))
    return monotone


def _interpolate_peak_memory_mb(
    points: list[tuple[float, float]], batch_cost: float
) -> float:
    if batch_cost <= points[0][0]:
        return points[0][1]
    for idx in range(1, len(points)):
        low_cost, low_peak = points[idx - 1]
        high_cost, high_peak = points[idx]
        if batch_cost <= high_cost:
            if math.isclose(high_cost, low_cost):
                return max(low_peak, high_peak)
            fraction = (batch_cost - low_cost) / (high_cost - low_cost)
            return low_peak + fraction * (high_peak - low_peak)
    return points[-1][1]


def _extrapolate_peak_memory_mb(
    points: list[tuple[float, float]], batch_cost: float
) -> float:
    n = len(points)
    mean_cost = sum(cost for cost, _ in points) / n
    mean_peak = sum(peak for _, peak in points) / n
    denominator = sum((cost - mean_cost) ** 2 for cost, _ in points)
    if denominator == 0.0:
        return points[-1][1]
    slope = sum((cost - mean_cost) * (peak - mean_peak) for cost, peak in points)
    slope = max(0.0, slope / denominator)
    intercept = mean_peak - slope * mean_cost
    return max(points[-1][1], intercept + slope * batch_cost)


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _unjsonable(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_unjsonable(item) for item in value)
    if isinstance(value, dict):
        return tuple((key, _unjsonable(item)) for key, item in sorted(value.items()))
    return value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "y", "on"):
            return True
        if lowered in ("0", "false", "no", "n", "off"):
            return False
    raise ValueError(f"cannot parse boolean batching config value: {value!r}")
