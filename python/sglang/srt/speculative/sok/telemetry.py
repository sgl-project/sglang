"""Lightweight dispatch telemetry for SOK shape collection.

Aggregates kernel dispatch events without sitting on the hot path.
The inference loop calls `record_dispatch()` after each kernel; this
module batches observations into ShapeProfile and KernelCache.

Sampling is controlled by SOKConfig.telemetry_sample_rate:
  1.0 = record every dispatch (default, safe for small model counts)
  0.1 = record 10% of dispatches (use if profiling overhead measurable)
"""

import logging
import time
from typing import Optional

from sglang.srt.speculative.sok.config import SOKConfig
from sglang.srt.speculative.sok.kernel_cache import KernelCache
from sglang.srt.speculative.sok.shape_profile import ShapeKey, ShapeProfile

logger = logging.getLogger(__name__)


class DispatchTelemetry:
    """Collects per-dispatch telemetry and routes to cache + profile.

    Not a context manager — the caller provides timing externally to
    avoid adding overhead to the kernel dispatch hot path.
    """

    def __init__(
        self,
        config: SOKConfig,
        cache: KernelCache,
        profile: ShapeProfile,
    ):
        self._config = config
        self._cache = cache
        self._profile = profile
        self._dispatch_count = 0
        self._sampled_count = 0
        self._total_latency_us = 0.0
        self._enabled = config.enable_telemetry

    def record_dispatch(
        self,
        kernel_name: str,
        triton_cache_key: str,
        shape: ShapeKey,
        latency_us: float = 0.0,
        was_cache_hit: bool = True,
    ):
        """Record one kernel dispatch event.

        Args:
            kernel_name: Triton kernel function name.
            triton_cache_key: Cache directory key for this compiled variant.
            shape: Dispatch shape descriptor.
            latency_us: Wall-clock dispatch time in microseconds (0 = not timed).
            was_cache_hit: True if kernel was already compiled; False if JIT.
        """
        if not self._enabled:
            return

        self._dispatch_count += 1

        # Sampling gate
        rate = self._config.telemetry_sample_rate
        if rate < 1.0:
            # Deterministic modulo sampling (no RNG overhead)
            interval = max(1, int(1.0 / rate))
            if self._dispatch_count % interval != 0:
                return

        self._sampled_count += 1

        # Route to cache
        if was_cache_hit:
            self._cache.record_hit(kernel_name, triton_cache_key)
        else:
            self._cache.record_jit(kernel_name, triton_cache_key)

        # Route to shape profile
        self._profile.record(shape, latency_us)

        if latency_us > 0:
            self._total_latency_us += latency_us

    def get_summary(self) -> dict:
        """Summary for periodic logging."""
        cache_stats = self._cache.get_stats()
        profile_stats = self._profile.get_stats()
        return {
            "dispatches": self._dispatch_count,
            "sampled": self._sampled_count,
            "sample_rate": self._config.telemetry_sample_rate,
            "avg_latency_us": (
                round(self._total_latency_us / max(self._sampled_count, 1), 2)
            ),
            "cache_hit_rate": cache_stats["hit_rate"],
            "unique_shapes": profile_stats["shapes"],
            "top_shapes": profile_stats["top3"],
        }
