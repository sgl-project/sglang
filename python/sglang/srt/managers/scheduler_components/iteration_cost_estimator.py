# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Lightweight iteration-cost estimator for cost-aware chunked prefill.

Tracks recent decode-only and mixed-batch iteration latencies and uses the
*slowdown ratio* of mixed vs pure-decode to dynamically reduce the prefill
chunk size, protecting decode latency.

Key design decisions:
* No torch.cuda.synchronize() in the hot path.
* TPOT is approximated by the decode iteration wall-clock latency (each
  decode request produces one token per iteration, so per-request inter-token
  latency ~= iteration latency).
* The controller is *relative*: it compares mixed-batch latency to the
  pure-decode baseline, not to an absolute constant. This prevents
  permanent throttling when pure decode alone is slow (e.g. 128K context).
* No division of iteration latency by decode batch size.
* Hysteresis and bounded rate-of-change prevent chunk-size oscillation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(slots=True, kw_only=True)
class IterationCostEstimator:
    """Estimate iteration cost and choose prefill chunk size adaptively.

    The controller maintains three EMA tracks:

    * ``_ema_decode_ms``: pure decode-only iteration latency.
    * ``_ema_mixed_ms``: mixed prefill+decode iteration latency.
    * ``_ema_prefill_ms``: prefill-only iteration latency (and token count
      for per-token normalisation).

    Throttling logic:
        slowdown = _ema_mixed_ms / max(_ema_decode_ms, epsilon)
        if slowdown > max_slowdown_ratio:
            reduce chunk size
        else:
            relax toward base chunk size

    This ensures that:
    * When pure decode is already slow (e.g. 150K context), the controller
      does not collapse prefill to minimum — it only limits *additional*
      slowdown caused by prefill interference.
    * When pure decode is fast, prefill is allowed to proceed at full speed
      unless it causes decode to slow down beyond the ratio.
    """

    # ---- Configurable parameters ----
    ema_alpha: float = 0.15
    """EMA smoothing factor for all latency tracks."""

    max_slowdown_ratio: float = 1.5
    """Maximum tolerated ratio of mixed-batch latency to pure-decode latency.
    When the ratio exceeds this, prefill chunk size is reduced."""

    min_chunk_ratio: float = 0.25
    """Minimum chunk size as a fraction of base_chunk_size. Configurable via
    server arg ``cost_aware_min_chunk_ratio``."""

    max_prefill_wait_iters: int = 64
    """Maximum iterations a prefill request may wait before force-admission.
    Configurable via server arg ``cost_aware_max_prefill_wait_iters``."""

    absolute_latency_limit_ms: float = 500.0
    """Hard cap on mixed-batch iteration latency. Regardless of the slowdown
    ratio, if mixed latency exceeds this, prefill is throttled."""

    warmup_iters: int = 10
    """Number of decode observations before the controller starts throttling.
    During warmup, full chunk size is returned."""

    # ---- Internal state ----
    _ema_decode_ms: float = field(default=0.0, repr=False)
    _ema_mixed_ms: float = field(default=0.0, repr=False)
    _ema_prefill_ms: float = field(default=0.0, repr=False)
    _last_prefill_tokens: int = field(default=0, repr=False)
    _decode_obs_count: int = field(default=0, repr=False)
    _prefill_wait_count: int = field(default=0, repr=False)
    _current_chunk_ratio: float = field(default=1.0, repr=False)
    _enabled: bool = field(default=False, repr=False)

    def enable(self) -> None:
        self._enabled = True
        logger.info(
            "IterationCostEstimator enabled: max_slowdown_ratio=%.2f, "
            "min_chunk_ratio=%.2f, max_wait=%d, abs_limit=%.0fms",
            self.max_slowdown_ratio,
            self.min_chunk_ratio,
            self.max_prefill_wait_iters,
            self.absolute_latency_limit_ms,
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def update_observation(
        self,
        *,
        batch_type: str,
        iteration_ms: float,
        num_prefill_tokens: int = 0,
    ) -> None:
        """Record a completed iteration's latency.

        Args:
            batch_type: One of "decode", "mixed", or "prefill".
            iteration_ms: Wall-clock latency of the iteration in milliseconds.
            num_prefill_tokens: Number of new prefill tokens (for prefill/mixed).
        """
        if not self._enabled:
            return
        if iteration_ms <= 0 or iteration_ms > 60000:
            return  # skip outliers

        alpha = self.ema_alpha

        if batch_type == "decode":
            if self._ema_decode_ms > 0:
                self._ema_decode_ms = self._ema_decode_ms * (1 - alpha) + iteration_ms * alpha
            else:
                self._ema_decode_ms = iteration_ms
            self._decode_obs_count += 1

        elif batch_type == "mixed":
            if self._ema_mixed_ms > 0:
                self._ema_mixed_ms = self._ema_mixed_ms * (1 - alpha) + iteration_ms * alpha
            else:
                self._ema_mixed_ms = iteration_ms

        elif batch_type == "prefill":
            if self._ema_prefill_ms > 0:
                self._ema_prefill_ms = self._ema_prefill_ms * (1 - alpha) + iteration_ms * alpha
            else:
                self._ema_prefill_ms = iteration_ms
            self._last_prefill_tokens = num_prefill_tokens

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def choose_prefill_chunk_size(
        self,
        base_chunk_size: int,
        has_decode_work: bool,
        max_chunk_size: int,
        alignment: int = 1,
    ) -> int:
        """Choose a prefill chunk size that respects decode latency.

        Args:
            base_chunk_size: Default chunked_prefill_size.
            has_decode_work: Whether decode requests are active in this iteration.
            max_chunk_size: Hard maximum (max_prefill_tokens).
            alignment: Page-size alignment.

        Returns:
            Chosen chunk size.
        """
        if not self._enabled:
            return min(base_chunk_size, max_chunk_size)

        if not has_decode_work:
            # No decode pressure: full throughput, reset wait counter
            self._prefill_wait_count = 0
            self._current_chunk_ratio = 1.0
            return min(base_chunk_size, max_chunk_size)

        # Warmup: not enough decode observations yet
        if self._decode_obs_count < self.warmup_iters:
            self._prefill_wait_count = 0
            return min(base_chunk_size, max_chunk_size)

        # Need a pure-decode baseline to compare against
        if self._ema_decode_ms <= 0:
            self._prefill_wait_count = 0
            return min(base_chunk_size, max_chunk_size)

        # Starvation prevention: if prefill has been throttled for too long,
        # force-admit at full chunk size to guarantee progress.
        if self._prefill_wait_count >= self.max_prefill_wait_iters:
            self._prefill_wait_count = 0
            self._current_chunk_ratio = 1.0
            return min(base_chunk_size, max_chunk_size)

        # Determine target ratio
        target_ratio = self._compute_target_ratio()

        # Bounded rate-of-change: move _current_chunk_ratio toward target_ratio
        # by at most 10% per iteration to prevent oscillation.
        max_step = 0.10
        diff = target_ratio - self._current_chunk_ratio
        diff = max(-max_step, min(max_step, diff))
        self._current_chunk_ratio += diff

        # Clamp
        self._current_chunk_ratio = max(
            self.min_chunk_ratio, min(1.0, self._current_chunk_ratio)
        )

        chunk_size = int(base_chunk_size * self._current_chunk_ratio)

        # Apply alignment
        if alignment > 1:
            chunk_size = (chunk_size // alignment) * alignment

        # Clamp to valid range
        chunk_size = max(alignment, min(chunk_size, max_chunk_size, base_chunk_size))

        # Update wait counter: only count iterations where prefill was
        # actually throttled (chunk < base).  When full chunk is returned,
        # reset the counter — prefill is making full progress.
        if chunk_size >= base_chunk_size:
            self._prefill_wait_count = 0
        else:
            self._prefill_wait_count += 1

        return chunk_size

    def _compute_target_ratio(self) -> float:
        """Compute the desired chunk-size ratio based on current EMAs.

        Returns a value in [min_chunk_ratio, 1.0].
        """
        decode_baseline = self._ema_decode_ms
        mixed_latency = self._ema_mixed_ms

        if mixed_latency <= 0:
            # No mixed observations yet: allow full prefill
            return 1.0

        slowdown = mixed_latency / max(decode_baseline, 1.0)

        # Absolute cap: if mixed latency is dangerously high, throttle hard
        if mixed_latency > self.absolute_latency_limit_ms:
            return self.min_chunk_ratio

        if slowdown <= self.max_slowdown_ratio:
            # Within tolerated slowdown: relax to full
            return 1.0

        # Linearly reduce from 1.0 at max_slowdown_ratio to min_chunk_ratio
        # at 2x max_slowdown_ratio
        excess_start = self.max_slowdown_ratio
        excess_end = self.max_slowdown_ratio * 2.0
        if slowdown >= excess_end:
            return self.min_chunk_ratio

        # Linear interpolation
        t = (slowdown - excess_start) / (excess_end - excess_start)
        return 1.0 - t * (1.0 - self.min_chunk_ratio)

    def reset_wait_counter(self) -> None:
        """Call when prefill is successfully admitted (any chunk size)."""
        self._prefill_wait_count = 0

    # ------------------------------------------------------------------
    # Diagnostics (for tests, not hot path)
    # ------------------------------------------------------------------

    @property
    def ema_decode_ms(self) -> float:
        return self._ema_decode_ms

    @property
    def ema_mixed_ms(self) -> float:
        return self._ema_mixed_ms

    @property
    def current_chunk_ratio(self) -> float:
        return self._current_chunk_ratio

    @property
    def decode_obs_count(self) -> int:
        return self._decode_obs_count

    # ------------------------------------------------------------------
    # Batch-type classification (used by scheduler for cost tracking)
    # ------------------------------------------------------------------

    @staticmethod
    def classify_batch_type(
        forward_mode_is_decode_or_idle: bool,
        forward_mode_is_target_verify: bool,
        forward_mode_is_extend: bool,
        has_decode_work: bool = False,
    ) -> str:
        """Classify a batch's forward mode into a cost-tracking category.

        Returns one of:
        * ``"decode"`` — pure decode, idle, or target-verify (all decode-shaped).
        * ``"mixed"`` — extend (prefill) mixed with active decode requests.
        * ``"prefill"`` — extend-only (prefill) with no active decode.

        TARGET_VERIFY is classified as decode because it performs decode-shaped
        work (fixed token rows per request, no variable-length prefill).
        """
        if forward_mode_is_decode_or_idle or forward_mode_is_target_verify:
            return "decode"
        if forward_mode_is_extend:
            return "mixed" if has_decode_work else "prefill"
        return "decode"
