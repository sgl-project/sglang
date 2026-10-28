"""The step-time totals the load snapshot reports, accumulated as each step
commits."""

from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING, Deque, Optional, Tuple

from sglang.srt.managers.utils import is_health_check_generate_req

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler_components.pending_steps import PendingStep

STEP_MAX_US = 2_000_000
# Pause windows kept for prefill busy accounting.
PAUSE_WINDOWS_MAX = 1024


def _accumulate_decode_moment(
    totals: list[float],
    batch_size: int,
    step_us: int,
    generated: int,
) -> None:
    if batch_size <= 0 or step_us <= 0:
        return
    b = float(batch_size)
    t = float(step_us)
    g = float(generated)
    totals[0] += 1.0
    totals[1] += b
    totals[2] += t
    totals[3] += b * b
    totals[4] += b * t
    totals[5] += g


class StepCounters:
    def __init__(self) -> None:
        self.total_prefill_uncached_tokens = 0
        self.total_prefill_busy_us = 0
        self.decode_moment_totals: list[float] = [0.0] * 6
        self._prev_step: Optional[Tuple[int, float, bool]] = None
        self._prev_prefill_end_ts: Optional[float] = None
        # Closed pause windows a later prefill step may still overlap; not busy time.
        self._pause_intervals: Deque[Tuple[float, float]] = deque(
            maxlen=PAUSE_WINDOWS_MAX
        )
        self._pause_started_at: Optional[float] = None

    def begin_pause(self, now: float) -> None:
        """An engine pause starts; a repeated pause keeps the first start."""
        if self._pause_started_at is None:
            self._pause_started_at = now

    def end_pause(self, now: float) -> None:
        """The engine resumes: close the open pause window."""
        if self._pause_started_at is not None:
            self._pause_intervals.append((self._pause_started_at, now))
            self._pause_started_at = None

    def record(self, record: PendingStep) -> None:
        batch = record.step
        mode = batch.forward_mode
        is_prefill = mode.is_extend_without_speculative()
        if not (is_prefill or mode.is_decode() or mode.is_target_verify()):
            return
        if all(is_health_check_generate_req(req) for req in batch.reqs):
            return
        start_iter, start_ts = batch.forward_iter, record.launch_ts
        prev = self._prev_step
        self._prev_step = (batch.forward_iter, record.launch_ts, is_prefill)
        # An idle pass keeps forward_iter contiguous (forward_ct advances in run_batch).
        contiguous = False
        if prev is not None:
            prev_iter, prev_launch_ts, prev_is_prefill = prev
            contiguous = (
                not record.after_idle_gap
                and prev_iter + 1 == start_iter
                and prev_is_prefill == is_prefill
            )
        if is_prefill:
            # Completion boundaries include the burst tail and scheduling overhead
            # without double-counting overlapping launch-to-result spans.
            end_ts = time.monotonic()
            prev_end_ts = self._prev_prefill_end_ts
            self._prev_prefill_end_ts = end_ts
            if prev_end_ts is not None:
                # A new burst or mode can still overlap an earlier prefill.
                start_ts = prev_end_ts if contiguous else max(start_ts, prev_end_ts)
            pauses = list(self._pause_intervals)
            if self._pause_started_at is not None:
                # A result processed while still paused excludes the pause so far.
                pauses.append((self._pause_started_at, end_ts))
            paused = sum(max(0.0, min(end_ts, b) - max(start_ts, a)) for a, b in pauses)
            step_us = int((end_ts - start_ts - paused) * 1e6)
            # Later prefill steps start at or after this end.
            while self._pause_intervals and self._pause_intervals[0][1] <= end_ts:
                self._pause_intervals.popleft()
        else:
            # Decode regression uses launch cadence and skips boundaries.
            if not contiguous:
                return
            step_us = int((record.launch_ts - prev_launch_ts) * 1e6)
        # Only decode cadence is capped; a long prefill chunk is real busy time.
        if step_us <= 0 or (not is_prefill and step_us >= STEP_MAX_US):
            return
        if is_prefill:
            self.total_prefill_busy_us += step_us
            self.total_prefill_uncached_tokens += batch.extend_num_tokens
        else:
            batch_size = len(batch.reqs)
            _accumulate_decode_moment(
                self.decode_moment_totals,
                batch_size,
                step_us,
                record.result.get_num_generated_tokens(batch_size),
            )
