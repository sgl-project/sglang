from __future__ import annotations

import statistics
import time
from collections import deque
from typing import Callable, Optional

from sglang.srt.speculative.dspark_components.dspark_sps_table import (
    SpsCostTable,
    build_batch_size_sweep,
    floor_probe_index,
    is_uninitialized_sps_table,
)

ONLINE_SAMPLE_WINDOW = 128
ONLINE_MAX_STEP_INTERVAL_SECONDS = 1.0


class OnlineSpsProfiler:

    def __init__(
        self,
        *,
        initial_table: SpsCostTable,
        rebuild_interval_steps: int,
        min_bin_samples: int,
        sample_window: int = ONLINE_SAMPLE_WINDOW,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if rebuild_interval_steps < 1:
            raise ValueError(
                f"rebuild_interval_steps must be >= 1, got {rebuild_interval_steps}."
            )
        if min_bin_samples < 1:
            raise ValueError(f"min_bin_samples must be >= 1, got {min_bin_samples}.")
        self._initial_table = initial_table
        self._initial_is_profiled = not is_uninitialized_sps_table(initial_table)
        if self._initial_is_profiled:
            self._bin_edges = list(initial_table.sample_batch_tokens)
        else:
            self._bin_edges = build_batch_size_sweep(initial_table.max_batch_tokens)
        self._samples: list[deque] = [
            deque(maxlen=sample_window) for _ in self._bin_edges
        ]
        self._rebuild_interval_steps = rebuild_interval_steps
        self._min_bin_samples = min_bin_samples
        self._clock = clock
        self._prev_stamp: Optional[tuple[float, int]] = None
        self._steps_since_rebuild = 0

    def note_non_decode_step(self) -> None:
        self._prev_stamp = None

    def observe_step(self, *, batch_tokens: int) -> Optional[SpsCostTable]:
        now = self._clock()
        prev = self._prev_stamp
        self._prev_stamp = (now, batch_tokens)
        if prev is not None:
            prev_time, prev_batch_tokens = prev
            dt = now - prev_time
            if 0.0 < dt <= ONLINE_MAX_STEP_INTERVAL_SECONDS:
                self._samples[self._bin_index(prev_batch_tokens)].append(dt)
        self._steps_since_rebuild += 1
        if self._steps_since_rebuild < self._rebuild_interval_steps:
            return None
        self._steps_since_rebuild = 0
        return self._rebuild()

    def num_measured_bins(self) -> int:
        return sum(
            1 for samples in self._samples if len(samples) >= self._min_bin_samples
        )

    def num_bins(self) -> int:
        return len(self._bin_edges)

    def _bin_index(self, batch_tokens: int) -> int:
        return floor_probe_index(self._bin_edges, batch_tokens)

    def _rebuild(self) -> Optional[SpsCostTable]:
        measured: list[Optional[float]] = [
            (
                1.0 / statistics.median(samples)
                if len(samples) >= self._min_bin_samples
                else None
            )
            for samples in self._samples
        ]
        if all(value is None for value in measured):
            return None
        sample_steps_per_sec = [
            (
                value
                if value is not None
                else self._fallback_sps(measured=measured, idx=idx)
            )
            for idx, value in enumerate(measured)
        ]
        return SpsCostTable(
            sample_batch_tokens=list(self._bin_edges),
            sample_steps_per_sec=_enforce_strictly_decreasing(
                _pava_non_increasing(sample_steps_per_sec)
            ),
            max_batch_tokens=max(
                self._initial_table.max_batch_tokens, self._bin_edges[-1]
            ),
        )

    def _fallback_sps(self, *, measured: list, idx: int) -> float:
        if self._initial_is_profiled:
            return self._initial_table.lookup(self._bin_edges[idx])
        for j in range(idx - 1, -1, -1):
            if measured[j] is not None:
                return measured[j]
        for j in range(idx + 1, len(measured)):
            if measured[j] is not None:
                return measured[j]
        raise AssertionError("_rebuild guarantees at least one measured bin.")


def _pava_non_increasing(values: list[float]) -> list[float]:
    blocks: list[tuple[float, int]] = []
    for value in values:
        blocks.append((value, 1))
        while len(blocks) > 1 and blocks[-1][0] > blocks[-2][0]:
            mean_hi, weight_hi = blocks.pop()
            mean_lo, weight_lo = blocks.pop()
            merged_weight = weight_lo + weight_hi
            merged_mean = (mean_lo * weight_lo + mean_hi * weight_hi) / merged_weight
            blocks.append((merged_mean, merged_weight))
    smoothed: list[float] = []
    for mean, weight in blocks:
        smoothed.extend([mean] * weight)
    return smoothed


_STRICT_DECREASE_RELATIVE_STEP = 1e-3


def _enforce_strictly_decreasing(values: list[float]) -> list[float]:
    out: list[float] = []
    for value in values:
        if out:
            value = min(value, out[-1] * (1.0 - _STRICT_DECREASE_RELATIVE_STEP))
        out.append(value)
    return out
