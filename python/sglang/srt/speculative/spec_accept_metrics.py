"""Per-position acceptance statistics for speculative decoding.

The aggregate speculative metrics report how many draft tokens were accepted
overall.  This module additionally tracks the conditional acceptance rate at
each draft position::

    alpha[j] = P(position j is accepted | positions 0..j-1 are accepted)

The denominator must only include requests whose current speculative budget
contains position ``j``.  Keeping explicit ``eligible`` and ``accepted``
counters therefore matters when adaptive speculative decoding changes the
budget between batches: a position that was not proposed is censored, not
rejected.

The tracker is intentionally independent of logging and Prometheus.  The
scheduler metrics reporter owns one instance and publishes its snapshots.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class SpecAcceptMetricsScopeSnapshot:
    """Acceptance statistics for one scope (window or lifetime)."""

    accept_rates: tuple[float, ...]
    mean_accept_length: float
    num_requests: int


@dataclass(frozen=True)
class SpecAcceptMetricsSnapshot:
    """A decode-window snapshot paired with the lifetime snapshot."""

    window: SpecAcceptMetricsScopeSnapshot
    lifetime: SpecAcceptMetricsScopeSnapshot


@dataclass
class _PerPositionCounters:
    # ``eligible[j]``: position j was within the current budget and every
    # previous draft position was accepted.
    eligible: list[int] = field(default_factory=list)
    # ``accepted[j]``: position j itself was also accepted.
    accepted: list[int] = field(default_factory=list)
    total_accept_length: int = 0
    num_requests: int = 0

    def observe(
        self,
        num_correct_drafts_per_req: Sequence[int],
        num_draft_tokens: int,
    ) -> None:
        # ``num_draft_tokens`` includes the always-emitted bonus token, so the
        # number of conditional draft positions is one smaller.
        num_positions = max(int(num_draft_tokens) - 1, 0)
        if not num_correct_drafts_per_req:
            return

        self._ensure_num_positions(num_positions)
        for raw_num_correct in num_correct_drafts_per_req:
            # Runtime values are expected to be in range.  Clamp defensively so
            # metrics cannot make serving fail if a backend reports a bad value.
            num_correct = min(max(int(raw_num_correct), 0), num_positions)
            self.total_accept_length += num_correct + 1  # include bonus token
            self.num_requests += 1

            # If c drafts were accepted, positions 0..c-1 were accepted and
            # position c was eligible but rejected (unless c filled the budget).
            for position in range(min(num_correct + 1, num_positions)):
                self.eligible[position] += 1
            for position in range(num_correct):
                self.accepted[position] += 1

    def snapshot(self, width: int | None = None) -> SpecAcceptMetricsScopeSnapshot:
        width = len(self.eligible) if width is None else width
        rates = []
        for position in range(width):
            eligible = self.eligible[position] if position < len(self.eligible) else 0
            accepted = self.accepted[position] if position < len(self.accepted) else 0
            rates.append(accepted / eligible if eligible > 0 else math.nan)
        mean_accept_length = (
            self.total_accept_length / self.num_requests
            if self.num_requests > 0
            else 0.0
        )
        return SpecAcceptMetricsScopeSnapshot(
            accept_rates=tuple(rates),
            mean_accept_length=mean_accept_length,
            num_requests=self.num_requests,
        )

    def reset(self) -> None:
        self.eligible.clear()
        self.accepted.clear()
        self.total_accept_length = 0
        self.num_requests = 0

    def _ensure_num_positions(self, num_positions: int) -> None:
        missing = num_positions - len(self.eligible)
        if missing > 0:
            self.eligible.extend([0] * missing)
            self.accepted.extend([0] * missing)


class SpecAcceptMetrics:
    """Track per-position acceptance over a decode window and process lifetime."""

    def __init__(self) -> None:
        self._window = _PerPositionCounters()
        self._lifetime = _PerPositionCounters()

    def observe(
        self,
        num_correct_drafts_per_req: Sequence[int],
        num_draft_tokens: int | None,
    ) -> None:
        if num_draft_tokens is None:
            return
        self._window.observe(num_correct_drafts_per_req, num_draft_tokens)
        self._lifetime.observe(num_correct_drafts_per_req, num_draft_tokens)

    def snapshot_and_reset_window(self) -> SpecAcceptMetricsSnapshot:
        # Pad the window to the lifetime width.  A NaN at a deeper position
        # explicitly means "not observed in this window" and prevents an old
        # Prometheus gauge value from looking current after the budget shrinks.
        width = max(len(self._window.eligible), len(self._lifetime.eligible))
        snapshot = SpecAcceptMetricsSnapshot(
            window=self._window.snapshot(width),
            lifetime=self._lifetime.snapshot(width),
        )
        self._window.reset()
        return snapshot

    def reset(self) -> None:
        self._window.reset()
        self._lifetime.reset()
