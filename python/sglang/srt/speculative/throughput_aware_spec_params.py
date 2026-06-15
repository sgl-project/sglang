"""Throughput-aware speculative decoding: per-position acceptance tracking and cost table.

Two components:
  - PositionAcceptanceTracker  -- per-position sliding-window accept-rate tracker,
                                   shared across all batch sizes.
  - BatchSizeCostTable         -- runtime-profiled cost table indexed by
                                   (batch_size, num_steps) only (no seqlen dimension).
"""

from __future__ import annotations

import bisect
import logging
import math
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


class PositionAcceptanceTracker:
    """Per-position acceptance-rate tracker using a sliding window.

    Position k is accepted when ``num_correct_drafts > k``.  Each position
    keeps a circular buffer of size ``window_size``; the mean is used as its
    rate.  Positions with no data are extrapolated geometrically from known
    positions.  Rates are shared across all batch sizes.
    """

    def __init__(
        self,
        max_steps: int,
        window_size: int = 20,
    ):
        self.max_steps = max_steps
        self.window_size = window_size

        self._windows: list[deque] = [
            deque(maxlen=window_size) for _ in range(max_steps)
        ]

    def update(self, accept_lengths: list[int], current_steps: int) -> None:
        """Update per-position rates from one verify batch (positions 0..current_steps-1).

        Single pass over accept_lengths: histogram + suffix counts give
        p[k] = (#reqs with a > k) / n for all k in O(bs + current_steps).
        """
        n = len(accept_lengths)
        if n == 0:
            return

        s = min(current_steps, self.max_steps)
        if s <= 0:
            return

        # counts[c] = #reqs whose accepted-position count is exactly c (capped at s)
        counts = [0] * (s + 1)
        for a in accept_lengths:
            counts[a if a < s else s] += 1

        # p[k] = (n - sum(counts[0..k])) / n  ==  fraction with a > k
        cum = 0
        for k in range(s):
            cum += counts[k]
            self._windows[k].append((n - cum) / n)

    def clear_positions_above(self, steps: int) -> None:
        """Clear buffers for positions >= *steps* (called on step-count decrease)."""
        for k in range(steps, self.max_steps):
            self._windows[k].clear()

    def all_positions_warmed(self, target_steps: int) -> bool:
        """Return True when every position in [0, target_steps) has a full window."""
        for k in range(min(target_steps, self.max_steps)):
            if len(self._windows[k]) < self.window_size:
                return False
        return True

    def get_expected_tokens(self, target_steps: int) -> Optional[float]:
        """E[tokens produced | target_steps drafted] = 1 + sum(p[k] for k in range(S)).

        Returns None at cold start (no data to extrapolate from).
        """
        if target_steps <= 0:
            return None

        rates: list[float] = []
        for k in range(target_steps):
            rate = self._get_rate_or_extrapolate(k, rates)
            if rate is None:
                return None
            rates.append(rate)

        return 1.0 + sum(rates)

    def snapshot_position_rates(self, num_positions: int) -> list[Optional[float]]:
        """Per-position rates (window mean or extrapolated) for logging."""
        if num_positions <= 0:
            return []
        known: list[float] = []
        out: list[Optional[float]] = []
        for k in range(num_positions):
            rate = self._get_rate_or_extrapolate(k, known)
            out.append(rate)
            if rate is not None:
                known.append(rate)
        return out

    def is_position_extrapolated(self, k: int) -> bool:
        """Return True if position k has no real data (its rate would be extrapolated)."""
        return not bool(self._windows[k])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _best_rate(self, k: int) -> Optional[float]:
        w = self._windows[k]
        return sum(w) / len(w) if w else None

    def _get_rate_or_extrapolate(
        self, k: int, known_rates: list[float]
    ) -> Optional[float]:
        """Window mean if available; else geometric extrapolation from known_rates."""
        real = self._best_rate(k)
        if real is not None:
            return real
        if not known_rates:
            return None
        if len(known_rates) == 1:
            return known_rates[0]
        # p[i] = p[0] * alpha^i  ⟹  alpha = (p[N-1] / p[0])^(1/(N-1))
        p0, pn = known_rates[0], known_rates[-1]
        if p0 <= 0:
            return None
        alpha = (pn / p0) ** (1.0 / (len(known_rates) - 1))
        return known_rates[-1] * alpha


class BatchSizeCostTable:
    """(batch_size, num_steps) -> decode cycle latency (ms). BS lookup uses ceiling."""

    def __init__(self) -> None:
        self._data: dict[tuple[int, int], float] = {}
        self._batch_sizes: list[int] = []  # sorted profiled batch sizes

    def set(self, batch_size: int, num_steps: int, cost_ms: float) -> None:
        """Record a measured cost entry."""
        self._data[(batch_size, num_steps)] = cost_ms
        if batch_size not in self._batch_sizes:
            self._batch_sizes = sorted(set(self._batch_sizes) | {batch_size})

    def lookup(self, batch_size: int, num_steps: int) -> Optional[float]:
        """Return cost in ms, or ``None`` if no entry exists for *num_steps*."""
        if not self._data:
            return None

        # Ceiling batch_size: first profiled bs >= requested
        idx = bisect.bisect_left(self._batch_sizes, batch_size)
        if idx >= len(self._batch_sizes):
            idx = len(self._batch_sizes) - 1
        nearest_bs = self._batch_sizes[idx]

        return self._data.get((nearest_bs, num_steps))

    def is_empty(self) -> bool:
        return len(self._data) == 0

    def summary(self) -> str:
        """Return a human-readable summary of the table for logging."""
        if self.is_empty():
            return "{}"
        lines = []
        for (bs, steps), cost in sorted(self._data.items()):
            lines.append(f"(bs={bs}, steps={steps}): {cost:.2f}ms")
        return "{" + ", ".join(lines) + "}"


def score_candidates(
    tracker: PositionAcceptanceTracker,
    cost_table: BatchSizeCostTable,
    candidate_steps: list[int],
    batch_size: int,
) -> list[dict]:
    """Return per-candidate dicts with keys: steps, expected, cost_ms, score."""
    rows = []
    for steps in candidate_steps:
        expected = tracker.get_expected_tokens(steps)
        cost_ms = cost_table.lookup(batch_size, steps)
        score: Optional[float] = None
        if expected is not None and cost_ms is not None and cost_ms > 0:
            score = expected / cost_ms
        rows.append(
            {
                "steps": steps,
                "expected": expected,
                "cost_ms": cost_ms,
                "score": score,
            }
        )
    return rows


def pick_best_step(rows: list[dict], fallback: int) -> int:
    """Return the step with the highest score, or fallback if no valid score exists."""
    best_step = fallback
    best_score = -math.inf
    for row in rows:
        s = row["score"]
        if s is not None and s > best_score:
            best_score = s
            best_step = row["steps"]
    return best_step


def format_position_rates(
    tracker: PositionAcceptanceTracker, num_positions: int
) -> str:
    """Format per-position rates for logging, marking extrapolated ones."""
    rates = tracker.snapshot_position_rates(num_positions)
    parts = []
    for k, rate in enumerate(rates):
        if rate is None:
            parts.append(f"p[{k}]=?")
        elif tracker.is_position_extrapolated(k):
            parts.append(f"p[{k}]={rate:.3f}*")  # * = extrapolated
        else:
            parts.append(f"p[{k}]={rate:.3f}")
    return "[" + ", ".join(parts) + "]"


def format_score_rows(rows: list[dict], best_steps: int) -> str:
    """Format score breakdown for logging."""
    parts = []
    for row in rows:
        s = row["steps"]
        expected = row["expected"]
        cost_ms = row["cost_ms"]
        score = row["score"]
        marker = "★" if s == best_steps else ""
        if score is not None:
            parts.append(
                f"S={s}:E={expected:.2f}/cost={cost_ms:.1f}ms→{score:.4f}{marker}"
            )
        elif expected is not None:
            parts.append(f"S={s}:E={expected:.2f}/cost=?{marker}")
        else:
            parts.append(f"S={s}:?{marker}")
    return "[" + ", ".join(parts) + "]"
