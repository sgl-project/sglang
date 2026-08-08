"""Atomic latest-snapshot store."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Collection, Sequence
from typing import Optional

from sglang.srt.managers.load_snapshot import LoadSnapshot

# ---------------------------------------------------------------------------
# Integer range constants
# ---------------------------------------------------------------------------

_INT32_MIN = -(2**31)
_INT32_MAX = 2**31 - 1
_INT64_MAX = 2**63 - 1

# ---------------------------------------------------------------------------
# Validated field name tuples
# ---------------------------------------------------------------------------

_NON_NEGATIVE_INT64_FIELDS = (
    "num_running_reqs",
    "num_waiting_reqs",
    "num_waiting_uncached_tokens",
    "num_used_tokens",
    "num_total_tokens",
    "max_total_num_tokens",
    "max_running_requests",
)

_FINITE_FLOAT_FIELDS = (
    "token_usage",
    "gen_throughput",
    "cache_hit_rate",
    "utilization",
)


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class RankSnapshot:
    dp_rank: int
    snapshot_time_unix_ms: int
    num_running_reqs: int
    num_waiting_reqs: int
    num_waiting_uncached_tokens: int
    num_used_tokens: int
    num_total_tokens: int
    max_total_num_tokens: int
    max_running_requests: int
    token_usage: float
    gen_throughput: float
    cache_hit_rate: float
    utilization: float


@dataclasses.dataclass(frozen=True, slots=True)
class SnapshotView:
    ranks: tuple[RankSnapshot, ...]
    last_success_unix_ms: Optional[int]
    last_success_monotonic: Optional[float]
    last_error: Optional[str]

    @classmethod
    def empty(cls) -> SnapshotView:
        """Return the initial view before any successful sample."""
        return cls((), None, None, "no successful load sample")


class SnapshotValidationError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _require_non_negative_int64(field: str, value: object) -> int:
    """Validate one non-negative protobuf int64 field."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise SnapshotValidationError(f"{field} must be an integer")
    if value < 0 or value > _INT64_MAX:
        raise SnapshotValidationError(
            f"{field} must be in protobuf int64 range [0, {_INT64_MAX}]"
        )
    return value


def _require_finite_float(field: str, value: object) -> float:
    """Validate and normalize one finite floating-point field."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SnapshotValidationError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SnapshotValidationError(f"{field} must be finite")
    return result


def _snapshot_time_unix_ms(load: LoadSnapshot, collected_at_unix_ms: int) -> int:
    """Return the scheduler timestamp or the collection-time fallback."""
    timestamp = load.timestamp
    if (
        isinstance(timestamp, (int, float))
        and not isinstance(timestamp, bool)
        and math.isfinite(float(timestamp))
        and timestamp > 0
    ):
        timestamp_ms = int(float(timestamp) * 1000)
        if timestamp_ms > _INT64_MAX:
            raise SnapshotValidationError(
                "timestamp is outside protobuf int64 millisecond range"
            )
        return timestamp_ms
    return collected_at_unix_ms


def _rank_snapshot_from_load(
    load: LoadSnapshot, *, collected_at_unix_ms: int
) -> RankSnapshot:
    """Validate one scheduler snapshot and freeze its core metrics."""
    dp_rank = load.dp_rank
    if isinstance(dp_rank, bool) or not isinstance(dp_rank, int):
        raise SnapshotValidationError("dp_rank must be an integer")
    if dp_rank < _INT32_MIN or dp_rank > _INT32_MAX:
        raise SnapshotValidationError("dp_rank is outside protobuf int32 range")

    counts = {
        field: _require_non_negative_int64(field, getattr(load, field))
        for field in _NON_NEGATIVE_INT64_FIELDS
    }
    if counts["num_used_tokens"] > counts["max_total_num_tokens"]:
        raise SnapshotValidationError(
            "num_used_tokens must not exceed max_total_num_tokens"
        )
    if counts["num_running_reqs"] > counts["max_running_requests"]:
        raise SnapshotValidationError(
            "num_running_reqs must not exceed max_running_requests"
        )

    floats = {
        field: _require_finite_float(field, getattr(load, field))
        for field in _FINITE_FLOAT_FIELDS
    }
    return RankSnapshot(
        dp_rank=dp_rank,
        snapshot_time_unix_ms=_snapshot_time_unix_ms(load, collected_at_unix_ms),
        **counts,
        **floats,
    )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class LatestSnapshotStore:
    def __init__(self) -> None:
        """Initialize the store with an unreachable empty view."""
        self._view = SnapshotView.empty()

    def view(self) -> SnapshotView:
        """Return the current immutable snapshot view."""
        # SnapshotView/RankSnapshot/tuple are immutable, so the same reference is safe.
        return self._view

    def apply_full_snapshot(
        self,
        loads: Sequence[LoadSnapshot],
        *,
        expected_dp_ranks: Collection[int],
        collected_at_unix_ms: int,
        collected_at_monotonic: float,
    ) -> SnapshotView:
        """Validate and atomically publish one authoritative full snapshot."""
        collected_at_unix_ms = _require_non_negative_int64(
            "collected_at_unix_ms", collected_at_unix_ms
        )
        collected_at_monotonic = _require_finite_float(
            "collected_at_monotonic", collected_at_monotonic
        )
        if collected_at_monotonic < 0:
            raise SnapshotValidationError(
                "collected_at_monotonic must be finite and non-negative"
            )

        expected = frozenset(expected_dp_ranks)
        for dp_rank in expected:
            if isinstance(dp_rank, bool) or not isinstance(dp_rank, int):
                raise SnapshotValidationError(
                    "expected_dp_ranks must contain only integers"
                )
            if dp_rank < _INT32_MIN or dp_rank > _INT32_MAX:
                raise SnapshotValidationError(
                    "expected dp_rank is outside protobuf int32 range"
                )

        # All operations below modify local variables only; self._view is
        # replaced after every field and rank has validated successfully.
        candidates: dict[int, RankSnapshot] = {}
        for load in loads:
            candidate = _rank_snapshot_from_load(
                load, collected_at_unix_ms=collected_at_unix_ms
            )
            if candidate.dp_rank in candidates:
                raise SnapshotValidationError(
                    f"duplicate dp_rank {candidate.dp_rank} in full snapshot"
                )
            candidates[candidate.dp_rank] = candidate

        actual = frozenset(candidates)
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise SnapshotValidationError(
                f"incomplete rank set: missing={missing}, unexpected={unexpected}"
            )

        previous_by_rank = {rank.dp_rank: rank for rank in self._view.ranks}
        merged: list[RankSnapshot] = []
        for dp_rank in sorted(expected):
            incoming = candidates[dp_rank]
            previous = previous_by_rank.get(dp_rank)
            if (
                previous is not None
                and previous.snapshot_time_unix_ms > incoming.snapshot_time_unix_ms
            ):
                # Prevent an older sample from overwriting an already-published value.
                merged.append(previous)
            else:
                # Equal timestamp: use raw metrics from this full sample.
                merged.append(incoming)

        new_view = SnapshotView(
            ranks=tuple(merged),
            last_success_unix_ms=collected_at_unix_ms,
            last_success_monotonic=collected_at_monotonic,
            last_error=None,
        )
        self._view = new_view
        return new_view

    def record_error(self, error: BaseException | str) -> SnapshotView:
        """Publish a sampling error while preserving the last good ranks."""
        message = str(error).strip()
        if not message:
            message = (
                type(error).__name__
                if isinstance(error, BaseException)
                else "unknown load sampling error"
            )
        current = self._view
        new_view = SnapshotView(
            ranks=current.ranks,
            last_success_unix_ms=current.last_success_unix_ms,
            last_success_monotonic=current.last_success_monotonic,
            last_error=message,
        )
        self._view = new_view
        return new_view
