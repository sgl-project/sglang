"""Stateless single-pull snapshot validation."""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence

import msgspec

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


class RankSnapshot(msgspec.Struct, frozen=True):
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


class SnapshotValidationError(ValueError):
    """One pull's data violates the validation contract."""


class RankSetMismatchError(SnapshotValidationError):
    """Duplicate, missing, or unexpected ranks (retryable, unlike field errors)."""


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


def _snapshot_time_unix_ms(load: LoadSnapshot, fallback_time_unix_ms: int) -> int:
    """Return the scheduler timestamp or the fallback wall clock."""
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
    return fallback_time_unix_ms


def _rank_snapshot_from_load(
    load: LoadSnapshot, *, fallback_time_unix_ms: int
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
        snapshot_time_unix_ms=_snapshot_time_unix_ms(load, fallback_time_unix_ms),
        **counts,
        **floats,
    )


def _validate_expected_dp_ranks(expected_dp_ranks: Collection[int]) -> frozenset[int]:
    """Validate the expected DP-rank set elements."""
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
    return expected


def validate_full_snapshot(
    loads: Sequence[LoadSnapshot],
    *,
    expected_dp_ranks: Collection[int],
    fallback_time_unix_ms: int,
) -> tuple[RankSnapshot, ...]:
    """Validate one pull's snapshots and return a rank-sorted frozen tuple."""
    fallback_time_unix_ms = _require_non_negative_int64(
        "fallback_time_unix_ms", fallback_time_unix_ms
    )
    expected = _validate_expected_dp_ranks(expected_dp_ranks)

    candidates: dict[int, RankSnapshot] = {}
    for load in loads:
        candidate = _rank_snapshot_from_load(
            load, fallback_time_unix_ms=fallback_time_unix_ms
        )
        if candidate.dp_rank in candidates:
            raise RankSetMismatchError(
                f"duplicate dp_rank {candidate.dp_rank} in full snapshot"
            )
        candidates[candidate.dp_rank] = candidate

    actual = frozenset(candidates)
    if actual != expected:
        raise RankSetMismatchError(
            "incomplete rank set: "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    return tuple(candidates[rank] for rank in sorted(expected))
