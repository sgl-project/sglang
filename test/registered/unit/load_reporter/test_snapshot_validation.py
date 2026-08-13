"""Unit tests for stateless full-snapshot validation."""

from __future__ import annotations

import sys

import pytest

from sglang.srt.load_reporter.snapshot_validation import (
    RankSetMismatchError,
    SnapshotValidationError,
    validate_full_snapshot,
)
from sglang.srt.managers.load_snapshot import LoadSnapshot
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_snapshot(**overrides) -> LoadSnapshot:
    """Build a LoadSnapshot with valid defaults for every validated field."""
    fields = dict(
        timestamp=10.0,
        dp_rank=0,
        num_running_reqs=1,
        num_waiting_reqs=0,
        num_waiting_uncached_tokens=0,
        num_used_tokens=5,
        num_total_tokens=5,
        max_total_num_tokens=100,
        max_running_requests=8,
        token_usage=0.5,
        gen_throughput=1.0,
        cache_hit_rate=0.9,
        utilization=0.5,
    )
    fields.update(overrides)
    return LoadSnapshot(**fields)


def validate(loads, expected_dp_ranks, fallback_time_unix_ms=30_000):
    return validate_full_snapshot(
        loads,
        expected_dp_ranks=expected_dp_ranks,
        fallback_time_unix_ms=fallback_time_unix_ms,
    )


# ---------------------------------------------------------------------------
# Successful validation
# ---------------------------------------------------------------------------


class TestSuccessfulValidation:
    def test_returns_sorted_complete_ranks(self):
        ranks = validate(
            [
                make_snapshot(dp_rank=1, timestamp=20.0),
                make_snapshot(dp_rank=0, timestamp=10.0),
            ],
            {0, 1},
        )
        assert [rank.dp_rank for rank in ranks] == [0, 1]
        assert [rank.snapshot_time_unix_ms for rank in ranks] == [10_000, 20_000]

    def test_all_metrics_are_frozen(self):
        (rank,) = validate([make_snapshot(num_used_tokens=5)], {0})
        assert rank.num_running_reqs == 1
        assert rank.num_used_tokens == 5
        assert rank.num_total_tokens == 5
        assert rank.max_total_num_tokens == 100
        assert rank.max_running_requests == 8
        assert rank.token_usage == 0.5
        assert rank.gen_throughput == 1.0
        assert rank.cache_hit_rate == 0.9
        assert rank.utilization == 0.5

    def test_timestamp_regression_is_returned_without_substitution(self):
        first = validate(
            [make_snapshot(dp_rank=0, timestamp=20.0, num_running_reqs=5)],
            {0},
            fallback_time_unix_ms=21_000,
        )
        second = validate(
            [make_snapshot(dp_rank=0, timestamp=10.0, num_running_reqs=2)],
            {0},
            fallback_time_unix_ms=22_000,
        )
        assert first[0].snapshot_time_unix_ms == 20_000
        assert second[0].snapshot_time_unix_ms == 10_000
        assert second[0].num_running_reqs == 2

    def test_absent_timestamp_uses_fallback(self):
        (rank,) = validate(
            [make_snapshot(timestamp=0.0)], {0}, fallback_time_unix_ms=42_000
        )
        assert rank.snapshot_time_unix_ms == 42_000

    def test_negative_timestamp_uses_fallback(self):
        (rank,) = validate(
            [make_snapshot(timestamp=-1.0)], {0}, fallback_time_unix_ms=42_000
        )
        assert rank.snapshot_time_unix_ms == 42_000

    def test_empty_expected_ranks_accepts_empty_loads(self):
        # The builder, not the validator, rejects a report with no ranks.
        assert validate([], frozenset()) == ()


# ---------------------------------------------------------------------------
# Rank-set mismatch (retryable)
# ---------------------------------------------------------------------------


class TestRankSetMismatch:
    def test_duplicate_rank_raises(self):
        with pytest.raises(RankSetMismatchError):
            validate([make_snapshot(dp_rank=0), make_snapshot(dp_rank=0)], {0})

    def test_missing_rank_raises(self):
        with pytest.raises(RankSetMismatchError):
            validate([make_snapshot(dp_rank=0)], {0, 1})

    def test_unexpected_rank_raises(self):
        with pytest.raises(RankSetMismatchError):
            validate([make_snapshot(dp_rank=1)], {0})

    def test_mismatch_never_returns_partial_tuple(self):
        with pytest.raises(RankSetMismatchError):
            validate(
                [make_snapshot(dp_rank=0, num_running_reqs=1)],
                {0, 1},
            )

    def test_mismatch_is_distinguishable_from_field_errors(self):
        with pytest.raises(SnapshotValidationError) as exc_info:
            validate(
                [make_snapshot(dp_rank=0, num_running_reqs=-1)],
                {0},
            )
        assert not isinstance(exc_info.value, RankSetMismatchError)


# ---------------------------------------------------------------------------
# Invalid field values (not retryable)
# ---------------------------------------------------------------------------


class TestInvalidFields:
    @pytest.mark.parametrize(
        ("overrides", "message_part"),
        [
            (dict(num_running_reqs=True), "must be an integer"),
            (dict(num_running_reqs=-1), "int64 range"),
            (dict(num_running_reqs=1.5), "must be an integer"),
            (dict(max_total_num_tokens=2**63), "int64 range"),
        ],
    )
    def test_invalid_int64_field_raises(self, overrides, message_part):
        with pytest.raises(SnapshotValidationError) as exc_info:
            validate([make_snapshot(**overrides)], {0})
        assert message_part in str(exc_info.value)
        assert not isinstance(exc_info.value, RankSetMismatchError)

    @pytest.mark.parametrize(
        "overrides",
        [
            dict(token_usage=True),
            dict(token_usage="0.5"),
            dict(token_usage=float("inf")),
            dict(gen_throughput=float("nan")),
        ],
    )
    def test_invalid_float_field_raises(self, overrides):
        with pytest.raises(SnapshotValidationError) as exc_info:
            validate([make_snapshot(**overrides)], {0})
        assert not isinstance(exc_info.value, RankSetMismatchError)

    def test_used_tokens_exceeding_capacity_raises(self):
        with pytest.raises(SnapshotValidationError):
            validate(
                [make_snapshot(num_used_tokens=101, max_total_num_tokens=100)],
                {0},
            )

    def test_running_reqs_exceeding_capacity_raises(self):
        with pytest.raises(SnapshotValidationError):
            validate(
                [make_snapshot(num_running_reqs=9, max_running_requests=8)],
                {0},
            )

    def test_oversized_timestamp_raises(self):
        with pytest.raises(SnapshotValidationError):
            validate([make_snapshot(timestamp=2**63)], {0})

    def test_invalid_fallback_timestamp_raises(self):
        with pytest.raises(SnapshotValidationError):
            validate([make_snapshot()], {0}, fallback_time_unix_ms=-1)

    @pytest.mark.parametrize("bad_rank", [True, 1.5, 2**31])
    def test_invalid_expected_rank_raises(self, bad_rank):
        with pytest.raises(SnapshotValidationError):
            validate([make_snapshot(dp_rank=bad_rank)], {bad_rank})


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
