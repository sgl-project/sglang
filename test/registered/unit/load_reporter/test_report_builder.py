"""Unit tests for stateless LoadReport construction."""

from __future__ import annotations

import sys

import pytest

from sglang.srt.load_reporter.config import WorkerMetadata
from sglang.srt.load_reporter.proto import load_monitor_pb2 as pb
from sglang.srt.load_reporter.report_builder import ReportBuilder, SequenceAllocator
from sglang.srt.load_reporter.snapshot_validation import validate_full_snapshot
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


def make_identity(model: str | None = "test-model") -> WorkerMetadata:
    return WorkerMetadata(
        worker_addr="127.0.0.1:9999",
        worker_type=pb.WORKER_TYPE_REGULAR,
        model=model,
    )


def make_builder(stale_after_ms: int = 3000) -> ReportBuilder:
    return ReportBuilder("test-instance", stale_after_ms, SequenceAllocator())


def ranks(*overrides_list):
    """Validate one snapshot per override dict and return the rank tuple."""
    return validate_full_snapshot(
        [make_snapshot(**overrides) for overrides in overrides_list],
        expected_dp_ranks={o.get("dp_rank", 0) for o in overrides_list},
        fallback_time_unix_ms=20_000,
    )


# ---------------------------------------------------------------------------
# Build success path
# ---------------------------------------------------------------------------


class TestBuild:
    def test_build_uses_only_supplied_ranks(self):
        builder = make_builder()
        first = builder.build(
            ranks(dict(timestamp=20.0, num_running_reqs=5)),
            make_identity(),
            report_time_unix_ms=20_100,
        )
        second = builder.build(
            ranks(dict(timestamp=10.0, num_running_reqs=2)),
            make_identity(),
            report_time_unix_ms=20_200,
        )
        assert first.status == pb.REPORT_STATUS_HEALTHY
        assert first.ranks[0].num_running_reqs == 5
        assert first.ranks[0].snapshot_time_unix_ms == 20_000
        assert second.ranks[0].num_running_reqs == 2
        assert second.ranks[0].snapshot_time_unix_ms == 10_000
        assert second.status == pb.REPORT_STATUS_STALE
        assert second.sequence_id == first.sequence_id + 1

    def test_stale_boundary_is_strict(self):
        builder = make_builder(stale_after_ms=1000)
        # Age exactly at the threshold is healthy; one millisecond over is stale.
        at_boundary = builder.build(
            ranks(dict(timestamp=19.0)),
            make_identity(),
            report_time_unix_ms=20_000,
        )
        over_boundary = builder.build(
            ranks(dict(timestamp=18.999)),
            make_identity(),
            report_time_unix_ms=20_000,
        )
        assert at_boundary.status == pb.REPORT_STATUS_HEALTHY
        assert over_boundary.status == pb.REPORT_STATUS_STALE
        assert "stale by" in over_boundary.last_error

    def test_worker_metadata_propagates(self):
        report = make_builder().build(
            ranks(dict(timestamp=20.0)),
            make_identity(model="meta-model"),
            report_time_unix_ms=20_100,
        )
        assert report.worker.worker_addr == "127.0.0.1:9999"
        assert report.worker.worker_type == pb.WORKER_TYPE_REGULAR
        assert report.worker.model == "meta-model"

    def test_model_absent_omits_field(self):
        report = make_builder().build(
            ranks(dict(timestamp=20.0)),
            make_identity(model=None),
            report_time_unix_ms=20_100,
        )
        assert not report.worker.HasField("model")

    def test_sequence_ids_start_positive_and_are_monotonic(self):
        builder = make_builder()
        first = builder.build(
            ranks(dict(timestamp=20.0)), make_identity(), report_time_unix_ms=20_100
        )
        second = builder.build_unreachable(
            make_identity(), report_time_unix_ms=20_200, error="pull failed"
        )
        assert first.sequence_id >= 1
        assert second.sequence_id == first.sequence_id + 1

    def test_empty_ranks_are_rejected(self):
        with pytest.raises(ValueError):
            make_builder().build((), make_identity(), report_time_unix_ms=20_100)


# ---------------------------------------------------------------------------
# Unreachable path
# ---------------------------------------------------------------------------


class TestBuildUnreachable:
    def test_unreachable_has_no_last_good_ranks(self):
        report = make_builder().build_unreachable(
            make_identity(),
            report_time_unix_ms=30_000,
            error="snapshot pull timed out",
        )
        assert report.status == pb.REPORT_STATUS_UNREACHABLE
        assert not report.ranks
        assert report.last_error == "snapshot pull timed out"

    def test_empty_error_is_normalized(self):
        report = make_builder().build_unreachable(
            make_identity(), report_time_unix_ms=30_000, error="  "
        )
        assert report.last_error == "load snapshot unavailable"

    def test_exception_error_uses_its_message(self):
        report = make_builder().build_unreachable(
            make_identity(),
            report_time_unix_ms=30_000,
            error=RuntimeError("read failed"),
        )
        assert report.last_error == "read failed"

    def test_exception_with_empty_message_is_normalized(self):
        report = make_builder().build_unreachable(
            make_identity(), report_time_unix_ms=30_000, error=RuntimeError("")
        )
        assert report.last_error == "load snapshot unavailable"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
