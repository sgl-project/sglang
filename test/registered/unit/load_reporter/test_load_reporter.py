"""Consolidated CPU tests for the load reporter subsystem."""

from __future__ import annotations

import argparse
import ast
import asyncio
import re
import socket
import sys
import time
import types
from pathlib import Path
from typing import Any, AsyncIterator, List, Optional

import grpc
import grpc.aio
import pytest

from sglang.srt.load_reporter.config import WorkerMetadata
from sglang.srt.load_reporter.lifecycle import LoadReporterHandle, start_load_reporter
from sglang.srt.load_reporter.proto import load_monitor_pb2 as pb
from sglang.srt.load_reporter.proto import load_monitor_pb2_grpc as pb_grpc
from sglang.srt.load_reporter.report_builder import ReportBuilder
from sglang.srt.load_reporter.runtime import LoadReporterRuntime, _PeriodSchedule
from sglang.srt.load_reporter.service import add_service_to_server
from sglang.srt.load_reporter.snapshot_source import ManagerLoadSnapshotSource
from sglang.srt.load_reporter.snapshot_validation import (
    RankSetMismatchError,
    SnapshotValidationError,
    validate_full_snapshot,
)
from sglang.srt.managers.load_snapshot import LoadSnapshot
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=75, suite="base-a-test-cpu")

pytest_plugins = ("pytest_asyncio",)

# ============================================================================
# Protocol contract
# ============================================================================

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PROTO_PACKAGE = _REPO_ROOT / "python/sglang/srt/load_reporter/proto"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _service():
    return pb.DESCRIPTOR.services_by_name["LoadMonitorService"]


def _message(name):
    return pb.DESCRIPTOR.message_types_by_name[name]


def _oneof_fields_by_number(msg_name, oneof_name):
    """Return {field_number: field_name} for the named oneof in msg_name."""
    msg = _message(msg_name)
    oneof = msg.oneofs_by_name[oneof_name]
    return {f.number: f.name for f in oneof.fields}


def _fields_by_number(msg_name):
    return {f.number: f.name for f in _message(msg_name).fields}


class TestGeneratedRuntimeCompatibility:
    def test_protobuf_gencode_targets_declared_minimum(self):
        source = (_PROTO_PACKAGE / "load_monitor_pb2.py").read_text()
        match = re.search(r"^# Protobuf Python Version: (\S+)$", source, re.MULTILINE)

        assert match is not None
        assert match.group(1) == "6.31.1"

    def test_grpc_gencode_targets_declared_minimum(self):
        source = (_PROTO_PACKAGE / "load_monitor_pb2_grpc.py").read_text()
        module = ast.parse(source)
        generated_version = next(
            node.value.value
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "GRPC_GENERATED_VERSION"
                for target in node.targets
            )
            and isinstance(node.value, ast.Constant)
        )

        assert generated_version == "1.78.0"


# ---------------------------------------------------------------------------
# Service contract
# ---------------------------------------------------------------------------


class TestServiceDescriptor:
    def test_monitor_is_the_only_bidirectional_method(self):
        service = _service()
        assert service.full_name == "sglang.router.loadmonitor.v1.LoadMonitorService"
        assert list(service.methods_by_name) == ["Monitor"]

        method = service.methods_by_name["Monitor"]
        assert method.client_streaming is True
        assert method.server_streaming is True


# ---------------------------------------------------------------------------
# RouterFrame oneof
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("message_name", "expected_fields"),
    [
        (
            "RouterFrame",
            {1: "register", 2: "update_config", 3: "keep_alive", 4: "stop"},
        ),
        ("WorkerFrame", {1: "registered", 2: "report", 3: "error"}),
    ],
)
def test_frame_payload_contract(message_name, expected_fields):
    assert _oneof_fields_by_number(message_name, "payload") == expected_fields


# ---------------------------------------------------------------------------
# Preserved LoadReport field numbers
# ---------------------------------------------------------------------------


class TestLoadReportFieldNumbers:
    def test_preserved_field_numbers(self):
        fields = _fields_by_number("LoadReport")
        assert {number: fields[number] for number in (1, 2, 3, 4, 5, 7)} == {
            1: "source_instance_id",
            2: "sequence_id",
            3: "report_time_unix_ms",
            4: "worker",
            5: "status",
            7: "ranks",
        }


# ============================================================================
# Server arguments
# ============================================================================


def _parse(args: list[str]):
    """Parse args through ServerArgs CLI into a ServerArgs instance."""
    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    namespace = parser.parse_args(["--model-path", "dummy"] + args)
    return ServerArgs.from_cli_args(namespace)


class TestLoadReporterPortDefault:
    def test_none_means_disabled(self):
        """None is the only disabled sentinel; no extra enable flag required."""
        sa = _parse([])
        assert sa.load_reporter_port is None


class TestLoadReporterPortParsing:
    def test_valid_port(self):
        sa = _parse(["--load-reporter-port", "30100"])
        assert sa.load_reporter_port == 30100


class TestLoadReporterPortValidation:
    @pytest.mark.parametrize("port", [0, 65536])
    def test_invalid_port_is_rejected(self, port):
        with pytest.raises((ValueError, SystemExit)):
            _parse(["--load-reporter-port", str(port)])


# ============================================================================
# Snapshot validation
# ============================================================================

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

    def test_non_positive_timestamp_uses_fallback(self):
        (rank,) = validate(
            [make_snapshot(timestamp=0.0)], {0}, fallback_time_unix_ms=42_000
        )
        assert rank.snapshot_time_unix_ms == 42_000


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
            dict(token_usage="0.5"),
            dict(token_usage=float("inf")),
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

    @pytest.mark.parametrize("bad_rank", [True, 2**31])
    def test_invalid_expected_rank_raises(self, bad_rank):
        with pytest.raises(SnapshotValidationError):
            validate([make_snapshot(dp_rank=bad_rank)], {bad_rank})


# ============================================================================
# Report construction
# ============================================================================


def make_identity(model: str | None = "test-model") -> WorkerMetadata:
    return WorkerMetadata(
        worker_addr="127.0.0.1:9999",
        worker_type=pb.WORKER_TYPE_REGULAR,
        model=model,
    )


def make_builder(stale_after_ms: int = 3000) -> ReportBuilder:
    return ReportBuilder("test-instance", stale_after_ms)


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

    def test_empty_errors_are_normalized(self):
        for error in ("  ", RuntimeError("")):
            report = make_builder().build_unreachable(
                make_identity(), report_time_unix_ms=30_000, error=error
            )
            assert report.last_error == "load snapshot unavailable"

    def test_exception_error_uses_its_message(self):
        report = make_builder().build_unreachable(
            make_identity(),
            report_time_unix_ms=30_000,
            error=RuntimeError("read failed"),
        )
        assert report.last_error == "read failed"


# ============================================================================
# Snapshot source
# ============================================================================


class TestManagerLoadSnapshotSource:
    @pytest.mark.asyncio
    async def test_manager_source_requests_core_loads(self):
        snapshots = [make_load_snapshot()]

        class Manager:
            elastic_worker_count = 1

            async def get_loads(self, *, include):
                self.include = include
                return snapshots

        manager = Manager()
        source = ManagerLoadSnapshotSource(manager, {0})

        assert await source.get_loads() == snapshots
        assert manager.include == ["core"]

    def test_manager_source_tracks_elastic_worker_count(self):
        class Manager:
            elastic_worker_count = 1

        manager = Manager()
        source = ManagerLoadSnapshotSource(manager, {0})

        assert source.expected_dp_ranks() == frozenset({0})

        manager.elastic_worker_count = 3
        assert source.expected_dp_ranks() == frozenset({0, 1, 2})

        manager.elastic_worker_count = 2
        assert source.expected_dp_ranks() == frozenset({0, 1})


# ============================================================================
# Runtime sessions
# ============================================================================

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_server_args(
    *, load_reporter_port: Optional[int] = 9999, dp_size: int = 1, **overrides
) -> types.SimpleNamespace:
    values = {
        "host": "127.0.0.1",
        "load_reporter_port": load_reporter_port,
        "disaggregation_mode": "none",
        "served_model_name": "test-model",
        "dp_size": dp_size,
        "tokenizer_worker_num": 1,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
    finally:
        sock.close()


def make_load_snapshot(num_running_reqs: int = 1, dp_rank: int = 0):
    from sglang.srt.managers.load_snapshot import LoadSnapshot

    return LoadSnapshot(
        timestamp=time.time(),
        dp_rank=dp_rank,
        num_running_reqs=num_running_reqs,
        max_running_requests=8,
        max_total_num_tokens=1024,
    )


class SnapshotSource:
    """Return one valid snapshot per expected rank and count pulls."""

    def __init__(self, dp_size: int = 1) -> None:
        self._dp_size = dp_size
        self.get_loads_calls = 0

    async def get_loads(self) -> list:
        self.get_loads_calls += 1
        return [make_load_snapshot(1, dp_rank=rank) for rank in range(self._dp_size)]

    def expected_dp_ranks(self) -> frozenset:
        return frozenset(range(self._dp_size))


class HangingSnapshotSource:
    """Snapshot source whose in-flight read only ends when cancelled."""

    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def get_loads(self) -> list:
        self.started.set()
        await asyncio.Future()
        raise AssertionError("unreachable")

    def expected_dp_ranks(self) -> frozenset:
        return frozenset({0})


class ControlledSnapshotSource:
    """Return one valid snapshot only after the test releases it."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def get_loads(self) -> list:
        self.started.set()
        await self.release.wait()
        return [make_load_snapshot(1)]

    def expected_dp_ranks(self) -> frozenset:
        return frozenset({0})


class MutableSnapshotSource:
    """Return one valid snapshot per expected rank, tracked by the rank set."""

    def __init__(self) -> None:
        self.num_running_reqs = 1
        self.get_loads_calls = 0
        self._dp_size = 1
        self._rank_updates: list = []

    async def get_loads(self) -> list:
        self.get_loads_calls += 1
        return [
            make_load_snapshot(self.num_running_reqs, dp_rank=rank)
            for rank in range(self._dp_size)
        ]

    def expected_dp_ranks(self) -> frozenset:
        return frozenset(range(self._dp_size))

    def update_expected_dp_ranks(self, ranks) -> bool:
        self._rank_updates.append(frozenset(ranks))
        new = frozenset(ranks)
        changed = new != frozenset(range(self._dp_size))
        self._dp_size = len(list(ranks))
        return changed


class ScriptedSnapshotSource:
    """Return one scripted result (or raise it) per get_loads call; last step repeats."""

    def __init__(self, script: list, dp_size: int = 1) -> None:
        self._script = list(script)
        self.get_loads_calls = 0
        self._dp_size = dp_size

    async def get_loads(self) -> list:
        self.get_loads_calls += 1
        step = self._script[min(self.get_loads_calls, len(self._script)) - 1]
        if isinstance(step, BaseException):
            raise step
        return step

    def expected_dp_ranks(self) -> frozenset:
        return frozenset(range(self._dp_size))


class DelayedScriptedSnapshotSource:
    """Return one scripted result (or raise it) after a fixed delay per call."""

    def __init__(self, script: list, delay: float, dp_size: int = 1) -> None:
        self._script = list(script)
        self._delay = delay
        self.get_loads_calls = 0
        self._dp_size = dp_size

    async def get_loads(self) -> list:
        self.get_loads_calls += 1
        await asyncio.sleep(self._delay)
        step = self._script[min(self.get_loads_calls, len(self._script)) - 1]
        if isinstance(step, BaseException):
            raise step
        return step

    def expected_dp_ranks(self) -> frozenset:
        return frozenset(range(self._dp_size))


async def drain_queue(q: asyncio.Queue, count: int, timeout: float = 2.0) -> list:
    """Drain up to count non-None items from q within timeout seconds."""
    items = []
    deadline = time.monotonic() + timeout
    while len(items) < count:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            item = await asyncio.wait_for(q.get(), timeout=remaining)
            if item is None:
                break
            items.append(item)
        except asyncio.TimeoutError:
            break
    return items


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegisterSession:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("router_id", "report_interval_ms", "lease_ttl_ms", "error"),
        [
            ("", 500, 3000, "router_id"),
            ("r1", 0, 3000, "report_interval_ms"),
            ("r1", 500, 0, "lease_ttl_ms"),
        ],
    )
    async def test_register_rejects_invalid_session_config(
        self, router_id, report_interval_ms, lease_ttl_ms, error
    ):

        rt = LoadReporterRuntime(SnapshotSource(), make_server_args())
        try:
            with pytest.raises(ValueError, match=error):
                rt.register_session(router_id, report_interval_ms, lease_ttl_ms)
        finally:
            await rt.close()

    @pytest.mark.asyncio
    async def test_register_returns_ack(self):

        rt = LoadReporterRuntime(SnapshotSource(), make_server_args())
        try:
            ack, session = rt.register_session("r1", 500, 3000)
            assert ack.lease_ttl_ms == 3000
            assert ack.renew_after_ms == max(1, 3000 // 3)
            assert ack.renew_after_ms == 1000
        finally:
            session.stop()
            await rt.close()

    @pytest.mark.asyncio
    async def test_initial_report_waits_for_pull(self):

        source = ControlledSnapshotSource()
        rt = LoadReporterRuntime(source, make_server_args())
        try:
            _, session = rt.register_session("r1", 10_000, 30_000)
            await asyncio.wait_for(source.started.wait(), timeout=0.5)

            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(session.queue.get(), timeout=0.05)

            source.release.set()
            report = await asyncio.wait_for(session.queue.get(), timeout=0.5)

            assert report.status == pb.REPORT_STATUS_HEALTHY
            assert [rank.dp_rank for rank in report.ranks] == [0]
            assert report.ranks[0].num_running_reqs == 1
        finally:
            source.release.set()
            await rt.close()


# ---------------------------------------------------------------------------
# Fire loop behavior
# ---------------------------------------------------------------------------


class TestFireLoop:
    def test_harmonic_periods_share_reporter_epoch_boundary(self):

        epoch = 10.0
        now = epoch + 0.75
        period_500 = _PeriodSchedule(500, epoch, now)
        period_1000 = _PeriodSchedule(1000, epoch, now)

        assert period_500.next_deadline == pytest.approx(epoch + 1.0)
        assert period_1000.next_deadline == pytest.approx(epoch + 1.0)

    def test_missed_ticks_preserve_reporter_epoch_phase(self):

        epoch = 10.0
        now = epoch + 1.35
        period = _PeriodSchedule(400, epoch, epoch)
        period.advance(now)

        assert period.next_deadline == pytest.approx(epoch + 1.6)

    @pytest.mark.asyncio
    async def test_empty_period_schedule_is_removed(self):

        rt = LoadReporterRuntime(SnapshotSource(), make_server_args())
        try:
            _, session = rt.register_session("r1", 5000, 30000)
            session.stop()

            assert 5000 not in rt._period_schedules
        finally:
            await rt.close()

    @pytest.mark.asyncio
    async def test_coalesced_registration_shares_one_initial_pull(self):

        source = SnapshotSource()
        rt = LoadReporterRuntime(source, make_server_args())
        try:
            _, s1 = rt.register_session("r1", 5000, 30000)
            _, s2 = rt.register_session("r2", 5000, 30000)
            r1 = await asyncio.wait_for(s1.queue.get(), timeout=0.3)
            r2 = await asyncio.wait_for(s2.queue.get(), timeout=0.3)
            assert r1.ranks and r2.ranks
            assert r1.sequence_id == r2.sequence_id  # one report, broadcast
            assert source.get_loads_calls == 1
        finally:
            await rt.close()

    @pytest.mark.asyncio
    async def test_staggered_same_interval_sessions_share_periodic_pull(self):
        """Equal intervals share one periodic phase despite staggered registration."""

        source = SnapshotSource()
        rt = LoadReporterRuntime(source, make_server_args())
        try:
            _, first = rt.register_session("first", 400, 3000)
            await asyncio.wait_for(first.queue.get(), timeout=0.3)

            await asyncio.sleep(0.05)
            _, second = rt.register_session("second", 400, 3000)
            await asyncio.wait_for(second.queue.get(), timeout=0.3)
            calls_after_initial = source.get_loads_calls

            first_periodic = await asyncio.wait_for(first.queue.get(), timeout=0.6)
            second_periodic = await asyncio.wait_for(second.queue.get(), timeout=0.6)

            assert first_periodic.sequence_id == second_periodic.sequence_id
            assert source.get_loads_calls == calls_after_initial + 1
        finally:
            await rt.close()

    @pytest.mark.asyncio
    async def test_harmonic_periods_share_one_pull_at_common_boundary(self):

        source = SnapshotSource()
        rt = LoadReporterRuntime(source, make_server_args())
        try:
            _, fast = rt.register_session("fast", 100, 3000)
            _, slow = rt.register_session("slow", 200, 3000)
            await asyncio.wait_for(fast.queue.get(), timeout=0.3)
            await asyncio.wait_for(slow.queue.get(), timeout=0.3)

            # The 100 ms-only boundary sends only to fast.
            await asyncio.wait_for(fast.queue.get(), timeout=0.3)
            # At 200 ms both period buckets consume the same snapshot pull.
            fast_common = await asyncio.wait_for(fast.queue.get(), timeout=0.3)
            slow_common = await asyncio.wait_for(slow.queue.get(), timeout=0.3)

            assert fast_common.sequence_id == slow_common.sequence_id
            assert source.get_loads_calls == 3  # initial, 100 ms, shared 200 ms
        finally:
            await rt.close()

    @pytest.mark.asyncio
    async def test_no_background_pull_without_sessions(self):

        source = SnapshotSource()
        rt = LoadReporterRuntime(source, make_server_args())
        try:
            await asyncio.sleep(0.15)
            assert source.get_loads_calls == 0
        finally:
            await rt.close()

    @pytest.mark.asyncio
    async def test_periodic_cadence_not_anchored_at_first_report_completion(self):
        """A slow initial pull does not shift the reporter-epoch cadence."""

        source = ControlledSnapshotSource()
        rt = LoadReporterRuntime(source, make_server_args())
        try:
            reporter_started_at = time.monotonic()
            rt.register_session("r1", 500, 30000)

            # Hold the first pull for 200ms (< interval), then let it finish.
            await asyncio.wait_for(source.started.wait(), timeout=1.0)
            source.started.clear()
            await asyncio.sleep(0.2)
            source.release.set()

            await asyncio.wait_for(source.started.wait(), timeout=1.0)
            elapsed = time.monotonic() - reporter_started_at
            assert (
                0.42 <= elapsed < 0.62
            ), f"fire 2 started {elapsed:.3f}s after reporter epoch"
        finally:
            source.release.set()
            await rt.close()

    @pytest.mark.asyncio
    async def test_failed_fire_does_not_reuse_last_good_ranks(self):

        source = ScriptedSnapshotSource(
            [[make_load_snapshot(1)], RuntimeError("read failed")]
        )
        rt = LoadReporterRuntime(source, make_server_args())
        try:
            _, session = rt.register_session("r1", 80, 3000)
            healthy = await asyncio.wait_for(session.queue.get(), timeout=0.3)
            failed = await asyncio.wait_for(session.queue.get(), timeout=0.3)
            assert healthy.ranks
            assert failed.status == pb.REPORT_STATUS_UNREACHABLE
            assert not failed.ranks
            assert failed.last_error == "read failed"
            assert (
                source.get_loads_calls == 2
            )  # one pull per fire, no retry on read error
        finally:
            await rt.close()

    @pytest.mark.asyncio
    async def test_rank_set_mismatch_retries_once_then_succeeds(self):

        source = ScriptedSnapshotSource(
            [
                [make_load_snapshot(1)],  # missing rank 1 -> retryable mismatch
                [make_load_snapshot(1), make_load_snapshot(1, dp_rank=1)],
            ],
            dp_size=2,
        )
        rt = LoadReporterRuntime(source, make_server_args(dp_size=2))
        try:
            _, session = rt.register_session("r1", 5000, 30000)
            report = await asyncio.wait_for(session.queue.get(), timeout=0.3)
            assert report.status == pb.REPORT_STATUS_HEALTHY
            assert [r.dp_rank for r in report.ranks] == [0, 1]
            assert source.get_loads_calls == 2  # initial attempt + one retry
        finally:
            await rt.close()

    @pytest.mark.asyncio
    async def test_rank_set_retry_uses_remaining_budget(self, monkeypatch):
        """The retry shares the fire's timeout budget."""
        import sglang.srt.load_reporter.runtime as runtime_module

        monkeypatch.setattr(runtime_module, "SNAPSHOT_PULL_TIMEOUT_SECONDS", 0.2)
        source = DelayedScriptedSnapshotSource(
            script=[
                [make_load_snapshot(1)],  # missing rank 1 -> retryable mismatch
                [make_load_snapshot(1), make_load_snapshot(1, dp_rank=1)],
            ],
            delay=0.12,
            dp_size=2,
        )
        rt = LoadReporterRuntime(source, make_server_args(dp_size=2))
        started_at = time.monotonic()
        try:
            _, session = rt.register_session("r1", 5000, 30000)
            report = await asyncio.wait_for(session.queue.get(), timeout=0.5)

            assert report.status == pb.REPORT_STATUS_UNREACHABLE
            assert source.get_loads_calls == 2  # retry ran with the remainder
            assert time.monotonic() - started_at < 0.35  # not 2x the budget
        finally:
            await rt.close()

    @pytest.mark.asyncio
    async def test_initial_pull_hanging_is_bounded(self, monkeypatch):
        import sglang.srt.load_reporter.runtime as runtime_module

        monkeypatch.setattr(runtime_module, "SNAPSHOT_PULL_TIMEOUT_SECONDS", 0.05)
        source = ControlledSnapshotSource()
        rt = LoadReporterRuntime(source, make_server_args())
        started_at = time.monotonic()
        try:
            _, session = rt.register_session("r1", 10_000, 30_000)
            report = await asyncio.wait_for(session.queue.get(), timeout=0.2)

            assert time.monotonic() - started_at >= 0.04
            assert report.status == pb.REPORT_STATUS_UNREACHABLE
            assert not report.ranks
        finally:
            source.release.set()
            await rt.close()


# ---------------------------------------------------------------------------
# Config updates
# ---------------------------------------------------------------------------


class TestRuntimeUpdateConfig:
    @pytest.mark.asyncio
    async def test_interval_update_during_pull_uses_current_membership(self):
        """A session moved from a due period during a pull is not sent that fire."""
        source = ControlledSnapshotSource()
        rt = LoadReporterRuntime(source, make_server_args())
        try:
            _, steady = rt.register_session("steady", 200, 3000)
            _, moving = rt.register_session("moving", 200, 3000)

            await asyncio.wait_for(source.started.wait(), timeout=1.0)
            source.started.clear()
            source.release.set()
            await asyncio.wait_for(steady.queue.get(), timeout=1.0)
            await asyncio.wait_for(moving.queue.get(), timeout=1.0)

            source.release.clear()
            await asyncio.wait_for(source.started.wait(), timeout=1.0)
            rt.update_session_config(moving, report_interval_ms=5000)
            source.release.set()

            assert await asyncio.wait_for(steady.queue.get(), timeout=1.0) is not None
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(moving.queue.get(), timeout=0.15)
        finally:
            source.release.set()
            await rt.close()

    @pytest.mark.asyncio
    async def test_update_config_joins_existing_period_without_resetting_it(self):

        rt = LoadReporterRuntime(SnapshotSource(), make_server_args())
        try:
            _, moving = rt.register_session("moving", 5000, 30000)
            _, existing = rt.register_session("existing", 3000, 30000)
            schedule = rt._period_schedules[3000]
            deadline = schedule.next_deadline

            rt.update_session_config(moving, report_interval_ms=3000)

            assert 5000 not in rt._period_schedules
            assert rt._period_schedules[3000] is schedule
            assert schedule.next_deadline == deadline
            assert schedule.sessions == {moving, existing}
        finally:
            await rt.close()

    @pytest.mark.asyncio
    async def test_update_config_reanchors_lease_deadline(self):

        rt = LoadReporterRuntime(SnapshotSource(), make_server_args())
        try:
            _, session = rt.register_session("r1", 1000, 3000)
            initial_report = await asyncio.wait_for(session.queue.get(), timeout=1.0)
            assert initial_report is not None

            rt.update_session_config(session, lease_ttl_ms=30)

            sentinel = await asyncio.wait_for(session.queue.get(), timeout=0.2)
            assert sentinel is None
        finally:
            await rt.close()

    @pytest.mark.asyncio
    async def test_update_config_rejects_all_fields_atomically(self):

        rt = LoadReporterRuntime(SnapshotSource(), make_server_args())
        try:
            _, session = rt.register_session("r1", 500, 3000)
            initial_report = await asyncio.wait_for(session.queue.get(), timeout=1.0)
            assert initial_report is not None

            with pytest.raises(ValueError, match="report_interval_ms"):
                rt.update_session_config(session, report_interval_ms=-1, lease_ttl_ms=1)

            assert session.report_interval_ms == 500
            assert rt._period_schedules[500].sessions == {session}
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(session.queue.get(), timeout=0.05)
        finally:
            session.stop()
            await rt.close()


# ---------------------------------------------------------------------------
# Lease handling
# ---------------------------------------------------------------------------


class TestLeaseExpiry:
    @pytest.mark.asyncio
    async def test_expiring_session_does_not_shrink_shared_pull_timeout(self):
        """A short lease must not bound a shared fire's pull."""

        source = ControlledSnapshotSource()
        rt = LoadReporterRuntime(source, make_server_args())
        try:
            _, expiring = rt.register_session("expiring", 60, 80)
            _, healthy = rt.register_session("healthy", 60, 3000)

            # Fire 1: release the initial pull immediately.
            await asyncio.wait_for(source.started.wait(), timeout=1.0)
            source.started.clear()
            source.release.set()
            first = await asyncio.wait_for(healthy.queue.get(), timeout=1.0)
            assert first.status == pb.REPORT_STATUS_HEALTHY
            source.release.clear()  # arm the gate before fire 2's pull awaits it

            # Fire 2: hold the pull past the expiring session's 80ms lease.
            await asyncio.wait_for(source.started.wait(), timeout=1.0)
            source.started.clear()
            await asyncio.sleep(0.05)
            source.release.set()

            # The healthy session shares the deadline and must still get the report, not a lease-shortened timeout.
            second = await asyncio.wait_for(healthy.queue.get(), timeout=1.0)
            assert second.status == pb.REPORT_STATUS_HEALTHY

            # The expiring session is reaped once its lease ends.
            sentinel = await asyncio.wait_for(expiring.queue.get(), timeout=1.0)
            if sentinel is not None:
                sentinel = await asyncio.wait_for(expiring.queue.get(), timeout=1.0)
            assert sentinel is None
        finally:
            source.release.set()
            await rt.close()

    @pytest.mark.asyncio
    async def test_keepalive_does_not_publish_before_report_deadline(self):
        """Lease renewals must not accelerate the periodic report cadence."""

        rt = LoadReporterRuntime(SnapshotSource(), make_server_args())
        keepalive_task = None
        try:
            _, session = rt.register_session("r1", 500, 200)
            initial_report = await asyncio.wait_for(session.queue.get(), timeout=1.0)
            assert initial_report is not None

            async def keep_lease_alive():
                while True:
                    await asyncio.sleep(0.05)
                    session.refresh_lease()

            keepalive_task = asyncio.create_task(keep_lease_alive())

            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(session.queue.get(), timeout=0.3)
        finally:
            if keepalive_task is not None:
                keepalive_task.cancel()
                await asyncio.gather(keepalive_task, return_exceptions=True)
            await rt.close()

    @pytest.mark.asyncio
    async def test_lease_expiry_stops_session(self):

        rt = LoadReporterRuntime(SnapshotSource(), make_server_args())
        try:
            # Very short lease: session should expire quickly.
            _, session = rt.register_session("r1", 5000, 20)
            # Wait for None sentinel in queue (lease expires)
            try:
                sentinel = await asyncio.wait_for(session.queue.get(), timeout=1.5)
                # might be a report first, then None
                if sentinel is not None:
                    sentinel = await asyncio.wait_for(session.queue.get(), timeout=1.5)
                assert sentinel is None, "Expected None sentinel after lease expiry"
            except asyncio.TimeoutError:
                pytest.fail("Session did not stop after lease expiry")
        finally:
            await rt.close()


# ---------------------------------------------------------------------------
# Router replacement
# ---------------------------------------------------------------------------


class TestSameRouterIdReplacement:
    @pytest.mark.asyncio
    async def test_replacement_does_not_corrupt_session_table(self):
        """Replacing a router session must leave the new generation registered."""

        source = SnapshotSource()
        rt = LoadReporterRuntime(source, make_server_args())
        try:
            _, session1 = rt.register_session("r1", 30, 3000)
            _, session2 = rt.register_session("r1", 30, 3000)

            sentinel = await asyncio.wait_for(session1.queue.get(), timeout=1.0)
            if sentinel is not None:
                sentinel = await asyncio.wait_for(session1.queue.get(), timeout=1.0)
            assert sentinel is None

            # The replacement session must still emit reports.
            reports = await drain_queue(session2.queue, 1, timeout=0.5)
            assert len(reports) >= 1, (
                "session2 must still emit reports after replacement; "
                "a generation-blind on_close would have removed it"
            )
        finally:
            await rt.close()


# ---------------------------------------------------------------------------
# Topology changes
# ---------------------------------------------------------------------------


class TestTopologyChangeEvents:
    @pytest.mark.asyncio
    async def test_next_fire_observes_rank_update_without_immediate_pull(self):

        source = MutableSnapshotSource()
        rt = LoadReporterRuntime(source, make_server_args())
        try:
            _, session = rt.register_session("r1", 200, 30000)
            initial = await asyncio.wait_for(session.queue.get(), timeout=0.5)
            assert [rank.dp_rank for rank in initial.ranks] == [0]

            before = source.get_loads_calls
            assert rt.update_expected_dp_ranks(range(2)) is True
            await asyncio.sleep(0.05)
            assert source.get_loads_calls == before
            assert len(source._rank_updates) == 1

            report = await asyncio.wait_for(session.queue.get(), timeout=0.5)
            assert [rank.dp_rank for rank in report.ranks] == [0, 1]
        finally:
            session.stop()
            await rt.close()


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


class TestShutdown:
    @pytest.mark.asyncio
    async def test_close_cancels_in_flight_pull_promptly(self):

        source = HangingSnapshotSource()
        rt = LoadReporterRuntime(source, make_server_args())
        _, session = rt.register_session("r1", 5000, 30000)
        await asyncio.wait_for(source.started.wait(), timeout=0.5)

        # Must not wait for the pull timeout: the fire task is cancelled.
        await asyncio.wait_for(rt.close(), timeout=0.5)
        # Stopped sessions receive the None sentinel.
        sentinel = await asyncio.wait_for(session.queue.get(), timeout=0.5)
        assert sentinel is None
        # close() stays idempotent.
        await asyncio.wait_for(rt.close(), timeout=0.1)

    @pytest.mark.asyncio
    async def test_close_without_sessions(self):

        rt = LoadReporterRuntime(SnapshotSource(), make_server_args())
        await asyncio.wait_for(rt.close(), timeout=0.5)


# ============================================================================
# gRPC service
# ============================================================================

# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------


async def start_test_server(runtime):
    """Start an in-process grpc.aio server; return (server, port)."""

    server = grpc.aio.server()
    add_service_to_server(runtime, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    return server, port


async def make_stub(port: int):
    """Return a Monitor stub connected to the given ephemeral port."""
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    return pb_grpc.LoadMonitorServiceStub(channel), channel


async def receive_frames(call, count: int, timeout: float = 3.0) -> list:
    """Receive up to count WorkerFrames within timeout seconds."""
    frames = []
    deadline = asyncio.get_event_loop().time() + timeout
    while len(frames) < count:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            break
        try:
            frame = await asyncio.wait_for(call.read(), timeout=remaining)
            if frame == grpc.aio.EOF:
                break
            frames.append(frame)
        except asyncio.TimeoutError:
            break
        except Exception:
            break
    return frames


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestServiceUpdateConfig:
    @pytest.mark.asyncio
    async def test_shorter_report_interval_takes_effect_from_update(self):

        rt = LoadReporterRuntime(SnapshotSource(), make_server_args())
        server, port = await start_test_server(rt)
        stub, channel = await make_stub(port)

        async def frames() -> AsyncIterator[pb.RouterFrame]:
            yield pb.RouterFrame(
                register=pb.RegisterRequest(
                    router_id="r1",
                    report_interval_ms=1000,
                    lease_ttl_ms=3000,
                )
            )
            await asyncio.sleep(0.05)
            yield pb.RouterFrame(
                update_config=pb.UpdateConfigRequest(report_interval_ms=30)
            )
            await asyncio.sleep(0.2)

        try:
            call = stub.Monitor(frames())
            received = await receive_frames(call, 5, timeout=1.0)
            reports = [f for f in received if f.WhichOneof("payload") == "report"]
            assert received[0].WhichOneof("payload") == "registered"
            assert len(reports) >= 2
        finally:
            await channel.close()
            await server.stop(grace=0)
            await rt.close()


class TestInvalidArguments:
    @pytest.mark.asyncio
    async def test_invalid_register_yields_terminal_error(self):

        rt = LoadReporterRuntime(SnapshotSource(), make_server_args())
        server, port = await start_test_server(rt)
        stub, channel = await make_stub(port)

        async def frames() -> AsyncIterator[pb.RouterFrame]:
            yield pb.RouterFrame(
                register=pb.RegisterRequest(
                    router_id="r1",
                    report_interval_ms=0,
                    lease_ttl_ms=3000,
                )
            )

        try:
            call = stub.Monitor(frames())
            received = await receive_frames(call, 2, timeout=1.0)
            assert len(received) == 1
            assert received[0].WhichOneof("payload") == "error"
            assert received[0].error.code == "INVALID_ARGUMENT"
        finally:
            await channel.close()
            await server.stop(grace=0)
            await rt.close()

    @pytest.mark.asyncio
    async def test_invalid_update_yields_terminal_error(self):

        rt = LoadReporterRuntime(SnapshotSource(), make_server_args())
        server, port = await start_test_server(rt)
        stub, channel = await make_stub(port)

        async def frames() -> AsyncIterator[pb.RouterFrame]:
            yield pb.RouterFrame(
                register=pb.RegisterRequest(
                    router_id="r1",
                    report_interval_ms=1000,
                    lease_ttl_ms=3000,
                )
            )
            yield pb.RouterFrame(
                update_config=pb.UpdateConfigRequest(report_interval_ms=0)
            )
            await asyncio.sleep(0.05)

        try:
            call = stub.Monitor(frames())
            received = await receive_frames(call, 4, timeout=1.0)
            errors = [f for f in received if f.WhichOneof("payload") == "error"]
            assert received[0].WhichOneof("payload") == "registered"
            assert len(errors) == 1
            assert errors[0].error.code == "INVALID_ARGUMENT"
        finally:
            await channel.close()
            await server.stop(grace=0)
            await rt.close()


class TestIllegalFirstFrame:
    @pytest.mark.asyncio
    async def test_non_register_first_frame_yields_error(self):

        rt = LoadReporterRuntime(SnapshotSource(), make_server_args())
        server, port = await start_test_server(rt)
        stub, channel = await make_stub(port)

        async def frames() -> AsyncIterator[pb.RouterFrame]:
            # Send a keep_alive as the first frame — illegal
            yield pb.RouterFrame(keep_alive=pb.KeepAlive())

        call = stub.Monitor(frames())
        received = await receive_frames(call, 1, timeout=2.0)
        assert len(received) == 1
        assert received[0].WhichOneof("payload") == "error"
        assert received[0].error.code == "INVALID_FIRST_FRAME"

        await channel.close()
        await server.stop(grace=0)
        await rt.close()

    @pytest.mark.asyncio
    async def test_empty_stream_no_crash(self):
        """Stream ending before any frame: server exits cleanly."""

        rt = LoadReporterRuntime(SnapshotSource(), make_server_args())
        server, port = await start_test_server(rt)
        stub, channel = await make_stub(port)

        async def frames() -> AsyncIterator[pb.RouterFrame]:
            return
            yield  # make it a generator

        call = stub.Monitor(frames())
        assert await asyncio.wait_for(call.read(), timeout=1.5) == grpc.aio.EOF
        assert await call.code() == grpc.StatusCode.OK

        await channel.close()
        await server.stop(grace=0)
        await rt.close()


class TestClientCancel:
    @pytest.mark.asyncio
    async def test_client_cancel_does_not_hang(self):

        rt = LoadReporterRuntime(SnapshotSource(), make_server_args())
        server, port = await start_test_server(rt)
        stub, channel = await make_stub(port)

        async def frames() -> AsyncIterator[pb.RouterFrame]:
            yield pb.RouterFrame(
                register=pb.RegisterRequest(
                    router_id="r1",
                    report_interval_ms=500,
                    lease_ttl_ms=3000,
                )
            )
            await asyncio.sleep(10)  # keep alive

        call = stub.Monitor(frames())
        # Read ack then cancel
        frames_received = await receive_frames(call, 1, timeout=2.0)
        assert frames_received[0].WhichOneof("payload") == "registered"
        call.cancel()

        # Server should clean up without hanging
        await asyncio.sleep(0.2)

        await channel.close()
        await server.stop(grace=0)
        await rt.close()


# ============================================================================
# Reporter lifecycle
# ============================================================================

# ---------------------------------------------------------------------------
# Fixtures (all test-only, live in test/)
# ---------------------------------------------------------------------------


class RankAwareSource(SnapshotSource):
    """Snapshot source that supports update_expected_dp_ranks."""

    def __init__(self, dp_size: int = 1) -> None:
        super().__init__(dp_size)
        self._update_calls: list = []

    def update_expected_dp_ranks(self, ranks) -> bool:
        self._update_calls.append(frozenset(ranks))
        new = frozenset(ranks)
        changed = new != self._expected_dp_ranks()
        self._dp_size = len(ranks)
        return changed

    def _expected_dp_ranks(self) -> frozenset:
        return frozenset(range(self._dp_size))


# ---------------------------------------------------------------------------
# Group 1: disabled — port is None
# ---------------------------------------------------------------------------


class TestDisabled:
    @pytest.mark.asyncio
    async def test_returns_none_when_port_none(self):

        handle = await start_load_reporter(
            make_server_args(load_reporter_port=None),
            SnapshotSource(),
        )
        assert handle is None

    @pytest.mark.asyncio
    async def test_rejects_missing_snapshot_source_when_enabled(self):

        with pytest.raises(ValueError, match="snapshot_source is required"):
            await start_load_reporter(
                make_server_args(load_reporter_port=30100),
                None,
            )

    def test_no_grpc_import_when_disabled(self):
        """Calling with port=None must not import grpc (checked in a clean subprocess)."""
        import os
        import subprocess
        import sys
        import textwrap

        code = textwrap.dedent("""
            import asyncio, sys, types
            from sglang.srt.load_reporter.lifecycle import start_load_reporter

            args = types.SimpleNamespace(
                host="127.0.0.1", load_reporter_port=None,
                disaggregation_mode="none", served_model_name="m",
                dp_size=1, tokenizer_worker_num=1,
            )
            h = asyncio.get_event_loop().run_until_complete(
                start_load_reporter(args, None)
            )
            assert h is None
            bad = [m for m in sys.modules if m == "grpc" or m.startswith("grpc.")]
            assert not bad, f"grpc imported while disabled: {bad}"
            print("OK")
            """)
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                os.pardir,
                os.pardir,
                os.pardir,
                os.pardir,
                "python",
            ),
        )
        assert "OK" in result.stdout, result.stderr


# ---------------------------------------------------------------------------
# Group 2: owner path — real listener, real grpc client
# ---------------------------------------------------------------------------


class TestOwnerPath:
    @pytest.mark.asyncio
    async def test_listener_serves_register_and_report(self):

        port = free_port()
        source = SnapshotSource()
        args = make_server_args(load_reporter_port=port)
        handle = await start_load_reporter(args, source)
        assert handle is not None
        try:
            stub, channel = await make_stub(port)

            async def frames() -> AsyncIterator[pb.RouterFrame]:
                yield pb.RouterFrame(
                    register=pb.RegisterRequest(
                        router_id="r1", report_interval_ms=40, lease_ttl_ms=10000
                    )
                )
                await asyncio.sleep(0.6)

            call = stub.Monitor(frames())
            received = await receive_frames(call, 3, timeout=2.0)
            assert received[0].WhichOneof("payload") == "registered"
            reports = [f for f in received if f.WhichOneof("payload") == "report"]
            assert len(reports) >= 1
            assert reports[0].report.status == pb.REPORT_STATUS_HEALTHY
            assert [rank.dp_rank for rank in reports[0].report.ranks] == [0]
            await channel.close()
        finally:
            await handle.close()


# ---------------------------------------------------------------------------
# Group 4: fixed port occupied — explicit failure, no state corruption
# ---------------------------------------------------------------------------


class TestLifecycleFixedPortOccupied:
    @pytest.mark.asyncio
    async def test_occupied_port_raises_and_recovers(self):

        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        occupied = blocker.getsockname()[1]
        try:
            with pytest.raises(RuntimeError):
                await start_load_reporter(
                    make_server_args(load_reporter_port=occupied),
                    SnapshotSource(),
                )
        finally:
            blocker.close()

        # A later start on a free port must succeed (no global state corrupted by the failed bind).
        available_port = free_port()
        handle = await start_load_reporter(
            make_server_args(load_reporter_port=available_port), SnapshotSource()
        )
        assert handle is not None
        await handle.close()


# ---------------------------------------------------------------------------
# Group 5: handle delegation (used by MultiTokenizerRouter)
# ---------------------------------------------------------------------------


class TestHandleDelegation:
    @pytest.mark.asyncio
    async def test_update_expected_dp_ranks_through_handle(self):
        """Expected-rank update propagates through the handle to the runtime."""

        port = free_port()
        source = RankAwareSource(dp_size=1)
        handle = await start_load_reporter(
            make_server_args(load_reporter_port=port), source
        )
        assert handle is not None
        try:
            assert handle.update_expected_dp_ranks(range(2)) is True
            assert len(source._update_calls) == 1
            assert source._update_calls[0] == frozenset([0, 1])

            # No-op when ranks unchanged.
            assert handle.update_expected_dp_ranks(range(2)) is False
            assert len(source._update_calls) == 2
        finally:
            await handle.close()


class TestCloseCancellationSafety:
    """A close() cancelled mid-await must not abandon remaining teardown."""

    @pytest.mark.asyncio
    async def test_cancelled_close_still_completes_teardown(self):

        steps: List[str] = []
        server_entered = asyncio.Event()
        release_server = asyncio.Event()

        class FakeServer:
            async def stop(self, grace=None):
                steps.append("server.stop.enter")
                server_entered.set()
                await release_server.wait()
                steps.append("server.stop.exit")

        class FakeRuntime:
            async def close(self):
                steps.append("runtime.close")

        handle = LoadReporterHandle(runtime=FakeRuntime(), server=FakeServer())

        # First caller enters close() and is cancelled while inside server.stop.
        first = asyncio.create_task(handle.close())
        await asyncio.wait_for(server_entered.wait(), timeout=1.0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        # Shared teardown must survive the cancelled caller: release the step, let a second caller await it.
        release_server.set()
        await handle.close()

        assert steps == [
            "server.stop.enter",
            "server.stop.exit",
            "runtime.close",
        ]

    @pytest.mark.asyncio
    async def test_second_caller_awaits_shared_teardown(self):

        runtime_closed = asyncio.Event()
        release_runtime = asyncio.Event()
        steps: List[str] = []

        class FakeRuntime:
            async def close(self):
                runtime_closed.set()
                await release_runtime.wait()
                steps.append("runtime.close")

        class FakeServer:
            async def stop(self, grace=None):
                steps.append("server.stop")

        handle = LoadReporterHandle(runtime=FakeRuntime(), server=FakeServer())

        first = asyncio.create_task(handle.close())
        await asyncio.wait_for(runtime_closed.wait(), timeout=1.0)
        # Second caller joins while the shared teardown is still in-flight.
        second = asyncio.create_task(handle.close())
        await asyncio.sleep(0)
        assert not first.done()
        assert not second.done()

        release_runtime.set()
        await asyncio.gather(first, second)
        # Teardown ran exactly once even though two callers awaited it.
        assert steps == ["server.stop", "runtime.close"]


# ============================================================================
# Standalone SMG lifecycle
# ============================================================================

# ---------------------------------------------------------------------------
# Fakes (test-only)
# ---------------------------------------------------------------------------


def make_standalone_server_args(*, port: Optional[int], sidecar_port: int) -> Any:
    """Build resolved standalone gRPC arguments for runtime-context tests.

    Args:
        port: Optional load reporter listener port.
        sidecar_port: HTTP sidecar listener port.

    Returns:
        A lightweight real ServerArgs instance whose config namespaces can be
        published by the standalone gRPC entry point.
    """
    from sglang.srt.server_args import ServerArgs

    return ServerArgs(
        model_path="dummy",
        load_reporter_port=port,
        port=sidecar_port - 1,
        smg_http_sidecar_port=sidecar_port,
        enable_metrics=False,
    )


class FakeRequestManager:
    """Stand-in for smg's GrpcRequestManager load-snapshot interface."""

    def __init__(self, server_args: Any) -> None:
        self.server_args = server_args
        self.get_loads_calls = 0

    async def get_loads(self, include=None) -> list:
        self.get_loads_calls += 1
        return [make_load_snapshot()]


def port_is_free(port: int) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = probe.connect_ex(("127.0.0.1", port))
    probe.close()
    return result != 0


def install_fake_smg(capability: bool):
    """Inject a fake smg_grpc_servicer.sglang.server module."""
    holder = types.SimpleNamespace(
        request_manager=None,
        stop_event=None,  # set per-test to control server lifetime
    )

    if capability:

        async def fake_serve_grpc(
            server_args, model_info, on_request_manager_ready=None
        ):
            rm = FakeRequestManager(server_args)
            holder.request_manager = rm
            if on_request_manager_ready is not None:
                await on_request_manager_ready(rm, server_args, {})
            if holder.stop_event is not None:
                await holder.stop_event.wait()

    else:

        async def fake_serve_grpc(server_args, model_info):
            rm = FakeRequestManager(server_args)
            holder.request_manager = rm
            if holder.stop_event is not None:
                await holder.stop_event.wait()

    holder.serve = fake_serve_grpc

    pkg = types.ModuleType("smg_grpc_servicer")
    sub = types.ModuleType("smg_grpc_servicer.sglang")
    server_mod = types.ModuleType("smg_grpc_servicer.sglang.server")
    server_mod.serve_grpc = fake_serve_grpc
    sys.modules["smg_grpc_servicer"] = pkg
    sys.modules["smg_grpc_servicer.sglang"] = sub
    sys.modules["smg_grpc_servicer.sglang.server"] = server_mod
    return holder


def uninstall_fake_smg():
    for name in (
        "smg_grpc_servicer.sglang.server",
        "smg_grpc_servicer.sglang",
        "smg_grpc_servicer",
    ):
        sys.modules.pop(name, None)


@pytest.fixture
def isolate_sidecar(monkeypatch):
    """Isolate sidecar I/O and leave runtime config unpublished."""
    from sglang.srt.entrypoints import grpc_server
    from sglang.srt.runtime_context import reset_context

    async def _noop_sidecar(host, port, app):
        return types.SimpleNamespace(cleanup=_acleanup)

    async def _acleanup():
        return None

    reset_context()
    monkeypatch.setattr(grpc_server, "_start_sidecar_server", _noop_sidecar)
    monkeypatch.setattr(grpc_server, "_add_admin_routes", lambda app, rm: None)
    yield
    reset_context()


class TestReporterEnabledWithCapability:
    @pytest.mark.asyncio
    async def test_client_connects_and_reports(self, isolate_sidecar):
        from sglang.srt.entrypoints import grpc_server

        port = free_port()
        args = make_standalone_server_args(port=port, sidecar_port=free_port())
        holder = install_fake_smg(capability=True)
        holder.stop_event = asyncio.Event()
        try:
            serve_task = asyncio.ensure_future(grpc_server.serve_grpc(args))
            # Wait for the ready callback to bind the reporter port.
            for _ in range(50):
                await asyncio.sleep(0.02)
                if holder.request_manager is not None and not port_is_free(port):
                    break
            rm = holder.request_manager
            assert rm is not None

            stub, channel = await make_stub(port)

            async def frames() -> AsyncIterator[pb.RouterFrame]:
                yield pb.RouterFrame(
                    register=pb.RegisterRequest(
                        router_id="r1", report_interval_ms=40, lease_ttl_ms=10000
                    )
                )
                await asyncio.sleep(0.5)

            call = stub.Monitor(frames())
            received = await receive_frames(call, 2, timeout=2.0)
            assert received[0].WhichOneof("payload") == "registered"
            assert any(f.WhichOneof("payload") == "report" for f in received)
            await channel.close()
        finally:
            holder.stop_event.set()
            await asyncio.wait_for(serve_task, timeout=3.0)
            uninstall_fake_smg()

        # After serve_grpc returns, the reporter listener is released.
        assert port_is_free(port)


class TestCapabilityGuard:
    @pytest.mark.asyncio
    async def test_reporter_enabled_missing_capability_raises(self, isolate_sidecar):
        from sglang.srt.entrypoints import grpc_server

        port = free_port()
        args = make_standalone_server_args(port=port, sidecar_port=free_port())
        holder = install_fake_smg(capability=False)
        try:
            with pytest.raises(RuntimeError, match="load-reporter-port"):
                await grpc_server.serve_grpc(args)
            # _serve_grpc must not have been invoked.
            assert holder.request_manager is None
        finally:
            uninstall_fake_smg()

    @pytest.mark.asyncio
    async def test_reporter_disabled_missing_capability_is_compatible(
        self, isolate_sidecar
    ):
        from sglang.srt.entrypoints import grpc_server

        args = make_standalone_server_args(port=None, sidecar_port=free_port())
        holder = install_fake_smg(capability=False)
        holder.stop_event = asyncio.Event()
        try:
            serve_task = asyncio.ensure_future(grpc_server.serve_grpc(args))
            for _ in range(50):
                await asyncio.sleep(0.02)
                if holder.request_manager is not None:
                    break
            assert holder.request_manager is not None  # served normally, no error
        finally:
            holder.stop_event.set()
            await asyncio.wait_for(serve_task, timeout=3.0)
            uninstall_fake_smg()


class TestServeGrpcCleanup:
    @pytest.mark.asyncio
    async def test_cancellation_cleans_up_reporter(self, isolate_sidecar):
        from sglang.srt.entrypoints import grpc_server

        port = free_port()
        args = make_standalone_server_args(port=port, sidecar_port=free_port())
        holder = install_fake_smg(capability=True)
        holder.stop_event = asyncio.Event()  # never set: we cancel instead
        try:
            serve_task = asyncio.ensure_future(grpc_server.serve_grpc(args))
            for _ in range(50):
                await asyncio.sleep(0.02)
                if holder.request_manager is not None and not port_is_free(port):
                    break
            serve_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await serve_task
        finally:
            uninstall_fake_smg()
        assert port_is_free(port)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
