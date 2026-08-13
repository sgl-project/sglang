"""Unit tests for the LoadReporterRuntime push-channel fire loop."""

from __future__ import annotations

import asyncio
import sys
import time
import types

import pytest

pytest_plugins = ("pytest_asyncio",)

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_server_args(dp_size: int = 1) -> types.SimpleNamespace:
    args = types.SimpleNamespace()
    args.host = "127.0.0.1"
    args.load_reporter_port = 9999
    args.disaggregation_mode = "none"
    args.served_model_name = "test-model"
    args.dp_size = dp_size
    return args


def make_load_snapshot(num_running_reqs: int = 1, dp_rank: int = 0):
    from sglang.srt.managers.load_snapshot import LoadSnapshot

    return LoadSnapshot(
        timestamp=time.time(),
        dp_rank=dp_rank,
        num_running_reqs=num_running_reqs,
        max_running_requests=8,
        max_total_num_tokens=1024,
    )


class FakeSnapshotSource:
    """Return one valid rank-0 snapshot per pull and count calls."""

    def __init__(self, dp_size: int = 1) -> None:
        self._dp_size = dp_size
        self.get_loads_calls = 0

    async def get_loads(self) -> list:
        self.get_loads_calls += 1
        return [make_load_snapshot(1)]

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
            ("   ", 500, 3000, "router_id"),
            ("r1", 0, 3000, "report_interval_ms"),
            ("r1", -1, 3000, "report_interval_ms"),
            ("r1", 500, 0, "lease_ttl_ms"),
            ("r1", 500, -1, "lease_ttl_ms"),
        ],
    )
    async def test_register_rejects_invalid_session_config(
        self, router_id, report_interval_ms, lease_ttl_ms, error
    ):
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        rt = LoadReporterRuntime(FakeSnapshotSource(), make_server_args())
        try:
            with pytest.raises(ValueError, match=error):
                rt.register_session(router_id, report_interval_ms, lease_ttl_ms)
        finally:
            await rt.close()

    @pytest.mark.asyncio
    async def test_register_returns_ack(self):
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        rt = LoadReporterRuntime(FakeSnapshotSource(), make_server_args())
        try:
            ack, session = rt.register_session("r1", 500, 3000)
            assert ack.lease_ttl_ms == 3000
            assert ack.renew_after_ms == max(1, 3000 // 3)
            assert ack.renew_after_ms == 1000
        finally:
            session.stop()
            await rt.close()

    @pytest.mark.asyncio
    async def test_initial_report_enqueued(self):
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        rt = LoadReporterRuntime(FakeSnapshotSource(), make_server_args())
        try:
            _, session = rt.register_session("r1", 500, 3000)
            # The next fire delivers the initial report immediately.
            reports = await drain_queue(session.queue, 1)
            assert len(reports) == 1, "Expected 1 initial report"
            assert reports[0].ranks
        finally:
            session.stop()
            await rt.close()

    @pytest.mark.asyncio
    async def test_initial_report_waits_for_pull(self):
        from sglang.srt.load_reporter.proto import load_monitor_pb2 as pb
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

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
    @pytest.mark.asyncio
    async def test_coalesced_registration_shares_one_initial_pull(self):
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        source = FakeSnapshotSource()
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
    async def test_shared_fire_broadcasts_one_report_to_aligned_sessions(self):
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        source = FakeSnapshotSource()
        rt = LoadReporterRuntime(source, make_server_args())
        try:
            _, s1 = rt.register_session("r1", 100, 30000)
            _, s2 = rt.register_session("r2", 100, 30000)
            for _ in range(2):
                r1 = await asyncio.wait_for(s1.queue.get(), timeout=0.5)
                r2 = await asyncio.wait_for(s2.queue.get(), timeout=0.5)
            # The periodic broadcast shares one report across aligned sessions.
            assert r1.sequence_id == r2.sequence_id
            assert source.get_loads_calls == 2  # initial fire + one periodic fire
        finally:
            await rt.close()

    @pytest.mark.asyncio
    async def test_long_interval_session_receives_only_its_own_deadlines(self):
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        source = FakeSnapshotSource()
        rt = LoadReporterRuntime(source, make_server_args())
        try:
            _, fast = rt.register_session("fast", 40, 30000)
            _, slow = rt.register_session("slow", 300, 30000)
            await asyncio.wait_for(fast.queue.get(), timeout=0.3)
            await asyncio.wait_for(slow.queue.get(), timeout=0.3)
            # slow has no further report due before ~300ms.
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(slow.queue.get(), timeout=0.15)
            slow_report = await asyncio.wait_for(slow.queue.get(), timeout=0.3)
            assert slow_report.ranks
        finally:
            await rt.close()

    @pytest.mark.asyncio
    async def test_no_pull_between_initial_report_and_periodic_deadline(self):
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        source = FakeSnapshotSource()
        rt = LoadReporterRuntime(source, make_server_args())
        try:
            _, session = rt.register_session("r1", 5000, 30000)
            await asyncio.wait_for(session.queue.get(), timeout=0.3)
            calls_after_initial = source.get_loads_calls
            await asyncio.sleep(0.2)
            assert source.get_loads_calls == calls_after_initial
        finally:
            await rt.close()

    @pytest.mark.asyncio
    async def test_no_background_pull_without_sessions(self):
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        source = FakeSnapshotSource()
        rt = LoadReporterRuntime(source, make_server_args())
        try:
            await asyncio.sleep(0.15)
            assert source.get_loads_calls == 0
        finally:
            await rt.close()

    @pytest.mark.asyncio
    async def test_periodic_cadence_anchored_at_registration(self):
        """R4: the cadence anchor is registration, not first-report completion (2nd fire at ~500ms, not ~700ms)."""
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        source = ControlledSnapshotSource()
        rt = LoadReporterRuntime(source, make_server_args())
        try:
            registered_at = time.monotonic()
            rt.register_session("r1", 500, 30000)

            # Hold the first pull for 200ms (< interval), then let it finish.
            await asyncio.wait_for(source.started.wait(), timeout=1.0)
            source.started.clear()
            await asyncio.sleep(0.2)
            source.release.set()

            await asyncio.wait_for(source.started.wait(), timeout=1.0)
            elapsed = time.monotonic() - registered_at
            assert (
                0.42 <= elapsed < 0.62
            ), f"fire 2 started {elapsed:.3f}s after registration"
        finally:
            source.release.set()
            await rt.close()

    @pytest.mark.asyncio
    async def test_reports_flow_periodically(self):
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        rt = LoadReporterRuntime(FakeSnapshotSource(), make_server_args())
        try:
            _, session = rt.register_session("r1", 30, 3000)
            # Drain the immediate report plus at least 2 periodic ones.
            reports = await drain_queue(session.queue, 3, timeout=2.0)
            assert len(reports) >= 2, f"Expected >=2 reports, got {len(reports)}"
        finally:
            session.stop()
            await rt.close()

    @pytest.mark.asyncio
    async def test_failed_fire_does_not_reuse_last_good_ranks(self):
        from sglang.srt.load_reporter.proto import load_monitor_pb2 as pb
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

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
        from sglang.srt.load_reporter.proto import load_monitor_pb2 as pb
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

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
        """R3: the rank-set retry shares the fire's timeout budget (pre-fix, each attempt got the full 0.2s)."""
        import sglang.srt.load_reporter.runtime as runtime_module
        from sglang.srt.load_reporter.proto import load_monitor_pb2 as pb
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

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
        from sglang.srt.load_reporter.proto import load_monitor_pb2 as pb
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

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


class TestUpdateConfig:
    @pytest.mark.asyncio
    async def test_update_config_reanchors_report_deadline(self):
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        rt = LoadReporterRuntime(FakeSnapshotSource(), make_server_args())
        try:
            _, session = rt.register_session("r1", 1000, 3000)
            initial_report = await asyncio.wait_for(session.queue.get(), timeout=1.0)
            assert initial_report is not None

            session.update_config(report_interval_ms=30)

            report = await asyncio.wait_for(session.queue.get(), timeout=0.2)
            assert report is not None
            assert session.report_interval_ms == 30
        finally:
            session.stop()
            await rt.close()

    @pytest.mark.asyncio
    async def test_update_config_reanchors_lease_deadline(self):
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        rt = LoadReporterRuntime(FakeSnapshotSource(), make_server_args())
        try:
            _, session = rt.register_session("r1", 1000, 3000)
            initial_report = await asyncio.wait_for(session.queue.get(), timeout=1.0)
            assert initial_report is not None

            session.update_config(lease_ttl_ms=30)

            sentinel = await asyncio.wait_for(session.queue.get(), timeout=0.2)
            assert sentinel is None
        finally:
            await rt.close()

    @pytest.mark.asyncio
    async def test_update_config_rejects_all_fields_atomically(self):
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        rt = LoadReporterRuntime(FakeSnapshotSource(), make_server_args())
        try:
            _, session = rt.register_session("r1", 500, 3000)
            initial_report = await asyncio.wait_for(session.queue.get(), timeout=1.0)
            assert initial_report is not None

            with pytest.raises(ValueError, match="report_interval_ms"):
                session.update_config(report_interval_ms=-1, lease_ttl_ms=1)

            assert session.report_interval_ms == 500
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(session.queue.get(), timeout=0.05)
        finally:
            session.stop()
            await rt.close()

    @pytest.mark.asyncio
    async def test_update_config_refreshes_lease(self):
        """After update_config extends the lease, session should keep reporting."""
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        rt = LoadReporterRuntime(FakeSnapshotSource(), make_server_args())
        try:
            # Short initial lease of 50ms.
            _, session = rt.register_session("r1", 10, 50)
            # Extend the lease to 5000ms before the original 50ms expires.
            await asyncio.sleep(0.02)
            session.update_config(lease_ttl_ms=5000)
            # Wait well past the original 50ms window.
            await asyncio.sleep(0.1)
            # Session should still be emitting reports (queue not terminated).
            reports = await drain_queue(session.queue, 1, timeout=0.3)
            assert (
                len(reports) >= 1
            ), "Session should still report after update_config extended the lease"
        finally:
            session.stop()
            await rt.close()


# ---------------------------------------------------------------------------
# Lease handling
# ---------------------------------------------------------------------------


class TestLeaseExpiry:
    @pytest.mark.asyncio
    async def test_expiring_session_does_not_shrink_shared_pull_timeout(self):
        """R1: a short lease must not bound a shared fire's pull (pre-fix, an expiring session shrank the timeout)."""
        from sglang.srt.load_reporter.proto import load_monitor_pb2 as pb
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

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
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        rt = LoadReporterRuntime(FakeSnapshotSource(), make_server_args())
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
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        rt = LoadReporterRuntime(FakeSnapshotSource(), make_server_args())
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
    async def test_same_router_id_replaces_session(self):
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        rt = LoadReporterRuntime(FakeSnapshotSource(), make_server_args())
        try:
            _, session1 = rt.register_session("r1", 500, 3000)
            # Re-register same router_id — replaces session1.
            _, session2 = rt.register_session("r1", 200, 3000)
            # session1 should receive None sentinel (stopped).
            sentinel = await asyncio.wait_for(session1.queue.get(), timeout=1.0)
            # consume any initial report first, then None
            if sentinel is not None:
                sentinel = await asyncio.wait_for(session1.queue.get(), timeout=1.0)
            assert sentinel is None, "Old session should have been stopped"
            assert session2 is not session1
        finally:
            await rt.close()

    @pytest.mark.asyncio
    async def test_replacement_does_not_corrupt_session_table(self):
        """C1: a generation-blind on_close deleted the replacement session; the fire loop must keep serving it."""
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        source = FakeSnapshotSource()
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
# Multiple routers
# ---------------------------------------------------------------------------


class TestMultiRouter:
    @pytest.mark.asyncio
    async def test_different_routers_coexist(self):
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        rt = LoadReporterRuntime(FakeSnapshotSource(), make_server_args())
        try:
            _, session1 = rt.register_session("r1", 30, 3000)
            _, session2 = rt.register_session("r2", 30, 3000)
            # Both sessions receive reports independently.
            reports1 = await drain_queue(session1.queue, 1)
            reports2 = await drain_queue(session2.queue, 1)
            assert len(reports1) >= 1
            assert len(reports2) >= 1
        finally:
            await rt.close()

    @pytest.mark.asyncio
    async def test_fire_cadence_tracks_shortest_interval(self):
        """The fire loop must wake at the shortest session interval."""
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        source = FakeSnapshotSource()
        rt = LoadReporterRuntime(source, make_server_args())
        try:
            # s1 wants 1000ms, s2 wants 30ms: fires should run ~at 30ms cadence.
            _, s1 = rt.register_session("r1", 1000, 5000)
            _, s2 = rt.register_session("r2", 30, 5000)
            before = source.get_loads_calls
            await asyncio.sleep(0.2)
            after = source.get_loads_calls
            # At 30ms over 200ms expect >=4 fires; at 1000ms expect <=1; assert a majority.
            assert (
                after - before >= 3
            ), f"Expected >=3 fires at 30ms min interval, got {after - before}"
        finally:
            s1.stop()
            s2.stop()
            await rt.close()


# ---------------------------------------------------------------------------
# Topology changes
# ---------------------------------------------------------------------------


class TestTopologyChangeEvents:
    @pytest.mark.asyncio
    async def test_update_expected_dp_ranks_does_not_pull(self):
        """A topology change only updates the expected set; it never pulls."""
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        source = MutableSnapshotSource()
        rt = LoadReporterRuntime(source, make_server_args())
        try:
            _, session = rt.register_session("r1", 5000, 30000)
            await asyncio.wait_for(session.queue.get(), timeout=0.5)
            before = source.get_loads_calls
            assert rt.update_expected_dp_ranks(range(2)) is True
            await asyncio.sleep(0.15)
            assert (
                source.get_loads_calls == before
            ), "topology change must not trigger a snapshot pull"
            assert len(source._rank_updates) == 1
        finally:
            session.stop()
            await rt.close()

    @pytest.mark.asyncio
    async def test_next_fire_observes_new_rank_set(self):
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        source = MutableSnapshotSource()
        rt = LoadReporterRuntime(source, make_server_args())
        try:
            _, session = rt.register_session("r1", 30, 30000)
            initial = await asyncio.wait_for(session.queue.get(), timeout=0.5)
            assert [rank.dp_rank for rank in initial.ranks] == [0]

            assert rt.update_expected_dp_ranks(range(2)) is True

            report = await asyncio.wait_for(session.queue.get(), timeout=0.3)
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
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

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
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        rt = LoadReporterRuntime(FakeSnapshotSource(), make_server_args())
        await asyncio.wait_for(rt.close(), timeout=0.5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
