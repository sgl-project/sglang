"""Unit tests for LoadReporterRuntime inbound Router session management.

Tests exercise runtime.register_session, session lifecycle, sampler
activation, and time-controlled lease expiry using monkeypatching.
No grpc.aio server is needed; sessions are driven directly.
"""

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


def make_load_snapshot(num_running_reqs: int):
    from sglang.srt.managers.load_snapshot import LoadSnapshot

    return LoadSnapshot(
        timestamp=time.time(),
        dp_rank=0,
        num_running_reqs=num_running_reqs,
        max_running_requests=8,
        max_total_num_tokens=1024,
    )


class FakeSnapshotSource:
    """Minimal LoadSnapshotSource for testing."""

    def __init__(self, dp_size: int = 1) -> None:
        self._dp_size = dp_size
        self.get_loads_calls = 0

    async def get_loads(self) -> list:
        self.get_loads_calls += 1
        return []

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
    """Return the current running-request count on every sample."""

    def __init__(self) -> None:
        self.num_running_reqs = 1
        self.get_loads_calls = 0

    async def get_loads(self) -> list:
        self.get_loads_calls += 1
        return [make_load_snapshot(self.num_running_reqs)]

    def expected_dp_ranks(self) -> frozenset:
        return frozenset({0})


class BlockingAfterInitialSnapshotSource:
    """Complete the initial sample, then hold the next one in flight."""

    def __init__(self) -> None:
        self.get_loads_calls = 0
        self.blocked_sample_started = asyncio.Event()
        self.release = asyncio.Event()

    async def get_loads(self) -> list:
        self.get_loads_calls += 1
        if self.get_loads_calls == 1:
            return [make_load_snapshot(1)]

        self.blocked_sample_started.set()
        await self.release.wait()
        return [make_load_snapshot(9)]

    def expected_dp_ranks(self) -> frozenset:
        return frozenset({0})


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


class TestSnapshotSources:
    @pytest.mark.asyncio
    async def test_manager_source_can_read_without_entering_manager_event_loop(self):
        from sglang.srt.load_reporter.sampler import ManagerLoadSnapshotSource

        expected_loads = [object()]

        class Reader:
            def read_all(self):
                return expected_loads

        class Manager:
            async def get_loads(self, include):
                raise AssertionError("background reporter entered manager event loop")

        reader = Reader()
        source = ManagerLoadSnapshotSource(Manager(), {0}, snapshot_reader=reader)

        assert await source.get_loads() is expected_loads

    def test_manager_source_tracks_elastic_worker_count(self):
        from sglang.srt.load_reporter.sampler import ManagerLoadSnapshotSource

        class Manager:
            elastic_worker_count = 1

        manager = Manager()
        source = ManagerLoadSnapshotSource(manager, {0})

        assert source.expected_dp_ranks() == frozenset({0})

        manager.elastic_worker_count = 3
        assert source.expected_dp_ranks() == frozenset({0, 1, 2})

        manager.elastic_worker_count = 2
        assert source.expected_dp_ranks() == frozenset({0, 1})


# ---------------------------------------------------------------------------
# Tests
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
            ack, session = rt.register_session("r1", 500, 3000)
            # The session enqueues the first report immediately.
            reports = await drain_queue(session.queue, 1)
            assert len(reports) == 1, "Expected 1 initial report"
        finally:
            session.stop()
            await rt.close()

    @pytest.mark.asyncio
    async def test_initial_report_waits_for_first_completed_snapshot(self):
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

    @pytest.mark.asyncio
    async def test_initial_report_is_bounded_when_first_sample_hangs(self, monkeypatch):
        import sglang.srt.load_reporter.runtime as runtime_module
        from sglang.srt.load_reporter.proto import load_monitor_pb2 as pb
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        monkeypatch.setattr(runtime_module, "INITIAL_SAMPLE_TIMEOUT_SECONDS", 0.05)
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

    @pytest.mark.asyncio
    async def test_periodic_reports(self):
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        rt = LoadReporterRuntime(FakeSnapshotSource(), make_server_args())
        try:
            ack, session = rt.register_session("r1", 30, 3000)
            # drain the immediate report + wait for at least 2 periodic ones
            reports = await drain_queue(session.queue, 3, timeout=2.0)
            assert len(reports) >= 2, f"Expected >=2 reports, got {len(reports)}"
        finally:
            session.stop()
            await rt.close()


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
    async def test_report_interval_update_reschedules_sampler(self):
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        source = FakeSnapshotSource()
        rt = LoadReporterRuntime(source, make_server_args())
        try:
            _, session = rt.register_session("r1", 1000, 3000)
            await asyncio.sleep(0.1)
            calls_before = source.get_loads_calls

            session.update_config(report_interval_ms=30)
            await asyncio.sleep(0.15)

            assert source.get_loads_calls - calls_before >= 2
        finally:
            session.stop()
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
            ack, session = rt.register_session("r1", 10, 50)
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


class TestLeaseExpiry:
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
            ack, session = rt.register_session("r1", 5000, 20)
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


class TestSameRouterIdReplacement:
    @pytest.mark.asyncio
    async def test_same_router_id_replaces_session(self):
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        rt = LoadReporterRuntime(FakeSnapshotSource(), make_server_args())
        try:
            ack1, session1 = rt.register_session("r1", 500, 3000)
            # Re-register same router_id — replaces session1.
            ack2, session2 = rt.register_session("r1", 200, 3000)
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
        """C1 regression: old session's on_close must not delete the new session.

        Bug mechanism: _on_session_closed was generation-blind — it did
        self._sessions.pop(router_id) unconditionally, so when the old session's
        cleanup ran asynchronously it silently removed the new session from the
        table, deactivated the sampler, and leaked the new session's task.

        This test reproduces the exact asynchronous interleaving and asserts the
        behavioral invariants that would have failed on the buggy code:
        - sampler stays active after replacement (session2 is in the table)
        - session2 emits a fresh report (write loop is still running)
        """
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        source = FakeSnapshotSource()
        rt = LoadReporterRuntime(source, make_server_args())
        try:
            ack1, session1 = rt.register_session("r1", 30, 3000)
            ack2, session2 = rt.register_session("r1", 30, 3000)

            # Let old session's _run() finally block execute (needs event loop yield).
            # Drain session1 until we see the None sentinel.
            sentinel = None
            for _ in range(10):
                await asyncio.sleep(0.01)
                while not session1.queue.empty():
                    item = session1.queue.get_nowait()
                    if item is None:
                        sentinel = item
                        break
                if sentinel is None and session1._done.is_set():
                    sentinel = None  # done event set; on_close has fired
                    break

            # Allow a bit more time for on_close to execute.
            await asyncio.sleep(0.05)

            # Sampler must still be active: session2 is in the table.
            calls_before = source.get_loads_calls
            await asyncio.sleep(0.15)
            calls_after = source.get_loads_calls
            assert calls_after > calls_before, (
                "Sampler must stay active after same-router-id replacement; "
                "generation-blind on_close would have deactivated it"
            )

            # session2 must still emit reports.
            reports = await drain_queue(session2.queue, 1, timeout=0.5)
            assert len(reports) >= 1, (
                "session2 must still emit reports after replacement; "
                "generation-blind on_close would have leaked its task"
            )
        finally:
            await rt.close()


class GatedReRegistrationSource:
    """Return a snapshot immediately until gated, then block one sample.

    Lets a test drive: (1) a first session's initial sample completing, then
    (2) a later session registering while the *next* sample is held in flight.
    """

    def __init__(self) -> None:
        self.gate = False
        self.num_running = 1
        self.gated_started = asyncio.Event()
        self.release = asyncio.Event()

    async def get_loads(self) -> list:
        if self.gate:
            self.gated_started.set()
            await self.release.wait()
        return [make_load_snapshot(self.num_running)]

    def expected_dp_ranks(self) -> frozenset:
        return frozenset({0})


class TestReRegistrationFreshness:
    @pytest.mark.asyncio
    async def test_new_registration_waits_for_post_registration_sample(
        self, monkeypatch
    ):
        """I5: a session registering after the sampler went idle must wait for a
        fresh sample, not reuse a globally-latched completion from an earlier
        session's sample.
        """
        import sglang.srt.load_reporter.runtime as runtime_module
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        # Long initial-sample timeout so the test asserts the generation barrier,
        # not a timeout fallback.
        monkeypatch.setattr(runtime_module, "INITIAL_SAMPLE_TIMEOUT_SECONDS", 10.0)

        source = GatedReRegistrationSource()
        rt = LoadReporterRuntime(source, make_server_args())
        try:
            # Router A registers; its initial sample (num_running=1) completes.
            _, session_a = rt.register_session("router-a", 10_000, 30_000)
            first = await asyncio.wait_for(session_a.queue.get(), timeout=1.0)
            assert first is not None
            assert first.ranks[0].num_running_reqs == 1

            # Router A closes; with no active sessions the sampler deactivates.
            session_a.stop()
            sentinel = await asyncio.wait_for(session_a.queue.get(), timeout=1.0)
            assert sentinel is None
            await asyncio.sleep(0.05)  # let on_close deactivate the sampler

            # The next sample will block in flight and would report num_running=5.
            source.gate = True
            source.num_running = 5

            # Router B registers and reactivates the sampler.  A stale global
            # one-shot event would let it emit the old num_running=1 immediately;
            # the generation barrier must make it wait for the fresh sample.
            _, session_b = rt.register_session("router-b", 10_000, 30_000)
            await asyncio.wait_for(source.gated_started.wait(), timeout=1.0)
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(session_b.queue.get(), timeout=0.1)

            # Once the fresh sample lands, Router B emits it.
            source.release.set()
            report = await asyncio.wait_for(session_b.queue.get(), timeout=1.0)
            assert report.ranks[0].num_running_reqs == 5
        finally:
            source.release.set()
            await rt.close()


class TestMultiRouter:
    @pytest.mark.asyncio
    async def test_different_routers_coexist(self):
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        rt = LoadReporterRuntime(FakeSnapshotSource(), make_server_args())
        try:
            ack1, session1 = rt.register_session("r1", 30, 3000)
            ack2, session2 = rt.register_session("r2", 30, 3000)
            # Both sessions receive reports independently.
            reports1 = await drain_queue(session1.queue, 1)
            reports2 = await drain_queue(session2.queue, 1)
            assert len(reports1) >= 1
            assert len(reports2) >= 1
        finally:
            await rt.close()

    @pytest.mark.asyncio
    async def test_min_interval_sampling(self):
        """Sampler must use the shortest session interval (behavioral)."""
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        source = FakeSnapshotSource()
        rt = LoadReporterRuntime(source, make_server_args())
        try:
            # s1 wants 1000ms, s2 wants 30ms: sampler should run ~at 30ms cadence.
            ack1, s1 = rt.register_session("r1", 1000, 5000)
            ack2, s2 = rt.register_session("r2", 30, 5000)
            before = source.get_loads_calls
            await asyncio.sleep(0.2)
            after = source.get_loads_calls
            # At 30ms interval over 200ms we expect at least 4 samples; at
            # 1000ms interval we'd expect at most 1.  Assert a clear majority.
            assert (
                after - before >= 3
            ), f"Expected >=3 samples at 30ms min interval, got {after - before}"
        finally:
            s1.stop()
            s2.stop()
            await rt.close()


class TestSamplerActivation:
    @pytest.mark.asyncio
    async def test_sampler_activates_on_first_session(self):
        """Sampler must sample after a session is registered (behavioral)."""
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        source = FakeSnapshotSource()
        rt = LoadReporterRuntime(source, make_server_args())
        try:
            before = source.get_loads_calls
            ack, session = rt.register_session("r1", 500, 3000)
            await asyncio.sleep(0.15)
            after = source.get_loads_calls
            assert (
                after > before
            ), "Sampler should start sampling when first session is registered"
        finally:
            session.stop()
            await rt.close()

    @pytest.mark.asyncio
    async def test_sampler_deactivates_on_last_close(self):
        """Sampler must stop sampling after the last session closes (behavioral)."""
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        source = FakeSnapshotSource()
        rt = LoadReporterRuntime(source, make_server_args())
        try:
            ack, session = rt.register_session("r1", 500, 3000)
            session.stop()
            # Allow on_close to fire and deactivate the sampler.
            await asyncio.sleep(0.1)
            # Capture call count after deactivation.
            snapshot = source.get_loads_calls
            await asyncio.sleep(0.15)
            after = source.get_loads_calls
            assert (
                after == snapshot
            ), "Sampler should stop sampling after last session closes"
        finally:
            await rt.close()


class TestShutdown:
    @pytest.mark.asyncio
    async def test_timeout_cancels_hanging_sampler_and_sessions(self, monkeypatch):
        import sglang.srt.load_reporter.runtime as runtime_module
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        monkeypatch.setattr(runtime_module, "SHUTDOWN_TIMEOUT_SECONDS", 0.05)
        source = HangingSnapshotSource()
        rt = LoadReporterRuntime(source, make_server_args())
        _, session = rt.register_session("r1", 1000, 3000)
        await asyncio.wait_for(source.started.wait(), timeout=0.5)

        sampler_task = rt._sampler._task
        assert sampler_task is not None
        try:
            await asyncio.wait_for(rt.close(), timeout=0.5)

            assert sampler_task.done()
            assert session._task.done()
            await asyncio.wait_for(rt.close(), timeout=0.1)
        finally:
            tasks = [sampler_task, session._task]
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


class TestDecoratorEvents:
    @pytest.mark.asyncio
    async def test_request_end_refreshes_coalesce_without_early_report(self):
        """Refresh hints update state, but only the deadline publishes it."""
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        source = MutableSnapshotSource()
        rt = LoadReporterRuntime(source, make_server_args())
        try:
            _, session = rt.register_session("r1", 400, 3000)
            initial_report = await asyncio.wait_for(session.queue.get(), timeout=0.5)
            assert initial_report.ranks[0].num_running_reqs == 1

            source.num_running_reqs = 7
            for _ in range(10):
                rt.notify_refresh()

            deadline = time.monotonic() + 0.2
            while source.get_loads_calls < 2 and time.monotonic() < deadline:
                await asyncio.sleep(0.005)
            assert source.get_loads_calls == 2

            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(session.queue.get(), timeout=0.1)

            report = await asyncio.wait_for(session.queue.get(), timeout=0.4)
            assert report.ranks[0].num_running_reqs == 7
        finally:
            await rt.close()

    @pytest.mark.asyncio
    async def test_inflight_sample_does_not_delay_deadline_report(self):
        """A deadline reads the latest completed snapshot without awaiting I/O."""
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        source = BlockingAfterInitialSnapshotSource()
        rt = LoadReporterRuntime(source, make_server_args())
        try:
            _, session = rt.register_session("r1", 150, 3000)
            initial_report = await asyncio.wait_for(session.queue.get(), timeout=0.5)
            assert initial_report.ranks[0].num_running_reqs == 1

            rt.notify_refresh()
            await asyncio.wait_for(source.blocked_sample_started.wait(), timeout=0.2)

            report = await asyncio.wait_for(session.queue.get(), timeout=0.3)
            assert not source.release.is_set()
            assert report.ranks[0].num_running_reqs == 1
        finally:
            source.release.set()
            await rt.close()

    @pytest.mark.asyncio
    async def test_notify_refresh_wakes_sampler(self):
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        source = FakeSnapshotSource()
        rt = LoadReporterRuntime(source, make_server_args())
        try:
            ack, session = rt.register_session("r1", 5000, 30000)
            before = source.get_loads_calls
            rt.notify_refresh()
            await asyncio.sleep(0.1)
            after = source.get_loads_calls
            assert after > before, "notify_refresh should trigger a sample"
        finally:
            session.stop()
            await rt.close()

    @pytest.mark.asyncio
    async def test_notify_source_changed_wakes_sampler(self):
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        source = FakeSnapshotSource()
        rt = LoadReporterRuntime(source, make_server_args())
        try:
            ack, session = rt.register_session("r1", 5000, 30000)
            before = source.get_loads_calls
            rt.notify_source_changed()
            await asyncio.sleep(0.1)
            after = source.get_loads_calls
            assert after > before
        finally:
            session.stop()
            await rt.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
