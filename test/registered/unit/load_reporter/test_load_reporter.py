"""Core unit coverage for engine-initiated load reporting.

The suite intentionally focuses on the public control boundary, request-level
refresh coalescing, sampler lifecycle, and multi-worker IPC correlation. The
GPU test covers the assembled server path.
"""

from __future__ import annotations

import asyncio
import unittest
from ipaddress import IPv4Address
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.load_reporter.ipc import (
    LoadReporterControlProxy,
    LoadReporterRefreshNotifier,
)
from sglang.srt.load_reporter.registration import (
    StartReportingRequest,
    start_reporting,
)
from sglang.srt.load_reporter.runtime import LoadReporterRuntime
from sglang.srt.load_reporter.sampler import (
    LoadSampler,
    RouterLoadSnapshotSource,
    TokenizerManagerLoadSnapshotSource,
)
from sglang.srt.managers.io_struct import (
    LoadReporterIpcCode,
    LoadReporterRefreshReason,
    LoadReporterStartIpcReqOutput,
    LoadReporterStateBroadcastReq,
)
from sglang.srt.managers.load_snapshot import LoadSnapshot
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class AsyncCustomTestCase(CustomTestCase, unittest.IsolatedAsyncioTestCase):
    """Run async tests with SGLang's standard retry and cleanup behavior."""


class _Manager:
    """Minimal tokenizer-manager source used by adapter tests."""

    elastic_worker_count = 2

    def __init__(self) -> None:
        """Initialize the captured include arguments.

        Returns:
            None.
        """
        self.includes: list[object] = []

    async def get_loads(self, include=None) -> list[LoadSnapshot]:
        """Return two rank snapshots and record the requested sections.

        Args:
            include: Optional load-snapshot sections requested by the caller.

        Returns:
            One snapshot for each simulated DP rank.
        """
        self.includes.append(include)
        return [LoadSnapshot(dp_rank=0), LoadSnapshot(dp_rank=1)]


class _Reader:
    """Minimal shared-memory reader used by the router adapter test."""

    def read_all(self) -> list[LoadSnapshot]:
        """Return the currently published rank snapshots.

        Returns:
            A single simulated DP-rank snapshot.
        """
        return [LoadSnapshot(dp_rank=0)]


class _CountingSource:
    """Snapshot source that exposes a deterministic call counter."""

    def __init__(self, *, blocked: bool = False) -> None:
        """Initialize the source.

        Args:
            blocked: Whether reads should wait for an explicit release.

        Returns:
            None.
        """
        self.call_count = 0
        self._release = asyncio.Event()
        if not blocked:
            self._release.set()

    async def get_loads(self) -> list[LoadSnapshot]:
        """Count one read, wait for release, and return one rank.

        Returns:
            A single simulated DP-rank snapshot.
        """
        self.call_count += 1
        await self._release.wait()
        return [LoadSnapshot(dp_rank=0)]

    def expected_dp_ranks(self) -> frozenset[int]:
        """Return the authoritative DP-rank set.

        Returns:
            The single expected rank.
        """
        return frozenset({0})

    def release(self) -> None:
        """Unblock pending snapshot reads.

        Returns:
            None.
        """
        self._release.set()


class _FakeStore:
    """No-op snapshot destination used to isolate sampler behavior."""

    def apply_full_snapshot(
        self,
        loads,
        *,
        expected_dp_ranks,
        collected_at_unix_ms,
        collected_at_monotonic,
    ) -> None:
        """Accept a completed sampler publication.

        Args:
            loads: Rank snapshots returned by the source.
            expected_dp_ranks: Authoritative ranks for validation.
            collected_at_unix_ms: Completion wall-clock time.
            collected_at_monotonic: Completion monotonic time.

        Returns:
            None.
        """
        del (
            loads,
            expected_dp_ranks,
            collected_at_unix_ms,
            collected_at_monotonic,
        )

    def record_error(self, exc: Exception) -> None:
        """Accept a sampler error without retaining it.

        Args:
            exc: Sampling exception raised by the source or store.

        Returns:
            None.
        """
        del exc


async def _wait_for_calls(source: _CountingSource, expected: int) -> None:
    """Wait until a source reaches an expected call count.

    Args:
        source: Counting source to observe.
        expected: Minimum call count required.

    Returns:
        None.

    Raises:
        TimeoutError: If the expected count is not reached promptly.
    """
    deadline = asyncio.get_running_loop().time() + 1.0
    while source.call_count < expected:
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError(
                f"source reached {source.call_count} calls; expected {expected}"
            )
        await asyncio.sleep(0)


def _payload(port: int = 8080) -> StartReportingRequest:
    """Build a valid load-reporter registration payload.

    Args:
        port: Target Router gRPC port.

    Returns:
        A validated registration request.
    """
    return StartReportingRequest(
        ip=IPv4Address("127.0.0.1"),
        port=port,
        report_interval_ms=1000,
        lease_ttl_ms=5000,
    )


def _ok(request_id: str) -> LoadReporterStartIpcReqOutput:
    """Build a successful correlated IPC response.

    Args:
        request_id: Correlation identifier copied from the request.

    Returns:
        A successful start-reporting response.
    """
    return LoadReporterStartIpcReqOutput(
        request_id=request_id,
        code=LoadReporterIpcCode.OK,
        status="reporting",
        lease_ttl_ms=5000,
        renew_after_ms=2500,
    )


class TestLoadSampler(AsyncCustomTestCase):
    """Cover source adaptation, request coalescing, and activation lifecycle."""

    async def test_snapshot_sources_preserve_rank_contract(self) -> None:
        """Both source adapters expose snapshots and authoritative ranks."""
        manager = _Manager()
        tokenizer_source = TokenizerManagerLoadSnapshotSource(manager)
        self.assertEqual(
            [snapshot.dp_rank for snapshot in await tokenizer_source.get_loads()],
            [0, 1],
        )
        self.assertEqual(manager.includes, [["core"]])
        self.assertEqual(tokenizer_source.expected_dp_ranks(), frozenset({0, 1}))

        router_source = RouterLoadSnapshotSource(_Reader(), {0})
        self.assertEqual(
            [snapshot.dp_rank for snapshot in await router_source.get_loads()],
            [0],
        )
        self.assertFalse(router_source.update_expected_dp_ranks({0}))
        self.assertTrue(router_source.update_expected_dp_ranks({0, 1}))

    async def test_request_refreshes_coalesce_during_inflight_read(self) -> None:
        """Many request hints during one read produce one follow-up read."""
        source = _CountingSource(blocked=True)
        sampler = LoadSampler(source, _FakeStore(), interval_provider=lambda: None)
        try:
            sampler.activate()
            await _wait_for_calls(source, 1)
            for _ in range(4):
                sampler.notify_refresh()
            source.release()
            await _wait_for_calls(source, 2)
            await asyncio.sleep(0)
            self.assertEqual(source.call_count, 2)
        finally:
            await sampler.close()

    async def test_deactivate_blocks_refresh_until_reactivation(self) -> None:
        """The last monitor deactivates sampling without preventing reuse."""
        source = _CountingSource()
        sampler = LoadSampler(source, _FakeStore(), interval_provider=lambda: None)
        try:
            sampler.activate()
            await _wait_for_calls(source, 1)
            sampler.deactivate()
            await asyncio.sleep(0)
            baseline = source.call_count

            sampler.notify_refresh()
            await asyncio.sleep(0.02)
            self.assertEqual(source.call_count, baseline)

            sampler.activate()
            await _wait_for_calls(source, baseline + 1)
        finally:
            await sampler.close()

    def test_runtime_schedule_drives_sampler_activation(self) -> None:
        """Runtime transitions activate, reschedule, and deactivate the sampler."""
        runtime = object.__new__(LoadReporterRuntime)
        runtime._sampler = Mock()
        runtime._manager = SimpleNamespace(monitor_count=1)
        runtime._last_active = False
        runtime._active_changed = Mock()

        runtime._on_schedule_changed()
        runtime._sampler.activate.assert_called_once_with()
        runtime._active_changed.assert_called_once_with(True)

        runtime._on_schedule_changed()
        runtime._sampler.notify_schedule_changed.assert_called_once_with()

        runtime._manager.monitor_count = 0
        runtime._on_schedule_changed()
        runtime._sampler.deactivate.assert_called_once_with()
        self.assertEqual(runtime._active_changed.call_args_list[-1].args, (False,))


class TestLoadReporterControl(AsyncCustomTestCase):
    """Cover the internal control endpoint and multi-worker IPC semantics."""

    def test_start_reporting_has_no_dedicated_authentication(self) -> None:
        """Verify the internal Router control endpoint has no dedicated auth policy.

        Returns:
            None.
        """
        self.assertFalse(hasattr(start_reporting, "_auth_level"))

    async def test_out_of_order_ipc_responses_match_request_ids(self) -> None:
        """Reversed owner responses still resolve the corresponding callers."""
        sent = []
        proxy = LoadReporterControlProxy(sent.append, timeout_seconds=1)
        first = asyncio.create_task(proxy.start_reporting(_payload(1), "worker"))
        second = asyncio.create_task(proxy.start_reporting(_payload(2), "worker"))
        await asyncio.sleep(0)

        proxy.handle_response(_ok(sent[1].request_id))
        proxy.handle_response(_ok(sent[0].request_id))

        self.assertEqual((await first).status, "reporting")
        self.assertEqual((await second).status, "reporting")
        self.assertEqual(proxy.pending_count, 0)

    async def test_request_notifications_coalesce_with_abort_precedence(self) -> None:
        """Request-level events merge into one highest-priority refresh hint."""
        sent = []
        notifier = LoadReporterRefreshNotifier("worker", sent.append)
        await notifier.start()
        try:
            notifier.handle_state(
                LoadReporterStateBroadcastReq(active=True, coalesce_window_ms=10)
            )
            notifier.notify(LoadReporterRefreshReason.DISPATCH, 2)
            notifier.notify(LoadReporterRefreshReason.COMPLETION, 3)
            notifier.notify(LoadReporterRefreshReason.ABORT, 1)
            await asyncio.sleep(0.03)

            self.assertEqual(len(sent), 1)
            self.assertEqual(sent[0].event_count, 6)
            self.assertEqual(sent[0].reason, LoadReporterRefreshReason.ABORT)
        finally:
            await notifier.close()

    async def test_inactive_state_discards_pending_request_refresh(self) -> None:
        """A false active broadcast cancels an unsent request refresh window."""
        sent = []
        notifier = LoadReporterRefreshNotifier("worker", sent.append)
        await notifier.start()
        try:
            notifier.handle_state(
                LoadReporterStateBroadcastReq(active=True, coalesce_window_ms=20)
            )
            notifier.notify(LoadReporterRefreshReason.COMPLETION, 1)
            notifier.handle_state(
                LoadReporterStateBroadcastReq(active=False, coalesce_window_ms=20)
            )
            await asyncio.sleep(0.04)
            self.assertEqual(sent, [])
        finally:
            await notifier.close()


if __name__ == "__main__":
    unittest.main()
