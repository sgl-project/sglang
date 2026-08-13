"""Contract tests for the load reporter composition root."""

from __future__ import annotations

import asyncio
import socket
import sys
import types
from typing import AsyncIterator, List, Optional

import grpc
import grpc.aio
import pytest

from sglang.srt.load_reporter.proto import load_monitor_pb2 as pb
from sglang.srt.load_reporter.proto import load_monitor_pb2_grpc as pb_grpc

pytest_plugins = ("pytest_asyncio",)

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


# ---------------------------------------------------------------------------
# Fixtures (all test-only, live in test/)
# ---------------------------------------------------------------------------


def make_server_args(
    *, port: Optional[int], dp_size: int = 1, worker_num: int = 1
) -> types.SimpleNamespace:
    args = types.SimpleNamespace()
    args.host = "127.0.0.1"
    args.load_reporter_port = port
    args.disaggregation_mode = "none"
    args.served_model_name = "test-model"
    args.dp_size = dp_size
    args.tokenizer_worker_num = worker_num
    return args


class FakeSnapshotSource:
    """Minimal LoadSnapshotSource; counts get_loads calls."""

    def __init__(self, dp_size: int = 1) -> None:
        self._dp_size = dp_size
        self.get_loads_calls = 0

    async def get_loads(self) -> list:
        self.get_loads_calls += 1
        return []

    def expected_dp_ranks(self) -> frozenset:
        return frozenset(range(self._dp_size))


class RankAwareSource(FakeSnapshotSource):
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


async def start_client(port: int):
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    return pb_grpc.LoadMonitorServiceStub(channel), channel


async def receive_frames(call, count: int, timeout: float = 3.0) -> list:
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
        except (asyncio.TimeoutError, Exception):
            break
    return frames


# ---------------------------------------------------------------------------
# Group 1: disabled — port is None
# ---------------------------------------------------------------------------


class TestDisabled:
    @pytest.mark.asyncio
    async def test_returns_none_when_port_none(self):
        from sglang.srt.load_reporter.lifecycle import start_load_reporter

        handle = await start_load_reporter(
            make_server_args(port=None),
            FakeSnapshotSource(),
        )
        assert handle is None

    @pytest.mark.asyncio
    async def test_rejects_missing_snapshot_source_when_enabled(self):
        from sglang.srt.load_reporter.lifecycle import start_load_reporter

        with pytest.raises(ValueError, match="snapshot_source is required"):
            await start_load_reporter(
                make_server_args(port=30100),
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
        from sglang.srt.load_reporter.lifecycle import start_load_reporter

        # Bind an ephemeral port first, then hand it to the reporter.
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        source = FakeSnapshotSource()
        args = make_server_args(port=port)
        handle = await start_load_reporter(args, source)
        assert handle is not None
        try:
            stub, channel = await start_client(port)

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
            await channel.close()
        finally:
            await handle.close()

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self):
        from sglang.srt.load_reporter.lifecycle import start_load_reporter

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        handle = await start_load_reporter(
            make_server_args(port=port), FakeSnapshotSource()
        )
        assert handle is not None
        await handle.close()
        await handle.close()  # must not raise

    @pytest.mark.asyncio
    async def test_no_sample_before_periodic_deadline(self):
        """The owner waits for its periodic deadline after initial sampling."""
        from sglang.srt.load_reporter.lifecycle import start_load_reporter

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        source = FakeSnapshotSource()
        handle = await start_load_reporter(make_server_args(port=port), source)
        assert handle is not None
        try:
            stub, channel = await start_client(port)

            # Register with a very long interval so periodic sampling won't race our measurement.
            async def frames() -> AsyncIterator[pb.RouterFrame]:
                yield pb.RouterFrame(
                    register=pb.RegisterRequest(
                        router_id="r1",
                        report_interval_ms=100_000,
                        lease_ttl_ms=100_000,
                    )
                )
                await asyncio.sleep(1.0)

            call = stub.Monitor(frames())
            await receive_frames(call, 2, timeout=1.0)
            before = source.get_loads_calls
            await asyncio.sleep(0.3)
            after = source.get_loads_calls
            assert after == before, (
                "the fire loop must wait for the periodic deadline after its "
                "initial report"
            )
            await channel.close()
        finally:
            await handle.close()


# ---------------------------------------------------------------------------
# Group 3: HTTP single-tokenizer lifecycle
# ---------------------------------------------------------------------------


class TestHttpLifecycleAdapter:
    @pytest.mark.asyncio
    async def test_single_tokenizer_start_returns_handle_and_close_releases_port(self):
        """HTTP single-tokenizer calls start_load_reporter directly; close() releases the port."""
        from sglang.srt.load_reporter.lifecycle import start_load_reporter
        from sglang.srt.load_reporter.snapshot_source import ManagerLoadSnapshotSource

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        args = make_server_args(port=port)
        handle = await start_load_reporter(
            args,
            ManagerLoadSnapshotSource(FakeSnapshotSource(), range(args.dp_size)),
        )
        assert handle is not None

        await handle.close()

        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind(("127.0.0.1", port))
        finally:
            probe.close()


# ---------------------------------------------------------------------------
# Group 4: fixed port occupied — explicit failure, no state corruption
# ---------------------------------------------------------------------------


class TestFixedPortOccupied:
    @pytest.mark.asyncio
    async def test_occupied_port_raises_and_recovers(self):
        from sglang.srt.load_reporter.lifecycle import start_load_reporter

        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        occupied = blocker.getsockname()[1]
        try:
            with pytest.raises(RuntimeError):
                await start_load_reporter(
                    make_server_args(port=occupied),
                    FakeSnapshotSource(),
                )
        finally:
            blocker.close()

        # A later start on a free port must succeed (no global state corrupted by the failed bind).
        free = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        free.bind(("127.0.0.1", 0))
        free_port = free.getsockname()[1]
        free.close()
        handle = await start_load_reporter(
            make_server_args(port=free_port), FakeSnapshotSource()
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
        from sglang.srt.load_reporter.lifecycle import start_load_reporter

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        source = RankAwareSource(dp_size=1)
        handle = await start_load_reporter(make_server_args(port=port), source)
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
    """I1: a close() cancelled mid-await must not abandon remaining teardown."""

    @pytest.mark.asyncio
    async def test_cancelled_close_still_completes_teardown(self):
        from sglang.srt.load_reporter.lifecycle import LoadReporterHandle

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
        from sglang.srt.load_reporter.lifecycle import LoadReporterHandle

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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
