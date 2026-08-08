"""Contract tests for the load reporter composition root.

``start_load_reporter`` is the single serving-mode-agnostic entry point.  These
tests observe behaviour only through:
  - the returned handle (or ``None``)
  - a real ``grpc.aio`` client connecting to the owner-path listener
  - the real refresh notifier IPC ``send`` callback on the worker path
  - decorator-driven ``COMPLETION`` events waking the sampler

No test-only production getters, counters, or state introspection are used.
"""

from __future__ import annotations

import asyncio
import socket
import sys
import types
from typing import Any, AsyncIterator, List, Optional

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


class FakeOwner:
    """Owner with a statically decorated generate_request (HTTP/Engine shape)."""

    def __init__(self, port: Optional[int]) -> None:
        self.server_args = make_server_args(port=port)

    def make_generate(self):
        from sglang.srt.load_reporter.decorator import enable_load_monitor

        @enable_load_monitor("request_lifecycle")
        async def generate_request(self, n: int = 3):
            for i in range(n):
                yield i

        return generate_request


class FakeWorkerOwner:
    """Multi-tokenizer HTTP worker: statically decorated generate_request plus a
    synchronous _dispatch_to_scheduler that the notifier drives over IPC."""

    def __init__(self, port: Optional[int]) -> None:
        self.server_args = make_server_args(port=port, worker_num=2)
        self.sent: List[Any] = []

    def _dispatch_to_scheduler(self, obj: Any) -> None:
        self.sent.append(obj)

    def make_generate(self):
        from sglang.srt.load_reporter.decorator import enable_load_monitor

        @enable_load_monitor("request_lifecycle")
        async def generate_request(self, n: int = 2):
            for i in range(n):
                yield i

        return generate_request


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
            event_owner=FakeOwner(port=None),
        )
        assert handle is None

    def test_no_grpc_import_when_disabled(self):
        """Importing lifecycle + calling with port=None must not import grpc.

        Runs in a clean subprocess so already-imported grpc in this session
        does not mask a regression.
        """
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
                start_load_reporter(args, None, event_owner=None)
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
            cwd="python",
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
        handle = await start_load_reporter(args, source, event_owner=None)
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
            make_server_args(port=port), FakeSnapshotSource(), event_owner=None
        )
        assert handle is not None
        await handle.close()
        await handle.close()  # must not raise

    @pytest.mark.asyncio
    async def test_event_owner_binding_wakes_sampler(self):
        """A decorated request-lifecycle COMPLETION must wake the sampler."""
        from sglang.srt.load_reporter.lifecycle import start_load_reporter

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        source = FakeSnapshotSource()
        owner = FakeOwner(port=port)
        generate = owner.make_generate()
        handle = await start_load_reporter(
            make_server_args(port=port), source, event_owner=owner
        )
        assert handle is not None
        try:
            stub, channel = await start_client(port)

            async def frames() -> AsyncIterator[pb.RouterFrame]:
                yield pb.RouterFrame(
                    register=pb.RegisterRequest(
                        router_id="r1",
                        report_interval_ms=100000,  # effectively no periodic tick
                        lease_ttl_ms=100000,
                    )
                )
                await asyncio.sleep(1.0)

            call = stub.Monitor(frames())
            # Ack + initial report.
            await receive_frames(call, 2, timeout=1.0)
            before = source.get_loads_calls
            # Drive a decorated generate to completion → COMPLETION → notify.
            collected = [x async for x in generate(owner)]
            assert collected == [0, 1, 2]
            await asyncio.sleep(0.2)
            after = source.get_loads_calls
            assert after > before, "request-end COMPLETION should wake the sampler"
            await channel.close()
        finally:
            await handle.close()


class TestHttpLifecycleAdapter:
    @pytest.mark.asyncio
    async def test_single_tokenizer_start_returns_handle_and_close_releases_port(self):
        """HTTP single-tokenizer calls start_load_reporter directly (symmetric
        with the native-gRPC path); close() releases the listener port."""
        from sglang.srt.load_reporter.lifecycle import start_load_reporter
        from sglang.srt.load_reporter.sampler import ManagerLoadSnapshotSource

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        owner = FakeOwner(port=port)
        args = make_server_args(port=port)
        handle = await start_load_reporter(
            args,
            ManagerLoadSnapshotSource(owner, range(args.dp_size)),
            event_owner=owner,
        )
        assert handle is not None

        await handle.close()

        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind(("127.0.0.1", port))
        finally:
            probe.close()


# ---------------------------------------------------------------------------
# Group 3: IPC-worker path — multi-tokenizer HTTP worker
# ---------------------------------------------------------------------------


class TestIpcWorkerPath:
    @pytest.mark.asyncio
    async def test_worker_forwards_refresh_over_ipc_without_listener(self):
        from sglang.srt.load_reporter.lifecycle import start_load_reporter
        from sglang.srt.managers.io_struct import LoadReporterRefreshIpcReq

        owner = FakeWorkerOwner(port=40404)
        generate = owner.make_generate()
        handle = await start_load_reporter(owner.server_args, None, event_owner=owner)
        assert handle is not None
        try:
            # No listener should be bound on the worker path.
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            can_bind = probe.connect_ex(("127.0.0.1", 40404)) != 0
            probe.close()
            assert can_bind, "worker path must not bind a reporter port"

            # Drive a decorated generate to completion → notifier.notify → send.
            collected = [x async for x in generate(owner)]
            assert collected == [0, 1]
            # Wait past the coalesce window for the notifier to fire once.
            await asyncio.sleep(0.2)
            refreshes = [
                m for m in owner.sent if isinstance(m, LoadReporterRefreshIpcReq)
            ]
            assert len(refreshes) >= 1
        finally:
            await handle.close()


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
                    event_owner=None,
                )
        finally:
            blocker.close()

        # A subsequent start on a free port must still succeed (no global
        # state was corrupted by the failed bind).
        free = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        free.bind(("127.0.0.1", 0))
        free_port = free.getsockname()[1]
        free.close()
        handle = await start_load_reporter(
            make_server_args(port=free_port), FakeSnapshotSource(), event_owner=None
        )
        assert handle is not None
        await handle.close()


# ---------------------------------------------------------------------------
# Group 5: handle delegation (used by MultiTokenizerRouter)
# ---------------------------------------------------------------------------


class TestHandleDelegation:
    @pytest.mark.asyncio
    async def test_notify_refresh_delegates_to_runtime(self):
        from sglang.srt.load_reporter.lifecycle import start_load_reporter

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        source = FakeSnapshotSource()
        handle = await start_load_reporter(
            make_server_args(port=port), source, event_owner=None
        )
        assert handle is not None
        try:
            stub, channel = await start_client(port)

            async def frames() -> AsyncIterator[pb.RouterFrame]:
                yield pb.RouterFrame(
                    register=pb.RegisterRequest(
                        router_id="r1", report_interval_ms=100000, lease_ttl_ms=100000
                    )
                )
                await asyncio.sleep(1.0)

            call = stub.Monitor(frames())
            await receive_frames(call, 2, timeout=1.0)
            before = source.get_loads_calls
            handle.notify_refresh()
            await asyncio.sleep(0.2)
            assert source.get_loads_calls > before
            await channel.close()
        finally:
            await handle.close()

    @pytest.mark.asyncio
    async def test_notify_refresh_noop_on_ipc_worker_handle(self):
        from sglang.srt.load_reporter.lifecycle import start_load_reporter

        owner = FakeWorkerOwner(port=40405)
        handle = await start_load_reporter(owner.server_args, None, event_owner=owner)
        assert handle is not None
        try:
            handle.notify_refresh()  # must be a safe no-op
            assert handle.update_expected_dp_ranks(range(2)) is False
        finally:
            await handle.close()

    @pytest.mark.asyncio
    async def test_worker_completion_wakes_owner_sampler_before_timer(self):
        from sglang.srt.load_reporter.lifecycle import start_load_reporter
        from sglang.srt.managers.io_struct import LoadReporterRefreshIpcReq

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        source = FakeSnapshotSource()
        owner_handle = await start_load_reporter(
            make_server_args(port=port), source, event_owner=None
        )
        assert owner_handle is not None
        worker_handle = None
        channel = None
        try:
            stub, channel = await start_client(port)

            async def frames() -> AsyncIterator[pb.RouterFrame]:
                yield pb.RouterFrame(
                    register=pb.RegisterRequest(
                        router_id="multi-owner",
                        report_interval_ms=100_000,
                        lease_ttl_ms=100_000,
                    )
                )
                await asyncio.sleep(1.0)

            call = stub.Monitor(frames())
            await receive_frames(call, 2, timeout=1.0)

            worker = FakeWorkerOwner(port=port)

            def dispatch(message: Any) -> None:
                worker.sent.append(message)
                if isinstance(message, LoadReporterRefreshIpcReq):
                    owner_handle.notify_refresh()

            worker._dispatch_to_scheduler = dispatch
            generate = worker.make_generate()
            worker_handle = await start_load_reporter(
                worker.server_args, None, event_owner=worker
            )
            assert worker_handle is not None
            calls_before = source.get_loads_calls

            assert [item async for item in generate(worker)] == [0, 1]
            deadline = asyncio.get_running_loop().time() + 0.5
            while (
                source.get_loads_calls == calls_before
                and asyncio.get_running_loop().time() < deadline
            ):
                await asyncio.sleep(0.01)

            assert any(isinstance(m, LoadReporterRefreshIpcReq) for m in worker.sent)
            assert source.get_loads_calls > calls_before
        finally:
            if worker_handle is not None:
                await worker_handle.close()
            if channel is not None:
                await channel.close()
            await owner_handle.close()


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

        class FakeNotifier:
            async def close(self):
                steps.append("notifier.close")

        handle = LoadReporterHandle()
        handle._server = FakeServer()
        handle._runtime = FakeRuntime()
        handle._notifier = FakeNotifier()
        handle._unbind = lambda: steps.append("unbind")
        handle._restore = lambda: steps.append("restore")

        # First caller enters close() and is cancelled while inside server.stop.
        first = asyncio.create_task(handle.close())
        await asyncio.wait_for(server_entered.wait(), timeout=1.0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        # The shared teardown must survive the cancelled caller: release the
        # blocking step and let a second caller await the same close task.
        release_server.set()
        await handle.close()

        assert steps == [
            "server.stop.enter",
            "server.stop.exit",
            "runtime.close",
            "notifier.close",
            "unbind",
            "restore",
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

        handle = LoadReporterHandle()
        handle._runtime = FakeRuntime()
        handle._unbind = lambda: steps.append("unbind")

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
        assert steps == ["runtime.close", "unbind"]


class TestLifecycleShadowRestoration:
    def test_restores_preexisting_instance_override(self):
        from sglang.srt.load_reporter.lifecycle import (
            LoadReporterHandle,
            _install_lifecycle_shadow,
        )

        class Owner:
            async def generate_request(self):
                yield "class"

        async def instance_override():
            yield "instance"

        owner = Owner()
        owner.generate_request = instance_override
        original = owner.__dict__["generate_request"]
        handle = LoadReporterHandle()

        _install_lifecycle_shadow(handle, owner, "generate_request")
        assert owner.__dict__["generate_request"] is not original

        assert handle._restore is not None
        handle._restore()

        assert owner.__dict__["generate_request"] is original


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
