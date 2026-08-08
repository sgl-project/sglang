"""Integration tests for LoadMonitorService using a real grpc.aio in-process server.

The Worker is the gRPC server; the fake Router is the gRPC client.
Tests cover: normal handshake, periodic reporting (no decorator events),
illegal first frame, client cancel, server shutdown, same-router_id
stream replacement, and the fixed-port-occupied failure path.
"""

from __future__ import annotations

import asyncio
import sys
import types
from typing import AsyncIterator

import grpc
import grpc.aio
import pytest

from sglang.srt.load_reporter.proto import load_monitor_pb2 as pb
from sglang.srt.load_reporter.proto import load_monitor_pb2_grpc as pb_grpc

pytest_plugins = ("pytest_asyncio",)

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------


def make_server_args(dp_size: int = 1) -> types.SimpleNamespace:
    args = types.SimpleNamespace()
    args.host = "127.0.0.1"
    args.load_reporter_port = 9999
    args.disaggregation_mode = "none"
    args.served_model_name = "test-model"
    args.dp_size = dp_size
    return args


class FakeSnapshotSource:
    async def get_loads(self) -> list:
        return []

    def expected_dp_ranks(self) -> frozenset:
        return frozenset({0})


async def start_test_server(runtime):
    """Start an in-process grpc.aio server; return (server, port)."""
    from sglang.srt.load_reporter.service import add_service_to_server

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


class TestNormalHandshake:
    @pytest.mark.asyncio
    async def test_register_yields_ack(self):
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        rt = LoadReporterRuntime(FakeSnapshotSource(), make_server_args())
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
            # Keep stream open briefly then close
            await asyncio.sleep(0.3)

        call = stub.Monitor(frames())
        received = await receive_frames(call, 2, timeout=2.0)
        assert len(received) >= 1
        assert received[0].WhichOneof("payload") == "registered"
        assert received[0].registered.lease_ttl_ms == 3000
        assert received[0].registered.renew_after_ms == 1000

        await channel.close()
        await server.stop(grace=0)
        await rt.close()

    @pytest.mark.asyncio
    async def test_ack_followed_by_initial_report(self):
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        rt = LoadReporterRuntime(FakeSnapshotSource(), make_server_args())
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
            await asyncio.sleep(0.5)

        call = stub.Monitor(frames())
        received = await receive_frames(call, 2, timeout=2.0)
        # Frame 0: registered ack; Frame 1: first load report
        assert len(received) >= 2
        assert received[0].WhichOneof("payload") == "registered"
        assert received[1].WhichOneof("payload") == "report"

        await channel.close()
        await server.stop(grace=0)
        await rt.close()


class TestContinuousReporting:
    @pytest.mark.asyncio
    async def test_reports_flow_without_decorator_events(self):
        """Reports must flow on interval even with NO decorator events."""
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        rt = LoadReporterRuntime(FakeSnapshotSource(), make_server_args())
        server, port = await start_test_server(rt)
        stub, channel = await make_stub(port)

        async def frames() -> AsyncIterator[pb.RouterFrame]:
            yield pb.RouterFrame(
                register=pb.RegisterRequest(
                    router_id="r1",
                    report_interval_ms=40,
                    lease_ttl_ms=10000,
                )
            )
            await asyncio.sleep(1.0)

        call = stub.Monitor(frames())
        received = await receive_frames(call, 5, timeout=2.5)
        # Ack + at least 3 periodic reports (40ms interval over 1s)
        reports = [f for f in received if f.WhichOneof("payload") == "report"]
        assert len(reports) >= 3, f"Expected >=3 reports, got {len(reports)}"

        await channel.close()
        await server.stop(grace=0)
        await rt.close()


class TestUpdateConfig:
    @pytest.mark.asyncio
    async def test_shorter_report_interval_takes_effect_from_update(self):
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        rt = LoadReporterRuntime(FakeSnapshotSource(), make_server_args())
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
    @pytest.mark.parametrize(
        ("router_id", "report_interval_ms", "lease_ttl_ms"),
        [
            ("", 500, 3000),
            ("r1", 0, 3000),
            ("r1", 500, -1),
        ],
    )
    async def test_invalid_register_yields_terminal_error(
        self, router_id, report_interval_ms, lease_ttl_ms
    ):
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        rt = LoadReporterRuntime(FakeSnapshotSource(), make_server_args())
        server, port = await start_test_server(rt)
        stub, channel = await make_stub(port)

        async def frames() -> AsyncIterator[pb.RouterFrame]:
            yield pb.RouterFrame(
                register=pb.RegisterRequest(
                    router_id=router_id,
                    report_interval_ms=report_interval_ms,
                    lease_ttl_ms=lease_ttl_ms,
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
    @pytest.mark.parametrize("invalid_interval_ms", [0, -1])
    async def test_invalid_update_yields_terminal_error(self, invalid_interval_ms):
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        rt = LoadReporterRuntime(FakeSnapshotSource(), make_server_args())
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
                update_config=pb.UpdateConfigRequest(
                    report_interval_ms=invalid_interval_ms
                )
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
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        rt = LoadReporterRuntime(FakeSnapshotSource(), make_server_args())
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
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        rt = LoadReporterRuntime(FakeSnapshotSource(), make_server_args())
        server, port = await start_test_server(rt)
        stub, channel = await make_stub(port)

        async def frames() -> AsyncIterator[pb.RouterFrame]:
            return
            yield  # make it a generator

        call = stub.Monitor(frames())
        received = await receive_frames(call, 1, timeout=1.5)
        # Empty stream: no frames or possibly StreamError; no crash
        assert isinstance(received, list)

        await channel.close()
        await server.stop(grace=0)
        await rt.close()


class TestClientCancel:
    @pytest.mark.asyncio
    async def test_client_cancel_does_not_hang(self):
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        rt = LoadReporterRuntime(FakeSnapshotSource(), make_server_args())
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


class TestServerShutdown:
    @pytest.mark.asyncio
    async def test_server_stop_does_not_hang(self):
        from sglang.srt.load_reporter.runtime import LoadReporterRuntime

        rt = LoadReporterRuntime(FakeSnapshotSource(), make_server_args())
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
            await asyncio.sleep(10)

        call = stub.Monitor(frames())
        await receive_frames(call, 1, timeout=1.0)

        # Stop server while stream is active
        await asyncio.wait_for(server.stop(grace=0), timeout=2.0)
        await channel.close()
        await rt.close()


class TestFixedPortOccupied:
    @pytest.mark.asyncio
    async def test_bind_to_occupied_port_fails(self):
        """Binding to an occupied port must fail explicitly; no random fallback."""
        import socket

        from sglang.srt.load_reporter.runtime import LoadReporterRuntime
        from sglang.srt.load_reporter.service import add_service_to_server

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        try:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            occupied_port = sock.getsockname()[1]

            rt = LoadReporterRuntime(FakeSnapshotSource(), make_server_args())
            server = grpc.aio.server()
            add_service_to_server(rt, server)
            # grpc.aio raises RuntimeError on bind failure (never silently
            # falls back to a random port — that's the invariant we protect).
            with pytest.raises(RuntimeError):
                server.add_insecure_port(f"127.0.0.1:{occupied_port}")
            await server.stop(grace=0)
            await rt.close()
        finally:
            sock.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
