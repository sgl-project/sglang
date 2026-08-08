"""Standalone SMG RPC reporter integration in ``grpc_server.serve_grpc``.

The external ``smg-grpc-servicer`` package is NOT required: these tests inject a
fake ``_serve_grpc`` at the existing import boundary
(``smg_grpc_servicer.sglang.server.serve_grpc``) and monkeypatch the sidecar
helpers.  They verify that SGLang applies the SAME reporter lifecycle +
``enable_load_monitor("request_lifecycle")`` decorator to the current
``GrpcRequestManager`` bound method, guarded by the ``inspect.signature``
capability check, and cleaned up on normal exit / failure / cancellation.
"""

from __future__ import annotations

import asyncio
import socket
import sys
import types
from typing import Any, AsyncIterator, Optional

import grpc
import grpc.aio
import pytest

from sglang.srt.load_reporter.proto import load_monitor_pb2 as pb
from sglang.srt.load_reporter.proto import load_monitor_pb2_grpc as pb_grpc

pytest_plugins = ("pytest_asyncio",)

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


# ---------------------------------------------------------------------------
# Fakes (test-only)
# ---------------------------------------------------------------------------


def make_server_args(
    *, port: Optional[int], sidecar_port: int
) -> types.SimpleNamespace:
    args = types.SimpleNamespace()
    args.host = "127.0.0.1"
    args.port = sidecar_port - 1
    args.smg_http_sidecar_port = sidecar_port
    args.enable_metrics = False
    args.load_reporter_port = port
    args.disaggregation_mode = "none"
    args.served_model_name = "test-model"
    args.dp_size = 1
    return args


class FakeRequestManager:
    """Stand-in for smg's GrpcRequestManager: exposes server_args, get_loads,
    and an async-generator generate_request (undecorated in the class body)."""

    def __init__(self, server_args: Any) -> None:
        self.server_args = server_args
        self.get_loads_calls = 0

    async def get_loads(self, include=None) -> list:
        self.get_loads_calls += 1
        return []

    async def generate_request(self, obj: Any = None, request: Any = None):
        for i in range(3):
            yield i
            await asyncio.sleep(0)


def free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def port_is_free(port: int) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = probe.connect_ex(("127.0.0.1", port))
    probe.close()
    return result != 0


def install_fake_smg(capability: bool):
    """Inject a fake smg_grpc_servicer.sglang.server module.

    Returns a holder whose ``.serve`` is the fake ``_serve_grpc``.  When
    ``capability`` is True the fake accepts ``on_request_manager_ready`` (so the
    inspect.signature check passes); otherwise it does not.
    """
    holder = types.SimpleNamespace(
        request_manager=None,
        stop_event=None,  # set per-test to control server lifetime
        raise_on_serve=None,
    )

    if capability:

        async def fake_serve_grpc(
            server_args, model_info, on_request_manager_ready=None
        ):
            rm = FakeRequestManager(server_args)
            holder.request_manager = rm
            if on_request_manager_ready is not None:
                await on_request_manager_ready(rm, server_args, {})
            if holder.raise_on_serve is not None:
                raise holder.raise_on_serve
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
    """Neutralise the aiohttp sidecar so tests exercise only reporter wiring."""
    from sglang.srt.entrypoints import grpc_server

    async def _noop_sidecar(host, port, app):
        return types.SimpleNamespace(cleanup=_acleanup)

    async def _acleanup():
        return None

    monkeypatch.setattr(grpc_server, "_start_sidecar_server", _noop_sidecar)
    monkeypatch.setattr(grpc_server, "_add_admin_routes", lambda app, rm: None)
    yield


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


class TestReporterEnabledWithCapability:
    @pytest.mark.asyncio
    async def test_client_connects_and_shadow_restored_on_exit(self, isolate_sidecar):
        from sglang.srt.entrypoints import grpc_server

        port = free_port()
        args = make_server_args(port=port, sidecar_port=free_port())
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
            # generate_request is shadowed on THIS instance only.
            assert "generate_request" in rm.__dict__

            stub, channel = await start_client(port)

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

        # After serve_grpc returns: port freed, instance shadow removed.
        assert port_is_free(port)
        assert "generate_request" not in rm.__dict__

    @pytest.mark.asyncio
    async def test_request_end_wakes_sampler(self, isolate_sidecar):
        from sglang.srt.entrypoints import grpc_server

        port = free_port()
        args = make_server_args(port=port, sidecar_port=free_port())
        holder = install_fake_smg(capability=True)
        holder.stop_event = asyncio.Event()
        try:
            serve_task = asyncio.ensure_future(grpc_server.serve_grpc(args))
            for _ in range(50):
                await asyncio.sleep(0.02)
                if holder.request_manager is not None and not port_is_free(port):
                    break
            rm = holder.request_manager
            assert rm is not None

            stub, channel = await start_client(port)

            async def frames() -> AsyncIterator[pb.RouterFrame]:
                yield pb.RouterFrame(
                    register=pb.RegisterRequest(
                        router_id="r1",
                        report_interval_ms=100000,  # effectively no periodic tick
                        lease_ttl_ms=100000,
                    )
                )
                await asyncio.sleep(1.5)

            call = stub.Monitor(frames())
            await receive_frames(call, 2, timeout=1.0)  # ack + initial report
            before = rm.get_loads_calls
            # Drive the decorated (shadowed) generate_request to completion.
            collected = [x async for x in rm.generate_request()]
            assert collected == [0, 1, 2]
            await asyncio.sleep(0.2)
            assert rm.get_loads_calls > before, "request-end must wake the sampler"
            await channel.close()
        finally:
            holder.stop_event.set()
            await asyncio.wait_for(serve_task, timeout=3.0)
            uninstall_fake_smg()


class TestCapabilityGuard:
    @pytest.mark.asyncio
    async def test_reporter_enabled_missing_capability_raises(self, isolate_sidecar):
        from sglang.srt.entrypoints import grpc_server

        port = free_port()
        args = make_server_args(port=port, sidecar_port=free_port())
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

        args = make_server_args(port=None, sidecar_port=free_port())
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


class TestDisabledNoShadow:
    @pytest.mark.asyncio
    async def test_disabled_does_not_shadow_instance(self, isolate_sidecar):
        from sglang.srt.entrypoints import grpc_server

        args = make_server_args(port=None, sidecar_port=free_port())
        holder = install_fake_smg(capability=True)
        holder.stop_event = asyncio.Event()
        try:
            serve_task = asyncio.ensure_future(grpc_server.serve_grpc(args))
            for _ in range(50):
                await asyncio.sleep(0.02)
                if holder.request_manager is not None:
                    break
            rm = holder.request_manager
            assert rm is not None
            assert "generate_request" not in rm.__dict__  # no decorator installed
        finally:
            holder.stop_event.set()
            await asyncio.wait_for(serve_task, timeout=3.0)
            uninstall_fake_smg()


class TestServeGrpcCleanup:
    @pytest.mark.asyncio
    async def test_serve_failure_cleans_up_reporter(self, isolate_sidecar):
        from sglang.srt.entrypoints import grpc_server

        port = free_port()
        args = make_server_args(port=port, sidecar_port=free_port())
        holder = install_fake_smg(capability=True)
        holder.raise_on_serve = RuntimeError("startup boom")
        try:
            with pytest.raises(RuntimeError, match="startup boom"):
                await grpc_server.serve_grpc(args)
            rm = holder.request_manager
            assert rm is not None
        finally:
            uninstall_fake_smg()
        # Reporter cleaned up despite the failure: port freed, shadow removed.
        assert port_is_free(port)
        assert "generate_request" not in rm.__dict__

    @pytest.mark.asyncio
    async def test_cancellation_cleans_up_reporter(self, isolate_sidecar):
        from sglang.srt.entrypoints import grpc_server

        port = free_port()
        args = make_server_args(port=port, sidecar_port=free_port())
        holder = install_fake_smg(capability=True)
        holder.stop_event = asyncio.Event()  # never set: we cancel instead
        try:
            serve_task = asyncio.ensure_future(grpc_server.serve_grpc(args))
            for _ in range(50):
                await asyncio.sleep(0.02)
                if holder.request_manager is not None and not port_is_free(port):
                    break
            rm = holder.request_manager
            assert rm is not None
            serve_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await serve_task
        finally:
            uninstall_fake_smg()
        assert port_is_free(port)
        assert "generate_request" not in rm.__dict__


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
