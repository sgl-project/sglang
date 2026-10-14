import pytest
import zmq

from sglang.multimodal_gen.runtime.disaggregation import orchestrator
from sglang.multimodal_gen.runtime.utils.common import get_zmq_socket


def _bind_any(context: zmq.Context) -> tuple[zmq.Socket, str]:
    sock = context.socket(zmq.ROUTER)
    port = sock.bind_to_random_port("tcp://127.0.0.1")
    return sock, f"tcp://127.0.0.1:{port}"


def test_same_port_bind_fails_instead_of_moving_to_another_port():
    # A peer that connects to the configured endpoint would reach whoever holds it.
    context = zmq.Context()
    holder, endpoint = _bind_any(context)
    try:
        with pytest.raises(zmq.ZMQError) as exc:
            get_zmq_socket(
                context, zmq.ROUTER, endpoint, True, max_bind_retries=2, same_port=True
            )
        assert exc.value.errno == zmq.EADDRINUSE
    finally:
        holder.close(linger=0)
        context.destroy(linger=0)


def test_same_port_bind_succeeds_once_port_is_free():
    context = zmq.Context()
    holder, endpoint = _bind_any(context)
    holder.close(linger=0)
    try:
        sock, actual = get_zmq_socket(
            context, zmq.ROUTER, endpoint, True, max_bind_retries=2, same_port=True
        )
        assert actual == endpoint
        sock.close(linger=0)
    finally:
        context.destroy(linger=0)


def test_scheduler_ingress_binds_configured_port_only(monkeypatch):
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.managers import scheduler as scheduler_module

    class _Stop(Exception):
        pass

    calls = []

    def _record(context, socket_type, endpoint, bind, **kwargs):
        calls.append((endpoint, bind, kwargs))
        raise _Stop

    monkeypatch.setattr(scheduler_module, "set_global_server_args", lambda **_: None)
    monkeypatch.setattr(scheduler_module, "get_zmq_socket", _record)
    server_args = SimpleNamespace(
        num_gpus=1,
        dp_size=1,
        scheduler_endpoint_for=lambda replica: "tcp://127.0.0.1:5555",
    )
    with pytest.raises(_Stop):
        scheduler_module.Scheduler(server_args, gpu_id=0, port_args=None)
    assert calls == [("tcp://127.0.0.1:5555", True, {"same_port": True})]


@pytest.mark.parametrize(
    "occupied", [None, "frontend", "encoder", "denoiser", "decoder"]
)
def test_orchestrator_fixed_endpoints_and_startup_failure(monkeypatch, occupied):
    context = zmq.Context()
    holder, endpoint = _bind_any(context)
    endpoints = {
        name: endpoint if name == occupied else f"inproc://test-{name}"
        for name in ("frontend", "encoder", "denoiser", "decoder")
    }
    calls = []

    def bind_socket(context, socket_type, endpoint, bind, **kwargs):
        if bind:
            calls.append((endpoint, kwargs))
        return get_zmq_socket(
            context, socket_type, endpoint, bind, max_bind_retries=1, **kwargs
        )

    monkeypatch.setattr(orchestrator, "get_zmq_socket", bind_socket)
    server = orchestrator.DiffusionServer(
        frontend_endpoint=endpoints["frontend"],
        encoder_work_endpoints=[],
        denoiser_work_endpoints=["inproc://test-worker"],
        decoder_work_endpoints=[],
        encoder_result_endpoint=endpoints["encoder"],
        denoiser_result_endpoint=endpoints["denoiser"],
        decoder_result_endpoint=endpoints["decoder"],
    )
    try:
        server.start()
        thread = server._thread
        if occupied is None:
            assert server.wait_ready(timeout=5)
            assert [ep for ep, _ in calls] == list(endpoints.values())
        else:
            with pytest.raises(RuntimeError, match="failed to bind sockets") as exc:
                server.wait_ready(timeout=5)
            assert isinstance(exc.value.__cause__, zmq.ZMQError)
            assert exc.value.__cause__.errno == zmq.EADDRINUSE
            assert calls[-1][0] == endpoint
            assert server._context.closed
        assert all(kwargs == {"same_port": True} for _, kwargs in calls)
    finally:
        server.stop()
        holder.close(linger=0)
        context.destroy(linger=0)
    assert not thread.is_alive()
    assert server._context.closed
