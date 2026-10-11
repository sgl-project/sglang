import pytest
import zmq

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
