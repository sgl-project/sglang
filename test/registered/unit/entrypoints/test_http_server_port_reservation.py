"""The HTTP server binds host:port only after the model is loaded, so until then
the kernel may hand that port out as the ephemeral source port of an outbound
connection of this process or a child. launch_server binds the port before the
load and serves on that socket, without listening early, and leaves the port to
the launch paths that bind it themselves.
"""

import asyncio
import errno
import logging
import socket
import sys
import threading
import time
from types import SimpleNamespace

import pytest
import uvicorn

from sglang.srt.entrypoints import http_server
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.network import get_free_port
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HOST = "127.0.0.1"


class _LoadFailed(Exception):
    """Ends launch_server inside the model load."""


def _bind_error(host, port):
    # A plain bind, as any socket without SO_REUSEADDR (and bind(0)) performs it.
    with socket.socket() as probe:
        try:
            probe.bind((host, port))
        except OSError as error:
            return error
    return None


def _connect_error(host, port):
    with socket.socket() as client:
        client.settimeout(5)
        try:
            client.connect((host, port))
        except OSError as error:
            return error
    return None


def _close(reserved_socket):
    if reserved_socket is not None:
        reserved_socket.close()


def _launch(monkeypatch, server_args, during_load, http_server_func):
    # The real launch_server, with the model load and the serving replaced.
    def launch_subprocesses(**kwargs):
        during_load()
        return (None, None, None, SimpleNamespace(scheduler_infos=[]), None, [])

    monkeypatch.setattr(http_server.Engine, "_launch_subprocesses", launch_subprocesses)
    monkeypatch.setattr(
        http_server, "get_parallel", lambda: SimpleNamespace(node_rank=0)
    )
    monkeypatch.setattr(http_server, "_setup_and_run_http_server", http_server_func)
    http_server.launch_server(server_args)


@pytest.fixture
def uvicorn_log(monkeypatch):
    # uvicorn.Config re-runs dictConfig, which drops handlers added to its loggers.
    messages = []
    monkeypatch.setattr(
        logging.getLogger("uvicorn.error"),
        "info",
        lambda message, *args, **kwargs: messages.append(message % args),
    )
    return messages


@pytest.fixture(autouse=True)
def no_rust_server(monkeypatch):
    monkeypatch.delenv("SGLANG_RUST_SERVER", raising=False)


def test_http_port_stays_taken_while_the_model_loads(monkeypatch):
    """During the load the port cannot be bound, and clients are refused at
    once rather than accepted by a socket nobody serves yet."""
    port = get_free_port()
    probe = {}

    def during_load():
        probe["bind"] = _bind_error(HOST, port)
        probe["connect"] = _connect_error(HOST, port)

    def run_http_server(*args, **kwargs):
        reserved_socket = kwargs["reserved_socket"]
        probe["inheritable"] = reserved_socket.get_inheritable()
        _close(reserved_socket)

    _launch(
        monkeypatch,
        ServerArgs(model_path="dummy", host=HOST, port=port),
        during_load,
        run_http_server,
    )
    assert probe["bind"] is not None and probe["bind"].errno == errno.EADDRINUSE
    assert isinstance(probe["connect"], ConnectionRefusedError)
    assert probe["inheritable"] is False


def test_port_is_released_when_the_load_fails(monkeypatch):
    port = get_free_port()

    def during_load():
        raise _LoadFailed

    with pytest.raises(_LoadFailed):
        _launch(
            monkeypatch,
            ServerArgs(model_path="dummy", host=HOST, port=port),
            during_load,
            lambda *args, **kwargs: None,
        )
    assert _bind_error(HOST, port) is None


def test_busy_port_falls_back_to_the_servers_own_bind(monkeypatch):
    """A port in use at launch may free up during the load, so the launch goes on."""
    received = {}
    with socket.socket() as holder:
        holder.bind((HOST, 0))
        holder.listen()
        port = holder.getsockname()[1]

        def run_http_server(*args, **kwargs):
            received["reserved_socket"] = kwargs["reserved_socket"]

        _launch(
            monkeypatch,
            ServerArgs(model_path="dummy", host=HOST, port=port),
            lambda: None,
            run_http_server,
        )
    assert received["reserved_socket"] is None


@pytest.mark.parametrize(
    ("server_args", "rust_server"),
    [
        ({"nnodes": 2, "node_rank": 1}, None),
        ({}, "1"),
        ({"host": "localhost"}, None),
    ],
    ids=["follower-node", "rust-server", "host-name"],
)
def test_no_reservation_where_the_port_is_bound_elsewhere(
    monkeypatch, server_args, rust_server
):
    """A follower node and the Rust listeners bind host:port in another process;
    for a host name uvicorn binds every address it resolves to."""
    port = get_free_port()
    probe = {}
    if rust_server is not None:
        monkeypatch.setenv("SGLANG_RUST_SERVER", rust_server)

    def during_load():
        probe["error"] = _bind_error(HOST, port)
        raise _LoadFailed

    with pytest.raises(_LoadFailed):
        _launch(
            monkeypatch,
            ServerArgs(model_path="dummy", port=port, **{"host": HOST, **server_args}),
            during_load,
            lambda *args, **kwargs: None,
        )
    assert probe["error"] is None


def _ipv6_available():
    try:
        with socket.socket(socket.AF_INET6) as sock:
            sock.bind(("::1", 0))
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _ipv6_available(), reason="no IPv6")
def test_ipv6_reservation_matches_the_listener_uvicorn_would_bind():
    uvloop = pytest.importorskip("uvloop")

    def v6only(sock):
        return sock.getsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY)

    async def single_worker_v6only():
        server = await asyncio.get_running_loop().create_server(
            asyncio.Protocol, host="::", port=0
        )
        try:
            return v6only(server.sockets[0])
        finally:
            server.close()

    loop = uvloop.new_event_loop()
    try:
        expected_single = loop.run_until_complete(single_worker_v6only())
    finally:
        loop.close()
    with uvicorn.Config("app:app", host="::", port=0).bind_socket() as sock:
        expected_workers = v6only(sock)

    for workers, expected in ((1, expected_single), (2, expected_workers)):
        reserved_socket = http_server._reserve_http_socket(
            ServerArgs(
                model_path="dummy",
                host="::",
                port=get_free_port(),
                tokenizer_worker_num=workers,
            )
        )
        try:
            assert v6only(reserved_socket) == expected
        finally:
            reserved_socket.close()


def _serving(**overrides):
    options = dict(
        tokenizer_worker_num=1,
        enable_http2=False,
        enable_ssl_refresh=False,
        enable_http_worker_reuseport=False,
        host=HOST,
        port=0,
        fastapi_root_path="",
        api_key=None,
        admin_api_key=None,
        ssl_certfile=None,
        ssl_keyfile=None,
        ssl_ca_certs=None,
        ssl_keyfile_password=None,
        http2_max_concurrent_streams=None,
        http2_initial_connection_window_size=None,
    )
    options.update(overrides)
    return SimpleNamespace(**options)


@pytest.mark.parametrize(
    ("serving", "server", "served"),
    [
        ({}, "uvicorn", True),
        ({"enable_ssl_refresh": True}, "uvicorn-ssl-refresh", True),
        ({"enable_http2": True}, "granian", False),
        ({"tokenizer_worker_num": 2}, "uvicorn-workers", True),
        ({"tokenizer_worker_num": 2, "enable_http2": True}, "granian", False),
        (
            {"tokenizer_worker_num": 2, "enable_http_worker_reuseport": True},
            "reuseport-workers",
            False,
        ),
    ],
    ids=[
        "uvicorn",
        "ssl-refresh",
        "granian",
        "workers",
        "granian-workers",
        "reuseport-workers",
    ],
)
def test_each_server_serves_on_the_reservation_or_binds_after_its_release(
    monkeypatch, uvicorn_log, serving, server, served
):
    """uvicorn listens on the reserved socket; Granian and the SO_REUSEPORT
    workers bind host:port themselves, so it is closed before they start."""
    port = get_free_port()
    reserved_socket = http_server._reserve_http_socket(
        ServerArgs(model_path="dummy", host=HOST, port=port)
    )
    started = {}

    def start(name, sockets=None):
        started.update(name=name, sockets=sockets, closed=reserved_socket.fileno() < 0)

    def run(self, sockets=None):
        start("uvicorn", sockets)

    async def serve(self, sockets=None):
        start("uvicorn-ssl-refresh", sockets)

    class ReusePortMultiprocess:
        def __init__(self, config):
            start("reuseport-workers")

        def run(self):
            pass

    monkeypatch.setattr(uvicorn.Server, "run", run)
    monkeypatch.setattr(uvicorn.Server, "serve", serve)
    monkeypatch.setattr(
        http_server, "_run_granian_server", lambda **kwargs: start("granian")
    )
    monkeypatch.setattr(
        http_server,
        "_run_uvicorn_workers",
        lambda app_path, reserved_socket=None, **kwargs: start(
            "uvicorn-workers", [reserved_socket]
        ),
    )
    monkeypatch.setattr(http_server, "_ReusePortMultiprocess", ReusePortMultiprocess)
    # The real module needs watchfiles, which the CPU image may not have.
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.entrypoints.ssl_utils",
        SimpleNamespace(
            SSLCertRefresher=lambda *args: SimpleNamespace(stop=lambda: None)
        ),
    )
    monkeypatch.setattr(
        http_server, "get_serving", lambda: _serving(port=port, **serving)
    )
    monkeypatch.setattr(
        http_server,
        "get_observability",
        lambda: SimpleNamespace(log_level_http=None, log_level="error"),
    )
    monkeypatch.setattr(http_server, "set_global_state", lambda state: None)
    monkeypatch.setattr(http_server, "_global_state", None)
    monkeypatch.setattr(http_server, "set_uvicorn_logging_configs", lambda args: None)
    monkeypatch.setattr(
        http_server, "write_data_for_multi_tokenizer", lambda *args: None
    )
    tokenizer_manager = SimpleNamespace(
        set_server_stop_hook=lambda hook: None, startup_time=0.0
    )

    try:
        http_server._setup_and_run_http_server(
            ServerArgs(model_path="dummy", host=HOST, port=port),
            tokenizer_manager,
            None,
            None,
            [{}],
            None,
            reserved_socket=reserved_socket,
        )
    finally:
        reserved_socket.close()

    assert started["name"] == server
    if served:
        assert started["sockets"] == [reserved_socket]
        assert not started["closed"]
    else:
        assert started["closed"]
    if server in ("uvicorn", "uvicorn-ssl-refresh"):
        assert (
            f"Uvicorn running on http://{HOST}:{port} (Press CTRL+C to quit)"
            in uvicorn_log
        )


def test_uvicorn_workers_share_the_reservation(monkeypatch, uvicorn_log):
    port = get_free_port()
    reserved_socket = http_server._reserve_http_socket(
        ServerArgs(model_path="dummy", host=HOST, port=port, tokenizer_worker_num=2)
    )
    supervised = {}

    class Supervisor:
        def __init__(self, config, target=None, sockets=None):
            supervised["sockets"] = sockets

        def run(self):
            pass

    monkeypatch.setattr(http_server, "_SigquitForwardingMultiprocess", Supervisor)
    try:
        http_server._run_uvicorn_workers(
            "app:app", reserved_socket=reserved_socket, host=HOST, port=port, workers=2
        )
        assert supervised["sockets"] == [reserved_socket]
        assert reserved_socket.get_inheritable()
    finally:
        reserved_socket.close()
    assert (
        f"Uvicorn running on http://{HOST}:{port} (Press CTRL+C to quit)" in uvicorn_log
    )


def test_uvicorn_serves_on_a_bound_socket_it_did_not_listen_on():
    port = get_free_port()
    reserved_socket = http_server._reserve_http_socket(
        ServerArgs(model_path="dummy", host=HOST, port=port)
    )
    config = uvicorn.Config(
        "starlette.applications:Starlette",
        host=HOST,
        port=port,
        factory=True,
        lifespan="off",
        log_level="error",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, kwargs={"sockets": [reserved_socket]})
    thread.start()
    try:
        deadline = time.monotonic() + 60
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.02)
        with socket.create_connection((HOST, port), timeout=60) as connection:
            connection.sendall(b"GET /openapi.json HTTP/1.0\r\n\r\n")
            response = connection.recv(1024)
    finally:
        server.should_exit = True
        thread.join(timeout=60)
        reserved_socket.close()
    assert b"404 Not Found" in response


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
