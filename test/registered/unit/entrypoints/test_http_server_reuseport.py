"""With --enable-http-worker-reuseport every uvicorn worker must own a private
SO_REUSEPORT listener on the configured address, a replaced worker must get a
fresh one, and a second server on the same fixed host:port must still fail to
start instead of silently joining the first server's reuseport group.
"""

import copy
import signal
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from unittest.mock import Mock

import pytest
import uvicorn
from uvicorn.supervisors.multiprocess import SIGNALS

from sglang.srt.entrypoints import http_server
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def restore_signal_handlers():
    # Multiprocess.__init__ replaces process-wide signal handlers.
    handlers = {sig: signal.getsignal(sig) for sig in SIGNALS}
    yield
    for sig, handler in handlers.items():
        signal.signal(sig, handler)


@pytest.fixture
def config(free_tcp_port):
    return uvicorn.Config(
        "starlette.applications:Starlette",
        host="127.0.0.1",
        port=free_tcp_port,
        workers=4,
        factory=True,
        lifespan="off",
        log_level="error",
    )


def _start(config, monkeypatch):
    children = []

    class FakeProcess:
        def __init__(self, _config, _target, sockets):
            self.sockets = sockets
            self.alive = True
            self.pid = len(children) + 1

        def start(self):
            self.listener = self.sockets[0].dup()
            children.append(self.listener)

        def is_alive(self, timeout):
            return self.alive

        def kill(self):
            self.alive = False
            self.listener.close()

        terminate = kill

        def join(self):
            return None

    monkeypatch.setattr(http_server, "Process", FakeProcess)
    supervisor = http_server._ReusePortMultiprocess(config, target=None)
    supervisor.init_processes()
    return supervisor, children


def test_worker_listener_lifecycle(monkeypatch, config):
    resolve = Mock(wraps=socket.getaddrinfo)
    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    supervisor, children = _start(config, monkeypatch)
    assert resolve.call_count == 1
    try:
        assert all(p.sockets[0].fileno() == -1 for p in supervisor.processes)
        assert len({sock.fileno() for sock in children}) == config.workers
        assert all(
            sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT) for sock in children
        )

        old = supervisor.processes[0]
        old.alive = False
        supervisor.keep_subprocess_alive()
        assert supervisor.processes[0] is not old and old.listener.fileno() == -1

        before_restart = list(supervisor.processes)
        supervisor.restart_all()
        assert all(process.listener.fileno() == -1 for process in before_restart)
    finally:
        for sock in children:
            sock.close()


def test_partial_spawn_failure_cleans_up_workers(monkeypatch, config):
    config.workers = 3
    supervisor = http_server._ReusePortMultiprocess(config, target=None)
    supervisor._reserved_socket.close()
    workers = [Mock(), Mock()]
    monkeypatch.setattr(
        supervisor, "_spawn", Mock(side_effect=[*workers, RuntimeError("failed")])
    )
    with pytest.raises(RuntimeError):
        supervisor.init_processes()
    for worker in workers:
        worker.terminate.assert_called_once()
        worker.join.assert_called_once()


def test_unresolvable_host_exits_cleanly(monkeypatch, caplog, config):
    resolve = Mock(side_effect=socket.gaierror("bad host"))
    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    with pytest.raises(SystemExit):
        http_server._ReusePortMultiprocess(config, target=None)
    assert "bad host" in caplog.text


@pytest.mark.parametrize(
    ("hosts", "ephemeral", "expected_winners"),
    [
        (("127.0.0.1",) * 2, False, 1),
        (("127.0.0.1",) * 2, True, 2),
        (("127.0.0.1", "127.0.0.2"), False, 2),
    ],
)
def test_concurrent_supervisor_reservation(
    monkeypatch, config, hosts, ephemeral, expected_winners
):
    barrier = threading.Barrier(2)
    bind = http_server._bind_reuseport_socket

    def synchronized_bind(*args):
        with suppress(threading.BrokenBarrierError):
            barrier.wait(timeout=1)
        return bind(*args)

    def create_supervisor(index):
        worker_config = copy.copy(config)
        worker_config.host = hosts[index]
        worker_config.port = 0 if ephemeral else config.port
        try:
            return http_server._ReusePortMultiprocess(worker_config, target=None)
        except SystemExit:
            return None

    monkeypatch.setattr(http_server.Multiprocess, "__init__", Mock(return_value=None))
    monkeypatch.setattr(http_server, "_bind_reuseport_socket", synchronized_bind)
    with ThreadPoolExecutor(max_workers=2) as pool:
        supervisors = list(pool.map(create_supervisor, range(2)))

    winners = [supervisor for supervisor in supervisors if supervisor is not None]
    for supervisor in winners:
        supervisor._reserved_socket.close()
    assert len(winners) == expected_winners


def test_real_uvicorn_process_accepts_on_private_listener(config):
    config.workers = 1
    supervisor = http_server._ReusePortMultiprocess(config, uvicorn.Server(config).run)
    supervisor._reserved_socket.close()
    supervisor._reserved_socket = None
    process = supervisor._spawn()
    try:
        with socket.create_connection(
            (config.host, config.port), timeout=5
        ) as connection:
            connection.sendall(b"GET /openapi.json HTTP/1.0\r\n\r\n")
            assert b"404 Not Found" in connection.recv(1024)
    finally:
        process.terminate()
        process.join()
