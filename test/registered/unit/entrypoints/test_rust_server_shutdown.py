"""The Python side of the Rust server's SIGTERM drain: the main process relays
SIGTERM to the listener-hosting schedulers and waits for the schedulers with
Python mode's bound; the listener's scheduler drains on SIGTERM and stops the
schedulers once its drain completed."""

import contextlib
import ctypes
import multiprocessing as mp
import os
import signal
import socket
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from sglang.srt.entrypoints import engine
from sglang.srt.entrypoints import sidecar as sidecar_module
from sglang.srt.managers.io_struct import ShutdownReq
from sglang.srt.rust_server import shutdown as rust_shutdown
from sglang.srt.rust_server.server import RustServer, claim_sigterm_before_compiles
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.fixture
def restore_sigterm():
    handler = signal.getsignal(signal.SIGTERM)
    yield
    signal.signal(signal.SIGTERM, handler)
    signal.set_wakeup_fd(-1)


class _FakeServer:
    """The pyo3 ``Server`` surface the drain uses."""

    def __init__(self):
        self.complete = False
        self.drains = []
        self.read_fd, self.write_fd = os.pipe()
        os.set_blocking(self.read_fd, False)
        os.set_blocking(self.write_fd, False)

    def recv_requests(self, limit):
        return []

    def drain_complete(self):
        return self.complete

    def begin_drain(self, force=False):
        self.drains.append(force)

    def drain_signal_fd(self, signum, force=False, notify=None):
        self.signum, self.notify = signum, notify
        return self.write_fd


def test_listener_drains_on_sigterm_and_stops_schedulers_once(restore_sigterm):
    fake = _FakeServer()
    server = RustServer(fake, http_port=0)
    server._broadcasts_shutdown = True
    server.install_sigterm_drain(notify="launcher")
    assert (fake.signum, fake.notify) == (signal.SIGTERM, "launcher")

    os.kill(os.getpid(), signal.SIGTERM)  # must not terminate the scheduler
    assert fake.drains == [False]
    # The wakeup fd, which a Rust thread reads, carries the signal number.
    assert os.read(fake.read_fd, 16) == bytes([signal.SIGTERM])

    assert server.drain(8) == []
    fake.complete = True
    assert [type(req) for req in server.drain(8)] == [ShutdownReq]
    assert server.drain(8) == []

    # Only the rank that broadcasts to every scheduler stops them, and only it
    # reports the drain's end to the main process.
    other = RustServer(fake, http_port=0)
    other.install_sigterm_drain(notify="launcher")
    assert fake.notify is None
    assert other.drain(8) == []


def test_force_shutdown_env_forces_the_drain(restore_sigterm, monkeypatch):
    monkeypatch.setenv("SGL_FORCE_SHUTDOWN", "1")
    fake = _FakeServer()
    RustServer(fake, http_port=0).install_sigterm_drain()
    os.kill(os.getpid(), signal.SIGTERM)
    assert fake.drains == [True]


def _sigaction(signum, new=None):
    """Raw sigaction(2): the disposition before `new` (if any) applied, as an
    opaque buffer whose first field is the handler."""
    libc = ctypes.CDLL(None, use_errno=True)
    old = ctypes.create_string_buffer(512)
    assert libc.sigaction(signum, new, old) == 0, ctypes.get_errno()
    return old


def test_claimed_sigterm_survives_an_llvm_style_restore(restore_sigterm):
    """Triton's LLVM saves every disposition on its first compile and restores
    them all when one of its handlers fires; a SIGTERM saved at SIG_DFL would
    then silently undo the drain handler. Claimed first, the saved disposition
    is Python's own C handler, which runs the drain handler."""
    claim_sigterm_before_compiles()
    saved = _sigaction(signal.SIGTERM)  # LLVM's registration
    fake = _FakeServer()
    RustServer(fake, http_port=0).install_sigterm_drain()
    _sigaction(signal.SIGTERM, saved)  # LLVM's restore
    handler = ctypes.c_void_p.from_buffer(_sigaction(signal.SIGTERM)).value
    assert handler not in (None, 0), "SIGTERM is back at SIG_DFL"
    os.kill(os.getpid(), signal.SIGTERM)
    assert fake.drains == [False]


def _claimed_child(ready):
    claim_sigterm_before_compiles()
    ready.set()
    while True:
        time.sleep(0.05)


def test_claimed_sigterm_keeps_the_default_effect():
    """Before the drain handler is installed, SIGTERM still terminates."""
    ctx = mp.get_context("fork")
    ready = ctx.Event()
    child = ctx.Process(target=_claimed_child, args=(ready,))
    child.start()
    assert ready.wait(30)
    os.kill(child.pid, signal.SIGTERM)
    child.join(10)
    assert child.exitcode == -signal.SIGTERM


def _exits_later():
    time.sleep(2.5)


def test_a_stray_datagram_does_not_start_the_bound(monkeypatch):
    """Any local process can reach the abstract socket: without a SIGTERM, a
    datagram must not start the exit bound (which would kill the schedulers)."""
    monkeypatch.setattr(rust_shutdown, "_SCHEDULER_EXIT_TIMEOUT_SECS", 1)
    ctx = mp.get_context("fork")
    name = _name()
    child = ctx.Process(target=_exits_later)
    child.start()
    drain = rust_shutdown.RustServerSigtermDrain([child], [child], None, name)
    _notify(name)
    start = time.monotonic()
    drain.wait()
    assert child.exitcode == 0
    assert time.monotonic() - start >= 2


def _notify(name):
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
        sock.sendto(b"drained", "\0" + name)


def _listener_child(ready, done, name):
    # A listener-hosting rank-0 scheduler: on SIGTERM it drains for a while
    # (a long stream), reports the listener closed, then the schedulers exit.
    sigterm = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: sigterm.set())
    ready.set()
    sigterm.wait()
    time.sleep(2)
    _notify(name)
    done.set()


def _worker_child(ready, done):
    # Any other rank: a forwarded SIGTERM would kill it (default action).
    ready.set()
    done.wait()


def _wedged_child(ready):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready.set()
    time.sleep(60)


class _FakeWatchdog:
    stopped = False

    def stop(self):
        self.stopped = True


def _name():
    return f"sglang-drain-unit-{os.getpid()}-{time.monotonic_ns()}"


def test_main_process_relays_sigterm_and_waits_for_the_drain(
    restore_sigterm, monkeypatch
):
    """SIGTERM goes to listener ranks only, and Python's exit bound does not
    start while the listener is still draining."""
    monkeypatch.setattr(rust_shutdown, "_SCHEDULER_EXIT_TIMEOUT_SECS", 1)
    ctx = mp.get_context("fork")
    name = _name()
    ready = [ctx.Event(), ctx.Event()]
    done = ctx.Event()
    listener = ctx.Process(target=_listener_child, args=(ready[0], done, name))
    worker = ctx.Process(target=_worker_child, args=(ready[1], done))
    listener.start()
    worker.start()
    assert all(event.wait(30) for event in ready)
    watchdog = _FakeWatchdog()
    drain = rust_shutdown.RustServerSigtermDrain(
        [listener, worker], [listener], watchdog, name
    )
    drain.install()

    start = time.monotonic()
    os.kill(os.getpid(), signal.SIGTERM)
    drain.wait()
    assert watchdog.stopped
    assert (listener.exitcode, worker.exitcode) == (0, 0)
    assert time.monotonic() - start >= 2


def _follower_listener_child(ready, done):
    # A listener-hosting scheduler on a node >= 1: it drains on the SIGTERM
    # its launcher relays, then exits as rank 0's ShutdownReq would stop it.
    # Without a relay it gives up and exits 3.
    sigterm = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: sigterm.set())
    ready.set()
    relayed = sigterm.wait(10)
    time.sleep(1)
    done.set()
    sys.exit(0 if relayed else 3)


def test_follower_node_relays_sigterm_to_its_listeners(restore_sigterm, monkeypatch):
    """A node >= 1 launcher relays SIGTERM to the Rust listeners on its node,
    and only to them, so they drain as node 0's do; it then waits for its
    schedulers. A sidecar running meanwhile is stopped first, and SIGTERM
    then reaches the drain."""
    monkeypatch.setattr(engine, "get_parallel", lambda: SimpleNamespace(node_rank=1))
    sidecar_stops = []

    @contextlib.contextmanager
    def sidecar(node_hosts_listener):
        assert node_hosts_listener
        with sidecar_module._stop_on_sigterm(lambda: sidecar_stops.append(True)):
            yield

    monkeypatch.setattr(engine, "rust_server_sidecar", sidecar)
    ctx = mp.get_context("fork")
    ready = [ctx.Event(), ctx.Event()]
    done = ctx.Event()
    listener = ctx.Process(target=_follower_listener_child, args=(ready[0], done))
    worker = ctx.Process(target=_worker_child, args=(ready[1], done))
    listener.start()
    worker.start()
    assert all(event.wait(30) for event in ready)
    procs = [listener, worker]
    result = engine.SchedulerInitResult(
        scheduler_infos=[],
        block_until_scheduler_exits=lambda: [proc.join() for proc in procs],
        scheduler_procs=procs,
        rust_listener_procs=[listener],
    )

    threading.Timer(0.5, os.kill, args=(os.getpid(), signal.SIGTERM)).start()
    engine._block_until_follower_schedulers_exit(result, procs, _name(), True)
    assert (listener.exitcode, worker.exitcode) == (0, 0)
    assert sidecar_stops == [True]


def test_main_process_bounds_a_wedged_scheduler(restore_sigterm, monkeypatch):
    """A wedged scheduler never exits; Python's exit bound starts once its
    listener reports the (forced) drain ended, not before, and the scheduler
    is killed and reaped when the bound expires."""
    monkeypatch.setattr(rust_shutdown, "_SCHEDULER_EXIT_TIMEOUT_SECS", 1)
    ctx = mp.get_context("fork")
    name = _name()
    ready = ctx.Event()
    wedged = ctx.Process(target=_wedged_child, args=(ready,))
    wedged.start()
    assert ready.wait(30)
    drain = rust_shutdown.RustServerSigtermDrain([wedged], [wedged], None, name)
    drain.install()
    try:
        os.kill(os.getpid(), signal.SIGTERM)
        start = time.monotonic()
        threading.Timer(2, _notify, args=(name,)).start()
        drain.wait()
        elapsed = time.monotonic() - start
        assert 2 + 1 <= elapsed < 2 + 1 + 3, elapsed
        assert wedged.exitcode == -signal.SIGKILL
    finally:
        wedged.kill()
        wedged.join()


def test_main_process_raises_on_an_unreaped_scheduler(restore_sigterm, monkeypatch):
    """A killed scheduler not reaped within the bound raises, as Python's
    ``kill_process_tree`` does, instead of blocking the main process."""
    monkeypatch.setattr(rust_shutdown, "_SCHEDULER_EXIT_TIMEOUT_SECS", 0)
    monkeypatch.setattr(rust_shutdown, "_REAP_TIMEOUT_SECS", 1)
    ctx = mp.get_context("fork")
    name = _name()
    ready = ctx.Event()
    wedged = ctx.Process(target=_wedged_child, args=(ready,))
    wedged.start()
    assert ready.wait(30)
    # Stands in for a scheduler that SIGKILL cannot end (stuck in the kernel).
    monkeypatch.setattr(wedged, "kill", lambda: None)
    drain = rust_shutdown.RustServerSigtermDrain([wedged], [wedged], None, name)
    drain.install()
    try:
        os.kill(os.getpid(), signal.SIGTERM)
        _notify(name)
        start = time.monotonic()
        with pytest.raises(RuntimeError, match=f"pids=\\[{wedged.pid}\\]"):
            drain.wait()
        assert time.monotonic() - start < 1 + 3
    finally:
        if wedged.exitcode is None:
            os.kill(wedged.pid, signal.SIGKILL)
        wedged.join()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
