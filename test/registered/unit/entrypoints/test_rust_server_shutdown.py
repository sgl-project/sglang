"""The Python side of the Rust server's SIGTERM drain: the main process relays
SIGTERM to the listener-hosting schedulers and waits for the schedulers with
Python mode's bound; the listener's scheduler drains on SIGTERM and stops the
schedulers once its drain completed, with attention DP once every DP
listener closed."""

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
from sglang.srt.managers.scheduler_components.dp_attn import DRAIN_STATE_NO_LISTENER
from sglang.srt.runtime_context import get_parallel
from sglang.srt.rust_server import shutdown as rust_shutdown
from sglang.srt.rust_server.server import (
    DRAIN_CLOSED,
    DRAIN_DRAINING,
    DRAIN_FORCE_REQUESTED,
    RustServer,
    claim_sigterm_before_compiles,
)
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
        self.closed = False
        self.held = False
        self.release_calls = 0
        self.releases = 0
        self.read_fd, self.write_fd = os.pipe()
        os.set_blocking(self.read_fd, False)
        os.set_blocking(self.write_fd, False)

    def recv_requests(self, limit):
        return []

    def drain_complete(self):
        return self.complete

    def begin_drain(self, force=False):
        self.drains.append(force)

    def drain_state(self):
        state = DRAIN_DRAINING if self.drains else 0
        if any(self.drains):
            state |= DRAIN_FORCE_REQUESTED
        return state | (DRAIN_CLOSED if self.closed else 0)

    def hold_drain_end(self):
        assert not self.drains, "held after the drain began"
        self.held = True

    def release_drain_end(self):
        self.release_calls += 1
        if not self.drains:
            return False
        self.releases += 1
        return True

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
    monkeypatch.setenv("SGLANG_FORCE_SHUTDOWN", "1")
    fake = _FakeServer()
    RustServer(fake, http_port=0).install_sigterm_drain()
    os.kill(os.getpid(), signal.SIGTERM)
    assert fake.drains == [True]


DRAINING, FORCED, CLOSED = DRAIN_DRAINING, DRAIN_FORCE_REQUESTED, DRAIN_CLOSED
NO_LISTENER = DRAIN_STATE_NO_LISTENER


def _dp_listener(fake, *, hold, attn_dp_size=3, force_env=False, monkeypatch=None):
    if force_env:
        monkeypatch.setenv("SGLANG_FORCE_SHUTDOWN", "1")
    server = RustServer(fake, http_port=0)
    server._broadcasts_shutdown = hold
    with get_parallel().override(attn_dp_size=attn_dp_size):
        server.install_sigterm_drain(hold_drain_end=hold)
    return server


def test_any_draining_listener_drains_every_listener(restore_sigterm):
    fake = _FakeServer()
    server = _dp_listener(fake, hold=False)
    server.on_drain_rows([0, 0, NO_LISTENER])
    assert fake.drains == []
    for _ in range(2):  # a step can gather twice
        server.on_drain_rows([0, DRAINING | CLOSED, 0])
    assert fake.drains == [False]
    # A peer forced later upgrades the graceful drain; a failed probe, which
    # shows only as closed, does not.
    server.on_drain_rows([DRAINING, CLOSED | DRAINING, 0])
    assert fake.drains == [False]
    server.on_drain_rows([DRAINING, DRAINING | FORCED, 0])
    server.on_drain_rows([DRAINING | FORCED, DRAINING | FORCED, 0])
    assert fake.drains == [False, True]
    assert fake.releases == 0


def test_force_shutdown_env_forces_a_propagated_drain(restore_sigterm, monkeypatch):
    fake = _FakeServer()
    server = _dp_listener(fake, hold=False, force_env=True, monkeypatch=monkeypatch)
    server.on_drain_rows([0, DRAINING])
    assert fake.drains == [True]


def test_rank_0_ends_its_drain_once_every_listener_closed(restore_sigterm):
    fake = _FakeServer()
    server = _dp_listener(fake, hold=True)
    assert fake.held
    # Every listener closed, but no drain is under way here: no release.
    fake.closed = True
    server.on_drain_rows([CLOSED, CLOSED, CLOSED])
    assert fake.release_calls == 0
    fake.closed = False

    server.on_drain_rows([0, DRAINING, 0])
    assert fake.drains == [False]
    fake.closed = True
    for rows in (
        [DRAINING | CLOSED, DRAINING, DRAINING | CLOSED],
        [DRAINING | CLOSED, DRAINING | CLOSED, DRAINING],
    ):
        server.on_drain_rows(rows)
        assert fake.releases == 0, rows
    for _ in range(2):
        server.on_drain_rows([DRAINING | CLOSED] * 3)
    assert fake.releases == 1


def test_no_release_once_a_forced_drain_stopped_the_schedulers(restore_sigterm):
    fake = _FakeServer()
    server = _dp_listener(fake, hold=True)
    server.server.begin_drain(True)
    fake.complete = fake.closed = True  # a forced drain skips the hold
    assert [type(req) for req in server.drain(8)] == [ShutdownReq]
    server.on_drain_rows([DRAINING | FORCED | CLOSED] * 3)
    assert fake.release_calls == 0


def test_no_release_while_rank_0_has_not_begun_draining(restore_sigterm):
    """Rank 0's own state, read fresh, must be draining, even if every row
    shows a closed listener."""
    fake = _FakeServer()
    server = _dp_listener(fake, hold=True)
    fake.begin_drain = lambda force=False: None  # the drain never begins here
    server.on_drain_rows([CLOSED, DRAINING | CLOSED, CLOSED])
    assert fake.release_calls == 0


@pytest.mark.parametrize(
    "rows",
    [
        # Every other DP group's row is an inactive slot: vacuously closed.
        [CLOSED, NO_LISTENER, NO_LISTENER],
        # One listener row short.
        [CLOSED, CLOSED],
    ],
)
def test_a_mismatched_row_set_fails_open_with_one_error(restore_sigterm, caplog, rows):
    fake = _FakeServer()
    server = _dp_listener(fake, hold=True)
    caplog.set_level("ERROR")
    server.on_drain_rows(rows)
    server.on_drain_rows(rows)
    assert fake.releases == 0, "released with no drain under way"
    errors = [r for r in caplog.records if "drain barrier disabled" in r.message]
    assert len(errors) == 1

    server.server.begin_drain(False)
    server.on_drain_rows(rows)
    assert fake.releases == 0, "released before this listener closed"
    fake.closed = True
    server.on_drain_rows(rows)
    assert fake.releases == 1
    assert len([r for r in caplog.records if "drain barrier" in r.message]) == 1


def _hold_scheduler(**parallel):
    from sglang.srt.managers.scheduler import Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.require_mlp_sync = parallel.pop("require_mlp_sync", True)
    scheduler.rust_server_endpoints = [{}] * parallel.pop("endpoints", 2)
    fields = dict(tp_rank=0, attn_dp_size=2, dp_size=1, pp_size=1, num_dp_ranks=2)
    fields.update(parallel)
    return scheduler, fields


@pytest.mark.parametrize(
    ("case", "hold"),
    [
        (dict(), True),
        (dict(tp_rank=1), False),
        (dict(attn_dp_size=1, num_dp_ranks=1, endpoints=1), False),
        # `--dp-size` replicas with a gathered buffer: each its own listener.
        (dict(attn_dp_size=1, dp_size=2, endpoints=1), False),
        (dict(endpoints=1), False),
        (dict(require_mlp_sync=False), False),
        (dict(pp_size=2), False),
        (dict(skip_all_gather=True), False),
    ],
)
def test_rank_0_holds_its_drain_end_only_where_every_step_gathers(case, hold):
    from sglang.srt.environ import envs

    case = dict(case)
    skip = case.pop("skip_all_gather", False)
    scheduler, fields = _hold_scheduler(**case)
    with (
        get_parallel().override(**fields),
        envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(skip),
    ):
        assert scheduler._holds_rust_drain_end() is hold


def test_scheduler_gathers_its_listener_drain_state():
    from sglang.srt.environ import envs
    from sglang.srt.managers.scheduler import Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.rust_server = None
    with envs.SGLANG_RUST_SERVER.override(False):
        assert scheduler._dp_drain_column_callbacks() == {}
    with envs.SGLANG_RUST_SERVER.override(True):
        callbacks = scheduler._dp_drain_column_callbacks()
        assert callbacks["get_drain_state"]() == NO_LISTENER
        assert "on_drain_rows" not in callbacks
        fake = _FakeServer()
        scheduler.rust_server = RustServer(fake, http_port=0)
        callbacks = scheduler._dp_drain_column_callbacks()
        fake.closed = True
        assert callbacks["get_drain_state"]() == CLOSED
        callbacks["on_drain_rows"]([DRAINING])
        assert fake.drains == [False]


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


def _exits_soon():
    time.sleep(0.5)


def test_a_clean_exit_without_sigterm_bounds_the_others(monkeypatch):
    """No SIGTERM reached this launcher (another node's, or one scheduler's
    own, began the drain): the first scheduler exiting 0 starts the exit
    bound and stops the watchdog, so killing a straggler is not a crash."""
    from sglang.srt.utils.watchdog import SubprocessWatchdog

    monkeypatch.setattr(rust_shutdown, "_SCHEDULER_EXIT_TIMEOUT_SECS", 1)
    sigquits = []
    previous = signal.signal(signal.SIGQUIT, lambda *_: sigquits.append(True))
    ctx = mp.get_context("fork")
    ready = ctx.Event()
    clean = ctx.Process(target=_exits_soon)
    wedged = ctx.Process(target=_wedged_child, args=(ready,))
    clean.start()
    wedged.start()
    try:
        assert ready.wait(30)
        watchdog = SubprocessWatchdog([clean, wedged], interval=0.1)
        watchdog.start()
        drain = rust_shutdown.RustServerSigtermDrain(
            [clean, wedged], [clean], watchdog, _name()
        )
        start = time.monotonic()
        drain.wait()
        assert 0.5 + 1 <= time.monotonic() - start < 0.5 + 1 + 3
        assert (clean.exitcode, wedged.exitcode) == (0, -signal.SIGKILL)
        time.sleep(0.5)  # past several watchdog rounds
        assert sigquits == [], "the straggler's kill read as a crash"
    finally:
        signal.signal(signal.SIGQUIT, previous)
        for proc in (clean, wedged):
            if proc.exitcode is None:
                proc.kill()
            proc.join()


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
