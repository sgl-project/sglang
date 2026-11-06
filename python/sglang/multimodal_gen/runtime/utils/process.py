# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/utils.py

import contextlib
import ctypes
import gc
import os
import signal
import sys
import threading
from collections.abc import Iterator

import psutil


def kill_itself_when_parent_died() -> None:
    if sys.platform != "linux":
        return

    # keep GPU workers tied to the CLI process even if the parent is SIGKILLed
    PR_SET_PDEATHSIG = 1
    # Capture parent before arming PDEATHSIG: if the parent already died in the
    # fork->prctl window, PDEATHSIG won't fire, so detect the reparent explicitly.
    parent_pid = os.getppid()
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL) != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    # getppid() changing means we were reparented (parent gone). Comparing to the
    # captured pid instead of "== 1" avoids self-killing when PID 1 is the real
    # parent (e.g. running as a container's init process).
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGKILL)


_full_gc_lock = threading.Lock()
_full_gc_deferrals = 0
_gc_thresholds = gc.get_threshold()
# The largest third threshold the collector accepts, a C int, so never reached.
_NO_FULL_GC = 2**31 - 1


@contextlib.contextmanager
def defer_full_gc_for_imports() -> Iterator[None]:
    """Run no automatic full garbage collection while a process imports its runtime.

    Imports create long-lived modules, classes and functions and little garbage,
    yet each full collection traces the whole growing heap, and how many of them
    fall inside an import depends on the import order rather than on the work
    done. Young collections still run, so short-lived cyclic garbage is freed as
    usual. Restoring the entry thresholds lets automatic full collections resume
    after the block; nested or concurrent blocks restore them when the last one
    exits.
    """
    global _full_gc_deferrals, _gc_thresholds
    # Python 3.14.0-3.14.4 uses incremental GC and ignores threshold2
    if (3, 14) <= sys.version_info < (3, 14, 5):
        yield
        return

    pid = os.getpid()
    with _full_gc_lock:
        if _full_gc_deferrals == 0:
            _gc_thresholds = gc.get_threshold()
            gc.set_threshold(*_gc_thresholds[:2], _NO_FULL_GC)
        _full_gc_deferrals += 1
    try:
        yield
    finally:
        # A child forked inside the block restored the thresholds at fork.
        if os.getpid() == pid:
            with _full_gc_lock:
                _full_gc_deferrals -= 1
                if _full_gc_deferrals == 0:
                    gc.set_threshold(*_gc_thresholds)


def _restore_gc_thresholds_in_forked_child() -> None:
    # The parent restores the thresholds in the parent only, so a child forked
    # inside the block would otherwise never run a full collection.
    global _full_gc_lock, _full_gc_deferrals
    _full_gc_lock = threading.Lock()
    if _full_gc_deferrals:
        _full_gc_deferrals = 0
        gc.set_threshold(*_gc_thresholds)


os.register_at_fork(after_in_child=_restore_gc_thresholds_in_forked_child)


def kill_process_tree(
    parent_pid,
    include_parent: bool = True,
    skip_pid: int = None,
    wait_timeout: float | None = 10,
):
    """Kill a process tree and wait for its resources to be released.

    SIGKILL is asynchronous. Waiting prevents a subsequent server test from
    racing the old process for ports or accelerator resources. Pass ``None``
    only in a context where blocking is not acceptable.
    """
    # Remove sigchld handler to avoid spammy logs.
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)

    if parent_pid is None:
        parent_pid = os.getpid()
        include_parent = False

    try:
        itself = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return

    children = itself.children(recursive=True)
    killed = []
    for child in children:
        if child.pid == skip_pid:
            continue
        try:
            child.kill()
            killed.append(child)
        except psutil.NoSuchProcess:
            pass

    if include_parent:
        try:
            if parent_pid == os.getpid():
                itself.kill()
                sys.exit(0)

            itself.kill()

            # Sometime processes cannot be killed with SIGKILL (e.g, PID=1 launched by kubernetes),
            # so we send an additional signal to kill them.
            itself.send_signal(signal.SIGQUIT)
            killed.append(itself)
        except psutil.NoSuchProcess:
            pass

    if wait_timeout is not None and killed:
        _, alive = psutil.wait_procs(killed, timeout=wait_timeout)
        if alive:
            raise RuntimeError(
                f"kill_process_tree: {len(alive)} process(es) did not exit "
                f"within {wait_timeout}s; pids={[process.pid for process in alive]}"
            )
