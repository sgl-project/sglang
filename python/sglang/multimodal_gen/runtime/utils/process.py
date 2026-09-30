# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/utils.py

import ctypes
import os
import signal
import sys
import threading

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


def kill_process_tree(parent_pid, include_parent: bool = True, skip_pid: int = None):
    """Kill the process and all its child processes."""
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
    for child in children:
        if child.pid == skip_pid:
            continue
        try:
            child.kill()
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
        except psutil.NoSuchProcess:
            pass
