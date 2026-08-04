"""Parent-death watchdog for MLX workers on Apple Silicon.

macOS has no ``PR_SET_PDEATHSIG`` equivalent, so the kernel will not signal a
worker process when its parent dies; the worker would be reparented to PID 1
and leak (holding GPU/host memory and ports). This module emulates PDEATHSIG
with a daemon thread that watches the parent PID via kqueue and SIGKILLs the
current process once it gets orphaned.
"""

import os
import select
import signal
import threading


def start_parent_death_watcher() -> None:
    """SIGKILL this process once its current parent exits (macOS only).

    kqueue with an ``EVFILT_PROC`` / ``NOTE_EXIT`` filter is the native,
    event-driven mechanism on macOS (exposed via ``select.kqueue`` /
    ``select.kevent``), so the watcher thread blocks until the parent actually
    exits instead of waking up to poll.

    ``SIGKILL`` is sent from this watcher thread and is uncatchable /
    unblockable, so it works even when the main thread is stuck inside a
    blocking native call (e.g. an MLX/Metal ``mx.eval`` / ``.tolist()``).
    """
    original_ppid = os.getppid()

    def _watch_parent():
        kq = select.kqueue()
        kev = select.kevent(
            original_ppid,
            filter=select.KQ_FILTER_PROC,
            flags=select.KQ_EV_ADD,
            fflags=select.KQ_NOTE_EXIT,
        )
        try:
            # Register the EVFILT_PROC / NOTE_EXIT watch on the parent PID.
            kq.control([kev], 0, None)
        except (ProcessLookupError, OSError):
            # The parent already exited before we could register the watch
            # (ESRCH); we are already orphaned.
            os.kill(os.getpid(), signal.SIGKILL)
            return
        # Guard against the race where the parent exits between reading
        # original_ppid and registering the watch above.
        if os.getppid() != original_ppid:
            os.kill(os.getpid(), signal.SIGKILL)
            return
        # Block until the parent exits, then terminate ourselves.
        kq.control(None, 1, None)
        os.kill(os.getpid(), signal.SIGKILL)

    watcher = threading.Thread(
        target=_watch_parent,
        name="parent-death-watcher",
        daemon=True,
    )
    watcher.start()
