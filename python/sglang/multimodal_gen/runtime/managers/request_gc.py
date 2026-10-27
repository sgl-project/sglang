# SPDX-License-Identifier: Apache-2.0
"""Move automatic cyclic GC out of worker execution, without freezing objects."""

import gc
import os
import threading
import time
from contextlib import contextmanager
from typing import Iterator


class RequestGarbageCollector:
    """Coordinate nested forwards and output threads sharing one interpreter.

    Reference counting and explicit collections still work. Automatic GC uses
    the caller's original settings whenever no protected execution is active.
    """

    MAX_DEFER_SECONDS = 30.0

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active = 0
        self._was_enabled = False
        self._started_at = 0.0
        self._warmup_pending = False

    @contextmanager
    def defer(self) -> Iterator[None]:
        pid = os.getpid()
        with self._lock:
            if self._active == 0:
                self._was_enabled = gc.isenabled()
                gc.disable()
                self._started_at = time.monotonic()
            self._active += 1
        try:
            yield
        finally:
            # a forked child has already restored its own collector state
            if os.getpid() == pid:
                with self._lock:
                    self._active -= 1
                    if self._active == 0 and self._was_enabled:
                        gc.enable()

    def needs_collection_window(self) -> bool:
        """Bound continuous forward/finalize overlap at the next admission."""
        with self._lock:
            return (
                self._active > 0
                and self._was_enabled
                and time.monotonic() - self._started_at >= self.MAX_DEFER_SECONDS
            )

    def mark_warmup(self) -> None:
        with self._lock:
            self._warmup_pending = True

    def collect_after_warmup(self) -> None:
        """Collect only once quiescent, after warmup frames have unwound."""
        with self._lock:
            if not self._warmup_pending or self._active:
                return
            self._warmup_pending = False
        # destructors may wait on other threads; never collect under our lock
        gc.collect()

    def after_fork(self) -> None:
        # locks and active scopes belong to the parent, not its new child
        if self._active and self._was_enabled:
            gc.enable()
        self._lock = threading.RLock()
        self._active = 0
        self._warmup_pending = False


# GC is interpreter-wide: workers and their output threads must share ownership
request_gc = RequestGarbageCollector()
if os.name == "posix":
    os.register_at_fork(after_in_child=request_gc.after_fork)
