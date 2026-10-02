"""Coordinate serving's permanent GC freeze with temporary graph capture."""

import gc
import threading
from contextlib import contextmanager

_lock = threading.RLock()
_permanent = False
_temporary_depth = 0


def freeze_gc_permanently():
    global _permanent
    with _lock:
        gc.freeze()
        _permanent = True


@contextmanager
def freeze_gc_for_capture(enable_cudagraph_gc: bool):
    global _temporary_depth
    with _lock:
        gc.collect()
        temporary = not enable_cudagraph_gc and not _permanent
        if temporary:
            if _temporary_depth == 0:
                gc.freeze()
            _temporary_depth += 1
    try:
        yield
    finally:
        if temporary:
            with _lock:
                _temporary_depth -= 1
                # The last graph scope releases only a temporary freeze. An
                # API request during capture can make that freeze permanent.
                if _temporary_depth == 0 and not _permanent:
                    gc.unfreeze()
                    gc.collect()
