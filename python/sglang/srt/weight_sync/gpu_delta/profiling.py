"""Opt-in CPU-thread NVTX annotations; never synchronize or control capture.

Set GPU_DELTA_NVTX before process startup: 0 disables annotations (default),
1 labels preparation and scheduler controls, and 2 adds worker/planning/enqueue
detail. Disabled decorators return the original callable, including hot batch
and worker paths. The benchmark owns the separate capture-trigger range.

The asynchronous gpu_delta.pause range ends only after a successful unpause.
Poisoned, failed-resume and keep_pause=True sessions intentionally leave it open;
this diagnostic span is not a completed duration and may outlive that session.
Synchronous call/phase ranges are always balanced on exceptions.
"""

import os
from functools import wraps

LEVEL = int(os.environ.get("GPU_DELTA_NVTX", "0"))
if LEVEL not in (0, 1, 2):
    raise ValueError("GPU_DELTA_NVTX must be 0, 1 or 2")
ENABLED = LEVEL > 0
DETAIL = LEVEL == 2

if ENABLED:
    from torch.cuda import nvtx


def annotate(name, detail=False):
    """Label a synchronous call on its executing thread, including failures."""

    def decorate(function):
        if not ENABLED or (detail and not DETAIL):
            return function

        @wraps(function)
        def call(*args, **kwargs):
            nvtx.range_push(name)
            try:
                return function(*args, **kwargs)
            finally:
                nvtx.range_pop()

        return call

    return decorate
