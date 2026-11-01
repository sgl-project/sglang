"""Opt-in CPU-thread NVTX annotations; never synchronize or control capture.

SGLANG_ENABLE_NVTX_OPERATIONS enables all preparation, scheduler, worker and
enqueue ranges. Disabled decorators return the original callable, including hot
batch and worker paths. The benchmark owns the separate capture-trigger range.

The asynchronous gpu_delta.pause range ends only after a successful unpause.
Poisoned, failed-resume and keep_pause=True sessions intentionally leave it open;
this diagnostic span is not a completed duration and may outlive that session.
Synchronous call/phase ranges are always balanced on exceptions.
"""

from functools import wraps

from sglang.srt.environ import envs

ENABLED = envs.SGLANG_ENABLE_NVTX_OPERATIONS.get()

if ENABLED:
    from torch.cuda import nvtx


def annotate(name):
    """Label a synchronous call on its executing thread, including failures."""

    def decorate(function):
        if not ENABLED:
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
