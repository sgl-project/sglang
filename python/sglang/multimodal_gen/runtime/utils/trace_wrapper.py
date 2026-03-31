"""Context-manager wrappers around sglang.srt.observability.trace for diffusion tracing.

All tracing helpers for the multimodal_gen subsystem are consolidated here so
that call sites can use simple ``with`` statements instead of manual
start/end bookkeeping.
"""

from __future__ import annotations

from contextlib import contextmanager
from enum import Enum


class DiffStage(str, Enum):
    """Named trace stages for the diffusion pipeline."""

    SCHEDULER_DISPATCH = "scheduler_dispatch"
    GPU_FORWARD = "gpu_forward"


@contextmanager
def trace_req(trace_ctx):
    """Ensure ``trace_req_finish()`` is called when a request scope exits.

    Usage::

        with trace_req(batch.trace_ctx):
            ...
    """
    try:
        yield trace_ctx
    finally:
        trace_ctx.trace_req_finish()


@contextmanager
def trace_slice(trace_ctx, stage: DiffStage, *, level: int = 1, **kwargs):
    """Context manager for a single trace slice (span).

    Usage::

        with trace_slice(req.trace_ctx, DiffStage.GPU_FORWARD, level=2):
            result = pipeline.forward(req, server_args)
    """
    trace_ctx.trace_slice_start(stage.value, level=level)
    try:
        yield trace_ctx
    finally:
        trace_ctx.trace_slice_end(stage.value, level=level, **kwargs)
