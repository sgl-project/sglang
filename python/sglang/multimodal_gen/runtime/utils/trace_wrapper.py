"""Context-manager wrappers around sglang.srt.observability.trace for diffusion tracing.

All tracing helpers for the multimodal_gen subsystem are consolidated here so
that call sites can use simple ``with`` statements instead of manual
start/end bookkeeping.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class DiffStageConfig:
    """A named trace stage with a default nesting level."""

    stage_name: str
    level: int = 0


class DiffStage:
    """Named trace stages for the diffusion pipeline."""

    SCHEDULER_DISPATCH = DiffStageConfig("scheduler_dispatch", level=1)
    GPU_FORWARD = DiffStageConfig("gpu_forward", level=2)


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
def trace_slice(trace_ctx, stage: DiffStageConfig, **kwargs):
    """Context manager for a single trace slice (span).

    Usage::

        with trace_slice(req.trace_ctx, DiffStage.GPU_FORWARD):
            result = pipeline.forward(req, server_args)
    """
    trace_ctx.trace_slice_start(stage.stage_name, level=stage.level)
    try:
        yield trace_ctx
    finally:
        trace_ctx.trace_slice_end(stage.stage_name, level=stage.level, **kwargs)
