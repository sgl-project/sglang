"""Bounded, opt-in evidence for a traced request's prefill admission check."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from sglang.srt.observability.req_time_stats import convert_time_to_realtime_ns
from sglang.srt.observability.trace import TraceEvent, TraceSliceContext

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


def record_admission_wait(req: Req, reason: str, **counts: int) -> None:
    """Emit once per reason on this trace context, at the rejecting branch.

    The caller gates this on SGLANG_TRACE_PREFILL_ADMISSION. A self-contained
    zero-duration slice survives cancellation while still queued and does not
    depend on a later slice consuming an event cache. It records a decision
    point, not the duration of the wait or the request's root cause.
    """
    ctx = req.time_stats.trace_ctx
    if not ctx.tracing_enable or ctx.trace_level < 3:
        return
    from opentelemetry import trace

    # tracing_enable describes process configuration, not this request's W3C
    # sampled flag. The root context also survives scheduler serialization.
    parent = trace.get_current_span(ctx.root_span_context).get_span_context()
    if not parent.trace_flags.sampled:
        return
    emitted = getattr(ctx, "_prefill_admission_reasons", None)
    if emitted is None:
        emitted = ctx._prefill_admission_reasons = set()
    if reason in emitted:
        return
    emitted.add(reason)
    now = convert_time_to_realtime_ns(time.perf_counter())
    attrs = {"reason": reason, **counts}
    ctx.trace_slice(
        TraceSliceContext(
            slice_name="prefill_admission_wait",
            start_time_ns=now,
            end_time_ns=now,
            level=3,
            attrs=attrs,
            events=[TraceEvent("prefill_admission_rejection", now, attrs)],
        )
    )
