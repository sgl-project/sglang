"""Trace instrumentation for HiCache L2↔L3 operations.

This module provides standalone tracing for the HiCache storage layer,
independent of request-level tracing. It enables correlation with
Mooncake C++ trace via the `hicache.hash_values` attribute.

Span design:
  - "enqueue" spans: instant (zero-duration) spans on the scheduler thread,
    created when an operation is put into the backup/prefetch queue.
  - Operation spans: real spans on the background threads, linked to the
    enqueue span via `add_link`.

  hicache_l3_backup_enqueue    ← scheduler thread (instant)
       ↘ linked-to → hicache_l3_backup        ← backup thread

  hicache_l3_prefetch_enqueue  ← scheduler thread (instant)
       ↘ linked-to → hicache_l3_prefetch_query    ← prefetch thread
                     hicache_l3_prefetch_transfer  ← IO aux thread

Correlation with Mooncake C++ trace:
  hicache.hash_values in SGLang ↔ key_strs in Mooncake C++
  (MooncakeStore._batch_preprocess appends suffix like "_{tp_rank}_k/v"
  to each hash_value to form key_strs)

Enable with: --trace-modules request,hicache
"""

import functools
import logging
import threading
from typing import Dict, Optional

from sglang.srt.observability import trace as _trace_mod
from sglang.srt.observability.trace import (
    _get_host_id,
    get_cur_time_ns,
    get_global_tracing_enabled,
)

logger = logging.getLogger(__name__)

# Cache host_id once at module level (avoid repeated I/O per attribute lookup).
_host_id: str = _get_host_id()


# ---------------------------------------------------------------------------
# Stage names
# ---------------------------------------------------------------------------


class HiCacheStage:
    """Stage names for HiCache L2↔L3 operations."""

    L3_BACKUP_ENQUEUE = "hicache_l3_backup_enqueue"
    L3_PREFETCH_ENQUEUE = "hicache_l3_prefetch_enqueue"
    L3_BACKUP = "hicache_l3_backup"
    L3_PREFETCH_QUERY = "hicache_l3_prefetch_query"
    L3_PREFETCH_TRANSFER = "hicache_l3_prefetch_transfer"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _hicache_tracing_enabled() -> bool:
    """Check if HiCache module tracing is enabled."""
    if not get_global_tracing_enabled():
        return False

    if _trace_mod.tracer is None:
        return False

    from sglang.srt.server_args import get_global_server_args

    server_args = get_global_server_args()
    return "hicache" in server_args.trace_modules.split(",")


# Cache: (controller_id, native_tid) → frozen attrs dict.
# Host / rank info never changes per (controller, thread), so we compute once.
_attrs_cache: Dict[tuple, Dict] = {}


def _controller_attrs(controller) -> Dict:
    """Return module/rank/pid attributes, cached per (controller, thread)."""
    cache_key = (id(controller), threading.get_native_id())
    cached = _attrs_cache.get(cache_key)
    if cached is not None:
        return cached

    native_tid = threading.get_native_id()
    thread_info = _trace_mod.threads_info.get(native_tid)
    thread_label = thread_info.thread_label if thread_info else "hicache_scheduler"

    attrs = {
        "module": "sglang::hicache",
        "pid": native_tid,
        "host_id": _host_id,
        "thread_label": thread_label,
    }
    tp_rank = getattr(controller, "tp_rank", None)
    dp_rank = getattr(controller, "dp_rank", None)
    pp_rank = getattr(controller, "pp_rank", None)
    if tp_rank is not None:
        attrs["tp_rank"] = tp_rank
    if dp_rank is not None:
        attrs["dp_rank"] = dp_rank
    if pp_rank is not None:
        attrs["pp_rank"] = pp_rank

    _attrs_cache[cache_key] = attrs
    return attrs


def _start_instant_span(stage_name: str, attrs: Optional[Dict] = None):
    """Create a zero-duration instant span. Returns SpanContext or None."""
    if not _hicache_tracing_enabled():
        return None

    start_time = get_cur_time_ns()
    span = _trace_mod.tracer.start_span(stage_name, start_time=start_time)
    if attrs:
        span.set_attributes(attrs)
    span.end()
    return span.get_span_context()


def _trace_wrapped(stage_name: str, attrs_fn, link_fn):
    """Generic decorator factory for HiCache operation spans.

    The decorated method's first positional arg (after self) must be the
    operation object.  ``self`` is the HiCacheController instance.

    Args:
        stage_name: OTel span name.
        attrs_fn: callable(self, operation) -> dict, builds span attributes.
        link_fn: callable(operation) -> SpanContext|None, provides the
                 cross-thread link context.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, operation, *args, **kwargs):
            if not _hicache_tracing_enabled():
                return func(self, operation, *args, **kwargs)
            start_time = get_cur_time_ns()
            span = _trace_mod.tracer.start_span(stage_name, start_time=start_time)

            # Always set controller-level attributes (module, pid, ranks)
            ctrl_attrs = _controller_attrs(self)
            span.set_attributes(ctrl_attrs)

            link_ctx = link_fn(operation)
            if link_ctx is not None:
                span.add_link(link_ctx)
            try:
                return func(self, operation, *args, **kwargs)
            finally:
                attrs = attrs_fn(self, operation)
                if attrs:
                    span.set_attributes(attrs)
                span.end(end_time=get_cur_time_ns())

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def trace_backup_enqueue(controller, operation) -> None:
    """Create an instant span for L2→L3 backup enqueue on the scheduler thread.

    Stores the resulting SpanContext on ``operation.trace_parent_context``.
    """
    if not _hicache_tracing_enabled():
        return

    attrs = _controller_attrs(controller)
    attrs.update(
        {
            "hicache.operation_id": operation.id,
            "hicache.num_tokens": len(operation.token_ids),
            "hicache.direction": "L2_to_L3",
        }
    )
    operation.trace_parent_context = _start_instant_span(
        HiCacheStage.L3_BACKUP_ENQUEUE,
        attrs=attrs,
    )


def trace_prefetch_enqueue(controller, operation) -> None:
    """Create an instant span for L3→L2 prefetch enqueue on the scheduler thread.

    Stores the resulting SpanContext on ``operation.trace_parent_context``.
    """
    if not _hicache_tracing_enabled():
        return

    attrs = _controller_attrs(controller)
    attrs.update(
        {
            "hicache.operation_id": operation.id,
            "hicache.num_tokens": len(operation.token_ids),
            "hicache.direction": "L3_to_L2",
        }
    )
    request_id = getattr(operation, "request_id", None)
    if request_id:
        attrs["hicache.request_id"] = request_id
    operation.trace_parent_context = _start_instant_span(
        HiCacheStage.L3_PREFETCH_ENQUEUE,
        attrs=attrs,
    )


# -- Decorators for background-thread methods --


def trace_backup(func):
    """Decorator for L2→L3 backup methods. First arg after self must be operation."""

    @_trace_wrapped(
        HiCacheStage.L3_BACKUP,
        attrs_fn=lambda self, op: {
            "hicache.operation_id": op.id,
            "hicache.hash_values": ",".join(op.hash_value),
            "hicache.num_pages": len(op.hash_value),
            "hicache.direction": "L2_to_L3",
        },
        link_fn=lambda op: op.trace_parent_context,
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def trace_prefetch_query(func):
    """Decorator for L3→L2 prefetch hit-query methods."""

    @_trace_wrapped(
        HiCacheStage.L3_PREFETCH_QUERY,
        attrs_fn=lambda self, op: None,
        link_fn=lambda op: op.trace_parent_context,
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def trace_prefetch_transfer(func):
    """Decorator for L3→L2 prefetch data-transfer methods."""

    @_trace_wrapped(
        HiCacheStage.L3_PREFETCH_TRANSFER,
        attrs_fn=lambda self, op: {
            "hicache.operation_id": op.id,
            "hicache.hash_values": ",".join(op.hash_value),
            "hicache.num_pages": len(op.hash_value),
            "hicache.direction": "L3_to_L2",
            **(
                {"hicache.request_id": op.request_id}
                if hasattr(op, "request_id")
                else {}
            ),
        },
        link_fn=lambda op: op.trace_parent_context,
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper
