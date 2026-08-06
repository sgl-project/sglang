"""Load monitor decorator seam.

Exports:
  enable_load_monitor(kind) — wraps sync functions or async generators to fire
    load-monitor events.
  bind_load_monitor(owner, notify) — binds a callback to owner via weak-key
    registry; returns an unbind closure.

Fast bypass: both wrappers check server_args.load_reporter_port first and
become transparent passthroughs when absent or None.
"""

from __future__ import annotations

import functools
import inspect
import logging
import weakref
from contextlib import aclosing
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Callback type: (reason, count) -> None
# reason is a LoadReporterRefreshReason enum value; typed Any here to avoid
# importing io_struct at module load time (keeps disabled overhead to zero).
_NotifyFn = Callable[[Any, int], None]

# Weak-key registry: owner instance -> callback.
# Owner garbage collection automatically removes the entry.
_REGISTRY: weakref.WeakKeyDictionary[Any, _NotifyFn] = weakref.WeakKeyDictionary()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def bind_load_monitor(owner: Any, notify: _NotifyFn) -> Callable[[], None]:
    """Bind notify callback to owner in the weak-key registry.

    Returns an idempotent unbind closure. Safe to call multiple times.
    """
    _REGISTRY[owner] = notify

    def unbind() -> None:
        if _REGISTRY.get(owner) is notify:
            _REGISTRY.pop(owner, None)

    return unbind


def enable_load_monitor(kind: str) -> Callable[..., Any]:
    """Decorator factory for load-monitor observation points.

    Args:
        kind: "scheduler_message" (sync method) or "request_lifecycle"
            (async generator). Fires events after successful calls.
    """
    if kind == "scheduler_message":
        return _make_scheduler_message_decorator
    if kind == "request_lifecycle":
        return _make_request_lifecycle_decorator
    raise ValueError(
        f"enable_load_monitor: unknown kind {kind!r}. "
        "Expected 'scheduler_message' or 'request_lifecycle'."
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_notify(self: Any) -> Optional[_NotifyFn]:
    """Return the bound callback for *self*, or None if absent or disabled."""
    if getattr(getattr(self, "server_args", None), "load_reporter_port", None) is None:
        return None
    return _REGISTRY.get(self)


def _classify_scheduler_obj(obj: Any) -> Optional[tuple[Any, int]]:
    """Return (LoadReporterRefreshReason, count) for scheduler payloads."""
    from sglang.srt.managers.io_struct import (
        AbortReq,
        BatchTokenizedEmbeddingReqInput,
        BatchTokenizedGenerateReqInput,
    )
    from sglang.srt.managers.io_struct import LoadReporterRefreshReason as Reason
    from sglang.srt.managers.io_struct import (
        TokenizedEmbeddingReqInput,
        TokenizedGenerateReqInput,
    )

    if isinstance(obj, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)):
        return (Reason.DISPATCH, 1)
    if isinstance(
        obj, (BatchTokenizedGenerateReqInput, BatchTokenizedEmbeddingReqInput)
    ):
        return (Reason.DISPATCH, len(obj.batch))
    if isinstance(obj, AbortReq):
        return (Reason.ABORT, 1)
    return None


def _make_scheduler_message_decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a synchronous scheduler-dispatch method."""

    @functools.wraps(fn)
    def _scheduler_wrapper(self: Any, obj: Any, *args: Any, **kwargs: Any) -> Any:
        result = fn(self, obj, *args, **kwargs)  # let exceptions propagate unchanged
        notify = _get_notify(self)
        if notify is None:
            return result
        classified = _classify_scheduler_obj(obj)
        if classified is None:
            return result
        try:
            notify(*classified)
        except Exception:
            logger.exception("Load monitor callback raised on scheduler_message")
        return result

    return _scheduler_wrapper


async def _finalize_request_lifecycle(source: Any, notify: _NotifyFn):
    """Async-generator finalization wrapper.

    Iterates source and fires exactly one COMPLETION event in the finally block,
    covering normal exhaustion, early aclose(), and cancellation.
    """
    from sglang.srt.managers.io_struct import LoadReporterRefreshReason as Reason

    try:
        async with aclosing(source) as owned_source:
            async for item in owned_source:
                yield item
    finally:
        try:
            notify(Reason.COMPLETION, 1)
        except Exception:
            logger.exception("Load monitor callback raised on request_lifecycle")


def _make_request_lifecycle_decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap an async generator to fire COMPLETION on exit."""
    # The wrappers are plain functions that RETURN the shared finalization
    # async generator (one layer, not a nested ``async for``).  This keeps
    # ``aclose()``/cancellation propagating straight into its ``finally`` so
    # COMPLETION fires exactly once and promptly.  Callers only ever ``async
    # for`` / ``__anext__`` the result, never ``await`` the call itself.
    if inspect.ismethod(fn):
        owner = fn.__self__

        @functools.wraps(fn)
        def _bound_wrapper(*args: Any, **kwargs: Any):
            source = fn(*args, **kwargs)
            notify = _get_notify(owner)
            if notify is None:
                return source
            return _finalize_request_lifecycle(source, notify)

        return _bound_wrapper

    @functools.wraps(fn)
    def _unbound_wrapper(self: Any, *args: Any, **kwargs: Any):
        source = fn(self, *args, **kwargs)
        notify = _get_notify(self)
        if notify is None:
            return source
        return _finalize_request_lifecycle(source, notify)

    return _unbound_wrapper
