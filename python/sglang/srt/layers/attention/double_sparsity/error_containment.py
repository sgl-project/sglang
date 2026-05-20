"""Mid-decode error containment for Double Sparsity.

When the DS selector or adapter raises a typed exception mid-batch, the
scheduler must NOT propagate the exception in a way that aborts the
entire batch or kills the worker. Instead:

* the failing request is marked failed with an error class,
* the request's partial per-request-summary state is cleared,
* siblings in the same batch continue,
* the Prometheus error counter is incremented.

This module is the seam where attention-path code wraps a single DS
step in a try/except, classifies the failure, and reports it.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from sglang.srt.layers.attention.double_sparsity.page_table_adapter import (
    DSAdapterError,
)
from sglang.srt.layers.attention.double_sparsity import metrics as _ds_metrics


_EXCEPTION_TO_CLASS = {
    DSAdapterError: "bad_adapter_input",
}


def _classify(exc: BaseException) -> str:
    for exc_cls, label in _EXCEPTION_TO_CLASS.items():
        if isinstance(exc, exc_cls):
            return label
    return "selector_runtime_error"


def try_run_ds_step(
    fn: Callable[[], Any],
    *,
    request_id: str,
    error_state: Dict[str, Any],
) -> Tuple[bool, Optional[Any]]:
    """Run a single DS attention step under containment.

    Args:
        fn: a no-arg callable that performs the DS step for one request.
            It may raise any exception. DS-typed exceptions are caught;
            non-DS exceptions are re-raised because they indicate a bug
            that is not safely contained.
        request_id: the request identifier used in observability output.
        error_state: a per-request mutable dict the caller owns. On
            failure, this function writes ``error_state["ds_error_cls"]``
            and ``error_state["ds_error_message"]`` so the scheduler can
            mark the request failed and clear partial state.

    Returns:
        ``(success, value)``. On success ``value`` is whatever ``fn``
        returned. On failure ``value`` is ``None``.
    """
    try:
        return True, fn()
    except DSAdapterError as exc:
        cls = _classify(exc)
        error_state["ds_error_cls"] = cls
        error_state["ds_error_message"] = str(exc)
        _ds_metrics.record_error(cls, message=str(exc), request_id=request_id)
        # Caller is responsible for clearing partial per_request_summary
        # state for this request_id (the scheduler glue owns that map).
        error_state.setdefault("clear_per_request_summary", True)
        return False, None
