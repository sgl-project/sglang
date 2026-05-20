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
from sglang.srt.layers.attention.double_sparsity.channel_mask import (
    DoubleSparsityChannelMaskCorrupt,
    DoubleSparsityChannelMaskMissing,
)
from sglang.srt.layers.attention.double_sparsity.selector import (
    DoubleSparsityTPMisconfigured,
)
from sglang.srt.layers.attention.double_sparsity import metrics as _ds_metrics


_CONTAINED_EXCEPTIONS = (
    DSAdapterError,
    DoubleSparsityChannelMaskCorrupt,
    DoubleSparsityChannelMaskMissing,
    DoubleSparsityTPMisconfigured,
    RuntimeError,  # selector runtime errors (e.g. placeholder guard)
)


def try_run_ds_step(
    fn: Callable[[], Any],
    *,
    request_id: str,
    error_state: Dict[str, Any],
    layer_id: Optional[int] = None,
    selector_id: Optional[str] = None,
) -> Tuple[bool, Optional[Any]]:
    """Run a single DS attention step under containment.

    Catches the DS-typed exception families plus generic RuntimeError
    (which covers placeholder-guard runtime errors), classifies into
    one of `DS_ERROR_CLASSES`, increments the Prometheus counter, and
    writes ``error_state`` so the scheduler clears partial per-request
    state. Non-typed exceptions still propagate so they are not silently
    swallowed.

    Args:
        fn: a no-arg callable that performs the DS step for one request.
        request_id: per-request identifier for observability.
        error_state: per-request mutable dict the caller owns.
        layer_id: attention layer index (for structured logs).
        selector_id: stable selector identifier (for structured logs).

    Returns:
        ``(success, value)``. On success ``value`` is whatever ``fn``
        returned. On failure ``value`` is ``None``.
    """
    try:
        return True, fn()
    except _CONTAINED_EXCEPTIONS as exc:
        cls = _ds_metrics.classify_ds_exception(exc)
        error_state["ds_error_cls"] = cls
        error_state["ds_error_message"] = str(exc)
        error_state["ds_original_exception"] = exc
        error_state.setdefault("clear_per_request_summary", True)
        _ds_metrics.record_error(
            cls,
            message=str(exc),
            request_id=request_id,
            layer_id=layer_id,
            selector_id=selector_id,
        )
        return False, None
