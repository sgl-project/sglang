"""Opt-in AlphaMoE observations at kernel submission and graph replay boundaries.

Capture-time submissions register graph contents, but are never reported as
request observations. The arm file must be created after server warmup.
These are dispatch receipts; successful model evaluation and a GPU trace are
the corresponding numerical and device-execution evidence.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from weakref import WeakKeyDictionary

logger = logging.getLogger(__name__)
_enabled = os.environ.get("SGLANG_FLASHINFER_ALPHAMOE_TRACE_SHAPES", "0") == "1"
_active_submissions: ContextVar[dict | None] = ContextVar(
    "alphamoe_active_submissions", default=None
)
_graph_submissions: WeakKeyDictionary = WeakKeyDictionary()
_reported_executions: set[tuple] = set()


@contextmanager
def observe_alphamoe_submissions():
    """Collect only kernel submissions made inside this execution scope."""
    submissions = {}
    token = _active_submissions.set(submissions)
    try:
        yield submissions
    finally:
        _active_submissions.reset(token)


@contextmanager
def observe_alphamoe_capture(backend, shape_key):
    """Bind successful capture registration to its backend and complete key."""
    with observe_alphamoe_submissions() as submissions:
        yield
    # A failed capture never publishes a registration. Re-capture replaces the
    # same graph key; another stream, variant, runner, or backend cannot match.
    _graph_submissions.setdefault(backend, {})[shape_key] = submissions


def record_alphamoe_kernel(
    *, kernel, hidden_states, num_experts, intermediate_size, top_k, block_m
) -> None:
    submissions = _active_submissions.get()
    if not _enabled or submissions is None:
        return
    from sglang.srt.model_executor.runner_utils import get_is_capture_mode

    m, hidden_size = hidden_states.shape
    captured = get_is_capture_mode()
    key = (kernel, m, num_experts, hidden_size, intermediate_size, top_k, captured)
    submissions[key] = {
        "kernel": kernel,
        "M": int(m),
        "E": int(num_experts),
        "H": int(hidden_size),
        "I_local": None if intermediate_size is None else int(intermediate_size),
        "top_k": int(top_k),
        "block_m": int(block_m),
        "captured": captured,
        "device": str(hidden_states.device),
    }


def record_alphamoe_execution(
    forward_batch,
    *,
    execution: str,
    padded_tokens: int,
    backend=None,
    graph_key=None,
    submissions=None,
):
    if not _enabled:
        return
    arm_file = os.environ.get("SGLANG_FLASHINFER_ALPHAMOE_TRACE_ARM_FILE")
    if not arm_file or not os.path.isfile(arm_file):
        return
    if backend is not None:
        submissions = _graph_submissions.get(backend, {}).get(graph_key, {})
    records = list((submissions or {}).values())
    mode = forward_batch.forward_mode.name
    real_tokens = len(forward_batch.input_ids)
    key = (
        execution,
        mode,
        id(backend),
        graph_key,
        padded_tokens,
        real_tokens,
        forward_batch.batch_size,
        tuple(record["kernel"] for record in records),
    )
    if key in _reported_executions:
        return
    _reported_executions.add(key)
    logger.warning(
        "ALPHAMOE_RUNTIME_EXECUTION %s",
        json.dumps(
            {
                "schema": "sglang-flashinfer-alphamoe-runtime-v2",
                "execution": execution,
                "forward_mode": mode,
                "M": int(padded_tokens),
                "real_tokens": int(real_tokens),
                "batch_size": int(forward_batch.batch_size),
                "attribution": (
                    "capture_registration"
                    if backend is not None
                    else "direct_submissions"
                ),
                "graph_backend": (
                    type(backend).__name__ if backend is not None else None
                ),
                "graph_key": repr(graph_key) if graph_key is not None else None,
                "kernels": records,
            },
            sort_keys=True,
        ),
    )
