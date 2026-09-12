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


logger = logging.getLogger(__name__)
_enabled = os.environ.get("SGLANG_FLASHINFER_ALPHAMOE_TRACE_SHAPES", "0") == "1"
_kernel_shapes: dict[tuple, dict] = {}
_reported_executions: set[tuple] = set()


def record_alphamoe_kernel(
    *, kernel, hidden_states, num_experts, intermediate_size, top_k, block_m
) -> None:
    if not _enabled:
        return
    from sglang.srt.model_executor.runner_utils import get_is_capture_mode

    m, hidden_size = hidden_states.shape
    captured = get_is_capture_mode()
    key = (kernel, m, num_experts, hidden_size, intermediate_size, top_k, captured)
    _kernel_shapes[key] = {
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


def record_alphamoe_execution(forward_batch, *, execution: str, padded_tokens: int):
    if not _enabled or not _kernel_shapes:
        return
    arm_file = os.environ.get("SGLANG_FLASHINFER_ALPHAMOE_TRACE_ARM_FILE")
    if not arm_file or not os.path.isfile(arm_file):
        return
    records = [
        record
        for record in _kernel_shapes.values()
        if record["M"] == padded_tokens
    ]
    mode = forward_batch.forward_mode.name
    real_tokens = len(forward_batch.input_ids)
    key = (execution, mode, padded_tokens, real_tokens, forward_batch.batch_size)
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
                "kernels": records,
            },
            sort_keys=True,
        ),
    )
