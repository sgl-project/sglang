"""Memory budget for the DSA indexer's `[num_q x num_k]` fp32 MQA logits.

Both indexer flavors (`dsa_indexer.Indexer` and `dsa_indexer_kpool.IndexerKPool`)
materialize this tensor during long-context prefill, and neither can size it from
the model config alone: it grows with the batch's context lengths. The budget
below is what decides whether the computation has to be split, so it lives in one
place — a second copy is how the kpool path ended up with no bound at all
(sgl-project/sglang#37712).

On ROCm the bound is a correctness bound, not only an out-of-memory guard: aiter's
`fp8_mqa_logits` only compiles below 2 GiB of logits (`buffer_store`).
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.runtime_context import get_schedule
from sglang.srt.utils import get_device_module, is_hip, is_xpu

_is_hip = is_hip()
_is_xpu = is_xpu()

MQA_LOGITS_BYTES_PER_ELEM = 4
MQA_LOGITS_STATIC_SKIP_ELEMS = 8_000_000
MQA_LOGITS_TOTAL_MEM_FRACTION = 0.3
# aiter's fp8_mqa_logits only compiles below 2 GiB of logits (buffer_store).
MQA_LOGITS_MAX_BYTES_ROCM = 2**31 - 1

_budget_bytes_by_device: Dict[int, int] = {}


def _free_mem_fraction() -> float:
    return envs.SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION.get()


def mqa_logits_budget_bytes(device_index: int) -> int:
    free_mem_fraction = _free_mem_fraction()
    cached_budget = _budget_bytes_by_device.get(device_index)
    if cached_budget is not None:
        return cached_budget

    total_mem = get_device_module().get_device_properties(device_index).total_memory

    total_mem_budget = int(total_mem * MQA_LOGITS_TOTAL_MEM_FRACTION)
    mem_fraction_static = get_schedule().mem_fraction_static
    if mem_fraction_static is None:
        static_budget = total_mem_budget
    else:
        static_free_mem = int(total_mem * max(0.0, 1.0 - mem_fraction_static))
        static_budget = min(
            int(static_free_mem * free_mem_fraction),
            total_mem_budget,
        )
    static_budget = max(1, static_budget)

    # Keep the static serving-memory guard during CUDA graph capture without
    # caching it. The first non-capture prefill path will cache the real
    # free-memory budget below.
    if get_is_capture_mode():
        return static_budget

    # Match the original free-memory guard: logits_bytes * 2 > free_mem.
    # Synchronizes the host; cache the result capped by serving-memory headroom.
    if _is_xpu:
        # On XPU, use total_mem budget as the free-memory estimate;
        # dynamic free-memory query is not supported the same way as CUDA.
        # TODO Use torch.xpu.mem_get_info() when available (planned end of 2026).
        budget_bytes = static_budget
    else:
        free_mem, _ = torch.cuda.mem_get_info(device_index)
        budget_bytes = min(int(free_mem * free_mem_fraction), static_budget)

    budget_bytes = max(1, budget_bytes)
    _budget_bytes_by_device[device_index] = budget_bytes
    return budget_bytes


def should_chunk_mqa_logits(
    num_q: int, num_k: int, device_index: int
) -> Tuple[bool, int]:
    """
    Detect whether we need to chunk the MQA logits computation to avoid OOM,
    and on ROCm to stay under aiter's 2 GiB logits limit
    Return: (need_chunk, logits_budget_bytes)
    """
    # Quick static check for normal batches
    if num_q * num_k < MQA_LOGITS_STATIC_SKIP_ELEMS:
        return False, 0

    logits_bytes = num_q * num_k * MQA_LOGITS_BYTES_PER_ELEM
    logits_budget_bytes = mqa_logits_budget_bytes(device_index)
    if _is_hip:
        logits_budget_bytes = min(logits_budget_bytes, MQA_LOGITS_MAX_BYTES_ROCM)

    need_chunk = logits_bytes > logits_budget_bytes
    return need_chunk, logits_budget_bytes


def mqa_logits_max_rows(logits_budget_bytes: int, num_k: int, num_q: int) -> int:
    """How many q rows of a `[num_q x num_k]` logits tensor fit in the budget."""
    bytes_per_row = num_k * MQA_LOGITS_BYTES_PER_ELEM
    max_rows = max(1, int(logits_budget_bytes // max(bytes_per_row, 1)))
    return min(max_rows, num_q)
