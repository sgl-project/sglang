"""Breakable prefill CUDA graph policy for DeepSeek-V4 on CUDA.

Breakable capture is the CUDA default for prefill. Argument resolution turns it
off for DeepSeek-V4 because capturing the default ~70 token buckets pins the c4
indexer scratch in the capture pool. Eager DeepSeek-V4 prefill is host-bound
instead: launching an 8K-token extend from Python takes about twice its GPU
time. A short geometric bucket list up to the prefill chunk keeps the capture
pool to a few GB while the large extends replay from graphs.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from sglang.srt.configs.model_config import is_deepseek_v4
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase
from sglang.srt.runtime_context import get_context, get_exec, get_parallel, get_schedule

logger = logging.getLogger(__name__)

# Smallest captured extend; with the runner's 2x padding cap, shorter extends stay eager.
_MIN_BUCKET_TOKENS = 512


def dsv4_prefill_graph_buckets(max_tokens: int) -> List[int]:
    """Geometric token buckets from _MIN_BUCKET_TOKENS up to and including max_tokens."""
    buckets = []
    size = _MIN_BUCKET_TOKENS
    while size < max_tokens:
        buckets.append(size)
        size *= 2
    buckets.append(max_tokens)
    return buckets


def _prefill_token_ceiling() -> Optional[int]:
    chunk = get_schedule().chunked_prefill_size
    if chunk is None or chunk <= 0:
        return None
    max_prefill = get_schedule().max_prefill_tokens
    return min(chunk, max_prefill) if max_prefill else chunk


def _supported_topology() -> bool:
    parallel = get_parallel()
    return (
        parallel.pp_size == 1
        and parallel.attn_dp_size == 1
        and parallel.attn_cp_size == 1
        and parallel.dcp_size == 1
        and not get_exec().overlap.enable_two_batch_overlap
    )


def maybe_enable_dsv4_prefill_cuda_graph(*, server_args: Any, hf_config: Any) -> bool:
    """Re-enable breakable prefill capture for DeepSeek-V4 when only the auto rule disabled it.

    An explicit --cuda-graph-backend-prefill / --cuda-graph-bs-prefill /
    --cuda-graph-config choice is left untouched.
    """
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend
    from sglang.srt.utils import is_cuda

    if (
        server_args.cuda_graph_backend_prefill is not None
        or server_args.cuda_graph_bs_prefill is not None
        or server_args.cuda_graph_config is not None
    ):
        return False
    cfg = get_exec().graph.cuda_graph_config
    if (
        cfg is None
        or cfg.prefill.backend != Backend.DISABLED
        or cfg.decode.backend == Backend.DISABLED
    ):
        return False
    if not (is_cuda() and is_deepseek_v4(hf_config) and _supported_topology()):
        return False
    if not get_moe_a2a_backend().is_none():
        return False
    ceiling = _prefill_token_ceiling()
    if ceiling is None or ceiling < _MIN_BUCKET_TOKENS:
        return False
    buckets = dsv4_prefill_graph_buckets(ceiling)
    get_context().override(
        "dsv4_prefill_cuda_graph",
        cuda_graph_config=with_phase(
            cfg, Phase.PREFILL, backend=Backend.BREAKABLE, bs=buckets, max_bs=ceiling
        ),
    )
    logger.info(
        "DeepSeek-V4 breakable prefill CUDA graph enabled, token buckets=%s", buckets
    )
    return True
