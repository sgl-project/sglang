"""Sizing rules for CUDA RoPE DSA under decode context parallelism (DCP).

Target KV is sharded across the DCP group. Index keys and the draft KV stay
replicated over the allocator-global slot space, and each rank runs sparse
attention for the query heads of the whole DCP group.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

from sglang.srt.configs.model_config import get_dsa_index_topk, is_deepseek_dsa
from sglang.srt.environ import envs
from sglang.srt.runtime_context import (
    get_parallel,
    get_schedule,
    max_prefill_buffer_tokens,
    max_speculative_num_draft_tokens,
)
from sglang.srt.utils import is_hip

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

_is_hip = is_hip()

# FlashInfer 0.7.0.post1's sparse RoPE kernel returns a wrong LSE above this.
_TRTLLM_LSE_MAX_HEADS = 32
# TRT-LLM puts the batch in a 16-bit grid dimension.
TRTLLM_MAX_BATCH_ROWS = 65535


def is_cuda_rope_dsa(kvc: KVCacheConfigurator) -> bool:
    return (
        str(kvc.device).startswith("cuda")
        and not _is_hip
        and kvc.model_config.qk_rope_head_dim > 0
    )


def dsa_indexer_dcp_scale(kvc: KVCacheConfigurator) -> int:
    """How many ranks' worth of slots the replicated index keys cover."""
    if is_cuda_rope_dsa(kvc) and not kvc.is_draft_worker:
        return get_parallel().attn_dcp_size
    return 1


def dsa_dcp_head_groups(num_q_heads: int) -> int:
    """Smallest even split of the query heads that fits the LSE kernel."""
    groups = math.ceil(num_q_heads / _TRTLLM_LSE_MAX_HEADS)
    while num_q_heads % groups:
        groups += 1
    return groups


def dsa_dcp_max_query_rows(
    model_config: ModelConfig, *, max_running_requests: Optional[int] = None
) -> int:
    """Largest number of query rows one sparse-attention forward can see."""
    schedule = get_schedule()
    parallel = get_parallel()
    if schedule.chunked_prefill_size is None or schedule.chunked_prefill_size <= 0:
        prefill_rows = max(schedule.max_prefill_tokens, model_config.context_len)
    else:
        prefill_rows = max_prefill_buffer_tokens()
    if max_running_requests is None:
        # --max-running-requests is global; the request pool defaults to 4096.
        max_running_requests = (
            schedule.max_running_requests // parallel.attn_dp_size
            if schedule.max_running_requests is not None
            else 4096
        )
    decode_rows = max_running_requests
    if parallel.dcp_enabled:
        decode_rows *= max_speculative_num_draft_tokens() or 1
    rows = max(prefill_rows, decode_rows)
    if parallel.attn_dp_size > 1:
        # Attention DP pads rows to a multiple of the attention TP size.
        rows = math.ceil(rows / parallel.attn_tp_size) * parallel.attn_tp_size
    return rows


def dsa_dcp_workspace_size_bytes(
    *, num_q_heads: int, dcp_size: int, max_query_rows: int
) -> int:
    """TRT-LLM workspace for the gathered heads, including its LSE scratch."""
    scratch_bytes = envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get()
    if dcp_size == 1:
        return scratch_bytes
    num_dcp_q_heads = num_q_heads * dcp_size
    rows_per_call = TRTLLM_MAX_BATCH_ROWS // dsa_dcp_head_groups(num_dcp_q_heads)
    # float2[heads * rows * 256] softmax stats plus a 1 MiB guard.
    stats_bytes = 8 * num_dcp_q_heads * min(max_query_rows, rows_per_call) * 256
    return scratch_bytes * dcp_size + stats_bytes + (1 << 20)


def dsa_dcp_merge_size_bytes(
    *, num_q_heads: int, dcp_size: int, max_query_rows: int, kv_lora_rank: int
) -> int:
    """Peak size of the buffers that merge partial outputs across ranks."""
    if dcp_size == 1:
        return 0
    rows_heads = max_query_rows * num_q_heads
    # a2a: BF16 send and receive tensors; ag_rs: one FP32 gathered output.
    gathered_outputs = rows_heads * dcp_size * kv_lora_rank * 4
    # ag_rs: local FP32 reduction plus the BF16 result.
    local_outputs = rows_heads * kv_lora_rank * 6
    # ag_rs gathers every peer's LSE; fi_a2a sends and receives two FP32 stats.
    lse_bytes = rows_heads * (max(4 * dcp_size**2, 16 * dcp_size) + 4 * dcp_size)
    return gathered_outputs + local_outputs + lse_bytes


def dsa_dcp_replicated_q_weight_bytes(
    model_config: ModelConfig, num_dcp_q_heads: int
) -> int:
    """BF16 ``q_b_proj`` and ``w_kc`` weights gathered over the DCP group."""
    if not get_parallel().dcp_replicate_q_proj:
        return 0
    hf_config = model_config.hf_config
    qk_head_dim = model_config.qk_nope_head_dim + model_config.qk_rope_head_dim
    q_in_dim = hf_config.q_lora_rank or hf_config.hidden_size
    per_layer = num_dcp_q_heads * (
        qk_head_dim * q_in_dim
        + model_config.qk_nope_head_dim * model_config.kv_lora_rank
    )
    return model_config.num_hidden_layers * per_layer * 2


def dsa_dcp_runtime_reservation_bytes(
    kvc: KVCacheConfigurator, *, available_bytes: Optional[int] = None
) -> int:
    """Memory DCP allocates after KV sizing, on top of the non-DCP path."""
    model_config = kvc.model_config
    if not is_deepseek_dsa(model_config.hf_config):
        return 0
    dcp_size = dsa_indexer_dcp_scale(kvc)
    if dcp_size == 1:
        return 0

    max_running_requests = None
    if available_bytes is not None and available_bytes > 0:
        # Bound requests by the capacity before this reservation is taken out.
        capacity = kvc.config_from_budget(available_bytes).max_total_num_tokens
        max_running_requests = kvc.resolve_max_num_reqs(capacity)
    max_rows = dsa_dcp_max_query_rows(
        model_config, max_running_requests=max_running_requests
    )
    num_q_heads = model_config.num_attention_heads // get_parallel().attn_tp_size
    num_dcp_q_heads = num_q_heads * dcp_size
    kv_lora_rank = model_config.kv_lora_rank
    head_groups = dsa_dcp_head_groups(num_dcp_q_heads)

    workspace_bytes = (
        dsa_dcp_workspace_size_bytes(
            num_q_heads=num_q_heads, dcp_size=dcp_size, max_query_rows=max_rows
        )
        - envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get()
    )
    # BF16 gathered query and partial output of the other ranks' heads.
    gathered_bytes = (
        max_rows
        * num_q_heads
        * (dcp_size - 1)
        * (2 * kv_lora_rank + model_config.qk_rope_head_dim)
        * 2
    )
    lse_bytes = max_rows * num_dcp_q_heads * 4
    merge_bytes = dsa_dcp_merge_size_bytes(
        num_q_heads=num_q_heads,
        dcp_size=dcp_size,
        max_query_rows=max_rows,
        kv_lora_rank=kv_lora_rank,
    )
    # One int32 counter per extra head; the buffer holds at least 8192 rows.
    counter_bytes = max(max_rows, 8192) * num_q_heads * (dcp_size - 1) * 4
    # Rank-local int32 page table and per-row counts.
    padded_topk = math.ceil(get_dsa_index_topk(model_config.hf_config) / 4) * 4
    page_table_bytes = max_rows * head_groups * (padded_topk + 1) * 4
    # A forward split over several TRT-LLM calls concatenates their outputs.
    concat_bytes = 0
    if max_rows > TRTLLM_MAX_BATCH_ROWS // head_groups:
        concat_bytes = max_rows * num_dcp_q_heads * kv_lora_rank * 2
    return (
        workspace_bytes
        + gathered_bytes
        + lse_bytes
        + merge_bytes
        + counter_bytes
        + page_table_bytes
        + concat_bytes
        + dsa_dcp_replicated_q_weight_bytes(model_config, num_dcp_q_heads)
    )
