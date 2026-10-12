"""RoPE DSA sparse attention under decode context parallelism (DCP)."""

from __future__ import annotations

import math
from typing import Optional

import msgspec
import torch

from sglang.kernels.ops.attention.dsa.transform_index import (
    transform_index_page_table_dcp,
)
from sglang.kernels.ops.attention.fixup_zero_kv import fixup_zero_kv_rows

# FlashInfer 0.7.0.post1 returns a wrong LSE above this; drop the head folding
# once test_folded_head_groups_match_single_group_attention passes with one group.
_TRTLLM_LSE_MAX_HEADS = 32
# TRT-LLM puts the batch in a 16-bit grid dimension.
TRTLLM_MAX_BATCH_ROWS = 65535


def dsa_dcp_head_groups(num_q_heads: int) -> int:
    groups = math.ceil(num_q_heads / _TRTLLM_LSE_MAX_HEADS)
    while num_q_heads % groups:
        groups += 1
    return groups


class DcpPageTableCache(msgspec.Struct):
    """Rank-local page table of the latest top-k, reused by shared-top-k layers.

    NOTE(vincentzed): 57 of GLM's 78 layers reuse the previous layer's top-k
    tensor; remapping once per fresh top-k cuts 78 launches/step to 21.
    """

    metadata: object = None
    layer_id: int = -1
    topk_indices: Optional[torch.Tensor] = None
    page_table: Optional[torch.Tensor] = None
    counts: Optional[torch.Tensor] = None


def localize_dcp_page_table(
    cache: DcpPageTableCache,
    *,
    metadata: object,
    layer_id: int,
    topk_indices: Optional[torch.Tensor],
    page_table: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    head_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    reuse = (
        topk_indices is not None
        and cache.metadata is metadata
        and cache.layer_id == layer_id - 1
        and cache.topk_indices is topk_indices
    )
    if not reuse:
        cache.metadata = metadata
        cache.topk_indices = topk_indices
        cache.page_table, cache.counts = transform_index_page_table_dcp(
            page_table, dcp_size, dcp_rank, repeat_rows=head_groups
        )
    cache.layer_id = layer_id
    return cache.page_table, cache.counts


def dcp_trtllm_sparse_attention(
    *,
    query: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    head_groups: int,
    cu_seqlens_q: torch.Tensor,
    **attention_kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    import flashinfer.decode

    rows, _, heads, dim = query.shape
    folded_rows = rows * head_groups
    folded_heads = heads // head_groups
    out_dtype = torch.bfloat16
    if rows == 0:
        return (
            torch.empty(
                (0, 1, heads, attention_kwargs["kv_lora_rank"]),
                dtype=out_dtype,
                device=query.device,
            ),
            torch.empty((0, heads), dtype=torch.float32, device=query.device),
        )
    query = query.view(folded_rows, 1, folded_heads, dim)
    lse = torch.empty((rows, heads), dtype=torch.float32, device=query.device)
    folded_lse = lse.view(folded_rows, folded_heads)
    rows_per_call = (TRTLLM_MAX_BATCH_ROWS // head_groups) * head_groups
    if folded_rows > rows_per_call:
        # The calls share one workspace, so they must not overlap.
        attention_kwargs["enable_pdl"] = False
    outputs = []
    for start in range(0, folded_rows, rows_per_call):
        end = min(start + rows_per_call, folded_rows)
        output, _ = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=query[start:end],
            block_tables=block_tables[start:end],
            seq_lens=seq_lens[start:end],
            return_lse=True,
            lse=folded_lse[start:end],
            **attention_kwargs,
        )
        output = output.view(end - start, folded_heads, -1)
        fixup_zero_kv_rows(
            output,
            folded_lse[start:end],
            seq_lens[start:end],
            cu_seqlens_q[: end - start + 1],
            1,
        )
        outputs.append(output)
    output = outputs[0] if len(outputs) == 1 else torch.cat(outputs)
    return output.view(rows, 1, heads, -1), lse
