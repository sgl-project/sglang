from __future__ import annotations

from typing import Optional

import msgspec
import torch

from sglang.kernels.ops.attention.dsa.transform_index import (
    transform_index_page_table_dcp,
)
from sglang.kernels.ops.attention.fixup_zero_kv import fixup_zero_kv_rows

TRTLLM_MAX_BATCH_ROWS = 65535
# FlashInfer carves heads * rows * 256 float2 softmax slots out of the workspace per call;
# the rest is kernel scratch.
_SOFTMAX_BYTES_PER_ROW_HEAD = 256 * 8
_SCRATCH_RESERVE_BYTES = 128 << 20


def trtllm_rows_per_call(*, workspace_bytes: int, heads: int) -> int:
    rows = (workspace_bytes - _SCRATCH_RESERVE_BYTES) // (
        heads * _SOFTMAX_BYTES_PER_ROW_HEAD
    )
    return max(1, min(TRTLLM_MAX_BATCH_ROWS, rows))


class DcpPageTableCache(msgspec.Struct):
    """NOTE(vincentzed): GLM-5.2/5.3 reuse the previous layer's top-k tensor in
    57 of 78 layers; remapping once per fresh top-k cuts 78 launches/step to 21.
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
            page_table, dcp_size, dcp_rank
        )
    cache.layer_id = layer_id
    return cache.page_table, cache.counts


def dcp_trtllm_sparse_attention(
    *,
    query: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    **attention_kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    import flashinfer.decode

    rows, _, heads, _ = query.shape
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
    lse = torch.empty((rows, heads), dtype=torch.float32, device=query.device)
    workspace = attention_kwargs["workspace_buffer"]
    step = trtllm_rows_per_call(
        workspace_bytes=workspace.numel() * workspace.element_size(), heads=heads
    )
    if rows > step:
        attention_kwargs["enable_pdl"] = False
    top_k_lens = attention_kwargs.pop("sparse_mla_top_k_lens", None)
    outputs = []
    for start in range(0, rows, step):
        end = min(start + step, rows)
        output, _ = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=query[start:end],
            block_tables=block_tables[start:end],
            seq_lens=seq_lens[start:end],
            return_lse=True,
            lse=lse[start:end],
            sparse_mla_top_k_lens=(
                None if top_k_lens is None else top_k_lens[start:end]
            ),
            **attention_kwargs,
        )
        output = output.view(end - start, heads, -1)
        fixup_zero_kv_rows(
            output,
            lse[start:end],
            seq_lens[start:end],
            cu_seqlens_q[: end - start + 1],
            1,
        )
        outputs.append(output)
    output = outputs[0] if len(outputs) == 1 else torch.cat(outputs)
    return output.view(rows, 1, heads, -1), lse
