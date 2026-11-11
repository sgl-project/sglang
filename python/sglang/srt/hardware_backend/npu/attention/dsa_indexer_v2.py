from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import torch

_INDEXER_LAYOUT = dict(layout_q="TND", layout_k="PA_BBND", mask_mode=3, cmp_ratio=1)


class DsaIndexerV2Metadata(NamedTuple):
    cu_seqlens_q: torch.Tensor
    kv_lens: torch.Tensor
    plan: torch.Tensor
    max_seqlen_q: int


@lru_cache(maxsize=1)
def load_dsa_indexer_v2_ops():
    try:
        from cann_ops_transformer.ops import (
            quant_lightning_indexer,
            quant_lightning_indexer_metadata,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Ascend DSA context parallel with FP8 indexer K requires "
            "QuantLightningIndexerV2 and its metadata op from cann_ops_transformer."
        ) from exc
    return quant_lightning_indexer, quant_lightning_indexer_metadata


def prepare_dsa_indexer_v2_metadata(
    query_ends: torch.Tensor,
    kv_lens: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    topk: int,
    max_seqlen_q: int,
    existing: DsaIndexerV2Metadata | None = None,
) -> DsaIndexerV2Metadata:
    _, metadata_op = load_dsa_indexer_v2_ops()
    cu_seqlens_q = (
        torch.cat((query_ends.new_zeros(1), query_ends))
        if existing is None
        else existing.cu_seqlens_q
    )
    plan = metadata_op(
        num_heads,
        1,
        head_dim,
        topk,
        1,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=kv_lens,
        batch_size=query_ends.numel(),
        max_seqlen_q=max_seqlen_q,
        **_INDEXER_LAYOUT,
    )
    if existing is not None:
        # Captured indexer calls hold the address of this plan buffer.
        existing.plan.copy_(plan)
        return existing
    return DsaIndexerV2Metadata(cu_seqlens_q, kv_lens, plan, max_seqlen_q)


def run_dsa_indexer_v2(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    block_table: torch.Tensor,
    topk: int,
    metadata: DsaIndexerV2Metadata,
) -> torch.Tensor:
    indexer_op, _ = load_dsa_indexer_v2_ops()
    # DP padding can add rows beyond those described by the request page table.
    num_query_rows = metadata.kv_lens.numel() * metadata.max_seqlen_q
    indices, _ = indexer_op(
        query[:num_query_rows],
        key,
        weights[:num_query_rows].float(),
        query_scale[:num_query_rows],
        key_scale,
        topk,
        1,
        cu_seqlens_q=metadata.cu_seqlens_q,
        seqused_k=metadata.kv_lens,
        block_table=block_table,
        metadata=metadata.plan,
        max_seqlen_q=metadata.max_seqlen_q,
        return_value=0,
        **_INDEXER_LAYOUT,
    )
    indices = indices.squeeze(1)
    if num_query_rows < query.shape[0]:
        indices = torch.nn.functional.pad(
            indices, (0, 0, 0, query.shape[0] - num_query_rows), value=-1
        )
    return indices
