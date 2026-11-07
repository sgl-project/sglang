# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""MiniMax-M3 index selection using AITER's FP8 score/top-k kernels.

Retains SGLang's NHD KV pool and existing sparse attention consumer.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class AiterMiniMaxSelection:
    topk: torch.Tensor
    block_table: torch.Tensor
    context_lens: torch.Tensor


@dataclass
class _Metadata:
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    max_seq_len: int
    query_len: int = 1
    cu_seqlens_q: Optional[torch.Tensor] = None
    max_query_len: int = 1
    row_req_id: Optional[torch.Tensor] = None
    kv_lens: Optional[torch.Tensor] = None
    num_valid_pages: Optional[torch.Tensor] = None


class AiterMiniMaxIndexer:
    def __init__(
        self,
        *,
        max_context_len: int,
        page_size: int,
        num_index_heads: int,
        num_kv_heads: int,
        head_dim: int,
        topk: int,
        init_blocks: int,
        local_blocks: int,
        index_cache: torch.Tensor,
    ):
        if page_size != 128 or head_dim != 128 or topk != 16:
            raise ValueError(
                "AITER MiniMax indexer requires page_size=128, head_dim=128, topk=16"
            )
        if num_index_heads != num_kv_heads:
            raise ValueError(
                "AITER MiniMax indexer requires one index head per local KV head"
            )
        if index_cache.dtype != torch.float8_e4m3fn or not index_cache.is_contiguous():
            raise ValueError(
                "AITER MiniMax indexer requires a contiguous FP8 E4M3 index cache"
            )
        if tuple(index_cache.shape[1:]) != (1, head_dim):
            raise ValueError(
                "AITER MiniMax indexer expects an NHD index cache with one KV head"
            )
        max_blocks = (max_context_len + page_size - 1) // page_size
        if max_blocks > 8192:
            raise ValueError(
                "AITER MiniMax indexer supports at most 8192 sparse blocks"
            )

        from aiter.ops.msa_attention import (
            pa_sparse_block_score_decode,
            pa_sparse_block_score_prefill,
            pa_sparse_block_topk,
        )

        self.score_op = pa_sparse_block_score_decode
        self.prefill_score_op = pa_sparse_block_score_prefill
        self.topk_op = pa_sparse_block_topk
        self.max_context_len = max_context_len
        self.page_size = page_size
        self.num_index_heads = num_index_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.topk = topk
        self.init_blocks = init_blocks
        self.local_blocks = local_blocks
        self.metadata: Optional[_Metadata] = None

    def prepare(self, req_to_token, req_pool_indices, seq_lens, query_len):
        if self.num_index_heads * query_len > 16:
            raise ValueError(
                "AITER MiniMax score supports at most 16 query-token/head pairs"
            )
        # A sparse block is an allocator page in this configuration. Rebuild
        # from the live slot table once per forward, including graph replay.
        block_table = torch.index_select(
            req_to_token[:, : self.max_context_len : self.page_size],
            0,
            req_pool_indices,
        )
        block_table = (block_table // self.page_size).to(torch.int32)
        self.metadata = _Metadata(
            block_table, seq_lens.to(torch.int32), self.max_context_len, query_len
        )

    def prepare_prefill(
        self,
        req_to_token,
        req_pool_indices,
        seq_lens,
        cu_seqlens_q,
        *,
        total_q: int,
        max_query_len: int,
        max_seq_len: int,
    ):
        """Build ragged causal rows once, shared by all indexer layers."""
        if not 0 < max_seq_len <= self.max_context_len:
            raise ValueError("Prefill context is outside the configured index cache")
        block_table = torch.index_select(
            req_to_token[:, : max_seq_len : self.page_size], 0, req_pool_indices
        )
        block_table = (block_table // self.page_size).to(torch.int32)
        cu_seqlens_q = cu_seqlens_q.to(torch.int32)
        seq_lens = seq_lens.to(torch.int32)
        rows = torch.arange(total_q, device=seq_lens.device, dtype=torch.int32)
        row_req_id = torch.searchsorted(cu_seqlens_q[1:], rows, right=True).to(
            torch.int32
        )
        # Bottom-right causal alignment: final KV length minus Q length,
        # plus this query's local position. Each verify/prefill row is distinct.
        kv_lens = seq_lens[row_req_id] - cu_seqlens_q[row_req_id + 1] + rows + 1
        self.metadata = _Metadata(
            block_table=block_table,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            cu_seqlens_q=cu_seqlens_q,
            max_query_len=max_query_len,
            row_req_id=row_req_id,
            kv_lens=kv_lens,
            num_valid_pages=(kv_lens + self.page_size - 1) // self.page_size,
        )

    def forward(self, index_query, index_cache):
        return self.select(index_query, index_cache).topk

    def select(self, index_query, index_cache):
        if self.metadata is None:
            raise RuntimeError("AITER MiniMax indexer metadata was not initialized")
        md = self.metadata
        total_q = index_query.shape[0]
        expected_q = (
            md.row_req_id.numel()
            if md.cu_seqlens_q is not None
            else md.seq_lens.numel() * md.query_len
        )
        if total_q != expected_q:
            raise ValueError(
                "AITER MiniMax query rows do not match the prepared request geometry"
            )
        q = index_query.to(torch.float8_e4m3fn).contiguous()
        key_cache = index_cache.view(-1, self.page_size, self.head_dim)
        max_blocks = (md.max_seq_len + self.page_size - 1) // self.page_size
        score_width = max(64, 1 << (max_blocks - 1).bit_length())
        score = torch.empty(
            (self.num_index_heads, total_q, score_width),
            dtype=torch.float32,
            device=q.device,
        )
        topk = torch.empty(
            (self.num_index_heads, total_q, self.topk),
            dtype=torch.int32,
            device=q.device,
        )
        # Main and index pools share allocator pages. A 128-token block has
        # eight page16 pages in each separate K/V plane; head index is minor.
        sparse_bt = torch.empty(
            (total_q * self.num_kv_heads, self.topk * 8),
            dtype=torch.int32,
            device=q.device,
        )
        sparse_ctx = torch.empty(
            total_q * self.num_kv_heads, dtype=torch.int32, device=q.device
        )
        if md.cu_seqlens_q is None:
            self.score_op(
                q,
                key_cache,
                score,
                md.block_table,
                md.seq_lens,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
                query_len=md.query_len,
                max_seq_len=md.max_seq_len,
            )
        else:
            self.prefill_score_op(
                q,
                key_cache,
                score,
                md.block_table,
                md.cu_seqlens_q,
                md.seq_lens,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
                max_query_len=md.max_query_len,
                max_seq_len=md.max_seq_len,
            )
        self.topk_op(
            score,
            topk,
            md.block_table,
            md.seq_lens,
            sparse_bt,
            sparse_ctx,
            max_seq_len=md.max_seq_len,
            block_size=self.page_size,
            query_len=md.query_len,
            num_valid_pages=md.num_valid_pages,
            row_req_id=md.row_req_id,
            kv_lens=md.kv_lens,
            num_kv_heads=self.num_kv_heads,
            pages_per_block=8,
        )
        return AiterMiniMaxSelection(topk, sparse_bt, sparse_ctx)
