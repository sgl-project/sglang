# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""AITER page16 sparse attention over separate in-place SHUFFLE K/V planes.

The allocator still owns 128-token pages. Only sparse-layer contents use
SHUFFLE storage; dense-layer and index-K buffers retain their NHD layout.
"""

from typing import Optional

import torch

from sglang.srt.layers.attention.minimax_sparse_ops.aiter_indexer import (
    AiterMiniMaxSelection,
)


def sparse_kv_views(k_cache: torch.Tensor, v_cache: torch.Tensor):
    if (
        k_cache.dtype != torch.float8_e4m3fn
        or v_cache.dtype != k_cache.dtype
        or k_cache.shape != v_cache.shape
        or k_cache.ndim != 3
        or k_cache.shape[-1] != 128
        or k_cache.shape[0] % 128
        or not k_cache.is_contiguous()
        or not v_cache.is_contiguous()
    ):
        raise ValueError(
            "MiniMax AITER sparse PA requires contiguous FP8 page128 HD128 buffers"
        )
    heads = k_cache.shape[1]
    return (
        k_cache.view(-1, heads, 8, 16, 16),
        v_cache.view(-1, heads, 1, 128, 16),
    )


def store_sparse_kv(k, v, k_cache, v_cache, loc, k_scale, v_scale):
    from aiter import reshape_and_cache

    kc, vc = sparse_kv_views(k_cache, v_cache)
    reshape_and_cache(
        k.reshape(-1, k_cache.shape[1], 128),
        v.reshape(-1, v_cache.shape[1], 128),
        kc,
        vc,
        loc,
        "fp8",
        k_scale,
        v_scale,
        asm_layout=True,
    )


class AiterMiniMaxSparsePA:
    def __init__(self, device):
        from aiter.ops.triton.gluon.pa_decode_gluon import (
            get_recommended_splits,
            pa_decode_gluon,
        )

        self.get_splits = get_recommended_splits
        self.attend = pa_decode_gluon
        self.workspace = {}
        self.unit_scale = torch.ones(1, dtype=torch.float32, device=device)

    def forward(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        selection: AiterMiniMaxSelection,
        *,
        softmax_scale: float,
        k_scale: Optional[torch.Tensor],
        v_scale: Optional[torch.Tensor],
    ):
        if q.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("MiniMax AITER sparse PA expects BF16/FP16 Q")
        for scale in (k_scale, v_scale):
            if scale is not None and (
                scale.numel() != 1 or scale.dtype != torch.float32
            ):
                raise ValueError(
                    "MiniMax AITER sparse PA requires scalar FP32 KV scales"
                )
        kc, vc = sparse_kv_views(k_cache, v_cache)
        total_q, q_heads, head_dim = q.shape
        kv_heads = k_cache.shape[1]
        if head_dim != 128 or q_heads % kv_heads:
            raise ValueError("MiniMax AITER sparse PA requires HD128 and integral GQA")
        rows = total_q * kv_heads
        group_size = q_heads // kv_heads
        q = q.contiguous().view(rows, group_size, head_dim)
        out = torch.empty_like(q)
        if rows == 0:
            return out.view(total_q, q_heads, head_dim)
        splits = self.get_splits(rows, 1)
        key = (rows, group_size, q.dtype, q.device)
        workspace = self.workspace.get(key)
        if workspace is None:
            shape = (rows, 1, splits, group_size)
            workspace = (
                torch.empty(shape, device=q.device, dtype=torch.float32),
                torch.empty(shape, device=q.device, dtype=torch.float32),
                torch.empty((*shape, head_dim), device=q.device, dtype=q.dtype),
            )
            # Decode/verify graph buckets reuse scratch across layers. Prefill
            # token counts vary, so do not retain an unbounded set of workspaces.
            if torch.cuda.is_current_stream_capturing():
                self.workspace[key] = workspace
        exp_sums, max_logits, temporary_output = workspace
        self.attend(
            output=out,
            query=q,
            key_cache=kc.view(-1, 1, *kc.shape[2:]),
            value_cache=vc.view(-1, 1, *vc.shape[2:]),
            context_lengths=selection.context_lens,
            block_tables=selection.block_table,
            softmax_scale=softmax_scale,
            query_length=1,
            max_context_partition_num=splits,
            context_partition_size=256,
            compute_type=torch.float8_e4m3fn,
            query_scale=None,
            key_scale=self.unit_scale if k_scale is None else k_scale,
            value_scale=self.unit_scale if v_scale is None else v_scale,
            exp_sums=exp_sums,
            max_logits=max_logits,
            temporary_output=temporary_output,
            alibi_slopes=None,
            sinks=None,
            sliding_window=-1,
            ps=True,
        )
        return out.view(total_q, q_heads, head_dim)
