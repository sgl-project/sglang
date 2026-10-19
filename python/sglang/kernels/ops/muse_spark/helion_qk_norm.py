"""Helion Q/K normalization and RoPE fusion for Muse Spark."""

from __future__ import annotations

import helion
import helion.language as hl
import torch

from sglang.kernels.ops.muse_spark.helion_runtime import (
    token_count_specialization_key,
)

QK_NORM_CONFIGS: dict[str, helion.Config] = {
    "short": helion.Config(
        atomic_indexing=[],
        block_sizes=[8],
        indexing=[
            "pointer",
            "pointer",
            "pointer",
            "pointer",
            "pointer",
            "pointer",
            "tensor_descriptor",
            "tensor_descriptor",
            "tensor_descriptor",
            "pointer",
        ],
        load_eviction_policies=["", "", "last", "first", "last", "last"],
        num_stages=2,
        num_warps=4,
        pid_type="flat",
        range_flattens=[None],
        range_multi_buffers=[None],
        range_num_stages=[0],
        range_unroll_factors=[0],
        reduction_loops=[None],
    ),
    "long": helion.Config(
        block_sizes=[32],
        num_stages=1,
        num_warps=8,
        pid_type="flat",
        reduction_loops=[None],
    ),
}


@helion.kernel(
    key=token_count_specialization_key,
    static_shapes=False,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def _qk_norm_helion_kernel(
    qkv: torch.Tensor,
    positions: torch.Tensor | None,
    cos_sin_cache: torch.Tensor | None,
    k_cache: torch.Tensor | None,
    v_cache: torch.Tensor | None,
    cache_loc: torch.Tensor | None,
    eps: hl.constexpr,
    num_q_heads: hl.constexpr,
    num_kv_heads: hl.constexpr,
    head_dim: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match ``FusedQKNorm`` including its optional side-effecting KV store."""
    num_tokens, _ = qkv.size()
    hl.specialize(qkv.stride())

    q_width = num_q_heads * head_dim
    kv_width = num_kv_heads * head_dim
    q_out = torch.empty((num_tokens, q_width), dtype=qkv.dtype, device=qkv.device)
    k_out = torch.empty((num_tokens, kv_width), dtype=qkv.dtype, device=qkv.device)

    norm_heads = num_q_heads + num_kv_heads
    num_rows = num_tokens * norm_heads
    qkv_flat = qkv.view(-1)
    q_out_flat = q_out.view(-1)
    k_out_flat = k_out.view(-1)
    if k_cache is not None:
        k_cache_flat = k_cache.view(-1)
        v_cache_flat = v_cache.view(-1)

    for tile_rows in hl.tile(num_rows):
        hl.inline_triton("tl.extra.cuda.gdc_wait()", args=(), output_like=None)
        row = tile_rows.index
        valid = row < num_rows
        token = row // norm_heads
        head = row % norm_heads
        is_q = head < num_q_heads
        is_k = head >= num_q_heads
        cols = hl.arange(head_dim)
        source_idx = (
            token[:, None] * qkv.stride(0) + head[:, None] * head_dim + cols[None, :]
        )
        raw = hl.load(
            qkv_flat,
            [source_idx],
            extra_mask=valid[:, None],
        ).to(torch.float32)

        variance = torch.sum(raw * raw, dim=-1) / head_dim
        rstd = torch.rsqrt(variance + eps)
        values = raw * rstd[:, None]

        if positions is not None:
            rounded = values.to(qkv.dtype).to(torch.float32)
            rounded_pairs = rounded.reshape(tile_rows, head_dim // 2, 2)
            even, odd = hl.split(rounded_pairs)
            selected_positions = hl.load(
                positions,
                [token],
                extra_mask=valid,
            )
            freq = hl.arange(head_dim // 2)
            cos = hl.load(
                cos_sin_cache,
                [selected_positions[:, None], freq[None, :]],
                extra_mask=valid[:, None],
            ).to(torch.float32)
            sin = hl.load(
                cos_sin_cache,
                [selected_positions[:, None], head_dim // 2 + freq[None, :]],
                extra_mask=valid[:, None],
            ).to(torch.float32)
            values = hl.join(even * cos - odd * sin, even * sin + odd * cos).reshape(
                tile_rows, head_dim
            )

        q_head = torch.where(is_q, head, 0)
        k_head = torch.where(is_k, head - num_q_heads, 0)
        q_idx = (token[:, None] * num_q_heads + q_head[:, None]) * head_dim + cols[
            None, :
        ]
        k_idx = (token[:, None] * num_kv_heads + k_head[:, None]) * head_dim + cols[
            None, :
        ]
        hl.store(
            q_out_flat,
            [q_idx],
            values.to(qkv.dtype),
            extra_mask=valid[:, None] & is_q[:, None],
        )
        hl.store(
            k_out_flat,
            [k_idx],
            values.to(qkv.dtype),
            extra_mask=valid[:, None] & is_k[:, None],
        )

        if k_cache is not None:
            cache_row = hl.load(cache_loc, [token], extra_mask=valid)
            cache_idx = (
                cache_row[:, None] * kv_width
                + k_head[:, None] * head_dim
                + cols[None, :]
            )
            hl.store(
                k_cache_flat,
                [cache_idx],
                values.to(qkv.dtype),
                extra_mask=valid[:, None] & is_k[:, None],
            )
            v_source_idx = (
                token[:, None] * qkv.stride(0)
                + (norm_heads + k_head)[:, None] * head_dim
                + cols[None, :]
            )
            v_values = hl.load(
                qkv_flat,
                [v_source_idx],
                extra_mask=valid[:, None] & is_k[:, None],
            )
            hl.store(
                v_cache_flat,
                [cache_idx],
                v_values,
                extra_mask=valid[:, None] & is_k[:, None],
            )
        hl.inline_triton(
            "tl.extra.cuda.gdc_launch_dependents()", args=(), output_like=None
        )

    return q_out, k_out


__all__ = ["QK_NORM_CONFIGS", "_qk_norm_helion_kernel"]
