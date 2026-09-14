from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.utils.custom_op import register_custom_op


@triton.jit
def _rope_and_quantize_query_kernel(
    q_ptr,
    k_ptr,
    q_out_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    q_stride_token: tl.constexpr,
    q_stride_head: tl.constexpr,
    k_stride_token: tl.constexpr,
    k_stride_head: tl.constexpr,
    q_out_stride_token: tl.constexpr,
    q_out_stride_head: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_k_heads: tl.constexpr,
    head_dim: tl.constexpr,
    padded_num_q_heads: tl.constexpr,
    padded_num_k_heads: tl.constexpr,
    padded_half_dim: tl.constexpr,
    is_neox: tl.constexpr,
):
    token_idx = tl.program_id(0)
    position = tl.load(positions_ptr + token_idx).to(tl.int64)
    half_dim = head_dim // 2
    pair_idx = tl.arange(0, padded_half_dim)[None, :]
    q_head_idx = tl.arange(0, padded_num_q_heads)[:, None]
    k_head_idx = tl.arange(0, padded_num_k_heads)[:, None]
    pair_mask = pair_idx < half_dim

    cos = tl.load(
        cos_sin_cache_ptr + position * head_dim + pair_idx,
        mask=pair_mask,
        other=0.0,
    )
    sin = tl.load(
        cos_sin_cache_ptr + position * head_dim + half_dim + pair_idx,
        mask=pair_mask,
        other=0.0,
    )

    q_mask = (q_head_idx < num_q_heads) & pair_mask
    q_row = token_idx * q_stride_token + q_head_idx * q_stride_head
    q_out_row = token_idx * q_out_stride_token + q_head_idx * q_out_stride_head
    if is_neox:
        q_first_offset = q_row + pair_idx
        q_second_offset = q_first_offset + half_dim
        q_out_first_offset = q_out_row + pair_idx
        q_out_second_offset = q_out_first_offset + half_dim
    else:
        q_first_offset = q_row + 2 * pair_idx
        q_second_offset = q_first_offset + 1
        q_out_first_offset = q_out_row + 2 * pair_idx
        q_out_second_offset = q_out_first_offset + 1
    q_first = tl.load(q_ptr + q_first_offset, mask=q_mask, other=0.0).to(tl.float32)
    q_second = tl.load(q_ptr + q_second_offset, mask=q_mask, other=0.0).to(tl.float32)
    tl.store(
        q_out_ptr + q_out_first_offset,
        q_first * cos - q_second * sin,
        mask=q_mask,
    )
    tl.store(
        q_out_ptr + q_out_second_offset,
        q_second * cos + q_first * sin,
        mask=q_mask,
    )

    k_mask = (k_head_idx < num_k_heads) & pair_mask
    k_row = token_idx * k_stride_token + k_head_idx * k_stride_head
    if is_neox:
        k_first_offset = k_row + pair_idx
        k_second_offset = k_first_offset + half_dim
    else:
        k_first_offset = k_row + 2 * pair_idx
        k_second_offset = k_first_offset + 1
    k_first = tl.load(k_ptr + k_first_offset, mask=k_mask, other=0.0).to(tl.float32)
    k_second = tl.load(k_ptr + k_second_offset, mask=k_mask, other=0.0).to(tl.float32)
    tl.store(
        k_ptr + k_first_offset,
        k_first * cos - k_second * sin,
        mask=k_mask,
    )
    tl.store(
        k_ptr + k_second_offset,
        k_second * cos + k_first * sin,
        mask=k_mask,
    )


@register_custom_op(mutates_args=["q_out", "k"])
def apply_rope_and_quantize_query(
    q: torch.Tensor,
    k: torch.Tensor,
    q_out: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    is_neox: bool,
) -> None:
    """Apply full-dimension RoPE and write Q directly to an FP8 output."""
    assert q.ndim == k.ndim == q_out.ndim == 3
    assert q.shape == q_out.shape
    assert q.shape[0] == k.shape[0]
    assert q.shape[2] == k.shape[2] == cos_sin_cache.shape[1]
    assert q.dtype == k.dtype
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert q_out.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    assert cos_sin_cache.dtype == torch.float32
    assert q.stride(2) == k.stride(2) == q_out.stride(2) == 1
    assert positions.ndim == 1 and positions.shape[0] == q.shape[0]
    assert positions.dtype in (torch.int32, torch.int64)
    assert q.device == k.device == q_out.device == cos_sin_cache.device
    assert positions.device == q.device

    num_tokens, num_q_heads, head_dim = q.shape
    num_k_heads = k.shape[1]
    padded_num_q_heads = triton.next_power_of_2(num_q_heads)
    padded_num_k_heads = triton.next_power_of_2(num_k_heads)
    padded_half_dim = triton.next_power_of_2(head_dim // 2)
    _rope_and_quantize_query_kernel[(num_tokens,)](
        q,
        k,
        q_out,
        cos_sin_cache,
        positions,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        q_out.stride(0),
        q_out.stride(1),
        num_q_heads,
        num_k_heads,
        head_dim,
        padded_num_q_heads,
        padded_num_k_heads,
        padded_half_dim,
        is_neox,
        num_warps=4,
    )
