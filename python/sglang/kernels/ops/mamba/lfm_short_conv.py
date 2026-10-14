"""Gate-fused BF16 short convolution used by LFM2 on CUDA.

LFM2 materializes ``B * x``, transposes it for the generic causal-conv kernel,
then materializes ``C * conv(B * x)``. These kernels preserve both BF16
materialization points while operating directly on the three projection views.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

_DISABLE_LFM_FUSED_CONV = os.getenv("SGLANG_DISABLE_LFM_FUSED_CONV", "0") == "1"


@triton.jit
def _lfm_short_conv_prefill_kernel(
    b_ptr,
    c_ptr,
    x_ptr,
    weight_ptr,
    state_ptr,
    query_start_loc_ptr,
    cache_indices_ptr,
    has_initial_state_ptr,
    out_ptr,
    b_stride_t: tl.constexpr,
    c_stride_t: tl.constexpr,
    x_stride_t: tl.constexpr,
    weight_stride_d: tl.constexpr,
    state_stride_slot: tl.constexpr,
    state_stride_d: tl.constexpr,
    state_stride_w: tl.constexpr,
    out_stride_t: tl.constexpr,
    blocks_per_seq,
    dim: tl.constexpr,
    PAD_SLOT_ID: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tile = tl.program_id(0)
    seq = tile // blocks_per_seq
    token_block = tile - seq * blocks_per_seq
    start = tl.load(query_start_loc_ptr + seq)
    end = tl.load(query_start_loc_ptr + seq + 1)
    t = start + token_block * BLOCK_T + tl.arange(0, BLOCK_T)
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    tt = t[:, None]
    dd = d[None, :]
    valid_d = dd < dim
    valid_t = tt < end

    slot = tl.load(cache_indices_ptr + seq)
    active = slot != PAD_SLOT_ID
    has_initial = tl.load(has_initial_state_ptr + seq).to(tl.int1)
    state_base = slot * state_stride_slot + dd * state_stride_d

    token0 = tt - 2
    in_sequence0 = (token0 >= start) & (token0 < end)
    in_mask0 = valid_d & in_sequence0 & active
    b0 = tl.load(b_ptr + token0 * b_stride_t + dd, mask=in_mask0, other=0.0).to(
        tl.float32
    )
    x0 = tl.load(x_ptr + token0 * x_stride_t + dd, mask=in_mask0, other=0.0).to(
        tl.float32
    )
    new_bx0 = (b0 * x0).to(tl.bfloat16)
    state_pos0 = token0 - start + 2
    old0 = tl.load(
        state_ptr + state_base + state_pos0 * state_stride_w,
        mask=(
            valid_d
            & active
            & has_initial
            & (state_pos0 >= 0)
            & (state_pos0 < 2)
            & ~in_sequence0
        ),
        other=0.0,
    )
    bx0 = tl.where(in_sequence0, new_bx0, old0).to(tl.bfloat16)

    token1 = tt - 1
    in_sequence1 = (token1 >= start) & (token1 < end)
    in_mask1 = valid_d & in_sequence1 & active
    b1 = tl.load(b_ptr + token1 * b_stride_t + dd, mask=in_mask1, other=0.0).to(
        tl.float32
    )
    x1 = tl.load(x_ptr + token1 * x_stride_t + dd, mask=in_mask1, other=0.0).to(
        tl.float32
    )
    new_bx1 = (b1 * x1).to(tl.bfloat16)
    state_pos1 = token1 - start + 2
    old1 = tl.load(
        state_ptr + state_base + state_pos1 * state_stride_w,
        mask=(
            valid_d
            & active
            & has_initial
            & (state_pos1 >= 0)
            & (state_pos1 < 2)
            & ~in_sequence1
        ),
        other=0.0,
    )
    bx1 = tl.where(in_sequence1, new_bx1, old1).to(tl.bfloat16)

    in_mask2 = valid_d & valid_t
    b2 = tl.load(b_ptr + tt * b_stride_t + dd, mask=in_mask2, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + tt * x_stride_t + dd, mask=in_mask2, other=0.0).to(tl.float32)
    bx2 = (b2 * x2).to(tl.bfloat16)

    w0 = tl.load(weight_ptr + dd * weight_stride_d, mask=valid_d, other=0.0).to(
        tl.float32
    )
    w1 = tl.load(weight_ptr + dd * weight_stride_d + 1, mask=valid_d, other=0.0).to(
        tl.float32
    )
    w2 = tl.load(weight_ptr + dd * weight_stride_d + 2, mask=valid_d, other=0.0).to(
        tl.float32
    )
    conv = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
    conv += bx0.to(tl.float32) * w0
    conv += bx1.to(tl.float32) * w1
    conv += bx2.to(tl.float32) * w2
    conv = conv.to(tl.bfloat16)
    c = tl.load(
        c_ptr + tt * c_stride_t + dd,
        mask=valid_d & valid_t,
        other=0.0,
    ).to(tl.float32)
    active_y = (c * conv.to(tl.float32)).to(tl.bfloat16)
    pad_y = (c * bx2.to(tl.float32)).to(tl.bfloat16)
    y = tl.where(active, active_y, pad_y)
    tl.store(
        out_ptr + tt * out_stride_t + dd,
        y,
        mask=valid_d & valid_t,
    )

    # Exactly one token tile per sequence commits the final two gated inputs.
    final_prev = tl.sum(tl.where(tt == end - 1, bx1.to(tl.float32), 0.0), axis=0).to(
        tl.bfloat16
    )
    final_cur = tl.sum(tl.where(tt == end - 1, bx2.to(tl.float32), 0.0), axis=0).to(
        tl.bfloat16
    )
    final_block = (token_block * BLOCK_T <= end - start - 1) & (
        end - start - 1 < (token_block + 1) * BLOCK_T
    )
    final_mask = valid_d & active & final_block
    tl.store(
        state_ptr + state_base,
        final_prev[None, :],
        mask=final_mask,
    )
    tl.store(
        state_ptr + state_base + state_stride_w,
        final_cur[None, :],
        mask=final_mask,
    )


@triton.jit
def _lfm_short_conv_decode_kernel(
    b_ptr,
    c_ptr,
    x_ptr,
    weight_ptr,
    state_ptr,
    cache_indices_ptr,
    out_ptr,
    b_stride_t: tl.constexpr,
    c_stride_t: tl.constexpr,
    x_stride_t: tl.constexpr,
    weight_stride_d: tl.constexpr,
    state_stride_slot: tl.constexpr,
    state_stride_d: tl.constexpr,
    state_stride_w: tl.constexpr,
    out_stride_t: tl.constexpr,
    dim: tl.constexpr,
    PAD_SLOT_ID: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    valid_d = d < dim
    slot = tl.load(cache_indices_ptr + token)
    active = slot != PAD_SLOT_ID
    state_base = slot * state_stride_slot + d * state_stride_d
    state_mask = valid_d & active

    prev0 = tl.load(state_ptr + state_base, mask=state_mask, other=0.0).to(tl.float32)
    prev1 = tl.load(
        state_ptr + state_base + state_stride_w,
        mask=state_mask,
        other=0.0,
    ).to(tl.float32)
    w0 = tl.load(weight_ptr + d * weight_stride_d, mask=valid_d, other=0.0).to(
        tl.float32
    )
    w1 = tl.load(weight_ptr + d * weight_stride_d + 1, mask=valid_d, other=0.0).to(
        tl.float32
    )
    w2 = tl.load(weight_ptr + d * weight_stride_d + 2, mask=valid_d, other=0.0).to(
        tl.float32
    )
    b = tl.load(b_ptr + token * b_stride_t + d, mask=valid_d, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + token * x_stride_t + d, mask=valid_d, other=0.0).to(tl.float32)
    bx_bf16 = (b * x).to(tl.bfloat16)
    bx = bx_bf16.to(tl.float32)
    conv = tl.zeros((BLOCK_D,), dtype=tl.float32)
    conv += prev0 * w0
    conv += prev1 * w1
    conv += bx * w2
    conv_bf16 = conv.to(tl.bfloat16)
    c = tl.load(c_ptr + token * c_stride_t + d, mask=valid_d, other=0.0).to(tl.float32)
    active_y = (c * conv_bf16.to(tl.float32)).to(tl.bfloat16)
    pad_y = (c * bx_bf16.to(tl.float32)).to(tl.bfloat16)
    y = tl.where(active, active_y, pad_y)

    tl.store(out_ptr + token * out_stride_t + d, y, mask=valid_d)
    tl.store(state_ptr + state_base, prev1.to(tl.bfloat16), mask=state_mask)
    tl.store(state_ptr + state_base + state_stride_w, bx_bf16, mask=state_mask)


def can_use_fused_lfm_short_conv(
    b: torch.Tensor,
    c: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    state: torch.Tensor,
) -> bool:
    """Return whether the exact LFM2 BF16 width-3 contract is satisfied."""
    return (
        not _DISABLE_LFM_FUSED_CONV
        and b.is_cuda
        and torch.version.hip is None
        and torch.cuda.get_device_capability(b.device) == (9, 0)
        and c.is_cuda
        and x.is_cuda
        and weight.is_cuda
        and state.is_cuda
        and b.device == c.device == x.device == weight.device == state.device
        and b.dtype == c.dtype == x.dtype == torch.bfloat16
        and b.shape == c.shape == x.shape
        and b.ndim == 2
        and b.stride(1) == c.stride(1) == x.stride(1) == 1
        and weight.dtype == b.dtype
        and weight.ndim == 2
        and weight.shape == (b.shape[1], 3)
        and weight.stride(1) == 1
        and bias is None
        and state.dtype == b.dtype
        and state.ndim == 3
        and state.shape[1:] == (b.shape[1], 2)
    )


def can_dispatch_fused_lfm_short_conv(
    b: torch.Tensor,
    c: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    state: torch.Tensor,
) -> bool:
    """Return whether this call is in the end-to-end-qualified serving domain."""
    return (
        b.ndim == 2
        and b.shape[1] == 2048
        and can_use_fused_lfm_short_conv(b, c, x, weight, bias, state)
    )


def fused_lfm_short_conv_prefill(
    b: torch.Tensor,
    c: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    """Compute and cache ``C * conv(B * x)`` for packed prefill tokens."""
    tokens, dim = b.shape
    out = torch.empty((tokens, dim), dtype=b.dtype, device=b.device)
    block_t = 32
    block_d = 64
    blocks_per_seq = triton.cdiv(max_seq_len, block_t)
    num_sequences = query_start_loc.numel() - 1
    _lfm_short_conv_prefill_kernel[
        (num_sequences * blocks_per_seq, triton.cdiv(dim, block_d))
    ](
        b,
        c,
        x,
        weight,
        state,
        query_start_loc,
        cache_indices,
        has_initial_state,
        out,
        b.stride(0),
        c.stride(0),
        x.stride(0),
        weight.stride(0),
        state.stride(0),
        state.stride(1),
        state.stride(2),
        out.stride(0),
        blocks_per_seq,
        dim,
        -1,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out


def fused_lfm_short_conv_decode(
    b: torch.Tensor,
    c: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    cache_indices: torch.Tensor,
) -> torch.Tensor:
    """Compute, cache, and gate one decode token per active request."""
    tokens, dim = b.shape
    out = torch.empty((tokens, dim), dtype=b.dtype, device=b.device)
    block_d = 256
    _lfm_short_conv_decode_kernel[(tokens, triton.cdiv(dim, block_d))](
        b,
        c,
        x,
        weight,
        state,
        cache_indices,
        out,
        b.stride(0),
        c.stride(0),
        x.stride(0),
        weight.stride(0),
        state.stride(0),
        state.stride(1),
        state.stride(2),
        out.stride(0),
        dim,
        -1,
        BLOCK_D=block_d,
        num_warps=8,
    )
    return out


__all__ = [
    "can_dispatch_fused_lfm_short_conv",
    "can_use_fused_lfm_short_conv",
    "fused_lfm_short_conv_decode",
    "fused_lfm_short_conv_prefill",
]
