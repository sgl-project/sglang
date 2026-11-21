# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PLE gate and three grouped RMSNorms from vLLM 5df63a7203, for NPU execution."""
import torch
import triton
import triton.language as tl


@triton.jit
def _round_bf16_scalar(x):
    # arch22 can fold an in-register BF16-to-FP32 round trip for a scalar.
    # Retain the same nearest-even BF16 value as SG's tensor intermediates.
    bits = x.to(tl.uint32, bitcast=True)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    is_nan = (bits & 0x7FFFFFFF) > 0x7F800000
    rounded = tl.where(is_nan, (bits & 0xFFFF0000) | 0x00400000, rounded)
    return rounded.to(tl.float32, bitcast=True)


@triton.jit
def _ple_gate_kernel(
    key_ptr,  # [T, HC*H] with row stride key_rs
    value_ptr,  # [T, H] with row stride value_rs
    hidden_ptr,  # [T, HC*H]
    nk_ptr,  # [HC*H] norm_key weight
    nq_ptr,  # [HC*H] norm_query weight
    ncw_ptr,  # [HC*H] norm_conv weight
    gated_ptr,  # [T, HC*H] out
    normed_ptr,  # [T, HC*H] out
    key_rs,
    value_rs,
    eps,
    H: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    t = tl.program_id(0)
    s = tl.program_id(1)
    lanes = tl.arange(0, BLOCK_H)
    mask = lanes < H
    offs = s * H + lanes
    dtype: tl.constexpr = key_ptr.dtype.element_ty

    k = tl.load(key_ptr + t * key_rs + offs, mask=mask, other=0.0).to(tl.float32)
    q = tl.load(hidden_ptr + t * HC * H + offs, mask=mask, other=0.0).to(tl.float32)
    nk = tl.load(nk_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    nq = tl.load(nq_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Match eager materialization at each intermediate tensor boundary.
    k_n = (k * tl.rsqrt(tl.sum(k * k) / H + eps) * (1.0 + nk)).to(dtype)
    q_n = (q * tl.rsqrt(tl.sum(q * q) / H + eps) * (1.0 + nq)).to(dtype)
    products = (k_n.to(tl.float32) * q_n.to(tl.float32)).to(dtype)
    dot = _round_bf16_scalar(tl.sum(products.to(tl.float32)))
    d = dot / tl.sqrt(float(H))
    d = _round_bf16_scalar(d)
    sign = tl.where(d < 0, -1.0, 0.0)
    sign = tl.where(d > 0, 1.0, sign)
    magnitude = _round_bf16_scalar(tl.sqrt(tl.maximum(tl.abs(d), 1e-6)))
    g = tl.sigmoid(sign * magnitude)
    g = _round_bf16_scalar(g)

    v = tl.load(value_ptr + t * value_rs + lanes, mask=mask, other=0.0).to(tl.float32)
    gated = (g * v).to(dtype)
    gf = gated.to(tl.float32)
    ncw = tl.load(ncw_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    normed = gf * tl.rsqrt(tl.sum(gf * gf) / H + eps) * (1.0 + ncw)

    tl.store(gated_ptr + t * HC * H + offs, gated, mask=mask)
    tl.store(normed_ptr + t * HC * H + offs, normed, mask=mask)


def ple_gate(
    key: torch.Tensor,
    value: torch.Tensor,
    hidden: torch.Tensor,
    norm_key_w: torch.Tensor,
    norm_query_w: torch.Tensor,
    norm_conv_w: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = hidden.shape[0]
    h = value.shape[-1]
    hc = hidden.shape[-1] // h
    assert key.stride(1) == 1 and value.stride(1) == 1
    assert hidden.is_contiguous()
    if key.dtype != value.dtype or key.dtype != hidden.dtype:
        raise ValueError("key, value, and hidden must have the same dtype")
    if key.dtype != torch.bfloat16:
        raise ValueError("PLE gate expects BF16 inputs")
    gated = torch.empty_like(hidden)
    normed = torch.empty_like(hidden)
    if num_tokens == 0:
        return gated, normed
    _ple_gate_kernel[(num_tokens, hc)](
        key,
        value,
        hidden,
        norm_key_w,
        norm_query_w,
        norm_conv_w,
        gated,
        normed,
        key.stride(0),
        value.stride(0),
        eps,
        H=h,
        HC=hc,
        BLOCK_H=triton.next_power_of_2(h),
        num_warps=4,
    )
    return gated, normed
