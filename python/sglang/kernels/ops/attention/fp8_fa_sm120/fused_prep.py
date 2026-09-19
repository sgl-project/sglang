# SPDX-License-Identifier: Apache-2.0
"""Fused FP8 preparation for the SM120 kernel: two Triton passes into the plan buffers.

Replaces a Torch chain (FP32 copy, amax, divide, cast,
padded copy, V transpose copy) with one per-head amax pass and one pack pass that
writes the kernel's own buffers directly:

    Q/K  [H, padded_S, 128]   same token/column order as the input
    V    [H, 128, padded_S]   transposed here, so the kernel stays unchanged

Scales stay per head and the quantization recipe is the same (amax/448, div.rn,
clamp, E4M3), so the kernel's numerics do not move. Padding rows are zeroed once
by the plan and never written here.
"""

import torch
import triton
import triton.language as tl


# Triton's `/` lowers to div.full.f32 (approximate); Torch divides with div.rn.
# The FP8 bytes must match the Torch prepare() path, so the exact division is kept.
@triton.jit
def _divide_rn(numerator, denominator):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [numerator, denominator],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _load_tile(
    base,
    head,
    tokens,
    columns,
    valid,
    head_stride: tl.constexpr,
    token_stride: tl.constexpr,
):
    offsets = head * head_stride + tokens[:, None] * token_stride + columns[None, :]
    return tl.load(base + offsets, mask=valid, other=0.0).to(tl.float32)


@triton.jit
def _quantize(values, scale):
    scaled = _divide_rn(values, scale)
    scaled = tl.maximum(tl.minimum(scaled, 448.0), -448.0)
    return scaled.to(tl.float8e4nv)


# Pass 1: one program per (head, token block); each reduces its Q/K/V tile to
# three scalars and folds them into maximums[operand, head].
@triton.jit
def _head_amax_qkv(
    q,
    k,
    v,
    maximums,
    sequence,
    head_stride: tl.constexpr,
    token_stride: tl.constexpr,
    heads: tl.constexpr,
    block_tokens: tl.constexpr,
):
    head = tl.program_id(0)
    tokens = tl.program_id(1) * block_tokens + tl.arange(0, block_tokens)
    columns = tl.arange(0, 128)
    valid = tokens[:, None] < sequence

    q_tile = _load_tile(q, head, tokens, columns, valid, head_stride, token_stride)
    tl.atomic_max(maximums + 0 * heads + head, tl.max(tl.abs(q_tile)))

    k_tile = _load_tile(k, head, tokens, columns, valid, head_stride, token_stride)
    tl.atomic_max(maximums + 1 * heads + head, tl.max(tl.abs(k_tile)))

    v_tile = _load_tile(v, head, tokens, columns, valid, head_stride, token_stride)
    tl.atomic_max(maximums + 2 * heads + head, tl.max(tl.abs(v_tile)))


# Torch evaluates `tensor / 448.0` as `tensor * (1/448)` with the reciprocal in
# FP32 (CPU-scalar divisor fast path). The same product keeps the scales bit-equal.
INVERSE_FP8_MAX = (torch.ones((), dtype=torch.float32) / 448.0).item()


# scales[operand, head] = amax * (1/448), or 1 for an all-zero head.
@triton.jit
def _head_scales(
    maximums,
    scales,
    inverse_fp8_max,
    count: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.arange(0, block)
    valid = offsets < count
    maximum = tl.load(maximums + offsets, mask=valid, other=0.0)
    scale = tl.where(maximum > 0.0, maximum * inverse_fp8_max, 1.0)
    tl.store(scales + offsets, scale, mask=valid)


# Pass 2: same grid as pass 1. Q/K tiles go out in [token, column] order;
# the V tile is transposed in registers/shared and goes out in [column, token].
@triton.jit
def _pack_qkv(
    q,
    k,
    v,
    q_fp8,
    k_fp8,
    v_fp8,
    scales,
    sequence,
    head_stride: tl.constexpr,
    token_stride: tl.constexpr,
    padded_queries: tl.constexpr,
    padded_keys: tl.constexpr,
    heads: tl.constexpr,
    block_tokens: tl.constexpr,
    permute_keys: tl.constexpr,
):
    head = tl.program_id(0)
    tokens = tl.program_id(1) * block_tokens + tl.arange(0, block_tokens)
    columns = tl.arange(0, 128)
    valid = tokens[:, None] < sequence

    q_scale = tl.load(scales + 0 * heads + head)
    k_scale = tl.load(scales + 1 * heads + head)
    v_scale = tl.load(scales + 2 * heads + head)

    # Q is padded to the 128-row query tile, K/V to the 32-key tile.
    hsd_offsets = head * padded_queries * 128 + tokens[:, None] * 128 + columns[None, :]
    k_offsets = head * padded_keys * 128 + tokens[:, None] * 128 + columns[None, :]

    q_tile = _load_tile(q, head, tokens, columns, valid, head_stride, token_stride)
    tl.store(q_fp8 + hsd_offsets, _quantize(q_tile, q_scale), mask=valid)

    k_tile = _load_tile(k, head, tokens, columns, valid, head_stride, token_stride)
    tl.store(k_fp8 + k_offsets, _quantize(k_tile, k_scale), mask=valid)

    # The kernel packs P straight from the score fragment; V^T's stored key
    # order must follow it: stored position p holds key
    # p//16*16 + 8*((p%4)//2) + 2*((p%16)//4) + p%2 (see key_order_for_positions).
    # The permutation is applied on the LOAD side as a row gather: every gathered
    # row is still 256 contiguous bytes, and the transposed store stays contiguous.
    # Permuting the store instead scatters bytes inside 16 B windows and doubled
    # the prepare time. Padding keys are never written.
    if permute_keys:
        source_tokens = (
            tokens // 16 * 16
            + 8 * ((tokens % 4) // 2)
            + 2 * ((tokens % 16) // 4)
            + tokens % 2
        )
    else:
        source_tokens = tokens
    valid_source = source_tokens[:, None] < sequence

    hds_offsets = (
        head * 128 * padded_keys + columns[:, None] * padded_keys + tokens[None, :]
    )
    valid_transposed = source_tokens[None, :] < sequence

    v_tile = _load_tile(
        v, head, source_tokens, columns, valid_source, head_stride, token_stride
    )
    tl.store(
        v_fp8 + hds_offsets, tl.trans(_quantize(v_tile, v_scale)), mask=valid_transposed
    )


def _attach_fused_state(plan, block_tokens):
    q, k, v = plan.inputs
    for tensor in (k, v):
        if tensor.stride() != q.stride():
            raise ValueError("Q/K/V must share strides for fused preparation")
    if q.stride(2) != 1:
        raise ValueError("Head dimension must be contiguous for fused preparation")
    if block_tokens & (block_tokens - 1) or block_tokens < 16:
        raise ValueError(
            "block_tokens must be a power of two >= 16 (key permutation groups are 16 wide)"
        )

    heads = q.shape[0]
    plan.block_tokens = block_tokens
    plan.head_stride = q.stride(0)
    plan.token_stride = q.stride(1)
    plan.maximums = torch.zeros((3, heads), device=q.device, dtype=torch.float32)


def fused_prepare(plan):
    """Three launches: per-head amax, scale finalize, direct pack into the plan buffers."""
    q, k, v = plan.inputs
    heads = q.shape[0]
    padded_queries = plan.q_fp8.shape[1]
    padded_keys = plan.k_fp8.shape[1]
    grid = (heads, triton.cdiv(plan.sequence, plan.block_tokens))

    with torch.cuda.device(q.device):
        plan.maximums.zero_()
        _head_amax_qkv[grid](
            q,
            k,
            v,
            plan.maximums,
            plan.sequence,
            head_stride=plan.head_stride,
            token_stride=plan.token_stride,
            heads=heads,
            block_tokens=plan.block_tokens,
            num_warps=8,
        )
        _head_scales[(1,)](
            plan.maximums,
            plan.scales,
            INVERSE_FP8_MAX,
            count=3 * heads,
            block=triton.next_power_of_2(3 * heads),
            num_warps=1,
        )
        _pack_qkv[grid](
            q,
            k,
            v,
            plan.q_fp8,
            plan.k_fp8,
            plan.v_fp8,
            plan.scales,
            plan.sequence,
            head_stride=plan.head_stride,
            token_stride=plan.token_stride,
            padded_queries=padded_queries,
            padded_keys=padded_keys,
            heads=heads,
            block_tokens=plan.block_tokens,
            permute_keys=plan.key_order is not None,
            num_warps=8,
        )
    plan._prepared = True
