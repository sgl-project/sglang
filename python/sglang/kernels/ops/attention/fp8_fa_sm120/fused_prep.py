# SPDX-License-Identifier: Apache-2.0
"""Fused FP8 preparation for the SM120 kernel: two Triton passes into the workspace.

Replaces a Torch chain (FP32 copy, amax, divide, cast,
padded copy, V transpose copy) with one per-head amax pass and one pack pass that
writes the kernel's own buffers directly:

    Q/K  [H, padded_S, 128]   same token/column order as the input
    V    [H, 128, padded_S]   transposed here, so the kernel stays unchanged

Scales stay per head and the quantization recipe is the same (amax/448, div.rn,
clamp, E4M3), so the kernel's numerics do not move. The pack pass writes every
position of the padded buffers, padding included, because the workspace comes
from torch.empty.
"""

import torch
import triton
import triton.language as tl

# Power of two >= 16: the V^T key permutation works in groups of 16 tokens.
BLOCK_TOKENS = 64


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


# Pass 2: one program per (head, token block) up to padded_queries. Q/K tiles go
# out in [token, column] order; the V tile is transposed in registers/shared and
# goes out in [column, token]. Loads are masked on the input token, stores on the
# buffer position, so padding positions receive quantized zeros.
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
    in_queries = tokens[:, None] < padded_queries
    in_keys = tokens[:, None] < padded_keys

    q_tile = _load_tile(q, head, tokens, columns, valid, head_stride, token_stride)
    tl.store(q_fp8 + hsd_offsets, _quantize(q_tile, q_scale), mask=in_queries)

    k_tile = _load_tile(k, head, tokens, columns, valid, head_stride, token_stride)
    tl.store(k_fp8 + k_offsets, _quantize(k_tile, k_scale), mask=in_keys)

    # The kernel packs P straight from the score fragment; V^T's stored key
    # order must follow it: stored position p holds key
    # p//16*16 + 8*((p%4)//2) + 2*((p%16)//4) + p%2 (see key_order_for_positions).
    # The permutation is applied on the LOAD side as a row gather: every gathered
    # row is still 256 contiguous bytes, and the transposed store stays contiguous.
    # Permuting the store instead scatters bytes inside 16 B windows and doubled
    # the prepare time.
    source_tokens = (
        tokens // 16 * 16
        + 8 * ((tokens % 4) // 2)
        + 2 * ((tokens % 16) // 4)
        + tokens % 2
    )
    valid_source = source_tokens[:, None] < sequence

    hds_offsets = (
        head * 128 * padded_keys + columns[:, None] * padded_keys + tokens[None, :]
    )
    in_keys_transposed = tokens[None, :] < padded_keys

    v_tile = _load_tile(
        v, head, source_tokens, columns, valid_source, head_stride, token_stride
    )
    tl.store(
        v_fp8 + hds_offsets,
        tl.trans(_quantize(v_tile, v_scale)),
        mask=in_keys_transposed,
    )


def fused_prepare(q, k, v, workspace):
    """Three launches: per-head amax, scale finalize, direct pack into the workspace.

    q, k, v are [H, S, 128] views with one shared stride set; the workspace buffers are
    sized for this S and H.
    """
    heads = q.shape[0]
    sequence = q.shape[1]
    padded_queries = workspace.q_fp8.shape[1]
    padded_keys = workspace.k_fp8.shape[1]
    head_stride = q.stride(0)
    token_stride = q.stride(1)

    with torch.cuda.device(q.device):
        _head_amax_qkv[(heads, triton.cdiv(sequence, BLOCK_TOKENS))](
            q,
            k,
            v,
            workspace.maximums,
            sequence,
            head_stride=head_stride,
            token_stride=token_stride,
            heads=heads,
            block_tokens=BLOCK_TOKENS,
            num_warps=8,
        )
        _head_scales[(1,)](
            workspace.maximums,
            workspace.scales,
            INVERSE_FP8_MAX,
            count=3 * heads,
            block=triton.next_power_of_2(3 * heads),
            num_warps=1,
        )
        _pack_qkv[(heads, triton.cdiv(padded_queries, BLOCK_TOKENS))](
            q,
            k,
            v,
            workspace.q_fp8,
            workspace.k_fp8,
            workspace.v_fp8,
            workspace.scales,
            sequence,
            head_stride=head_stride,
            token_stride=token_stride,
            padded_queries=padded_queries,
            padded_keys=padded_keys,
            heads=heads,
            block_tokens=BLOCK_TOKENS,
            num_warps=8,
        )
