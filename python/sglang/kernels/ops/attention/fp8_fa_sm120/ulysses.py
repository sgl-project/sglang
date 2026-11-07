# SPDX-License-Identifier: Apache-2.0
"""Ulysses attention that exchanges Q/K/V as E4M3 instead of BF16.

The kernel quantizes Q/K/V per head with amax / 448 over the whole sequence. Under
Ulysses each rank holds a sequence shard of every head, so the per-head amax is a
max over the ranks' shard maxima, which is exact. Each rank therefore reduces its
shard, takes the max across ranks, quantizes its shard with the resulting scales
and sends the E4M3 bytes. The receiver only reorders those bytes into the kernel
buffers. Every byte the kernel reads, and every scale, is what the sequential
path (BF16 exchange, then quantize the gathered rows) produces, so the output is
bit-identical, and the input exchange moves half the bytes.

    shard    q/k/v [s_local, H, 128] BF16, every head
    send     [world, 3, s_local, H / world, 128] E4M3, block p goes to rank p
    receive  [world, 3, s_local, H / world, 128] E4M3, block p came from rank p
    kernel   Q/K [H / world, padded_S, 128], V^T [H / world, 128, padded_S]
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fp8_fa_sm120.fused_prep import (
    BLOCK_TOKENS,
    INVERSE_FP8_MAX,
    _head_amax_qkv,
    _head_scales,
    _load_tile,
    _quantize,
)
from sglang.kernels.ops.attention.fp8_fa_sm120.plan import (
    run_prepared,
    validate_inputs,
)


# One program per (head, token block) of the shard; writes each operand's tile
# into the send block of the rank that owns the head.
@triton.jit
def _quantize_shard(
    q,
    k,
    v,
    scales,
    send,
    rows,
    head_stride: tl.constexpr,
    token_stride: tl.constexpr,
    heads: tl.constexpr,
    local_heads: tl.constexpr,
    block_tokens: tl.constexpr,
):
    head = tl.program_id(0)
    tokens = tl.program_id(1) * block_tokens + tl.arange(0, block_tokens)
    columns = tl.arange(0, 128)
    valid = tokens[:, None] < rows

    part_numel = rows * local_heads * 128
    offsets = (
        (head // local_heads) * 3 * part_numel
        + tokens[:, None] * (local_heads * 128)
        + (head % local_heads) * 128
        + columns[None, :]
    )
    q_tile = _load_tile(q, head, tokens, columns, valid, head_stride, token_stride)
    q_scale = tl.load(scales + 0 * heads + head)
    tl.store(send + offsets, _quantize(q_tile, q_scale), mask=valid)
    k_tile = _load_tile(k, head, tokens, columns, valid, head_stride, token_stride)
    k_scale = tl.load(scales + 1 * heads + head)
    tl.store(send + part_numel + offsets, _quantize(k_tile, k_scale), mask=valid)
    v_tile = _load_tile(v, head, tokens, columns, valid, head_stride, token_stride)
    v_scale = tl.load(scales + 2 * heads + head)
    tl.store(send + 2 * part_numel + offsets, _quantize(v_tile, v_scale), mask=valid)


@triton.jit
def _received_offsets(tokens, head, rows, local_heads: tl.constexpr):
    # gathered token t came from rank t // rows, where it was local token t % rows
    return (
        (tokens // rows) * (3 * rows * local_heads * 128)
        + (tokens % rows) * (local_heads * 128)
        + head * 128
    )


# Same layout and padding as fused_prep._pack_qkv, from received E4M3 bytes
# (as uint8) instead of quantized BF16: the padding gets the byte of +0.0.
@triton.jit
def _pack_received(
    received,
    q_fp8,
    k_fp8,
    v_fp8,
    rows,
    sequence,
    local_heads: tl.constexpr,
    padded_queries: tl.constexpr,
    padded_keys: tl.constexpr,
    block_tokens: tl.constexpr,
):
    head = tl.program_id(0)
    tokens = tl.program_id(1) * block_tokens + tl.arange(0, block_tokens)
    columns = tl.arange(0, 128)
    valid = tokens[:, None] < sequence
    part_numel = rows * local_heads * 128

    offsets = (
        _received_offsets(tokens, head, rows, local_heads)[:, None] + columns[None, :]
    )
    hsd_offsets = head * padded_queries * 128 + tokens[:, None] * 128 + columns[None, :]
    k_offsets = head * padded_keys * 128 + tokens[:, None] * 128 + columns[None, :]

    q_tile = tl.load(received + offsets, mask=valid, other=0)
    tl.store(q_fp8 + hsd_offsets, q_tile, mask=tokens[:, None] < padded_queries)
    k_tile = tl.load(received + part_numel + offsets, mask=valid, other=0)
    tl.store(k_fp8 + k_offsets, k_tile, mask=tokens[:, None] < padded_keys)

    # V^T key order, as in fused_prep._pack_qkv
    source_tokens = (
        tokens // 16 * 16
        + 8 * ((tokens % 4) // 2)
        + 2 * ((tokens % 16) // 4)
        + tokens % 2
    )
    source_offsets = (
        _received_offsets(source_tokens, head, rows, local_heads)[:, None]
        + columns[None, :]
    )
    v_tile = tl.load(
        received + 2 * part_numel + source_offsets,
        mask=source_tokens[:, None] < sequence,
        other=0,
    )
    hds_offsets = (
        head * 128 * padded_keys + columns[:, None] * padded_keys + tokens[None, :]
    )
    tl.store(v_fp8 + hds_offsets, tl.trans(v_tile), mask=tokens[None, :] < padded_keys)


def quantize_shard(q, k, v, world, all_reduce_max):
    """Per-head scales over every rank's shard and this shard's E4M3 send blocks.

    q/k/v are [s_local, H, 128] BF16 shards sharing strides; ``all_reduce_max``
    reduces a [3, H] FP32 tensor in place across the ranks. Returns (scales
    [3, H] FP32, send [world, 3, s_local, H / world, 128] E4M3).
    """
    rows, heads, _ = q.shape
    local_heads = heads // world
    device = q.device
    with torch.cuda.device(device):
        maximums = torch.zeros((3, heads), device=device, dtype=torch.float32)
        _head_amax_qkv[(heads, triton.cdiv(rows, BLOCK_TOKENS))](
            q,
            k,
            v,
            maximums,
            rows,
            head_stride=q.stride(1),
            token_stride=q.stride(0),
            heads=heads,
            block_tokens=BLOCK_TOKENS,
            num_warps=8,
        )
        all_reduce_max(maximums)
        scales = torch.empty_like(maximums)
        _head_scales[(1,)](
            maximums,
            scales,
            INVERSE_FP8_MAX,
            count=3 * heads,
            block=triton.next_power_of_2(3 * heads),
            num_warps=1,
        )
        send = torch.empty(
            (world, 3, rows, local_heads, 128),
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        _quantize_shard[(heads, triton.cdiv(rows, BLOCK_TOKENS))](
            q,
            k,
            v,
            scales,
            send,
            rows,
            head_stride=q.stride(1),
            token_stride=q.stride(0),
            heads=heads,
            local_heads=local_heads,
            block_tokens=BLOCK_TOKENS,
            num_warps=8,
        )
    return scales, send


def fp8_ulysses_attention(
    q, k, v, *, world, rank, all_reduce_max, all_to_all, softmax_scale=None
):
    """FP8 attention of this rank's heads over the gathered sequence.

    q/k/v are [s_local, H, 128] BF16 sequence shards with every head.
    ``all_reduce_max(t)`` maxes a [3, H] FP32 tensor in place across the ranks and
    ``all_to_all(send)`` exchanges a contiguous [world, n] uint8 tensor (row p
    to rank p) and returns the [world, n] rows received (row p from rank p).
    Returns [world * s_local, H / world, 128] BF16, the same bytes as the BF16
    exchange followed by ``fp8_attention`` on the gathered rows.
    """
    scale = validate_inputs(q, k, v, softmax_scale)
    rows, heads, _ = q.shape
    if heads % world:
        raise ValueError(f"{heads} heads do not split over {world} ranks")
    local_heads = heads // world
    sequence = world * rows
    scales, send = quantize_shard(q, k, v, world, all_reduce_max)
    received = all_to_all(send.view(torch.uint8).view(world, -1))

    def prepare(workspace):
        workspace.scales.copy_(scales[:, rank * local_heads : (rank + 1) * local_heads])
        padded_queries = workspace.q_fp8.shape[1]
        _pack_received[(local_heads, triton.cdiv(padded_queries, BLOCK_TOKENS))](
            received,
            workspace.q_fp8.view(torch.uint8),
            workspace.k_fp8.view(torch.uint8),
            workspace.v_fp8.view(torch.uint8),
            rows,
            sequence,
            local_heads=local_heads,
            padded_queries=padded_queries,
            padded_keys=workspace.k_fp8.shape[1],
            block_tokens=BLOCK_TOKENS,
            num_warps=8,
        )

    output, _ = run_prepared(sequence, local_heads, q.device, scale, prepare)
    return output
