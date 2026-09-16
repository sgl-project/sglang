"""Exact greedy selection from TP-local base logits and rounded Markov bias."""

import torch
import triton
import triton.language as tl


@triton.jit
def _partial(
    B,
    X,
    P,
    BS: tl.constexpr,
    XS: tl.constexpr,
    WIDTH: tl.constexpr,
    OFFSET: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, part = tl.program_id(0), tl.program_id(1)
    i = part * BLOCK + tl.arange(0, BLOCK)
    valid = i < WIDTH
    bias = tl.load(B + row * BS + i, valid, 0).to(tl.float32)
    base = tl.load(X + row * XS + i, valid, 0).to(tl.float32)
    value = base + bias
    nan = valid & (value != value)
    has_nan = tl.sum(nan.to(tl.int32), 0) > 0
    maximum = tl.max(tl.where(valid & ~nan, value, -float("inf")), 0)
    wins = tl.where(has_nan, nan, valid & (value == maximum))
    idx = tl.min(tl.where(wins, i + OFFSET, 2147483647), 0)
    maximum = tl.where(has_nan, float("nan"), maximum)
    tl.store(P + (row * PARTS + part) * 2, maximum)
    tl.store(P + (row * PARTS + part) * 2 + 1, idx.to(tl.float32, bitcast=True))


@triton.jit
def _finish(
    P,
    OUT,
    BS: tl.constexpr,
    PARTS: tl.constexpr,
    WORLD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    valid = i < PARTS * WORLD
    rank, part = i // PARTS, i % PARTS
    offset = ((rank * BS + row) * PARTS + part) * 2
    value = tl.load(P + offset, valid, -float("inf"))
    idx = tl.load(P + offset + 1, valid, 0).to(tl.int32, bitcast=True)
    # The NVLink push transport changes +0 to -0 as its Lamport sentinel.
    # Indices are nonnegative int32 bits, so strip that sign bit to recover ID 0.
    idx &= 2147483647
    valid &= idx != 2147483647
    nan = valid & (value != value)
    has_nan = tl.sum(nan.to(tl.int32), 0) > 0
    maximum = tl.max(tl.where(valid & ~nan, value, -float("inf")), 0)
    wins = tl.where(has_nan, nan, valid & (value == maximum))
    result = tl.min(tl.where(wins, idx, 2147483647), 0)
    tl.store(OUT + row, result.to(tl.int64))


def sharded_greedy_step(bias, base_local, *, group, vocab_start, gather=None):
    """Fused BuildStepLocal + vocab gather + argmax, without materializing logits.

    Equivalent to argmax of rank-ordered ``build_step_local``/all_gather over the
    sharded vocab, excluding padding, but each rank reduces its own shard first so
    the transport carries a few partial (value, index) pairs per row instead of
    the full local logits.

    ``bias`` is the original GEMM's already-rounded result. Communication carries
    one (value, global-index-bits) pair per 4096-wide block of the shard per row;
    indices are transported as bits and are never converted numerically to float.
    """
    assert bias.ndim == base_local.ndim == 2
    assert bias.shape[0] == base_local.shape[0]
    assert bias.shape[1] <= base_local.shape[1]
    assert bias.stride(1) == base_local.stride(1) == 1
    rows, width = bias.shape
    block = 4096
    parts = triton.cdiv(base_local.shape[1], block)
    assert parts > 0
    partial = torch.empty((rows, parts, 2), device=bias.device, dtype=torch.float32)
    _partial[(rows, parts)](
        bias,
        base_local,
        partial,
        bias.stride(0),
        base_local.stride(0),
        width,
        vocab_start,
        parts,
        block,
        num_warps=4,
    )
    # The padded partition width fixes the transport shape on all ranks;
    # WIDTH masks real entries, including a completely empty final shard.
    if gather is not None:
        gathered = gather(partial.view(rows, parts * 2))
    else:
        gathered = group.all_gather(partial, dim=0) if group.world_size > 1 else partial
    result = torch.empty(rows, device=bias.device, dtype=torch.int64)
    _finish[(rows,)](
        gathered,
        result,
        rows,
        parts,
        group.world_size,
        triton.next_power_of_2(parts * group.world_size),
        num_warps=4,
    )
    return result
