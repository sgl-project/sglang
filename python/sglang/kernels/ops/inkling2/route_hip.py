"""ROCm decode / MTP-verify routing: top-k, expert metadata and activation quantization
fused into one launch."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.kernels.ops.inkling2.experts_sm90 import (
    _fake_quantize_mxfp8_chunk,
    _fake_quantize_row_chunk,
)
from sglang.kernels.ops.inkling2.route import _metadata, _topk_row
from sglang.kernels.ops.moe.gate_topk import fpval_to_key, key_to_indx

# route_topk_metadata fuses the routing (and quantization) of up to this many tokens;
# beyond, one program's top-k of every row costs more than the launches it saves.
ROUTE_FUSED_MAX_TOKENS = 4
# route_metadata_quantize takes up to this many ids; above, route_metadata's 16-warp
# block slows the quantization programs that share its launch.
ROUTE_QUANT_MAX_IDS = 1024


@triton.jit
def _pick_rows(values, slots, i: tl.constexpr):
    return tl.sum(tl.where(slots[None, :] == i, values, 0.0), 1)


@triton.jit
def _logsigmoid_renorm8_rows(logits, slots):
    """_logsigmoid_renorm8 on each row of a (rows, 8) tile, in the same order."""
    log_probs = tl.minimum(logits, 0.0) - libdevice.log1p(
        libdevice.exp(-tl.abs(logits))
    )
    max_log_probs = tl.max(log_probs, 1)
    shifted = libdevice.exp(log_probs - max_log_probs[:, None])
    total = (
        (_pick_rows(shifted, slots, 0) + _pick_rows(shifted, slots, 2))
        + (_pick_rows(shifted, slots, 1) + _pick_rows(shifted, slots, 3))
    ) + (
        (_pick_rows(shifted, slots, 4) + _pick_rows(shifted, slots, 6))
        + (_pick_rows(shifted, slots, 5) + _pick_rows(shifted, slots, 7))
    )
    return libdevice.exp(log_probs - (max_log_probs + libdevice.log(total))[:, None])


@triton.jit
def _topk_rows(
    P,
    Bias,
    Logits,
    Ids,
    Weights,
    GlobalScale,
    route_scale,
    M,
    N: tl.constexpr,
    GROUPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    TOPK: tl.constexpr,
    BM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """_topk_row (top-8, renormalized) for rows 0..M-1 together (M <= BM): the same
    per-row arithmetic on a (BM, BLOCK_N) tile, the top-k along dim 1."""
    rows = tl.arange(0, BM)
    row_mask = rows < M
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]
    logits = tl.load(P + rows[:, None] * N + cols[None, :], mask, other=0)
    for group in tl.static_range(1, GROUPS):
        logits = logits + tl.load(
            P + (group * M + rows[:, None]) * N + cols[None, :], mask, other=0
        )
    tl.store(Logits + rows[:, None] * N + cols[None, :], logits, mask)
    selection = tl.div_rn(1.0, 1.0 + libdevice.exp(-logits))
    if HAS_BIAS:
        selection = selection + tl.load(Bias + cols, col_mask, other=0)[None, :]
    selection = tl.where(col_mask[None, :], selection, float("-inf"))
    keys = fpval_to_key(selection.to(tl.uint32, bitcast=True)).to(tl.uint64) << 16
    keys = keys | (BLOCK_N - cols).to(tl.uint64)[None, :]
    top = tl.sort(tl.topk(keys, 8, dim=1), dim=1, descending=True)
    slots = tl.arange(0, 8)
    ids = key_to_indx((top & 0xFFFF).to(tl.int32), BLOCK_N)
    out_mask = row_mask[:, None] & (slots < TOPK)[None, :]
    tl.store(Ids + rows[:, None] * TOPK + slots[None, :], ids, out_mask)
    tl.debug_barrier()
    weights = _logsigmoid_renorm8_rows(
        tl.load(Logits + rows[:, None] * N + ids, row_mask[:, None], other=0), slots
    )
    weights *= route_scale
    weights *= tl.load(GlobalScale)
    tl.store(Weights + rows[:, None] * TOPK + slots[None, :], weights, out_mask)


@triton.jit
def _route_topk_metadata(
    P,
    Bias,
    Logits,
    Ids,
    Weights,
    GlobalScale,
    route_scale,
    SortedIds,
    Src2Dst,
    Dst2Src,
    Gammas,
    Counts,
    Offsets,
    BlockOffsets,
    Schedule,
    X,
    XQ,
    XAmax,
    start,
    max_blocks,
    N: tl.constexpr,
    GROUPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    E: tl.constexpr,
    IDS_BLOCK: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SEARCH_N: tl.constexpr,
    SEARCH_E: tl.constexpr,
    QUANT: tl.constexpr,
    K: tl.constexpr,
    AMAX_BLOCK: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    M: tl.constexpr = 1,
    BM: tl.constexpr = 1,
):
    # Program 0: route_topk then route_metadata for M tokens, the same code (M > 1:
    # the rows' top-k together), the metadata reading the ids and weights the top-k
    # just wrote. Programs 1..: the tokens' NVFP4 (QUANT 1) or MXFP8 (QUANT 2) fake
    # quantization, one (row, chunk) each.
    pid = tl.program_id(0)
    if pid == 0:
        if M == 1:
            _topk_row(
                P, Bias, Logits, Ids, Weights, GlobalScale, route_scale, 1, 0, N,
                GROUPS, HAS_BIAS, TOPK, 8, BLOCK_N, True,
            )  # fmt: skip
        else:
            _topk_rows(
                P, Bias, Logits, Ids, Weights, GlobalScale, route_scale, M, N,
                GROUPS, HAS_BIAS, TOPK, BM, BLOCK_N,
            )  # fmt: skip
        tl.debug_barrier()
        _metadata(
            Ids, Weights, SortedIds, Src2Dst, Dst2Src, Gammas, Counts, Offsets,
            BlockOffsets, Schedule, M * TOPK, start, max_blocks, E, IDS_BLOCK,
            BLOCK_E, BLOCK_S, BLOCK_M, SEARCH_N, SEARCH_E,
        )  # fmt: skip
    else:
        CHUNKS: tl.constexpr = (K + QUANT_BLOCK - 1) // QUANT_BLOCK
        row = (pid - 1) // CHUNKS
        chunk = (pid - 1) % CHUNKS
        if QUANT == 1:
            _fake_quantize_row_chunk(
                X, XQ, XAmax, row, chunk, K, AMAX_BLOCK, QUANT_BLOCK
            )
        if QUANT == 2:
            _fake_quantize_mxfp8_chunk(X, XQ, row, chunk, K, QUANT_BLOCK)


def route_topk_metadata(
    partials,
    bias,
    route_scale,
    global_scale,
    start,
    num_local,
    block_size_m,
    x=None,
    quant=None,
    out_dtype=torch.bfloat16,
):
    """route_topk (top-8, renormalized) of up to ROUTE_FUSED_MAX_TOKENS tokens' router
    partials followed by route_metadata of their ids for experts [start, start +
    num_local), in one launch; bitwise the two calls. With the tokens' activations
    x and quant "nvfp4" or "mxfp8", other programs of the same launch fake-quantize
    x as experts_sm90.fake_quantize(x, out_dtype) / fake_quantize_mxfp8(x) do,
    bitwise. Returns (weights, ids, route_metadata's tuple, (xq, x_amax) or None);
    x_amax is None for MXFP8."""
    groups, m, n = partials.shape
    assert 1 <= m <= ROUTE_FUSED_MAX_TOKENS
    device = partials.device
    quantized, quant_args, programs = None, (partials, partials, partials), 1
    mode, width, amax_block, quant_block = 0, 1, 1, 256
    if x is not None:
        assert x.shape[0] == m and x.is_contiguous() and quant in ("nvfp4", "mxfp8")
        width = x.shape[1]
        programs += m * triton.cdiv(width, quant_block)
        if quant == "nvfp4":
            mode, amax_block = 1, triton.next_power_of_2(width)
            xq = torch.empty_like(x, dtype=out_dtype)
            x_amax = torch.empty(m, dtype=torch.float32, device=device)
            quant_args, quantized = (x, xq, x_amax), (xq, x_amax)
        else:
            mode = 2
            xq = torch.empty_like(x, dtype=torch.bfloat16)
            quant_args, quantized = (x, xq, xq), (xq, None)
    topk = 8
    max_blocks = triton.cdiv(m * topk, block_size_m) + num_local - 1
    logits = torch.empty((m, n), device=device, dtype=torch.float32)
    ids = torch.empty((m, topk), device=device, dtype=torch.int32)
    weights = torch.empty((m, topk), device=device, dtype=torch.float32)
    sorted_ids, src2dst, dst2src = (
        torch.empty(m * topk, device=device, dtype=torch.int32) for _ in range(3)
    )
    gammas = torch.empty(m * topk, device=device, dtype=torch.float32)
    counts = torch.empty(num_local, device=device, dtype=torch.int32)
    offsets = torch.empty(num_local + 1, device=device, dtype=torch.int32)
    block_offsets = torch.empty(num_local + 1, device=device, dtype=torch.int32)
    schedule = torch.empty(max_blocks, device=device, dtype=torch.int32)
    # route_metadata's block for m * topk ids.
    ids_block = max(triton.next_power_of_2(m * topk), 16)
    block_e = triton.next_power_of_2(num_local + 1)
    _route_topk_metadata[(programs,)](
        partials,
        bias,
        logits,
        ids,
        weights,
        global_scale,
        route_scale,
        sorted_ids,
        src2dst,
        dst2src,
        gammas,
        counts,
        offsets,
        block_offsets,
        schedule,
        *quant_args,
        start,
        max_blocks,
        n,
        groups,
        bias is not None,
        topk,
        triton.next_power_of_2(n),
        num_local,
        ids_block,
        block_e,
        triton.next_power_of_2(max_blocks),
        block_size_m,
        ids_block.bit_length(),
        block_e.bit_length(),
        mode,
        width,
        amax_block,
        quant_block,
        M=m,
        BM=triton.next_power_of_2(m),
        # Swept on MI350X: the top-k of 3-4 rows in one program wants 8 warps.
        num_warps=4 if m <= 2 else 8,
        enable_fp_fusion=False,
    )
    metadata = (
        src2dst,
        dst2src,
        gammas,
        counts,
        offsets,
        block_offsets,
        schedule,
        sorted_ids,
    )
    return weights, ids, metadata, quantized


@triton.jit
def _route_metadata_quantize(
    Ids,
    Weights,
    SortedIds,
    Src2Dst,
    Dst2Src,
    Gammas,
    Counts,
    Offsets,
    BlockOffsets,
    Schedule,
    X,
    XQ,
    XAmax,
    n,
    start,
    max_blocks,
    E: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SEARCH_N: tl.constexpr,
    SEARCH_E: tl.constexpr,
    QUANT: tl.constexpr,
    K: tl.constexpr,
    AMAX_BLOCK: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
):
    # Program 0: route_metadata. Programs 1..: the rows' NVFP4 (QUANT 1) or MXFP8
    # (QUANT 2) fake quantization, one (row, chunk) each, as fake_quantize runs it.
    pid = tl.program_id(0)
    if pid == 0:
        _metadata(
            Ids, Weights, SortedIds, Src2Dst, Dst2Src, Gammas, Counts, Offsets,
            BlockOffsets, Schedule, n, start, max_blocks, E, BLOCK_N, BLOCK_E,
            BLOCK_S, BLOCK_M, SEARCH_N, SEARCH_E,
        )  # fmt: skip
    else:
        CHUNKS: tl.constexpr = (K + QUANT_BLOCK - 1) // QUANT_BLOCK
        row = ((pid - 1) // CHUNKS).to(tl.int64)
        chunk = (pid - 1) % CHUNKS
        if QUANT == 1:
            _fake_quantize_row_chunk(
                X, XQ, XAmax, row, chunk, K, AMAX_BLOCK, QUANT_BLOCK
            )
        if QUANT == 2:
            _fake_quantize_mxfp8_chunk(X, XQ, row, chunk, K, QUANT_BLOCK)


def route_metadata_quantize(
    ids, weights, start, num_local, num_experts, block_size_m, x, quant, out_dtype
):
    """route_metadata(ids, weights, ...) and the fake quantization of x's rows
    (experts_sm90.fake_quantize(x, out_dtype) or fake_quantize_mxfp8(x)) in one
    launch, bitwise the separate calls, for at most ROUTE_QUANT_MAX_IDS ids.
    Returns (route_metadata's tuple, (xq, x_amax)); x_amax is None for MXFP8."""
    n = ids.numel()
    rows, k = x.shape
    assert n <= ROUTE_QUANT_MAX_IDS and start + num_local <= num_experts
    assert x.is_contiguous() and quant in ("nvfp4", "mxfp8")
    ids, weights = ids.to(torch.int32), weights.to(torch.float32)
    device = ids.device
    max_blocks = triton.cdiv(n, block_size_m) + num_local - 1
    sorted_ids, src2dst, dst2src = (
        torch.empty(n, device=device, dtype=torch.int32) for _ in range(3)
    )
    gammas = torch.empty(n, device=device, dtype=torch.float32)
    counts = torch.empty(num_local, device=device, dtype=torch.int32)
    offsets = torch.empty(num_local + 1, device=device, dtype=torch.int32)
    block_offsets = torch.empty(num_local + 1, device=device, dtype=torch.int32)
    schedule = torch.empty(max_blocks, device=device, dtype=torch.int32)
    quant_block = 256
    if quant == "nvfp4":
        mode, amax_block = 1, triton.next_power_of_2(k)
        xq = torch.empty_like(x, dtype=out_dtype)
        x_amax = torch.empty(rows, dtype=torch.float32, device=device)
        quantized = (xq, x_amax)
    else:
        mode, amax_block = 2, 1
        xq = torch.empty_like(x, dtype=torch.bfloat16)
        x_amax, quantized = xq, (xq, None)
    block_n = max(triton.next_power_of_2(n), 16)
    block_e = triton.next_power_of_2(num_local + 1)
    _route_metadata_quantize[(1 + rows * triton.cdiv(k, quant_block),)](
        ids,
        weights,
        sorted_ids,
        src2dst,
        dst2src,
        gammas,
        counts,
        offsets,
        block_offsets,
        schedule,
        x,
        xq,
        x_amax,
        n,
        start,
        max_blocks,
        num_local,
        block_n,
        block_e,
        triton.next_power_of_2(max_blocks),
        block_size_m,
        block_n.bit_length(),
        block_e.bit_length(),
        mode,
        k,
        amax_block,
        quant_block,
        # route_metadata's warps for this block.
        num_warps=min(16, max(4, block_n // 128)),
        enable_fp_fusion=False,
    )
    metadata = (
        src2dst,
        dst2src,
        gammas,
        counts,
        offsets,
        block_offsets,
        schedule,
        sorted_ids,
    )
    return metadata, quantized
