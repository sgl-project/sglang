"""NVFP4 and MXFP8 expert GEMMs on Hopper through the Marlin MoE kernel (weight-only
dequantization, BF16 MMA). Activations are fake-quantized to their quantized values
(NVFP4 without the per-row outer scale). For NVFP4 the Marlin global scale is 1, so
the BF16 output is the rounded raw accumulator and the outer scales are applied by
the consumer, like the native path's raw accumulator."""

import functools

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.kernels.jit.utils import load_jit
from sglang.kernels.ops.inkling2.experts_sm90 import (
    ROW_LOOP_MIN_ROWS,
    _nvfp4_values,
    _ue8m0,
)
from sglang.kernels.ops.inkling2.moe import _capped_swiglu, _mxfp8_block
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.quantization.gptq_marlin_repack import gptq_marlin_repack
from sglang.srt.layers.quantization.marlin_utils import (
    marlin_make_workspace,
    marlin_permute_scales,
)
from sglang.srt.layers.quantization.marlin_utils_fp4 import (
    _permute_moe_fp4_scales_for_marlin,
    _repack_moe_fp4_weight_for_marlin,
    nvfp4_marlin_process_global_scale,
    nvfp4_marlin_process_scales,
)
from sglang.srt.layers.quantization.marlin_utils_fp8 import (
    fp8_fused_exponent_bias_into_scales,
    pack_fp8_to_int32,
)
from sglang.srt.layers.quantization.utils import get_scalar_types

_, _scalar_types = get_scalar_types()
ROUTE_BLOCKS = 256
GEMV_MAX_EXPERTS = 128
SWIGLU_BLOCK = 256


def repack(weight, scales):
    """(experts, n, k // 2) packed weights and (experts, n, k // 16) e4m3 scales ->
    Marlin weights, Marlin scales, and unit global scales."""
    experts, n, half_k = weight.shape
    k = half_k * 2
    perm = torch.empty(0, dtype=torch.int, device=weight.device)
    marlin_weight = _repack_moe_fp4_weight_for_marlin(
        weight, num_experts=experts, size_n=n, size_k=k, perm=perm
    )
    marlin_scales = _permute_moe_fp4_scales_for_marlin(
        scales.to(torch.bfloat16),
        num_experts=experts,
        size_n=n,
        size_k=k,
        group_size=16,
        process_scales=nvfp4_marlin_process_scales,
    )
    global_scale = nvfp4_marlin_process_global_scale(
        torch.ones(experts, dtype=torch.bfloat16, device=weight.device)
    )
    return marlin_weight, marlin_scales, global_scale


def repack_mxfp8(weight, scales):
    """(experts, n, k) e4m3 weights and (experts, n, k // 32) e8m0 scales -> Marlin
    weights and Marlin BF16 scales (no global scale)."""
    experts, n, k = weight.shape
    if int(scales.max()) >= 135:
        raise ValueError("MXFP8 scales overflow Marlin's BF16 exponent-bias fold")
    perm = torch.empty(0, dtype=torch.int, device=weight.device)
    marlin_weight = torch.stack(
        [
            gptq_marlin_repack(
                b_q_weight=pack_fp8_to_int32(
                    weight[e], size_k_first=False
                ).T.contiguous(),
                perm=perm,
                size_k=k,
                size_n=n,
                num_bits=8,
            )
            for e in range(experts)
        ]
    )
    bf16_scales = torch.exp2(scales.float() - 127).to(torch.bfloat16)
    marlin_scales = torch.stack(
        [
            fp8_fused_exponent_bias_into_scales(
                marlin_permute_scales(
                    s=bf16_scales[e].T.contiguous(), size_k=k, size_n=n, group_size=32
                )
            )
            for e in range(experts)
        ]
    )
    return marlin_weight, marlin_scales, None


@functools.cache
def _workspace(device):
    return marlin_make_workspace(device, 4), torch.zeros(1, device=device)


@triton.jit
def _marlin_route(
    Counts,
    Offsets,
    Dst2Src,
    SortedIds,
    ExpertIds,
    NumPadded,
    rows,
    max_blocks,
    E: tl.constexpr,
    E_POW2: tl.constexpr,
    BSM: tl.constexpr,
    BLOCKS: tl.constexpr,
):
    experts = tl.arange(0, E_POW2)
    counts = tl.load(Counts + experts, experts < E, other=0)
    blocks = (counts + BSM - 1) // BSM
    ends = tl.cumsum(blocks, 0)
    starts = ends - blocks
    b = tl.program_id(0) * BLOCKS + tl.arange(0, BLOCKS)
    expert = tl.sum((ends[None, :] <= b[:, None]).to(tl.int32), 1)
    total = tl.sum(blocks, 0)
    valid = b < total
    in_range = b < max_blocks
    tl.store(ExpertIds + b, tl.where(valid, expert, -1), in_range)
    start = tl.sum(tl.where(experts[None, :] == expert[:, None], starts[None, :], 0), 1)
    count = tl.sum(tl.where(experts[None, :] == expert[:, None], counts[None, :], 0), 1)
    offset = tl.load(Offsets + expert, valid, other=0)
    slots = tl.arange(0, BSM)
    local = (b - start)[:, None] * BSM + slots[None, :]
    used = valid[:, None] & (local < count[:, None])
    ids = offset[:, None] + local
    if Dst2Src is not None:
        ids = tl.load(Dst2Src + ids, used, other=0)
    ids = tl.where(used, ids, rows)
    tl.store(SortedIds + b[:, None] * BSM + slots[None, :], ids, in_range[:, None])
    if tl.program_id(0) == 0:
        tl.store(NumPadded, total * BSM)


def route(counts, offsets, rows, tokens, dst2src=None, block_size=None):
    """Marlin block routing for the local experts' rows in [offsets[0], offsets[-1])
    of the expert-sorted order; with dst2src, the routed ids are source positions."""
    experts = counts.numel()
    block_size = block_size or _block_size(min(tokens, rows) / experts)
    max_blocks = triton.cdiv(rows, block_size) + experts
    sorted_ids = torch.empty(
        max_blocks * block_size, dtype=torch.int32, device=counts.device
    )
    expert_ids = torch.empty(max_blocks, dtype=torch.int32, device=counts.device)
    num_padded = torch.empty(1, dtype=torch.int32, device=counts.device)
    blocks = min(ROUTE_BLOCKS, triton.next_power_of_2(max_blocks))
    _marlin_route[(triton.cdiv(max_blocks, blocks),)](
        counts,
        offsets,
        dst2src,
        sorted_ids,
        expert_ids,
        num_padded,
        rows,
        max_blocks,
        experts,
        triton.next_power_of_2(experts),
        block_size,
        blocks,
    )
    return sorted_ids, expert_ids, num_padded, block_size


def _block_size(rows_per_expert):
    for size in (8, 16, 32, 64):
        if rows_per_expert / size < (0.5 if size == 8 else 0.9):
            return size
    return 64


def gemm(xq, marlin, out_size, routing, top_k):
    """BF16-rounded raw accumulator rows: row r of the result is xq[r // top_k] times
    the weights of the expert r is routed to (routed rows only)."""
    weight, scales, global_scale = marlin
    rows, width = xq.shape
    out = torch.empty((rows * top_k, out_size), dtype=torch.bfloat16, device=xq.device)
    if rows:
        sorted_ids, expert_ids, num_padded, size = routing
        moe_wna16_marlin_gemm(
            xq,
            out,
            weight,
            None,
            scales,
            global_scale,
            None,
            None,
            None,
            _workspace(xq.device)[0],
            sorted_ids,
            expert_ids,
            num_padded,
            _workspace(xq.device)[1],
            moe_block_size=size,
            top_k=top_k,
            mul_topk_weights=False,
            is_ep=False,
            b_q_type=(
                _scalar_types.float4_e2m1f
                if global_scale is not None
                else _scalar_types.float8_e4m3fn
            ),
            size_m=rows,
            size_n=out_size,
            size_k=width,
            is_k_full=True,
            use_atomic_add=False,
            use_fp32_reduce=True,
        )
    return out


@functools.cache
def _gemv_module(nvfp4, n, k, splits, warps):
    args = f"{str(nvfp4).lower()}, {n}, {k}, {splits}, {warps}"
    return load_jit(
        "inkling2_marlin_gemv",
        *(str(a) for a in (nvfp4, n, k, splits, warps)),
        cuda_files=["inkling2/marlin_gemv.cuh"],
        cuda_wrappers=[("run", f"inkling2_marlin_gemv::run<{args}>")],
    )


def uses_gemv(marlin, tokens, num_local):
    """Whether gemv() beats gemm() for this many tokens (rows per expert <= tokens)."""
    nvfp4 = marlin[2] is not None
    return num_local <= GEMV_MAX_EXPERTS and tokens <= (8 if nvfp4 else 2)


def _gemv_config(tokens, width):
    return (4, 4) if tokens <= 2 or width > 2304 else (1, 8)


def gemv(xq, marlin, out_size, metadata, top_k, tokens, splits=None, warps=None):
    """gemm() for few rows per expert (8 per pass), on the same Marlin weights:
    metadata is (counts, offsets, dst2src) of the local experts' rows; tokens
    picks the K split and warps."""
    weight, scales, global_scale = marlin
    counts, offsets, dst2src = metadata
    rows, width = xq.shape
    out = torch.empty((rows * top_k, out_size), dtype=torch.bfloat16, device=xq.device)
    if splits is None:
        splits, warps = _gemv_config(tokens, width)
    slots = min(counts.numel(), rows * top_k)
    if slots:
        module = _gemv_module(global_scale is not None, out_size, width, splits, warps)
        module.run(xq, weight, scales, out, counts, offsets, dst2src, slots, top_k)
    return out


@triton.jit
def _row_scale(XAmax, WAmax, WDenominator, expert, amax_row):
    denominator = 1536.0 * tl.load(WDenominator)
    return (tl.load(XAmax + amax_row) * tl.load(WAmax + expert)) * libdevice.rcp_rn(
        denominator
    )


@triton.jit
def _swiglu_fake_quantize(
    X,
    Ids,
    Gammas,
    XAmax,
    WAmax,
    WDenominator,
    Y,
    YAmax,
    start,
    num_local,
    TOP_K: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    GATE_CAP: tl.constexpr,
    UP_CAP: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    expert = tl.load(Ids + row) - start
    if (expert < 0) | (expert >= num_local):
        return
    scale = _row_scale(XAmax, WAmax, WDenominator, expert, row // TOP_K)
    gamma = tl.load(Gammas + row)
    amax = tl.zeros((), tl.float32)
    for chunk in range(tl.cdiv(N, BLOCK)):
        cols = chunk * BLOCK + tl.arange(0, BLOCK)
        y = _scaled_swiglu(X, row, cols, cols < N, scale, gamma, N, GATE_CAP, UP_CAP)
        amax = tl.maximum(amax, tl.max(tl.abs(y), 0))
    tl.store(YAmax + row, amax)
    for chunk in range(tl.cdiv(N, BLOCK)):
        _swiglu_quantize_chunk(
            X, Y, row, chunk, scale, gamma, amax, N, BLOCK, GATE_CAP, UP_CAP
        )


@triton.jit
def _swiglu_chunks(
    X,
    Ids,
    Gammas,
    XAmax,
    WAmax,
    WDenominator,
    Y,
    Partials,
    start,
    num_local,
    TOP_K: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    CHUNKS: tl.constexpr,
    GATE_CAP: tl.constexpr,
    UP_CAP: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    expert = tl.load(Ids + row) - start
    if (expert < 0) | (expert >= num_local):
        return
    scale = _row_scale(XAmax, WAmax, WDenominator, expert, row // TOP_K)
    cols = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    y = _scaled_swiglu(
        X, row, cols, mask, scale, tl.load(Gammas + row), N, GATE_CAP, UP_CAP
    )
    tl.store(Y + row * N + cols, y.to(tl.bfloat16), mask)
    tl.store(Partials + row * CHUNKS + chunk, tl.max(tl.abs(y), 0))


@triton.jit
def _quantize_chunks(
    Ids,
    Y,
    Partials,
    YAmax,
    start,
    num_local,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    CHUNKS: tl.constexpr,
    CHUNKS_POW2: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    expert = tl.load(Ids + row) - start
    if (expert < 0) | (expert >= num_local):
        return
    chunks = tl.arange(0, CHUNKS_POW2)
    amax = tl.max(
        tl.load(Partials + row * CHUNKS + chunks, chunks < CHUNKS, other=0), 0
    )
    if chunk == 0:
        tl.store(YAmax + row, amax)
    cols = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    y = tl.load(Y + row * N + cols, mask, other=0).to(tl.float32)
    q = tl.reshape(_nvfp4_values(tl.reshape(y, (BLOCK // 16, 16)), amax), (BLOCK,))
    tl.store(Y + row * N + cols, q.to(tl.bfloat16), mask)


@triton.jit
def _swiglu_quantize_chunk(
    X,
    Y,
    row,
    chunk,
    scale,
    gamma,
    amax,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    GATE_CAP: tl.constexpr,
    UP_CAP: tl.constexpr,
):
    cols = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    y = _scaled_swiglu(X, row, cols, mask, scale, gamma, N, GATE_CAP, UP_CAP)
    q = tl.reshape(_nvfp4_values(tl.reshape(y, (BLOCK // 16, 16)), amax), (BLOCK,))
    tl.store(Y + row * N + cols, q.to(tl.bfloat16), mask)


@triton.jit
def _scaled_swiglu(
    X,
    row,
    cols,
    mask,
    scale,
    gamma,
    N: tl.constexpr,
    GATE_CAP: tl.constexpr,
    UP_CAP: tl.constexpr,
):
    gate = tl.load(X + row * 2 * N + 2 * cols, mask, other=0)
    up = tl.load(X + row * 2 * N + 2 * cols + 1, mask, other=0)
    gate = (gate * scale).to(tl.bfloat16).to(tl.float32)
    up = (up * scale).to(tl.bfloat16).to(tl.float32)
    y = _capped_swiglu(gate, up, gamma, GATE_CAP, UP_CAP)
    return y.to(tl.bfloat16).to(tl.float32)


def swiglu_fake_quantize(acc, ids, gammas, scales, start, num_local, top_k, caps):
    """routed_swiglu of the raw w13 accumulator rows (source order) followed by the
    NVFP4 fake quantization of the w2 input; rows not routed here are left unset."""
    rows, width = acc.shape[0], acc.shape[1] // 2
    y = torch.empty((rows, width), dtype=torch.bfloat16, device=acc.device)
    amax = torch.empty(rows, dtype=torch.float32, device=acc.device)
    if not rows:
        return y, amax
    x_amax, w_amax, w_denominator = scales
    common = (start, num_local)
    if rows >= ROW_LOOP_MIN_ROWS:
        _swiglu_fake_quantize[(rows,)](
            acc,
            ids,
            gammas,
            x_amax,
            w_amax,
            w_denominator,
            y,
            amax,
            *common,
            top_k,
            width,
            SWIGLU_BLOCK,
            *caps,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return y, amax
    chunks = triton.cdiv(width, SWIGLU_BLOCK)
    partials = torch.empty((rows, chunks), dtype=torch.float32, device=acc.device)
    grid = (rows, chunks)
    _swiglu_chunks[grid](
        acc,
        ids,
        gammas,
        x_amax,
        w_amax,
        w_denominator,
        y,
        partials,
        *common,
        top_k,
        width,
        SWIGLU_BLOCK,
        chunks,
        *caps,
        enable_fp_fusion=False,
    )
    _quantize_chunks[grid](
        ids,
        y,
        partials,
        amax,
        *common,
        width,
        SWIGLU_BLOCK,
        chunks,
        triton.next_power_of_2(chunks),
        enable_fp_fusion=False,
    )
    return y, amax


@triton.jit
def _source_partial(
    X,
    Ids,
    XAmax,
    WAmax,
    WDenominator,
    Y,
    start,
    num_local,
    TOP_K: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    total = tl.zeros((BLOCK,), tl.float32)
    for k in tl.static_range(TOP_K):
        row = token * TOP_K + k
        expert = tl.load(Ids + row) - start
        if (expert >= 0) & (expert < num_local):
            x = tl.load(X + row * N + cols, mask, other=0)
            if XAmax is not None:
                scale = _row_scale(XAmax, WAmax, WDenominator, expert, row)
                x = (x * scale).to(tl.bfloat16)
            total += x.to(tl.float32)
    tl.store(Y + token * N + cols, total, mask)


def source_partial(x, ids, scales, start, num_local, top_k, out=None):
    """FP32 per-token sums, in top-k order, of the BF16-rounded (scaled, for NVFP4)
    rows routed to this rank's experts."""
    tokens = x.shape[0] // top_k
    if out is None:
        out = torch.empty((tokens, x.shape[1]), dtype=torch.float32, device=x.device)
    if out.numel():
        _source_partial[(tokens, triton.cdiv(out.shape[1], 1024))](
            x,
            ids,
            *scales,
            out,
            start,
            num_local,
            top_k,
            out.shape[1],
            1024,
            enable_fp_fusion=False,
        )
    return out


@triton.jit
def _swiglu_fake_quantize_mxfp8(
    X,
    Ids,
    Gammas,
    Y,
    start,
    num_local,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    GATE_CAP: tl.constexpr,
    UP_CAP: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    expert = tl.load(Ids + row) - start
    if (expert < 0) | (expert >= num_local):
        return
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    gate = tl.load(X + row * 2 * N + 2 * cols, mask, other=0).to(tl.float32)
    up = tl.load(X + row * 2 * N + 2 * cols + 1, mask, other=0).to(tl.float32)
    y = _capped_swiglu(gate, up, tl.load(Gammas + row), GATE_CAP, UP_CAP)
    y = y.to(tl.bfloat16).to(tl.float32)
    values, exponent = _mxfp8_block(tl.reshape(y, (BLOCK // 32, 32)))
    q = values.to(tl.float32) * _ue8m0(exponent)[:, None]
    tl.store(Y + row * N + cols, tl.reshape(q, (BLOCK,)).to(tl.bfloat16), mask)


def swiglu_fake_quantize_mxfp8(acc, ids, gammas, start, num_local, caps):
    """routed_swiglu of the BF16 w13 output rows (source order) followed by the MXFP8
    fake quantization of the w2 input; rows not routed here are left unset."""
    rows, width = acc.shape[0], acc.shape[1] // 2
    y = torch.empty((rows, width), dtype=torch.bfloat16, device=acc.device)
    if rows:
        _swiglu_fake_quantize_mxfp8[(rows, triton.cdiv(width, SWIGLU_BLOCK))](
            acc,
            ids,
            gammas,
            y,
            start,
            num_local,
            width,
            SWIGLU_BLOCK,
            *caps,
            enable_fp_fusion=False,
        )
    return y
