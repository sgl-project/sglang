"""Post-RMSNorm, gated residual add and pre-RMSNorm of one residual-stream boundary, on gfx950.

One program owns one row. Three kernels cover the token range, measured on MI350X at H=4096, 8192 and 16384:

- Up to one row per two CUs the call is latency bound. The TLX kernel pins every operand to one 8-wave register
  layout (8 contiguous elements per lane), issues all loads (branch, stream, gate, weights) before the first
  reduction, and uses buffer loads and stores off the row base.
- Up to 10 rows of 8192 elements per CU (counted in elements, so 20 rows at H=4096 and 5 at H=16384), plain Triton
  with the same early loads is faster: the pinned layout's FP32 accesses are not
  coalesced, which costs once bandwidth matters. Its reductions use the same layout, so the two are bitwise equal.
- Beyond that, persistent 8-wave programs walk rows with the same math and load the gate and weights once. Several
  programs share a CU (4 up to H=8192, 2 at H=16384), so one program's loads overlap another's reductions; holding
  the next row in registers instead costs occupancy (136 vs 121 VGPRs at H=8192) and is slower. From 3 * 2**23
  elements (about 300 MB of traffic; the last-level cache holds 256 MB) its row loads and stores are non-temporal:
  3-22% faster with the inputs evicted, and between 6% slower (near the threshold) and 7% faster with them cached.
  Below that size the hints slow calls whose inputs are cached by more than they gain, so they stay off.

All three reduce in the same layout, so a row's result does not depend on the batch it runs in.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

# Element offsets at or past this need 64-bit row offsets.
INT32_OFFSETS = 1 << 31
# Calls of at least this many elements use non-temporal row loads and stores.
STREAMING_ELEMENTS = 3 << 23
# The plain kernel takes up to this many elements per CU.
EARLY_ELEMENTS_PER_CU = 10 * 8192


def persistent_programs_per_cu(H: int) -> int:
    return 2 if H >= 16384 else 4


def use_streaming_accesses(rows: int, H: int) -> bool:
    return rows * H >= STREAMING_ELEMENTS


def pick_kernel(rows: int, H: int, num_cus: int) -> str:
    if rows <= num_cus // 2:
        return "tlx"
    if rows * H <= EARLY_ELEMENTS_PER_CU * num_cus:
        return "early"
    return "persistent"


@triton.jit
def _tlx_kernel(
    BRANCH,
    STREAM,
    OUTPUT,
    NEW_STREAM,
    POST_WEIGHT,
    PRE_WEIGHT,
    ALPHA,
    BETA,
    post_eps,
    pre_eps,
    H: tl.constexpr,
    POST_INPUT_SCALE: tl.constexpr,
    HAS_POST_WEIGHT: tl.constexpr,
    HAS_PRE_WEIGHT: tl.constexpr,
    GATE: tl.constexpr,
    GATE_PER_CHANNEL: tl.constexpr,
    PRE_NORM_BF16: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    layout: tl.constexpr = tlx.layout(
        shape=((64, 8), (8, H // (64 * 8 * 8))),
        stride=((8, 64 * 8), (1, 64 * 8 * 8)),
    )
    offs = tlx.require_layout(tl.arange(0, H), layout, pin=True)
    post_input = tlx.buffer_load(BRANCH + row * H, offs, contiguity=8).to(tl.float32)
    stream = tlx.require_layout(tlx.buffer_load(STREAM + row * H, offs, contiguity=4), layout, pin=True)
    if GATE_PER_CHANNEL:
        alpha = tlx.require_layout(tlx.buffer_load(ALPHA, offs, contiguity=4), layout, pin=True)
        beta = tlx.require_layout(tlx.buffer_load(BETA, offs, contiguity=4), layout, pin=True)
    if HAS_PRE_WEIGHT:
        pre_weight = tlx.buffer_load(PRE_WEIGHT, offs, contiguity=8)
    if HAS_POST_WEIGHT:
        post_weight = tlx.buffer_load(POST_WEIGHT, offs, contiguity=8).to(tl.float32)
        if POST_INPUT_SCALE:
            post_input = (post_input * post_weight).to(tl.bfloat16).to(tl.float32)
    normed_branch = post_input * tl.rsqrt(tl.sum(post_input * post_input) / H + post_eps)
    if HAS_POST_WEIGHT and not POST_INPUT_SCALE:
        normed_branch = normed_branch * post_weight
    normed_branch = normed_branch.to(tl.bfloat16).to(tl.float32)
    if GATE:
        if not GATE_PER_CHANNEL:
            alpha = tl.load(ALPHA)
            beta = tl.load(BETA)
        new_stream = alpha * stream + beta * normed_branch
    else:
        new_stream = stream + normed_branch
    new_stream = tlx.require_layout(new_stream, layout, pin=True)
    # A buffer access moves at most 128 bits: 8 BF16 or 4 FP32 elements.
    tlx.buffer_store(new_stream, NEW_STREAM + row * H, offs, contiguity=4)

    pre_input = new_stream
    if PRE_NORM_BF16:
        pre_input = pre_input.to(tl.bfloat16).to(tl.float32)
    normed = pre_input * tl.rsqrt(tl.sum(pre_input * pre_input) / H + pre_eps)
    if HAS_PRE_WEIGHT:
        normed = normed * pre_weight.to(tl.float32)
    tlx.buffer_store(tlx.require_layout(normed.to(tl.bfloat16), layout, pin=True), OUTPUT + row * H, offs, contiguity=8)


@triton.jit
def _plain_kernel(
    BRANCH,
    STREAM,
    OUTPUT,
    NEW_STREAM,
    POST_WEIGHT,
    PRE_WEIGHT,
    ALPHA,
    BETA,
    post_eps,
    pre_eps,
    H: tl.constexpr,
    POST_INPUT_SCALE: tl.constexpr,
    HAS_POST_WEIGHT: tl.constexpr,
    HAS_PRE_WEIGHT: tl.constexpr,
    GATE: tl.constexpr,
    GATE_PER_CHANNEL: tl.constexpr,
    PRE_NORM_BF16: tl.constexpr,
    ROW64: tl.constexpr,
):
    # 32-bit offsets let the compiler use buffer loads and stores.
    row = tl.program_id(0).to(tl.int64) if ROW64 else tl.program_id(0)
    offs = tl.arange(0, H)
    mask = offs < H
    branch = tl.load(BRANCH + row * H + offs, mask=mask, other=0.0)
    stream = tl.load(STREAM + row * H + offs)
    if GATE_PER_CHANNEL:
        alpha = tl.load(ALPHA + offs)
        beta = tl.load(BETA + offs)
    else:
        alpha = 0.0
        beta = 0.0
    if HAS_PRE_WEIGHT:
        pre_weight = tl.load(PRE_WEIGHT + offs).to(tl.float32)
    else:
        pre_weight = 0.0
    if HAS_POST_WEIGHT:
        post_weight = tl.load(POST_WEIGHT + offs, mask=mask, other=0.0).to(tl.float32)
    else:
        post_weight = 0.0
    _plain_row(branch, stream, alpha, beta, post_weight, pre_weight, row * H, OUTPUT, NEW_STREAM, ALPHA, BETA, post_eps,
               pre_eps, H, POST_INPUT_SCALE, HAS_POST_WEIGHT, HAS_PRE_WEIGHT, GATE, GATE_PER_CHANNEL, PRE_NORM_BF16)


@triton.jit
def _persistent_kernel(
    BRANCH,
    STREAM,
    OUTPUT,
    NEW_STREAM,
    POST_WEIGHT,
    PRE_WEIGHT,
    ALPHA,
    BETA,
    rows,
    post_eps,
    pre_eps,
    H: tl.constexpr,
    POST_INPUT_SCALE: tl.constexpr,
    HAS_POST_WEIGHT: tl.constexpr,
    HAS_PRE_WEIGHT: tl.constexpr,
    GATE: tl.constexpr,
    GATE_PER_CHANNEL: tl.constexpr,
    PRE_NORM_BF16: tl.constexpr,
    ROW64: tl.constexpr,
    STREAMING: tl.constexpr,
):
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    offs = tl.arange(0, H)
    mask = offs < H
    if GATE_PER_CHANNEL:
        alpha = tl.load(ALPHA + offs)
        beta = tl.load(BETA + offs)
    else:
        alpha = 0.0
        beta = 0.0
    if HAS_PRE_WEIGHT:
        pre_weight = tl.load(PRE_WEIGHT + offs).to(tl.float32)
    else:
        pre_weight = 0.0
    if HAS_POST_WEIGHT:
        post_weight = tl.load(POST_WEIGHT + offs, mask=mask, other=0.0).to(tl.float32)
    else:
        post_weight = 0.0
    for row in range(pid, rows, programs):
        base = row.to(tl.int64) * H if ROW64 else row * H
        branch = tl.load(BRANCH + base + offs, mask=mask, other=0.0, cache_modifier=".cg" if STREAMING else "")
        stream = tl.load(STREAM + base + offs, cache_modifier=".cg" if STREAMING else "")
        _plain_row(branch, stream, alpha, beta, post_weight, pre_weight, base, OUTPUT, NEW_STREAM, ALPHA, BETA,
                   post_eps, pre_eps, H, POST_INPUT_SCALE, HAS_POST_WEIGHT, HAS_PRE_WEIGHT, GATE, GATE_PER_CHANNEL,
                   PRE_NORM_BF16, ".cs" if STREAMING else "")


@triton.jit
def _plain_row(branch, stream, alpha, beta, post_weight, pre_weight, base, OUTPUT, NEW_STREAM, ALPHA, BETA, post_eps,
               pre_eps, H: tl.constexpr, POST_INPUT_SCALE: tl.constexpr, HAS_POST_WEIGHT: tl.constexpr,
               HAS_PRE_WEIGHT: tl.constexpr, GATE: tl.constexpr, GATE_PER_CHANNEL: tl.constexpr,
               PRE_NORM_BF16: tl.constexpr, STORE_CACHE: tl.constexpr = ""):
    offs = tl.arange(0, H)
    mask = offs < H
    post_input = branch.to(tl.float32)
    if HAS_POST_WEIGHT and POST_INPUT_SCALE:
        post_input = (post_input * post_weight).to(tl.bfloat16).to(tl.float32)
    post_sum = tl.sum(tl.where(mask, post_input * post_input, 0.0))
    normed_branch = post_input * tl.rsqrt(post_sum / H + post_eps)
    if HAS_POST_WEIGHT and not POST_INPUT_SCALE:
        normed_branch = normed_branch * post_weight
    normed_branch = normed_branch.to(tl.bfloat16).to(tl.float32)
    if GATE:
        if not GATE_PER_CHANNEL:
            alpha = tl.load(ALPHA)
            beta = tl.load(BETA)
        new_stream = alpha * stream + beta * normed_branch
    else:
        new_stream = stream + normed_branch
    tl.store(NEW_STREAM + base + offs, new_stream, mask=mask, cache_modifier=STORE_CACHE)

    pre_input = new_stream
    if PRE_NORM_BF16:
        pre_input = pre_input.to(tl.bfloat16).to(tl.float32)
    pre_sum = tl.sum(tl.where(mask, pre_input * pre_input, 0.0))
    normed = pre_input * tl.rsqrt(pre_sum / H + pre_eps)
    if HAS_PRE_WEIGHT:
        normed = normed * pre_weight
    tl.store(OUTPUT + base + offs, normed.to(tl.bfloat16), mask=mask, cache_modifier=STORE_CACHE)


def norm_residual_norm(stream, branch, post_weight, post_norm_input_scale, pre_weight, alpha, beta, post_eps, pre_eps,
                       pre_norm_bf16, out, new_stream):
    M, H = branch.shape
    if out is None:
        out = torch.empty(M, H, device=branch.device, dtype=torch.bfloat16)
    if new_stream is None:
        new_stream = torch.empty(M, H, device=branch.device, dtype=torch.float32)
    if M == 0:
        return out, new_stream
    args = (
        branch,
        stream,
        out,
        new_stream,
        post_weight if post_weight is not None else branch,
        pre_weight if pre_weight is not None else branch,
        alpha if alpha is not None else stream,
        beta if beta is not None else stream,
        float(post_eps),
        float(pre_eps),
        H,
    )
    flags = dict(
        POST_INPUT_SCALE=post_norm_input_scale,
        HAS_POST_WEIGHT=post_weight is not None,
        HAS_PRE_WEIGHT=pre_weight is not None,
        GATE=alpha is not None,
        GATE_PER_CHANNEL=alpha is not None and alpha.numel() == H,
        PRE_NORM_BF16=pre_norm_bf16,
    )
    num_cus = torch.cuda.get_device_properties(branch.device).multi_processor_count
    kind = pick_kernel(M, H, num_cus)
    if kind == "tlx":
        _tlx_kernel[(M, )](*args, **flags, num_warps=8, num_stages=1)
    elif kind == "early":
        _plain_kernel[(M, )](*args, **flags, ROW64=M * H >= INT32_OFFSETS, num_warps=8, num_stages=1)
    else:
        programs = min(M, persistent_programs_per_cu(H) * num_cus)
        _persistent_kernel[(programs, )](*args[:8], M, *args[8:], **flags, ROW64=M * H >= INT32_OFFSETS,
                                         STREAMING=use_streaming_accesses(M, H), num_warps=8, num_stages=1)
    return out, new_stream
