# ruff: noqa: BLE001, C408
"""Triton kernels for the Muse Spark MoE on ROCm (gfx950): BF16 and MXFP4 (Quark) experts.

Used by ``MuseSpark1xNativeExperts.forward`` on ROCm (a per-expert PyTorch loop that, under
graph capture, runs every local expert densely over all tokens):

1. ``fused_moe_gemm_kernel`` stage 1 -- grouped GEMM over tokens sorted by expert,
   w13 [E, 2I, K] with the SwiGLU epilogue fused (gate and up accumulated in the same
   program, output silu(gate) * up, shape [M*topk, I]).
2. ``fused_moe_gemm_kernel`` stage 2 -- grouped GEMM with w2 [E, H, I], output kept
   *per slot* ([M*topk, H]); no routing-weight multiply and no top-k reduction here,
   because Muse applies a per-expert RMSNorm before the weighted sum.
3. ``finalize_norm_weighted_sum_kernel`` -- per token: for each local slot,
   y = bf16(scale * out_slot); out += score_slot * y / rms(y); one 4096-wide row per
   program. The scale (stored gamma + gain center) is computed in-kernel.

MXFP4 weights (AMD Quark export, OCP MX: e2m1 values packed two per uint8 along K,
low nibble = even k, high nibble = odd k; one e8m0 scale per 32 consecutive k) are
dequantized inside the GEMM to bf16 -- exact, since fp4 x 2^e is representable -- and
multiplied against bf16 activations ("a16w4"). To avoid interleaving nibbles the K
reduction is split into even and odd halves: acc += A[:, 0::2] @ lo + A[:, 1::2] @ hi.

Alignment (sorting (token, slot) pairs by local expert, padded to BLOCK_M) is one
Triton id-map launch + sgl_kernel's ``moe_align_block_size`` (or a pure-torch fallback).
Everything is shape-static and free of host syncs so it can be captured in a HIP graph.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


# ----------------------------------------------------------------------------- GEMM
@triton.jit
def _fp4_e2m1_to_f32(n):
    """n: integer tensor holding e2m1 codes in the low 4 bits."""
    sign = (n >> 3) & 1
    e = (n >> 1) & 3
    m = (n & 1).to(tl.float32)
    mag = tl.where(e == 0, 0.5 * m, (1.0 + 0.5 * m) * tl.exp2((e - 1).to(tl.float32)))
    return tl.where(sign == 1, -mag, mag)


@triton.jit
def fused_moe_gemm_kernel(
    a_ptr,
    b_ptr,
    bs_ptr,  # MXFP4 only: e8m0 scales [E, N(+N), K//32] uint8
    c_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_se,
    stride_sn,
    stride_sk,
    stride_cm,
    stride_cn,
    stride_ck,  # SPLIT_K > 1: stride between K-partial slabs of the fp32 output
    TOP_K: tl.constexpr,
    A_ROW_DIV_TOPK: tl.constexpr,
    FUSE_SILU_MUL: tl.constexpr,
    W_MXFP4: tl.constexpr,
    FP4_DOT_SCALED: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """C[sorted_row, n] = sum_k A[a_row, k] * B[expert, n, k]  (B stored [E, N, K], or
    [E, N, K//2] packed fp4 when W_MXFP4).

    For stage 1 (FUSE_SILU_MUL) N is the intermediate size I and B holds
    [gate rows 0..I) ++ [up rows I..2I); the program accumulates both and stores
    silu(gate) * up.
    """
    pid = tl.program_id(0)
    pid_k = pid % SPLIT_K  # K-split index (0 when SPLIT_K == 1)
    pid = pid // SPLIT_K
    k0 = pid_k * (K // SPLIT_K)  # first k of this program's K range
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    if pid >= num_pid_m * num_pid_n:
        return

    # grouped (swizzled) program order for L2 reuse of B tiles
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_token_id = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens
    if A_ROW_DIV_TOPK:
        a_rows = offs_token // TOP_K
    else:
        a_rows = offs_token

    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if (
        off_expert < 0
    ):  # block of filtered (non-local / padded) slots: nothing to compute
        return
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    a_row_off = a_rows[:, None].to(tl.int64) * stride_am
    b_base = b_ptr + off_expert * stride_be

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if FUSE_SILU_MUL:
        acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if W_MXFP4 and FP4_DOT_SCALED:
        # gfx950 path: tl.dot_scaled upcasts e2m1 x e8m0 -> bf16 with the scaled-convert
        # instructions and feeds the MFMA. rhs tile is k-packed [BLOCK_K//2, BLOCK_N] uint8,
        # rhs_scale tile is [BLOCK_N, BLOCK_K//32] uint8 (not transposed), lhs is bf16.
        HALF_K: tl.constexpr = BLOCK_K // 2
        KS: tl.constexpr = BLOCK_K // 32
        offs_k = tl.arange(0, BLOCK_K)
        offs_j = tl.arange(0, HALF_K)
        offs_ks = tl.arange(0, KS)
        a_ptrs = a_ptr + a_row_off + offs_k[None, :] * stride_ak
        bq_ptrs = b_base + offs_j[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        bs_base = bs_ptr + off_expert * stride_se
        bs_ptrs = bs_base + offs_bn[:, None] * stride_sn + offs_ks[None, :] * stride_sk
        a_ptrs += k0 * stride_ak
        bq_ptrs += (k0 // 2) * stride_bk
        bs_ptrs += (k0 // 32) * stride_sk
        if FUSE_SILU_MUL:
            bq2_ptrs = bq_ptrs + N * stride_bn
            bs2_ptrs = bs_ptrs + N * stride_sn
        for _ in range(tl.cdiv(K // SPLIT_K, BLOCK_K)):
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            b = tl.load(bq_ptrs)
            bs = tl.load(bs_ptrs)
            acc = tl.dot_scaled(a, None, "bf16", b, bs, "e2m1", acc)
            if FUSE_SILU_MUL:
                b2 = tl.load(bq2_ptrs)
                bs2 = tl.load(bs2_ptrs)
                acc2 = tl.dot_scaled(a, None, "bf16", b2, bs2, "e2m1", acc2)
                bq2_ptrs += HALF_K * stride_bk
                bs2_ptrs += KS * stride_sk
            a_ptrs += BLOCK_K * stride_ak
            bq_ptrs += HALF_K * stride_bk
            bs_ptrs += KS * stride_sk
    elif W_MXFP4:
        HALF_K: tl.constexpr = BLOCK_K // 2
        offs_j = tl.arange(0, HALF_K)  # packed byte index within the tile
        a_even_ptrs = a_ptr + a_row_off + (2 * offs_j)[None, :] * stride_ak
        a_odd_ptrs = a_ptr + a_row_off + (2 * offs_j + 1)[None, :] * stride_ak
        bq_ptrs = b_base + offs_j[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        bs_base = bs_ptr + off_expert * stride_se
        bs_ptrs = (
            bs_base + (offs_j[:, None] // 16) * stride_sk + offs_bn[None, :] * stride_sn
        )
        a_even_ptrs += k0 * stride_ak
        a_odd_ptrs += k0 * stride_ak
        bq_ptrs += (k0 // 2) * stride_bk
        bs_ptrs += (k0 // 32) * stride_sk
        if FUSE_SILU_MUL:
            bq2_ptrs = bq_ptrs + N * stride_bn
            bs2_ptrs = bs_ptrs + N * stride_sn
        for _ in range(tl.cdiv(K // SPLIT_K, BLOCK_K)):
            a_e = tl.load(a_even_ptrs, mask=token_mask[:, None], other=0.0)
            a_o = tl.load(a_odd_ptrs, mask=token_mask[:, None], other=0.0)
            p = tl.load(bq_ptrs).to(tl.int32)
            s = tl.exp2(tl.load(bs_ptrs).to(tl.float32) - 127.0)
            lo = (_fp4_e2m1_to_f32(p & 0xF) * s).to(a_e.dtype)
            hi = (_fp4_e2m1_to_f32(p >> 4) * s).to(a_e.dtype)
            acc += tl.dot(a_e, lo)
            acc += tl.dot(a_o, hi)
            if FUSE_SILU_MUL:
                p2 = tl.load(bq2_ptrs).to(tl.int32)
                s2 = tl.exp2(tl.load(bs2_ptrs).to(tl.float32) - 127.0)
                lo2 = (_fp4_e2m1_to_f32(p2 & 0xF) * s2).to(a_e.dtype)
                hi2 = (_fp4_e2m1_to_f32(p2 >> 4) * s2).to(a_e.dtype)
                acc2 += tl.dot(a_e, lo2)
                acc2 += tl.dot(a_o, hi2)
                bq2_ptrs += HALF_K * stride_bk
                bs2_ptrs += (BLOCK_K // 32) * stride_sk
            a_even_ptrs += BLOCK_K * stride_ak
            a_odd_ptrs += BLOCK_K * stride_ak
            bq_ptrs += HALF_K * stride_bk
            bs_ptrs += (BLOCK_K // 32) * stride_sk
    else:
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + a_row_off + offs_k[None, :] * stride_ak
        b_ptrs = b_base + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        a_ptrs += k0 * stride_ak
        b_ptrs += k0 * stride_bk
        if FUSE_SILU_MUL:
            b2_ptrs = b_ptrs + N * stride_bn  # "up" half lives N rows after "gate"
        for _ in range(tl.cdiv(K // SPLIT_K, BLOCK_K)):
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            b = tl.load(b_ptrs)
            acc += tl.dot(a, b)
            if FUSE_SILU_MUL:
                b2 = tl.load(b2_ptrs)
                acc2 += tl.dot(a, b2)
                b2_ptrs += BLOCK_K * stride_bk
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

    if SPLIT_K == 1:
        if FUSE_SILU_MUL:
            acc = acc * tl.sigmoid(acc) * acc2
        c = acc.to(c_ptr.dtype.element_ty)
        c_ptrs = (
            c_ptr
            + offs_token[:, None].to(tl.int64) * stride_cm
            + offs_bn[None, :] * stride_cn
        )
        tl.store(c_ptrs, c, mask=token_mask[:, None])
    else:
        # K-partial slab pid_k of an fp32 buffer; stage 1 keeps gate and up separate
        # ([.., 0:N] gate, [.., N:2N] up) for the split-K SiLU epilogue kernel.
        c_ptrs = (
            c_ptr
            + pid_k * stride_ck
            + offs_token[:, None].to(tl.int64) * stride_cm
            + offs_bn[None, :] * stride_cn
        )
        tl.store(c_ptrs, acc, mask=token_mask[:, None])
        if FUSE_SILU_MUL:
            tl.store(c_ptrs + N * stride_cn, acc2, mask=token_mask[:, None])


@triton.jit
def _splitk_silu_mul_kernel(
    part_ptr,
    out_ptr,
    I,
    stride_pk,
    stride_pm,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """inter[slot, :] = silu(sum_k gate_k) * sum_k up_k over the SPLIT_K fp32 partial slabs."""
    slot = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < I
    base = part_ptr + slot * stride_pm + offs
    gate = tl.zeros((BLOCK,), dtype=tl.float32)
    up = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(SPLIT_K):
        gate += tl.load(base + k * stride_pk, mask=mask, other=0.0)
        up += tl.load(base + k * stride_pk + I, mask=mask, other=0.0)
    y = gate * tl.sigmoid(gate) * up
    tl.store(out_ptr + slot * I + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


# ------------------------------------------------------------------------- finalize
@triton.jit
def finalize_norm_weighted_sum_kernel(
    y_ptr,  # [M*TOP_K, H] per-slot expert outputs (rows of non-local slots are unread)
    w_ptr,  # [H] stored post-norm gamma (checkpoint dtype); effective = w + gain (in that dtype)
    scores_ptr,  # [M, TOP_K] fp32 routing weights
    ids_ptr,  # [M, TOP_K] int32 global expert ids
    out_ptr,  # [M, H]
    H,
    local_start,
    num_local,
    eps,
    gain,
    stride_yk,  # SPLIT_K > 1: stride between fp32 K-partial slabs of y
    TOP_K: tl.constexpr,
    HAS_NORM: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    m = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H
    if HAS_NORM and HAS_SCALE:
        # reference: (self.weight + gain_center) evaluated in the weight dtype, then weight * x
        w = tl.load(w_ptr + offs, mask=mask, other=0.0)
        scale = (w.to(tl.float32) + gain).to(w_ptr.dtype.element_ty).to(tl.float32)
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for s in tl.static_range(TOP_K):
        e = tl.load(ids_ptr + m * TOP_K + s)
        is_local = (e >= local_start) & (e < local_start + num_local)
        row = (m * TOP_K + s).to(tl.int64)
        if SPLIT_K == 1:
            y = tl.load(y_ptr + row * H + offs, mask=mask & is_local, other=0.0).to(
                tl.float32
            )
        else:
            y = tl.zeros((BLOCK_H,), dtype=tl.float32)
            for k in tl.static_range(SPLIT_K):
                y += tl.load(
                    y_ptr + k * stride_yk + row * H + offs,
                    mask=mask & is_local,
                    other=0.0,
                )
            y = y.to(tl.bfloat16).to(
                tl.float32
            )  # the expert output is bf16 in the reference
        sc = tl.load(scores_ptr + m * TOP_K + s).to(tl.float32)
        if HAS_NORM:
            # reference: x = bf16(weight * x); rms_norm in fp32; no affine after
            if HAS_SCALE:
                y = (y * scale).to(tl.bfloat16).to(tl.float32)
            ss = tl.sum(y * y, axis=0)
            inv = 1.0 / tl.sqrt(ss / H + eps)
            wgt = tl.where(is_local, sc * inv, 0.0)
        else:
            wgt = tl.where(is_local, sc, 0.0)
        acc += wgt * y
    tl.store(
        out_ptr + m.to(tl.int64) * H + offs, acc.to(out_ptr.dtype.element_ty), mask=mask
    )


# ------------------------------------------------------------------- block alignment
def moe_align_block_size_torch(
    topk_ids: torch.Tensor,
    num_local_experts: int,
    local_start: int,
    block_m: int,
):
    """Sort (token, slot) pairs by *local* expert and pad each expert to block_m.

    Pure torch, no host sync, static output shapes -> graph-capturable (~20 tiny kernels;
    kept as a fallback). Returns (sorted_token_ids[int32, cap], expert_ids[int32, cap //
    block_m], num_tokens_post_padded[int32, 1]). sorted_token_ids holds ``token * topk +
    slot``; padding slots hold ``M * topk``. Non-local slots are dropped.
    """
    M, topk = topk_ids.shape
    E = num_local_experts
    n = M * topk
    dev = topk_ids.device
    flat = topk_ids.reshape(-1).to(torch.int64)
    local = flat - local_start
    valid = (local >= 0) & (local < E)
    key = torch.where(valid, local, torch.full_like(local, E))  # bucket E = "not mine"

    counts = torch.zeros(E + 1, dtype=torch.int64, device=dev).scatter_add_(
        0, key, torch.ones_like(key)
    )
    counts_e = counts[:E]
    blocks_per_e = (counts_e + block_m - 1) // block_m
    padded_e = blocks_per_e * block_m
    cum_padded = torch.cumsum(padded_e, 0)
    pad_off = cum_padded - padded_e
    num_tokens_post_padded = cum_padded[-1:].to(torch.int32)

    order = torch.argsort(key, stable=True)
    sorted_key = key[order]
    grp_start = torch.cumsum(counts, 0) - counts
    rank = torch.arange(n, device=dev) - grp_start[sorted_key]

    cap = n + E * (block_m - 1)
    cap = ((cap + block_m - 1) // block_m) * block_m
    pad_off_full = torch.cat(
        [pad_off, torch.full((1,), cap, device=dev, dtype=torch.int64)]
    )
    dest = pad_off_full[sorted_key] + rank
    dest = torch.where(sorted_key < E, dest, torch.full_like(dest, cap))

    sorted_token_ids = torch.full((cap + 1,), n, dtype=torch.int32, device=dev)
    sorted_token_ids.scatter_(0, dest, order.to(torch.int32))
    sorted_token_ids = sorted_token_ids[:cap]

    num_blocks = cap // block_m
    block_start = torch.arange(num_blocks, device=dev) * block_m
    expert_ids = torch.searchsorted(cum_padded, block_start, right=True).to(torch.int32)
    expert_ids = torch.clamp(expert_ids, max=E - 1)
    return sorted_token_ids, expert_ids, num_tokens_post_padded


@triton.jit
def _map_local_ids_kernel(
    ids_ptr, out_ptr, n, local_start, num_local, BLOCK: tl.constexpr
):
    """out = g - local_start if that lies in [0, num_local) else -1  (one launch)."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    g = tl.load(ids_ptr + offs, mask=mask, other=-1).to(tl.int32)
    loc = g - local_start
    ok = (loc >= 0) & (loc < num_local)
    tl.store(out_ptr + offs, tl.where(ok, loc, -1), mask=mask)


try:  # sglang's AOT kernel (vLLM lineage): histogram + prefix-sum + scatter in one launch
    from sglang.kernels.ops.moe import (
        moe_align_block_size as _sgl_moe_align,  # type: ignore
    )

    _HAS_SGL_ALIGN = True
except Exception:  # pragma: no cover
    _sgl_moe_align = None
    _HAS_SGL_ALIGN = False

_ALIGN_ENV = os.environ.get("MUSE_MOE_ALIGN", "auto")
FP4_MODE = os.environ.get("MUSE_MOE_FP4_MODE", "dot_scaled")  # dot_scaled | manual
SPLIT_K_SMALL = int(
    os.environ.get("MUSE_MOE_SPLITK", "4")
)  # K-split factor for <=32 routed slots (1 = off)
SPLIT_K_MAX_SLOTS = 32
ALIGN_BACKEND = (
    "sgl"
    if (_ALIGN_ENV == "sgl" or (_ALIGN_ENV == "auto" and _HAS_SGL_ALIGN))
    else "torch"
)
if ALIGN_BACKEND == "sgl" and not _HAS_SGL_ALIGN:
    raise ImportError(
        "MUSE_MOE_ALIGN=sgl but sglang.kernels.ops.moe.moe_align_block_size is unavailable"
    )


def moe_align_block_size_sgl(
    topk_ids: torch.Tensor, num_local_experts: int, local_start: int, block_m: int
):
    """Same outputs as moe_align_block_size_torch, via sgl_kernel's single AOT kernel.

    Contract probed on 2026-09-04 (sglang 0.5.18 / sgl-kernel 0.4.6): pass ``num_experts + 1``
    buckets, ids must lie in [-1, num_experts-1]; -1 slots are grouped into blocks labelled
    ``expert_ids == -1`` (the GEMM skips those); padding sentinel is ``numel``. Values >=
    num_experts overflow the histogram, so non-local ids are mapped to -1 first.
    """
    E = num_local_experts
    n = topk_ids.numel()
    dev = topk_ids.device
    local = torch.empty_like(topk_ids, dtype=torch.int32)
    _map_local_ids_kernel[(triton.cdiv(n, 1024),)](
        topk_ids, local, n, local_start, E, BLOCK=1024
    )
    cap = n * block_m if n < E + 1 else n + (E + 1) * (block_m - 1)
    sorted_ids = torch.empty((cap,), dtype=torch.int32, device=dev)
    expert_ids = torch.empty(
        (triton.cdiv(cap, block_m),), dtype=torch.int32, device=dev
    )
    num_post = torch.empty((1,), dtype=torch.int32, device=dev)
    cumsum = torch.empty((E + 2,), dtype=torch.int32, device=dev)
    _sgl_moe_align(
        local, E + 1, block_m, sorted_ids, expert_ids, num_post, cumsum, True, False
    )
    return sorted_ids, expert_ids, num_post


@triton.jit
def _align_small_kernel(
    ids_ptr,
    sorted_ptr,
    expert_ptr,
    num_post_ptr,
    n,
    local_start,
    cap,
    num_blocks,
    E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK: tl.constexpr,
    CAP_BLOCK: tl.constexpr,
    NB_BLOCK: tl.constexpr,
):
    """One program: local-id mapping + per-expert counting + BLOCK_M-padded slot sort for n <= BLOCK
    routed slots. Same contract as the sgl_kernel path: sorted_ptr[cap] (sentinel = n for padding),
    expert_ptr[num_blocks] (-1 for unused blocks), num_post_ptr[1]."""
    offs = tl.arange(0, BLOCK)
    valid_slot = offs < n
    g = tl.load(ids_ptr + offs, mask=valid_slot, other=-1)
    local = g - local_start
    ok = valid_slot & (local >= 0) & (local < E)
    # sentinel / -1 fills first, then overwrite the used positions
    for c in tl.static_range(0, CAP_BLOCK, BLOCK):
        po = c + offs
        tl.store(sorted_ptr + po, tl.full((BLOCK,), 0, tl.int32) + n, mask=po < cap)
    bo = tl.arange(0, NB_BLOCK)
    tl.store(expert_ptr + bo, tl.full((NB_BLOCK,), -1, tl.int32), mask=bo < num_blocks)
    block_start = 0
    for e in tl.static_range(E):
        hit = ok & (local == e)
        cnt = tl.sum(hit.to(tl.int32), axis=0)
        rank = (
            tl.cumsum(hit.to(tl.int32), axis=0) - 1
        )  # 0-based rank among this expert's slots
        pos = block_start * BLOCK_M + rank
        tl.store(sorted_ptr + pos, offs.to(tl.int32), mask=hit)
        nb = (cnt + BLOCK_M - 1) // BLOCK_M
        tl.store(
            expert_ptr + bo,
            tl.full((NB_BLOCK,), e, tl.int32),
            mask=(bo >= block_start) & (bo < block_start + nb),
        )
        block_start += nb
    tl.store(num_post_ptr, (block_start * BLOCK_M).to(tl.int32))


ALIGN_SMALL_MAX = 64


def moe_align_block_size_small(
    topk_ids: torch.Tensor, num_local_experts: int, local_start: int, block_m: int
):
    n = topk_ids.numel()
    dev = topk_ids.device
    E = num_local_experts
    cap = n + E * (block_m - 1)
    num_blocks = triton.cdiv(cap, block_m)
    cap_block = max(64, triton.next_power_of_2(cap))
    sorted_ids = torch.empty(cap, dtype=torch.int32, device=dev)
    expert_ids = torch.empty(num_blocks, dtype=torch.int32, device=dev)
    num_post = torch.empty(1, dtype=torch.int32, device=dev)
    _align_small_kernel[(1,)](
        topk_ids.reshape(-1),
        sorted_ids,
        expert_ids,
        num_post,
        n,
        local_start,
        cap,
        num_blocks,
        E=E,
        BLOCK_M=block_m,
        BLOCK=64,
        CAP_BLOCK=cap_block,
        NB_BLOCK=max(2, triton.next_power_of_2(num_blocks)),
        num_warps=1,
    )
    return sorted_ids, expert_ids, num_post


def moe_align_block_size(topk_ids, num_local_experts, local_start, block_m):
    if topk_ids.numel() <= ALIGN_SMALL_MAX and ALIGN_BACKEND != "torch":
        return moe_align_block_size_small(
            topk_ids, num_local_experts, local_start, block_m
        )
    if ALIGN_BACKEND == "sgl":
        return moe_align_block_size_sgl(
            topk_ids, num_local_experts, local_start, block_m
        )
    return moe_align_block_size_torch(topk_ids, num_local_experts, local_start, block_m)


# ------------------------------------------------------------------------------ glue
# Tuned on MI355X (gfx950), Triton 3.7 AMD, 2026-09-04 (see tune_moe.py). Decode-size time is
# dominated by fixed overhead, GEMM tile choice matters little there.
_SMALL_M = dict(
    BLOCK_M=16,
    BLOCK_N=64,
    BLOCK_K=128,
    GROUP_M=1,
    num_warps=4,
    num_stages=2,
    waves_per_eu=2,
)  # K128: M=4 127->89us
_MID_M = dict(BLOCK_M=16, BLOCK_N=128, BLOCK_K=64, GROUP_M=4, num_warps=4, num_stages=2)
_LARGE_M = dict(
    BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, GROUP_M=8, num_warps=8, num_stages=2
)
CONFIG_TABLE = {"small": _SMALL_M, "mid": _MID_M, "large": _LARGE_M}
# MXFP4: the dequant costs VALU work per B tile; BLOCK_K=128 halves the per-k-step overhead.
_SMALL_M_FP4 = dict(
    BLOCK_M=16, BLOCK_N=64, BLOCK_K=128, GROUP_M=1, num_warps=4, num_stages=2
)
_MID_M_FP4 = dict(
    BLOCK_M=16, BLOCK_N=64, BLOCK_K=256, GROUP_M=4, num_warps=4, num_stages=2
)  # MI355X sweep 2026-09-07: M=64 313->193us, M=256 533->264us
_LARGE_M_FP4 = dict(
    BLOCK_M=64, BLOCK_N=64, BLOCK_K=128, GROUP_M=8, num_warps=4, num_stages=2
)  # M=1024 564->485us
CONFIG_TABLE_FP4 = {"small": _SMALL_M_FP4, "mid": _MID_M_FP4, "large": _LARGE_M_FP4}


def _pick_config(M: int, topk: int, num_local: int, mxfp4: bool = False):
    rows = M * topk
    table = CONFIG_TABLE_FP4 if mxfp4 else CONFIG_TABLE
    if rows <= 64:
        return table["small"]
    # mid/large crossover measured on MI355X (32 local experts, K=I=4096, graph replay):
    # MXFP4 mid wins up to M=768 (rows 6144), BF16 mid only up to M=256 (rows 2048).
    if rows <= (6144 if mxfp4 else 2048):
        return table["mid"]
    return table["large"]


def _launch_gemm(
    a,
    b,
    b_scale,
    c,
    sorted_ids,
    expert_ids,
    num_post_padded,
    N,
    K,
    num_valid,
    topk,
    a_row_div_topk,
    fuse_silu,
    mxfp4,
    cfg,
    split_k=1,
):
    cap = sorted_ids.numel()
    # Every expert that receives a slot owns at least one real M-block, so the number of
    # non-empty blocks is <= num_valid; the padded capacity (num_valid + E*(BLOCK_M-1) rows)
    # would otherwise launch ~E*(BLOCK_M-1)/BLOCK_M blocks per N-tile that exit immediately.
    num_m_blocks = min(triton.cdiv(cap, cfg["BLOCK_M"]), max(num_valid, 1))
    grid = (num_m_blocks * triton.cdiv(N, cfg["BLOCK_N"]) * split_k,)
    if mxfp4:
        s_strides = (b_scale.stride(0), b_scale.stride(1), b_scale.stride(2))
    else:
        b_scale = c  # dummy pointer
        s_strides = (0, 0, 0)
    fused_moe_gemm_kernel[grid](
        a,
        b,
        b_scale,
        c,
        sorted_ids,
        expert_ids,
        num_post_padded,
        N,
        K,
        num_valid,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        *s_strides,
        c.stride(0) if split_k == 1 else c.stride(1),
        c.stride(-1),
        c.stride(0) if split_k > 1 else 0,
        TOP_K=topk,
        A_ROW_DIV_TOPK=a_row_div_topk,
        FUSE_SILU_MUL=fuse_silu,
        W_MXFP4=mxfp4,
        FP4_DOT_SCALED=(FP4_MODE == "dot_scaled"),
        BLOCK_M=cfg["BLOCK_M"],
        BLOCK_N=cfg["BLOCK_N"],
        BLOCK_K=cfg["BLOCK_K"],
        GROUP_M=cfg["GROUP_M"],
        SPLIT_K=split_k,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
        **{
            k: cfg[k]
            for k in ("waves_per_eu", "matrix_instr_nonkdim", "kpack")
            if k in cfg
        },
    )


def muse_moe_triton(
    hidden_states: torch.Tensor,  # [M, K] bf16 (already pooled + pre-expert-normed)
    topk_scores: torch.Tensor,  # [M, topk] fp32
    topk_ids: torch.Tensor,  # [M, topk] int32/int64 global ids
    w13: torch.Tensor,  # [E_local, 2I, K] bf16   or  [E_local, 2I, K//2] uint8 (MXFP4)
    w2: torch.Tensor,  # [E_local, H, I] bf16     or  [E_local, H, I//2] uint8 (MXFP4)
    local_expert_start: int,
    post_norm_weight: torch.Tensor | None,  # [H] stored gamma (any dtype) or None
    eps: float = 1e-8,
    cfg: dict | None = None,
    post_norm_gain: float = 0.0,  # gain center added to gamma (zero-centered gamma); 0 if not
    has_norm: bool = True,  # False -> plain weighted sum (no per-expert RMSNorm)
    w13_scale: torch.Tensor | None = None,  # MXFP4: [E_local, 2I, K//32] uint8 e8m0
    w2_scale: torch.Tensor | None = None,  # MXFP4: [E_local, H, I//32] uint8 e8m0
) -> torch.Tensor:
    M, K = hidden_states.shape
    topk = topk_ids.shape[1]
    mxfp4 = w13.dtype == torch.uint8
    E, two_i, kw = w13.shape
    I = two_i // 2
    _, H, iw = w2.shape
    if mxfp4:
        assert w13_scale is not None and w2_scale is not None, (
            "MXFP4 weights need e8m0 scales"
        )
        assert kw * 2 == K and iw * 2 == I, (
            f"packed shape mismatch: w13 {tuple(w13.shape)} w2 {tuple(w2.shape)} K={K}"
        )
        assert w13_scale.shape == (E, two_i, K // 32) and w2_scale.shape == (
            E,
            H,
            I // 32,
        ), "scale shape mismatch"
        assert w13.stride(2) == 1 and w2.stride(2) == 1, (
            "packed fp4 must be contiguous along K"
        )
        assert hidden_states.dtype == torch.bfloat16, (
            "MXFP4 path expects bf16 activations"
        )
    else:
        assert K == kw and I == iw, "shape mismatch between w13/w2/hidden"
    cfg = cfg or _pick_config(M, topk, E, mxfp4)
    assert (
        K % cfg["BLOCK_K"] == 0 and I % cfg["BLOCK_K"] == 0 and cfg["BLOCK_K"] % 32 == 0
    )
    assert I % cfg["BLOCK_N"] == 0 and H % cfg["BLOCK_N"] == 0

    assert topk_ids.shape[0] == M and topk_scores.shape == topk_ids.shape, (
        f"routing/hidden shape mismatch: hidden {tuple(hidden_states.shape)} ids {tuple(topk_ids.shape)} scores {tuple(topk_scores.shape)}"
    )
    if not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()
    ids32 = topk_ids.to(torch.int32).contiguous()
    scores32 = topk_scores.to(torch.float32).contiguous()
    sorted_ids, expert_ids, num_post = moe_align_block_size(
        ids32, E, local_expert_start, cfg["BLOCK_M"]
    )
    num_valid = M * topk
    # Decode-sized batches: the per-program K loop (K/BLOCK_K serial MFMA steps on a mostly
    # empty M tile) is latency bound, so split K across programs and reduce fp32 partials.
    split_k = (
        SPLIT_K_SMALL if (num_valid <= SPLIT_K_MAX_SLOTS and SPLIT_K_SMALL > 1) else 1
    )
    while split_k > 1 and (
        K % (split_k * cfg["BLOCK_K"]) or I % (split_k * cfg["BLOCK_K"])
    ):
        split_k //= 2
    dev = hidden_states.device

    inter = torch.empty((num_valid, I), dtype=hidden_states.dtype, device=dev)
    if split_k > 1:
        part1 = torch.empty(
            (split_k, num_valid, 2 * I), dtype=torch.float32, device=dev
        )
        _launch_gemm(
            hidden_states,
            w13,
            w13_scale,
            part1,
            sorted_ids,
            expert_ids,
            num_post,
            I,
            K,
            num_valid,
            topk,
            True,
            True,
            mxfp4,
            cfg,
            split_k,
        )
        _splitk_silu_mul_kernel[(num_valid, triton.cdiv(I, 1024))](
            part1,
            inter,
            I,
            part1.stride(0),
            part1.stride(1),
            SPLIT_K=split_k,
            BLOCK=1024,
            num_warps=4,
        )
        y = torch.empty((split_k, num_valid, H), dtype=torch.float32, device=dev)
        _launch_gemm(
            inter,
            w2,
            w2_scale,
            y,
            sorted_ids,
            expert_ids,
            num_post,
            H,
            I,
            num_valid,
            topk,
            False,
            False,
            mxfp4,
            cfg,
            split_k,
        )
        stride_yk = y.stride(0)
    else:
        _launch_gemm(
            hidden_states,
            w13,
            w13_scale,
            inter,
            sorted_ids,
            expert_ids,
            num_post,
            I,
            K,
            num_valid,
            topk,
            True,
            True,
            mxfp4,
            cfg,
        )
        y = torch.empty((num_valid, H), dtype=hidden_states.dtype, device=dev)
        _launch_gemm(
            inter,
            w2,
            w2_scale,
            y,
            sorted_ids,
            expert_ids,
            num_post,
            H,
            I,
            num_valid,
            topk,
            False,
            False,
            mxfp4,
            cfg,
        )
        stride_yk = 0

    out = torch.empty((M, H), dtype=hidden_states.dtype, device=dev)
    has_scale = has_norm and post_norm_weight is not None
    w_arg = post_norm_weight if has_scale else out  # dummy pointer when unused
    finalize_norm_weighted_sum_kernel[(M,)](
        y,
        w_arg,
        scores32,
        ids32,
        out,
        H,
        local_expert_start,
        E,
        eps,
        float(post_norm_gain),
        stride_yk,
        TOP_K=topk,
        HAS_NORM=has_norm,
        HAS_SCALE=has_scale,
        BLOCK_H=triton.next_power_of_2(H),
        SPLIT_K=split_k,
        num_warps=4,
    )
    return out


# ----------------------------------------------------------------- MXFP4 reference
_E2M1_LUT = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=torch.float32,
)


def dequant_mxfp4(w_packed: torch.Tensor, scale_e8m0: torch.Tensor) -> torch.Tensor:
    """Reference dequant (verified against the BF16 checkpoint): [..., K//2] u8 + [..., K//32] u8 -> f32 [..., K]."""
    lut = _E2M1_LUT.to(w_packed.device)
    lo = lut[(w_packed & 0xF).long()]
    hi = lut[(w_packed >> 4).long()]
    vals = torch.stack([lo, hi], dim=-1).reshape(
        *w_packed.shape[:-1], w_packed.shape[-1] * 2
    )
    scale = torch.exp2(scale_e8m0.float() - 127.0)
    return vals * scale.repeat_interleave(32, dim=-1)
