"""BF16 routed MoE on gfx950, built on the shared ``tlx.ops.mm`` register core.

One call runs: local-expert block alignment, the stage-1 gate/up dual-B GEMM with fused SiLU
(or an FP32 split-K slab plus reduction), the stage-2 down GEMM, and a finalize that applies the
optional per-expert post-norm and the routing-weighted sum. Launch shapes depend only on the
token count, and nothing reads device values on the host, so the op is HIP-graph capturable.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

from sglang.kernels.tlx_ops.kernels.mm.gfx950 import _register_matmul_core

# Tuned on MI355X (gfx950) under HIP graph replay. Through ~3.5k tokens every tier streams expert weights
# once, so B loads bypass L2 (.cg) to keep it for the gathered activations.
_SMALL_M = dict(BLOCK_M=16, BLOCK_N=64, BLOCK_K=128, GROUP_M=1, num_warps=4, num_stages=3, waves_per_eu=2,
                CACHE_B_CG=True)
_DECODE_N32 = dict(BLOCK_M=16, BLOCK_N=32, BLOCK_K=64, GROUP_M=1, num_warps=2, num_stages=2, CACHE_B_CG=True)
_DECODE_N64 = dict(BLOCK_M=16, BLOCK_N=64, BLOCK_K=64, GROUP_M=1, num_warps=4, num_stages=2, CACHE_B_CG=True)
_LARGE_M = dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, GROUP_M=8, num_warps=8, num_stages=2, CACHE_B_CG=True)
_LARGE_M_WIDE_N = dict(_LARGE_M, BLOCK_N=256)
# BLOCK_N=256 here needs the whole 160 KiB LDS on some compiler versions; 128 with 4 warps matches its speed in half.
_HUGE_M_GATE_UP = dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, GROUP_M=8, num_warps=4, num_stages=2, CACHE_B_CG=True)
_HUGE_M_DOWN = dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, GROUP_M=8, num_warps=4, num_stages=2, CACHE_B_CG=True)
# 2x2 quadrant inter-wave pipeline (direct-to-LDS double buffers): wins once rows per expert fill its tiles.
_QUAD_128 = dict(BLOCK_M=128, BLOCK_K=64, GROUP_M=8, XCD_CHUNK=0, QUAD=True)
_QUAD_256 = dict(BLOCK_M=256, BLOCK_K=64, GROUP_M=4, XCD_CHUNK=8, QUAD=True)
SPLIT_K_SMALL = 4
# Split-K only pays between these slot counts; below, its extra reduction launch costs more than it saves.
SPLIT_K_MIN_SLOTS = 17
SPLIT_K_MAX_SLOTS = 32
ALIGN_BLOCK = 8192
QUAD_128_ABOVE_ROWS = 28672
QUAD_256_ABOVE_ROWS = 31744
_MAX_BUFFER_BYTES = (1 << 31) - 2


def pick_config(rows: int) -> tuple[dict, dict]:
    """(gate/up, down) configs; both share BLOCK_M because one alignment feeds both GEMMs."""
    if rows <= 112:
        return _SMALL_M, _SMALL_M
    if rows <= 160:
        return _DECODE_N32, _DECODE_N32
    if rows <= 384:
        return _DECODE_N64, _DECODE_N64
    if rows <= 2048:
        return _DECODE_N32, _DECODE_N32
    if rows <= 12288:
        return _LARGE_M_WIDE_N, _LARGE_M
    if rows <= QUAD_128_ABOVE_ROWS:
        return _HUGE_M_GATE_UP, _HUGE_M_DOWN
    # A second M block per expert re-streams that expert's weights; BLOCK_M=256 keeps ~one block per expert.
    if rows <= QUAD_256_ABOVE_ROWS:
        return _QUAD_128, _QUAD_128
    return _QUAD_256, _QUAD_256


def _fit_n(cfg, N):
    """The register core has no N tail; N is only guaranteed to be a multiple of 128."""
    if cfg.get("QUAD") or N % cfg["BLOCK_N"] == 0:
        return cfg
    return dict(cfg, BLOCK_N=128)


@triton.jit
def _moe_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    sorted_ids_ptr,
    expert_ids_ptr,
    num_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_ck,
    TOP_K: tl.constexpr,
    A_ROW_DIV_TOPK: tl.constexpr,
    FUSE_SILU_MUL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    XCD_CHUNK: tl.constexpr = 0,
    CACHE_B_CG: tl.constexpr = False,
):
    pid = tl.program_id(0)
    if XCD_CHUNK > 0:
        # Keep consecutive pids (M blocks sharing an expert's B tile) on one XCD's L2.
        grid = tl.num_programs(0)
        aligned = (grid // (8 * XCD_CHUNK)) * (8 * XCD_CHUNK)
        if pid < aligned:
            local_pid = pid // 8
            pid = (local_pid // XCD_CHUNK) * 8 * XCD_CHUNK + (pid % 8) * XCD_CHUNK + (local_pid % XCD_CHUNK)
    pid_k = pid % SPLIT_K
    pid = pid // SPLIT_K
    k0 = pid_k * (K // SPLIT_K)
    num_tokens_post_padded = tl.load(num_post_padded_ptr)
    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    if pid >= num_pid_m * num_pid_n:
        return

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_token_id = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_token = tl.load(sorted_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens
    if A_ROW_DIV_TOPK:
        a_rows = offs_token // TOP_K
    else:
        a_rows = offs_token

    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_expert < 0:
        return
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    b_base = b_ptr + off_expert * stride_be
    b2_base = b_base + N * stride_bn if FUSE_SILU_MUL else b_base
    # Gather coordinates are not contiguous: only the dense MM caller adds hints.
    acc, acc2 = _register_matmul_core(
        a_ptr,
        b_base,
        b2_base,
        a_rows.to(tl.int64),
        offs_bn.to(tl.int64),
        token_mask,
        N,
        K // SPLIT_K,
        k0,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        MASK_ROWS=True,
        DUAL_B=FUSE_SILU_MUL,
        CACHE_B_CG=CACHE_B_CG,
    )

    if SPLIT_K == 1:
        if FUSE_SILU_MUL:
            acc = acc * tl.sigmoid(acc) * acc2
        c = acc.to(c_ptr.dtype.element_ty)
        c_ptrs = c_ptr + offs_token[:, None].to(tl.int64) * stride_cm + offs_bn[None, :] * stride_cn
        tl.store(c_ptrs, c, mask=token_mask[:, None])
    else:
        c_ptrs = c_ptr + pid_k * stride_ck + offs_token[:, None].to(tl.int64) * stride_cm + offs_bn[None, :] * stride_cn
        tl.store(c_ptrs, acc, mask=token_mask[:, None])
        if FUSE_SILU_MUL:
            tl.store(c_ptrs + N * stride_cn, acc2, mask=token_mask[:, None])


def _launch_gemm(a, b, c, sorted_ids, expert_ids, num_post, N, K, num_valid, topk, a_row_div_topk, fuse_silu, cfg,
                 split_k):
    if cfg.get("QUAD"):
        return _launch_quad(a, b, c, sorted_ids, expert_ids, num_post, N, K, num_valid, topk, a_row_div_topk, fuse_silu,
                            cfg)
    bm, bn = cfg["BLOCK_M"], cfg["BLOCK_N"]
    num_m_blocks = min(triton.cdiv(sorted_ids.numel(), bm), max(num_valid, 1))
    grid = (num_m_blocks * triton.cdiv(N, bn) * split_k, )
    opts = {"num_warps": cfg["num_warps"], "num_stages": cfg["num_stages"], "matrix_instr_nonkdim": 16}
    if "waves_per_eu" in cfg:
        opts["waves_per_eu"] = cfg["waves_per_eu"]
    _moe_gemm_kernel[grid](
        a,
        b,
        c,
        sorted_ids,
        expert_ids,
        num_post,
        N,
        K,
        num_valid,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0) if split_k == 1 else c.stride(1),
        c.stride(-1),
        c.stride(0) if split_k > 1 else 0,
        TOP_K=topk,
        A_ROW_DIV_TOPK=a_row_div_topk,
        FUSE_SILU_MUL=fuse_silu,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=cfg["BLOCK_K"],
        GROUP_M=cfg["GROUP_M"],
        SPLIT_K=split_k,
        XCD_CHUNK=cfg.get("XCD_CHUNK", 0),
        CACHE_B_CG=cfg.get("CACHE_B_CG", False),
        **opts,
    )


@triton.jit
def _moe_quad_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    sorted_ids_ptr,
    expert_ids_ptr,
    num_post_ptr,
    N,
    K: tl.constexpr,
    num_valid,
    stride_am,
    stride_be,
    stride_bn,
    stride_cm,
    TOP_K: tl.constexpr,
    A_ROW_DIV_TOPK: tl.constexpr,
    DUAL_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    GROUP_M: tl.constexpr,
    XCD_CHUNK: tl.constexpr,
):
    BLOCK_K: tl.constexpr = 64
    HALF_M: tl.constexpr = BLOCK_M // 2
    HALF_N: tl.constexpr = 128
    tl.static_assert(K % (2 * BLOCK_K) == 0)
    n_steps: tl.constexpr = K // BLOCK_K

    pid = tl.program_id(0)
    if XCD_CHUNK > 0:
        grid = tl.num_programs(0)
        aligned = (grid // (8 * XCD_CHUNK)) * (8 * XCD_CHUNK)
        if pid < aligned:
            local_pid = pid // 8
            pid = (local_pid // XCD_CHUNK) * 8 * XCD_CHUNK + (pid % 8) * XCD_CHUNK + (local_pid % XCD_CHUNK)
    num_pid_m = tl.cdiv(tl.load(num_post_ptr), BLOCK_M)
    num_pid_n = N // HALF_N if DUAL_B else N // (2 * HALF_N)
    if pid >= num_pid_m * num_pid_n:
        return
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    expert = tl.load(expert_ids_ptr + pid_m)
    if expert < 0:
        return

    # Bank-conflict-free padded-shared bases for K-contiguous [rows, 64] half-tiles (as in tlx.ops.mm): K bits,
    # then the row bits >= 16, then the low row bits.
    b_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [(512, 16)],
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0]],
        [HALF_N, BLOCK_K])
    if HALF_M == 128:
        a_layout: tl.constexpr = b_layout
    else:
        a_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
            [(512, 16)],
            [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [1, 0], [2, 0], [4, 0], [8, 0]],
            [HALF_M, BLOCK_K])
    smem_a_top = tlx.local_alloc((HALF_M, BLOCK_K), tlx.dtype_of(a_ptr), 2, layout=a_layout)
    smem_a_bot = tlx.local_alloc((HALF_M, BLOCK_K), tlx.dtype_of(a_ptr), 2, layout=a_layout)
    smem_b_left = tlx.local_alloc((HALF_N, BLOCK_K), tlx.dtype_of(b_ptr), 2, layout=b_layout)
    smem_b_right = tlx.local_alloc((HALF_N, BLOCK_K), tlx.dtype_of(b_ptr), 2, layout=b_layout)

    tok_top = tl.load(sorted_ids_ptr + pid_m * BLOCK_M + tl.arange(0, HALF_M))
    tok_bot = tl.load(sorted_ids_ptr + pid_m * BLOCK_M + HALF_M + tl.arange(0, HALF_M))
    valid_top = tok_top < num_valid
    valid_bot = tok_bot < num_valid
    if A_ROW_DIV_TOPK:
        rows_top = tl.where(valid_top, tok_top // TOP_K, 0)
        rows_bot = tl.where(valid_bot, tok_bot // TOP_K, 0)
    else:
        rows_top = tl.where(valid_top, tok_top, 0)
        rows_bot = tl.where(valid_bot, tok_bot, 0)

    stride_am = tl.multiple_of(stride_am, 8)
    stride_bn = tl.multiple_of(stride_bn, 8)
    if DUAL_B:
        n_left = pid_n * HALF_N
        n_right = N + pid_n * HALF_N
    else:
        n_left = pid_n * 2 * HALF_N
        n_right = n_left + HALF_N
    b_expert = b_ptr + expert.to(tl.int64) * stride_be
    b_left_ptr = tl.multiple_of(b_expert + n_left.to(tl.int64) * stride_bn, 16)
    b_right_ptr = tl.multiple_of(b_expert + n_right.to(tl.int64) * stride_bn, 16)

    offs_k = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_K), BLOCK_K), BLOCK_K)
    a_top_off = rows_top[:, None] * stride_am + offs_k[None, :]
    a_bot_off = rows_bot[:, None] * stride_am + offs_k[None, :]
    b_off = tl.arange(0, HALF_N)[:, None] * stride_bn + offs_k[None, :]
    kb1: tl.constexpr = BLOCK_K
    ka = tl.zeros([], dtype=tl.int32)

    acc_tl = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)
    acc_bl = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)
    acc_tr = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)
    acc_br = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)

    tlx.buffer_load_to_local(smem_b_left[0], b_left_ptr, b_off + ka, contiguity=8)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_a_top[0], a_ptr, a_top_off + ka, contiguity=8)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_a_bot[0], a_ptr, a_bot_off + ka, contiguity=8)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_b_right[0], b_right_ptr, b_off + ka, contiguity=8)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_b_left[1], b_left_ptr, b_off + (ka + kb1), contiguity=8)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_a_top[1], a_ptr, a_top_off + (ka + kb1), contiguity=8)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_a_bot[1], a_ptr, a_bot_off + (ka + kb1), contiguity=8)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_b_right[1], b_right_ptr, b_off + (ka + kb1), contiguity=8)
    tlx.async_load_commit_group()
    ka += BLOCK_K * 2

    tlx.async_load_wait_group(6)
    b_left = tlx.local_load(tlx.local_trans(smem_b_left[0]), relaxed=True)
    a_top = tlx.local_load(smem_a_top[0], relaxed=True)

    for k in tl.range(0, n_steps - 2, 2, num_stages=1):
        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_tl = tl.dot(a_top, b_left, acc_tl, allow_tf32=False)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_bot = tlx.local_load(smem_a_bot[0], relaxed=True)
            tlx.buffer_load_to_local(smem_b_left[0], b_left_ptr, b_off + ka, contiguity=8)
            tlx.async_load_commit_group()
        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_bl = tl.dot(a_bot, b_left, acc_bl, allow_tf32=False)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_right = tlx.local_load(tlx.local_trans(smem_b_right[0]), relaxed=True)
            tlx.buffer_load_to_local(smem_a_top[0], a_ptr, a_top_off + ka, contiguity=8)
            tlx.async_load_commit_group()
        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_tr = tl.dot(a_top, b_right, acc_tr, allow_tf32=False)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_left = tlx.local_load(tlx.local_trans(smem_b_left[1]), relaxed=True)
            tlx.buffer_load_to_local(smem_a_bot[0], a_ptr, a_bot_off + ka, contiguity=8)
            tlx.async_load_commit_group()
        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_br = tl.dot(a_bot, b_right, acc_br, allow_tf32=False)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_top = tlx.local_load(smem_a_top[1], relaxed=True)
            tlx.buffer_load_to_local(smem_b_right[0], b_right_ptr, b_off + ka, contiguity=8)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_tl = tl.dot(a_top, b_left, acc_tl, allow_tf32=False)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_bot = tlx.local_load(smem_a_bot[1], relaxed=True)
            tlx.buffer_load_to_local(smem_b_left[1], b_left_ptr, b_off + (ka + kb1), contiguity=8)
            tlx.async_load_commit_group()
        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_bl = tl.dot(a_bot, b_left, acc_bl, allow_tf32=False)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_right = tlx.local_load(tlx.local_trans(smem_b_right[1]), relaxed=True)
            tlx.buffer_load_to_local(smem_a_top[1], a_ptr, a_top_off + (ka + kb1), contiguity=8)
            tlx.async_load_commit_group()
        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_tr = tl.dot(a_top, b_right, acc_tr, allow_tf32=False)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_left = tlx.local_load(tlx.local_trans(smem_b_left[0]), relaxed=True)
            tlx.buffer_load_to_local(smem_a_bot[1], a_ptr, a_bot_off + (ka + kb1), contiguity=8)
            tlx.async_load_commit_group()
        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_br = tl.dot(a_bot, b_right, acc_br, allow_tf32=False)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_top = tlx.local_load(smem_a_top[0], relaxed=True)
            tlx.buffer_load_to_local(smem_b_right[1], b_right_ptr, b_off + (ka + kb1), contiguity=8)
            tlx.async_load_commit_group()
            ka += BLOCK_K * 2

    acc_tl = tl.dot(a_top, b_left, acc_tl, allow_tf32=False)
    tlx.async_load_wait_group(5)
    a_bot = tlx.local_load(tlx.local_view(smem_a_bot, 0), relaxed=True)
    acc_bl = tl.dot(a_bot, b_left, acc_bl, allow_tf32=False)
    tlx.async_load_wait_group(4)
    b_right = tlx.local_load(tlx.local_trans(tlx.local_view(smem_b_right, 0)), relaxed=True)
    acc_tr = tl.dot(a_top, b_right, acc_tr, allow_tf32=False)
    tlx.async_load_wait_group(3)
    b_left = tlx.local_load(tlx.local_trans(tlx.local_view(smem_b_left, 1)), relaxed=True)
    acc_br = tl.dot(a_bot, b_right, acc_br, allow_tf32=False)
    tlx.async_load_wait_group(2)
    a_top = tlx.local_load(tlx.local_view(smem_a_top, 1), relaxed=True)
    acc_tl = tl.dot(a_top, b_left, acc_tl, allow_tf32=False)
    tlx.async_load_wait_group(1)
    a_bot = tlx.local_load(tlx.local_view(smem_a_bot, 1), relaxed=True)
    acc_bl = tl.dot(a_bot, b_left, acc_bl, allow_tf32=False)
    tlx.async_load_wait_group(0)
    b_right = tlx.local_load(tlx.local_trans(tlx.local_view(smem_b_right, 1)), relaxed=True)
    acc_tr = tl.dot(a_top, b_right, acc_tr, allow_tf32=False)
    acc_br = tl.dot(a_bot, b_right, acc_br, allow_tf32=False)

    c_top = c_ptr + tok_top[:, None].to(tl.int64) * stride_cm
    c_bot = c_ptr + tok_bot[:, None].to(tl.int64) * stride_cm
    if DUAL_B:
        offs_cn = pid_n * HALF_N + tl.arange(0, HALF_N)
        h_top = acc_tl * tl.sigmoid(acc_tl) * acc_tr
        h_bot = acc_bl * tl.sigmoid(acc_bl) * acc_br
        tl.store(c_top + offs_cn[None, :], h_top.to(c_ptr.dtype.element_ty), mask=valid_top[:, None])
        tl.store(c_bot + offs_cn[None, :], h_bot.to(c_ptr.dtype.element_ty), mask=valid_bot[:, None])
    else:
        offs_l = n_left + tl.arange(0, HALF_N)
        offs_r = n_right + tl.arange(0, HALF_N)
        tl.store(c_top + offs_l[None, :], acc_tl.to(c_ptr.dtype.element_ty), mask=valid_top[:, None])
        tl.store(c_bot + offs_l[None, :], acc_bl.to(c_ptr.dtype.element_ty), mask=valid_bot[:, None])
        tl.store(c_top + offs_r[None, :], acc_tr.to(c_ptr.dtype.element_ty), mask=valid_top[:, None])
        tl.store(c_bot + offs_r[None, :], acc_br.to(c_ptr.dtype.element_ty), mask=valid_bot[:, None])


def _quad_fits(a_rows, a_row_stride, b, N, K, dual_b):
    """Direct-to-LDS buffer loads use i32 offsets: A's rows and one expert's B must span < 2 GiB."""
    n_tile = 128 if dual_b else 256
    return (K % 128 == 0 and N % n_tile == 0 and a_row_stride % 8 == 0 and b.stride(1) % 8 == 0 and b.stride(2) == 1
            and a_rows * a_row_stride * 2 <= _MAX_BUFFER_BYTES and b.stride(0) * 2 <= _MAX_BUFFER_BYTES)


def _launch_quad(a, b, c, sorted_ids, expert_ids, num_post, N, K, num_valid, topk, a_row_div_topk, dual_b, cfg):
    bm = cfg["BLOCK_M"]
    num_m_blocks = min(triton.cdiv(sorted_ids.numel(), bm), max(num_valid, 1))
    num_n = N // 128 if dual_b else N // 256
    # Post-RA scheduling breaks the inter-wave MFMA/memory interleave.
    _moe_quad_kernel[(num_m_blocks * num_n, )](a, b, c, sorted_ids, expert_ids, num_post, N, K, num_valid, a.stride(0),
                                               b.stride(0), b.stride(1), c.stride(0), TOP_K=topk,
                                               A_ROW_DIV_TOPK=a_row_div_topk, DUAL_B=dual_b, BLOCK_M=bm,
                                               GROUP_M=cfg["GROUP_M"], XCD_CHUNK=cfg["XCD_CHUNK"], num_warps=8,
                                               num_stages=1, matrix_instr_nonkdim=16,
                                               llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"), ),
                                               llvm_cl_opts=(("enable-post-misched", "false"), ))


@triton.jit
def _splitk_silu_mul_kernel(part_ptr, out_ptr, I, stride_pk, stride_pm, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    """inter[slot, :] = silu(sum_k gate_k) * sum_k up_k over the SPLIT_K fp32 partial slabs."""
    slot = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < I
    base = part_ptr + slot * stride_pm + offs
    gate = tl.zeros((BLOCK, ), dtype=tl.float32)
    up = tl.zeros((BLOCK, ), dtype=tl.float32)
    for k in tl.static_range(SPLIT_K):
        gate += tl.load(base + k * stride_pk, mask=mask, other=0.0)
        up += tl.load(base + k * stride_pk + I, mask=mask, other=0.0)
    y = gate * tl.sigmoid(gate) * up
    tl.store(out_ptr + slot * I + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _finalize_kernel(
    y_ptr,  # [T*TOP_K, H] per-slot expert outputs (rows of non-local slots are unread)
    w_ptr,  # [H] post-norm gamma; effective scale = w + gain, rounded to the weight dtype
    scores_ptr,  # [T, TOP_K] fp32 routing weights
    ids_ptr,  # [T, TOP_K] int32 or int64 global expert ids
    out_ptr,  # [T, H]
    H,
    local_start,
    num_local,
    eps,
    gain,
    stride_yk,  # SPLIT_K > 1: stride between fp32 K-partial slabs of y
    TOP_K: tl.constexpr,
    TOP_K_BLOCK: tl.constexpr,
    HAS_NORM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    m = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H
    # All TOP_K rows are loaded at once so the loads overlap; same rounding points as one-at-a-time.
    slots = tl.arange(0, TOP_K_BLOCK)
    slot_ok = slots < TOP_K
    e = tl.load(ids_ptr + m * TOP_K + slots, mask=slot_ok, other=-1)
    is_local = slot_ok & (e >= local_start) & (e < local_start + num_local)
    sc = tl.load(scores_ptr + m * TOP_K + slots, mask=slot_ok, other=0.0).to(tl.float32)
    rows = (m * TOP_K + slots).to(tl.int64)
    ptrs = y_ptr + rows[:, None] * H + offs[None, :]
    ymask = is_local[:, None] & mask[None, :]
    if SPLIT_K == 1:
        y = tl.load(ptrs, mask=ymask, other=0.0).to(tl.float32)
    else:
        y = tl.zeros((TOP_K_BLOCK, BLOCK_H), dtype=tl.float32)
        for k in tl.static_range(SPLIT_K):
            y += tl.load(ptrs + k * stride_yk, mask=ymask, other=0.0)
        # The expert output is BF16 on the unsplit path; round identically.
        y = y.to(tl.bfloat16).to(tl.float32)
    if HAS_NORM:
        w = tl.load(w_ptr + offs, mask=mask, other=0.0)
        scale = (w.to(tl.float32) + gain).to(w_ptr.dtype.element_ty).to(tl.float32)
        y = (y * scale[None, :]).to(tl.bfloat16).to(tl.float32)
        inv = 1.0 / tl.sqrt(tl.sum(y * y, axis=1) / H + eps)
        wgt = tl.where(is_local, sc * inv, 0.0)
    else:
        wgt = tl.where(is_local, sc, 0.0)
    acc = tl.sum(y * wgt[:, None], axis=0)
    tl.store(out_ptr + m.to(tl.int64) * H + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _align_kernel(
    ids_ptr,
    sorted_ptr,
    expert_ptr,
    num_post_ptr,
    n,
    local_start,
    cap,
    num_blocks,
    E: tl.constexpr,
    E_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Program e: count every local expert, then write expert e's slots (stable order) into its
    BLOCK_M-padded segment of sorted_ids and e into its expert_ids blocks. Padding slots hold n.
    Program 0 also writes num_post and fills the tail past it (sorted_ids = n, expert_ids = -1)."""
    e = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    experts = tl.arange(0, E_BLOCK)
    counts = tl.zeros((E_BLOCK, ), dtype=tl.int32)
    for c in range(0, n, BLOCK):
        slot = c + offs
        local = tl.load(ids_ptr + slot, mask=slot < n, other=-1) - local_start
        # Bin E (< E_BLOCK) collects non-local and out-of-range slots. Range-check before narrowing int64 ids.
        counts += tl.histogram(tl.where((slot < n) & (local >= 0) & (local < E), local, E).to(tl.int32), E_BLOCK)
    padded = tl.where(experts < E, (counts + BLOCK_M - 1) // BLOCK_M * BLOCK_M, 0)
    start = tl.sum(tl.where(experts < e, padded, 0), axis=0)
    cnt = tl.sum(tl.where(experts == e, counts, 0), axis=0)
    nb = (cnt + BLOCK_M - 1) // BLOCK_M

    seen = 0
    for c in range(0, n, BLOCK):
        slot = c + offs
        local = tl.load(ids_ptr + slot, mask=slot < n, other=-1) - local_start
        hit = (slot < n) & (local == e)
        rank = seen + tl.cumsum(hit.to(tl.int32), axis=0) - 1
        tl.store(sorted_ptr + start + rank, slot.to(tl.int32), mask=hit)
        seen += tl.sum(hit.to(tl.int32), axis=0)
    pad = cnt + tl.arange(0, BLOCK_M)
    tl.store(sorted_ptr + start + pad, tl.full((BLOCK_M, ), 0, tl.int32) + n, mask=pad < nb * BLOCK_M)
    for b in range(0, nb, BLOCK):
        bo = b + offs
        tl.store(expert_ptr + start // BLOCK_M + bo, tl.full((BLOCK, ), 0, tl.int32) + e, mask=bo < nb)

    if e == 0:
        total = tl.sum(padded, axis=0)
        tl.store(num_post_ptr, total)
        for c in range(total, cap, BLOCK):
            po = c + offs
            tl.store(sorted_ptr + po, tl.full((BLOCK, ), 0, tl.int32) + n, mask=po < cap)
        for b in range(total // BLOCK_M, num_blocks, BLOCK):
            bo = b + offs
            tl.store(expert_ptr + bo, tl.full((BLOCK, ), -1, tl.int32), mask=bo < num_blocks)


def _align(topk_ids, num_local, local_start, block_m):
    """Sort routed slots by local expert into BLOCK_M-padded blocks. One launch, no host sync."""
    n = topk_ids.numel()
    dev = topk_ids.device
    cap = triton.cdiv(n + num_local * (block_m - 1), block_m) * block_m
    num_blocks = cap // block_m
    sorted_ids = torch.empty(cap, dtype=torch.int32, device=dev)
    expert_ids = torch.empty(num_blocks, dtype=torch.int32, device=dev)
    num_post = torch.empty(1, dtype=torch.int32, device=dev)
    align_block = min(ALIGN_BLOCK, max(64, triton.next_power_of_2(n)))
    _align_kernel[(num_local, )](
        topk_ids.reshape(-1),
        sorted_ids,
        expert_ids,
        num_post,
        n,
        local_start,
        cap,
        num_blocks,
        E=num_local,
        E_BLOCK=triton.next_power_of_2(num_local + 1),
        BLOCK_M=block_m,
        BLOCK=align_block,
        num_warps=8 if align_block >= 2048 else 4,
    )
    return sorted_ids, expert_ids, num_post


def _split_k(num_valid, K, I, block_k):
    split_k = SPLIT_K_SMALL if SPLIT_K_MIN_SLOTS <= num_valid <= SPLIT_K_MAX_SLOTS else 1
    while split_k > 1 and (K % (split_k * block_k) or I % (split_k * block_k)):
        split_k //= 2
    return split_k


def moe(x, topk_weights, topk_ids, w13, w2, expert_start, post_norm_weight, post_norm_eps, post_norm_gain, out):
    T, K = x.shape
    E, two_i, _ = w13.shape
    I = two_i // 2
    H = w2.shape[1]
    topk = topk_ids.shape[1]
    num_valid = T * topk
    dev = x.device
    if out is None:
        out = torch.empty((T, H), dtype=torch.bfloat16, device=dev)
    if T == 0:
        return out

    cfg, cfg_down = pick_config(num_valid)
    if cfg.get("QUAD") and (x.stride(1) != 1 or x.stride(0) % 8):
        # Only the quad engine needs unit-stride, 16-byte-aligned rows; copying is cheaper than leaving it.
        x = x.contiguous()
    if cfg.get("QUAD") and not (_quad_fits(T, x.stride(0), w13, I, K, True)
                                and _quad_fits(num_valid, I, w2, H, I, False)):
        cfg, cfg_down = _HUGE_M_GATE_UP, _HUGE_M_DOWN
    cfg, cfg_down = _fit_n(cfg, I), _fit_n(cfg_down, H)
    ids = topk_ids.contiguous()
    scores32 = topk_weights.to(torch.float32).contiguous()
    sorted_ids, expert_ids, num_post = _align(ids, E, expert_start, cfg["BLOCK_M"])
    split_k = _split_k(num_valid, K, I, cfg["BLOCK_K"])

    inter = torch.empty((num_valid, I), dtype=torch.bfloat16, device=dev)
    if split_k > 1:
        gate_up = torch.empty((split_k, num_valid, 2 * I), dtype=torch.float32, device=dev)
    else:
        gate_up = inter
    _launch_gemm(x, w13, gate_up, sorted_ids, expert_ids, num_post, I, K, num_valid, topk, True, True, cfg, split_k)
    if split_k > 1:
        _splitk_silu_mul_kernel[(num_valid, triton.cdiv(I, 1024))](
            gate_up,
            inter,
            I,
            gate_up.stride(0),
            gate_up.stride(1),
            SPLIT_K=split_k,
            BLOCK=1024,
            num_warps=4,
        )

    if split_k > 1:
        y = torch.empty((split_k, num_valid, H), dtype=torch.float32, device=dev)
    else:
        y = torch.empty((num_valid, H), dtype=torch.bfloat16, device=dev)
    _launch_gemm(inter, w2, y, sorted_ids, expert_ids, num_post, H, I, num_valid, topk, False, False, cfg_down, split_k)

    has_norm = post_norm_weight is not None
    _finalize_kernel[(T, )](
        y,
        post_norm_weight.contiguous() if has_norm else out,
        scores32,
        ids,
        out,
        H,
        expert_start,
        E,
        float(post_norm_eps),
        float(post_norm_gain),
        y.stride(0) if split_k > 1 else 0,
        TOP_K=topk,
        TOP_K_BLOCK=triton.next_power_of_2(topk),
        HAS_NORM=has_norm,
        BLOCK_H=triton.next_power_of_2(H),
        SPLIT_K=split_k,
        num_warps=4,
    )
    return out
