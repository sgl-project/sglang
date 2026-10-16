# SPDX-License-Identifier: Apache-2.0
"""BF16 x per-channel FP8 GEMM against (16, 16)-preshuffled weights on gfx950.

The tile loop matches aiter's gemm_a16w8_blockscale_preshuffle, but split-K
partials are reduced inside the GEMM by the last CTA of each output tile
instead of a second reduce launch.

gfx950 dispatches workgroups to its XCDs round-robin by program id, and each
XCD has a private L2. Programs are numbered so that every split of a tile has
the same id modulo _XCD_GROUP, which places all splits of a tile on one XCD
(also when a partition holds 4, 2, or 1 XCDs). Partials then stay in one L2:
each split waits for its partial store before a relaxed device-scope counter
increment, and the last arrival reads the partials back through L2. A
device-scope release would write back L2 per CTA, which costs more than the
reduce launch it replaces. The last CTA sums partials in split order, so the
result does not depend on arrival order, and it resets the counter for reuse.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_gfx95_supported, is_hip

_is_gfx95 = is_hip() and is_gfx95_supported()

_XCD_GROUP = 8


@triton.jit
def _store_scaled_tile(
    acc,
    c_ptr,
    scale_ptr,
    M,
    N,
    stride_cm,
    pid_m,
    pid_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    scale = tl.load(scale_ptr + offs_cn, mask=offs_cn < N, other=0.0)
    tl.store(
        c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :],
        (acc * scale[None, :]).to(c_ptr.type.element_ty),
        mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N),
    )


@triton.jit
def _a16w8_preshuffle_kernel(
    a_ptr,
    b_ptr,
    scale_ptr,
    c_ptr,
    partials_ptr,
    counters_ptr,
    M,
    N,
    num_tiles,
    stride_am,
    stride_bn,
    stride_cm,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    XCD_GROUP: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    pid = tl.program_id(0)
    group = pid // (XCD_GROUP * NUM_KSPLIT)
    lane = pid % (XCD_GROUP * NUM_KSPLIT)
    pid_k = lane // XCD_GROUP
    tile = group * XCD_GROUP + lane % XCD_GROUP
    if tile < num_tiles:
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        pid_m = tile // num_pid_n
        pid_n = tile % num_pid_n

        split_k: tl.constexpr = K // NUM_KSPLIT
        offs_k = pid_k * split_k + tl.arange(0, BLOCK_SIZE_K)
        offs_k_shuffle = pid_k * split_k * 16 + tl.arange(0, BLOCK_SIZE_K * 16)
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * (BLOCK_SIZE_N // 16) + tl.arange(0, BLOCK_SIZE_N // 16)) % (
            N // 16
        )
        a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :]
        b_ptrs = b_ptr + offs_bn[:, None] * stride_bn + offs_k_shuffle[None, :]

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for _ in range(split_k // BLOCK_SIZE_K):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs, cache_modifier=cache_modifier)
            b = (
                b.reshape(1, BLOCK_SIZE_N // 16, BLOCK_SIZE_K // 32, 2, 16, 16)
                .permute(0, 1, 4, 2, 3, 5)
                .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K)
                .trans(1, 0)
            )
            acc += tl.dot(a, b.to(a_ptr.type.element_ty))
            a_ptrs += BLOCK_SIZE_K
            b_ptrs += BLOCK_SIZE_K * 16

        if NUM_KSPLIT == 1:
            _store_scaled_tile(
                acc,
                c_ptr,
                scale_ptr,
                M,
                N,
                stride_cm,
                pid_m,
                pid_n,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
            )
        else:
            tile_elems: tl.constexpr = BLOCK_SIZE_M * BLOCK_SIZE_N
            offs_tile = (
                tl.arange(0, BLOCK_SIZE_M)[:, None] * BLOCK_SIZE_N
                + tl.arange(0, BLOCK_SIZE_N)[None, :]
            )
            tl.store(
                partials_ptr + (pid_k * num_tiles + tile) * tile_elems + offs_tile,
                acc,
            )
            # The partial store must reach L2 before the counter increment.
            tl.inline_asm_elementwise(
                "s_waitcnt vmcnt(0)\n v_mov_b32 $0, $1",
                "=v,v",
                [pid_k],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
            count = tl.atomic_add(counters_ptr + tile, 1, sem="relaxed", scope="gpu")
            if count == NUM_KSPLIT - 1:
                total = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                for s in range(NUM_KSPLIT):
                    total += tl.load(
                        partials_ptr + (s * num_tiles + tile) * tile_elems + offs_tile,
                        cache_modifier=".cg",
                    )
                tl.atomic_xchg(counters_ptr + tile, 0, sem="relaxed", scope="gpu")
                _store_scaled_tile(
                    total,
                    c_ptr,
                    scale_ptr,
                    M,
                    N,
                    stride_cm,
                    pid_m,
                    pid_n,
                    BLOCK_SIZE_M,
                    BLOCK_SIZE_N,
                )


def a16w8_preshuffle_num_tiles(m: int, n: int, config: dict) -> int:
    return triton.cdiv(m, config["BLOCK_SIZE_M"]) * triton.cdiv(
        n, config["BLOCK_SIZE_N"]
    )


def a16w8_preshuffle_supported(n: int, k: int, config: dict) -> bool:
    return (
        _is_gfx95
        and n % 16 == 0
        and config["BLOCK_SIZE_N"] % 16 == 0
        and k % (config["NUM_KSPLIT"] * config["BLOCK_SIZE_K"]) == 0
    )


def a16w8_preshuffle_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    config: dict,
    tile_counters: torch.Tensor,
) -> torch.Tensor:
    """Y = X @ (W * scale)^T for a 2-D BF16 X, an (N, K) FP8 weight preshuffled
    with shuffle_weight(w, (16, 16)), and one FP32 scale per weight row.

    tile_counters is an int32 buffer of at least a16w8_preshuffle_num_tiles()
    zeros, used by one stream at a time; every call leaves it zeroed.
    """
    m, k = x.shape
    n = weight.shape[0]
    num_ksplit = config["NUM_KSPLIT"]
    num_tiles = a16w8_preshuffle_num_tiles(m, n, config)
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    partials = (
        torch.empty(
            (num_ksplit, num_tiles, config["BLOCK_SIZE_M"] * config["BLOCK_SIZE_N"]),
            dtype=torch.float32,
            device=x.device,
        )
        if num_ksplit > 1
        else out
    )
    grid = (triton.cdiv(num_tiles, _XCD_GROUP) * _XCD_GROUP * num_ksplit,)
    _a16w8_preshuffle_kernel[grid](
        x,
        weight.view(n // 16, k * 16),
        weight_scale,
        out,
        partials,
        tile_counters,
        m,
        n,
        num_tiles,
        x.stride(0),
        k * 16,
        out.stride(0),
        K=k,
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        NUM_KSPLIT=num_ksplit,
        XCD_GROUP=_XCD_GROUP,
        cache_modifier=config["cache_modifier"],
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
        waves_per_eu=config["waves_per_eu"],
        matrix_instr_nonkdim=config["matrix_instr_nonkdim"],
    )
    return out
