# SPDX-License-Identifier: Apache-2.0
"""Skinny bf16 GEMM for gfx950 decode shapes, written in Gluon.

A Triton version of this kernel reached 1.29x of the hand-written FlyDSL kernel
it is meant to replace and could go no further, because the three things that
kernel does cannot be said in Triton's language:

  * it fetches only the rows the caller has, where `tl.dot` forces a 16-row
    MFMA tile and a caller with 8 rows spends half of every A fetch on padding;
  * it picks the cache policy for B's global->LDS stream, which for a weight
    read once per decode step wants both levels skipped;
  * it hands N tiles to XCDs in runs, so one XCD's B rows do not land a fixed
    large stride apart and crowd a few L2 sets.

Gluon exposes all three: `buffer_load` takes a mask and a cache policy,
`AMDMFMALayout` fixes the instruction shape, and the tile-to-XCD mapping is
ordinary index arithmetic. The structure follows AITER's gfx950 Gluon
`gemm_a8w8`, with bf16's 16x16x32 MFMA in place of the scaled fp8 one.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _band_xcd(pid, n_tiles, NUM_XCDS: gl.constexpr, XCD_BAND: gl.constexpr):
    """Hand tiles to XCDs in runs of XCD_BAND rather than one at a time.

    With the plain order the tiles one XCD sees are NUM_XCDS apart, so its B
    rows land a fixed large stride apart. The tail beyond the last whole band
    is left as identity, which keeps the remap a bijection for any tile count.
    """
    if XCD_BAND == 1:
        return pid  # identity; skip the divisions the banded form needs
    span: gl.constexpr = NUM_XCDS * XCD_BAND
    whole = (n_tiles // span) * span
    if pid < whole:
        band = pid // span
        within = pid % span
        return band * span + (within % XCD_BAND) * NUM_XCDS + within // XCD_BAND
    return pid


@gluon.jit
def _emit(acc, tile, offs_cm0, offs_cn, bias_ptr, c_ptr, partial_ptr, M, N,
          stride_cm, stride_cn, stride_ps, pid_k, BLOCK_M: gl.constexpr,
          HAS_BIAS: gl.constexpr, SPLIT_K: gl.constexpr,
          ATOMIC: gl.constexpr):
    """Write one row tile's accumulator: to `out`, or into the split accumulator.

    With a [SPLIT_K, M, N] fp32 buffer the splits cost a write and a read of
    SPLIT_K planes -- at split_k = 8 on a 3072-wide output that is 9.4 MB
    against 37.7 MB of weights, a quarter of the traffic. Adding into one plane
    atomically trades that for a single pass.
    """
    offs_cm = offs_cm0 + tile * BLOCK_M
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if ATOMIC:
        gl.amd.cdna4.buffer_atomic_add(
            ptr=partial_ptr, offsets=offs_cm[:, None] * N + offs_cn[None, :],
            value=acc, mask=mask)
    elif SPLIT_K == 1:
        if HAS_BIAS:
            acc += gl.amd.cdna4.buffer_load(
                ptr=bias_ptr, offsets=offs_cn, mask=offs_cn < N, other=0.0,
            )[None, :].to(gl.float32)
        gl.amd.cdna4.buffer_store(
            acc.to(c_ptr.type.element_ty), ptr=c_ptr,
            offsets=offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn,
            mask=mask)
    else:
        gl.amd.cdna4.buffer_store(
            acc, ptr=partial_ptr,
            offsets=pid_k * stride_ps + offs_cm[:, None] * N + offs_cn[None, :],
            mask=mask)


@gluon.jit
def _skinny_gemm_gluon_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_ps,
    partial_ptr,
    HAS_BIAS: gl.constexpr,
    M_ROWS: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    K_PER_SPLIT: gl.constexpr,
    N_KT: gl.constexpr,
    SPLIT_K: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    XCD_BAND: gl.constexpr,
    B_CPOL: gl.constexpr,
    K_ROT: gl.constexpr,
    ROWS_PER_BLOCK: gl.constexpr,
    ATOMIC: gl.constexpr,
    PREFETCH_B: gl.constexpr,
    B_TO_LDS: gl.constexpr,
    EVEN_K: gl.constexpr,
    EVEN_N: gl.constexpr,
    EVEN_M: gl.constexpr,
    NT_REPEAT: gl.constexpr,
    TILES_PER_WARP: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    # The rows one block covers: the 16-row bf16 MFMA instruction, times the
    # tiles a warp owns. M_ROWS masks the remainder down.
    BLOCK_M: gl.constexpr = 16 * TILES_PER_WARP
    MFMA_K: gl.constexpr = 32
    K_WIDTH: gl.constexpr = 8

    n_tiles = gl.cdiv(N, BLOCK_N * NT_REPEAT)
    pid_n = _band_xcd(gl.program_id(0), n_tiles, NUM_XCDS, XCD_BAND) * NT_REPEAT
    pid_k = gl.program_id(1)
    pid_m = gl.program_id(2)

    blocked_mk: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[4, 16],
        warps_per_cta=[NUM_WARPS, 1], order=[1, 0],
    )
    blocked_kn: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[8, 1], threads_per_warp=[16, 4],
        warps_per_cta=[1, NUM_WARPS], order=[0, 1],
    )
    # tiles_per_warp lets one warp own several row tiles natively, which is how
    # AITER's t48x64x64 covers a 48-row call with a 16-row instruction. That is
    # a different thing from ROWS_PER_BLOCK, which replays the MFMA once per
    # tile with its own accumulator.
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, MFMA_K], transposed=True,
        warps_per_cta=[1, NUM_WARPS],
        tiles_per_warp=[TILES_PER_WARP, 1] if TILES_PER_WARP > 1 else None,
    )
    shared_a: gl.constexpr = gl.SwizzledSharedLayout(
        vec=K_WIDTH, per_phase=1, max_phase=16, order=[1, 0])
    shared_b: gl.constexpr = gl.SwizzledSharedLayout(
        vec=K_WIDTH, per_phase=1, max_phase=16, order=[0, 1])
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mfma_layout, K_WIDTH)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mfma_layout, K_WIDTH)

    k_start = pid_k * K_PER_SPLIT
    # Rounding each split up to a whole BLOCK_K can push the last splits past
    # K entirely -- k = 6144 with split_k = 8 and BLOCK_K = 512 gives a 1024
    # slice, and 8 of those start at 7168. Those blocks have no work, and
    # without this they walk off the end of B, because EVEN_K has by then
    # dropped the bounds masks.
    if k_start >= K:
        return
    k_len = min(K_PER_SPLIT, K - k_start)

    offs_ak = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, blocked_mk))
    offs_am = pid_m * (BLOCK_M * ROWS_PER_BLOCK) + gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(1, blocked_mk))
    offs_bk = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, blocked_kn))
    # The wrap and the masks are what a bounds-checked load costs on every
    # element. These shapes are latency bound -- 3.67 MB of weights against a
    # 1.93 us empty-kernel floor at 3584x512 -- so when the block covers whole
    # tiles the predicates are dropped rather than computed and discarded.
    offs_bn_raw = pid_n * BLOCK_N + gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(0, blocked_kn))
    offs_bn = offs_bn_raw if EVEN_N else offs_bn_raw % N

    a_ptr += k_start * stride_ak
    b_ptr += k_start * stride_bk
    offs_a = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    offs_b = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # Only the rows the caller has. M_ROWS is a constexpr so the padding lanes
    # fold away at compile time instead of riding every fetch.
    a_mask = offs_am[:, None] < M_ROWS
    smem_a = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [BLOCK_M, BLOCK_K], layout=shared_a)
    smem_b = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [BLOCK_K, BLOCK_N], layout=shared_b)


    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mfma_layout)
    # A compile-time trip count when every split covers whole tiles: it drops
    # the per-block division and, more than that, lets the compiler unroll the
    # k loop instead of treating its length as unknown.
    n_kt = N_KT if N_KT > 0 else gl.cdiv(k_len, BLOCK_K)
    # Stagger where each block starts in K. Every block still visits every k
    # tile, so the sum is unchanged; what changes is that blocks stop opening
    # the same DRAM pages at the same time.
    rot = (pid_n * K_ROT) % n_kt if K_ROT > 0 else 0

    # One block owns ROWS_PER_BLOCK row tiles and fetches B once for all of
    # them. Putting every row tile on the grid instead re-reads the whole B
    # tile per tile -- three passes at m = 48, six at m = 84 -- and B is the
    # whole cost here. Widening the MFMA tile to cover the rows does not work:
    # that was measured and the register pressure cost more than the traffic
    # saved. The tile stays 16 rows; only the count of them per block changes.
    #
    # There is no software pipeline here, and that is a measured choice rather
    # than an omission. What AITER carries as `stages` was tried four ways and
    # every one of them lost: holding the next tile's A and B in registers
    # (worst cell 1.25x -> 1.34x), holding only B, a second set of LDS tiles so
    # the store does not wait on the MFMA still reading the other (slower at
    # all nine shapes it applies to), and `gl.amd.warp_pipeline`, where
    # TritonAMDGPUWarpPipeline fails the pass at one warp and at four, with the
    # index arithmetic inside a stage and outside it.
    #
    # The two cases are written out rather than sharing one body with three
    # accumulators: carrying acc1 and acc2 through the single-tile case cost it
    # 10-13% even though the constexpr guard meant nothing ever read them.
    #
    # No manual double buffering either: holding the next tile's A and B live
    # across the MFMA measured worse at every shape (worst cell 1.25x -> 1.34x).
    if ROWS_PER_BLOCK == 1:
        acc0 = acc
        if NT_REPEAT > 1:
            acc1 = acc
        # Only B is prefetched. Carrying A as well was measured and lost; A is
        # a few KB a tile against B's tens, so holding it live buys nothing and
        # the registers it occupies cost occupancy.
        b_pf = gl.amd.cdna4.buffer_load(
            ptr=b_ptr, offsets=offs_b + ((rot % n_kt) * BLOCK_K) * stride_bk,
            mask=(offs_bk[:, None] + (rot % n_kt) * BLOCK_K) < k_len,
            other=0.0, cache=B_CPOL) if PREFETCH_B else acc
        for kt in range(n_kt):
            # The rotation costs a modulo every k tile. Every tuned row uses
            # K_ROT = 0, so the unrotated walk is the one that has to be cheap.
            kk = kt * BLOCK_K if K_ROT == 0 else ((kt + rot) % n_kt) * BLOCK_K
            if PREFETCH_B:
                b = b_pf
                if kt + 1 < n_kt:
                    kn = ((kt + 1 + rot) % n_kt) * BLOCK_K
                    b_pf = gl.amd.cdna4.buffer_load(
                        ptr=b_ptr, offsets=offs_b + kn * stride_bk,
                        mask=(offs_bk[:, None] + kn) < k_len, other=0.0,
                        cache=B_CPOL)
            elif B_TO_LDS:
                # Straight from global into LDS. The register round trip the
                # plain load takes is what AITER's b_to_lds=True avoids, and
                # the asm behind it carries the same cache policy.
                if EVEN_K:
                    gl.amd.cdna4.async_copy.buffer_load_to_shared(
                        dest=smem_b, ptr=b_ptr,
                        offsets=offs_b + kk * stride_bk, cache_modifier=B_CPOL)
                else:
                    gl.amd.cdna4.async_copy.buffer_load_to_shared(
                        dest=smem_b, ptr=b_ptr,
                        offsets=offs_b + kk * stride_bk,
                        mask=(offs_bk[:, None] + kk) < k_len,
                        cache_modifier=B_CPOL)
                gl.amd.cdna4.async_copy.commit_group()
            elif EVEN_K:
                b = gl.amd.cdna4.buffer_load(
                    ptr=b_ptr, offsets=offs_b + kk * stride_bk, cache=B_CPOL)
            else:
                b = gl.amd.cdna4.buffer_load(
                    ptr=b_ptr, offsets=offs_b + kk * stride_bk,
                    mask=(offs_bk[:, None] + kk) < k_len,
                    other=0.0, cache=B_CPOL)
            if EVEN_K and EVEN_M:
                a0 = gl.amd.cdna4.buffer_load(
                    ptr=a_ptr, offsets=offs_a + kk * stride_ak)
            else:
                a0 = gl.amd.cdna4.buffer_load(
                    ptr=a_ptr, offsets=offs_a + kk * stride_ak,
                    mask=a_mask & ((offs_ak[None, :] + kk) < k_len), other=0.0)
            if not B_TO_LDS or PREFETCH_B:
                smem_b.store(b)
            smem_a.store(a0)
            gl.amd.cdna4.async_copy.wait_group(0)
            cur_a = smem_a.load(dot_a)
            acc0 = gl.amd.cdna4.mfma(cur_a, smem_b.load(dot_b), acc0)
            if NT_REPEAT > 1:
                # A is re-read once per n tile, and at K = 512 that is 5.5 MB
                # against B's 3.7 MB -- more traffic than the weights. Taking
                # NT_REPEAT tiles per block loads it once for all of them.
                b1 = gl.amd.cdna4.buffer_load(
                    ptr=b_ptr,
                    offsets=offs_b + kk * stride_bk + BLOCK_N * stride_bn,
                    mask=None if EVEN_K else (offs_bk[:, None] + kk) < k_len,
                    other=None if EVEN_K else 0.0, cache=B_CPOL)
                smem_b.store(b1)
                gl.amd.cdna4.async_copy.wait_group(0)
                acc1 = gl.amd.cdna4.mfma(cur_a, smem_b.load(dot_b), acc1)
        if NT_REPEAT == 1:
            acc1 = acc
        acc2 = acc
    else:
        acc0 = acc
        acc1 = acc
        acc2 = acc
        for kt in range(n_kt):
            # The rotation costs a modulo every k tile. Every tuned row uses
            # K_ROT = 0, so the unrotated walk is the one that has to be cheap.
            kk = kt * BLOCK_K if K_ROT == 0 else ((kt + rot) % n_kt) * BLOCK_K
            b = gl.amd.cdna4.buffer_load(
                ptr=b_ptr, offsets=offs_b + kk * stride_bk,
                mask=(offs_bk[:, None] + kk) < k_len, other=0.0, cache=B_CPOL)
            smem_b.store(b)
            gl.amd.cdna4.async_copy.wait_group(0)
            cur_b = smem_b.load(dot_b)

            a0 = gl.amd.cdna4.buffer_load(
                ptr=a_ptr, offsets=offs_a + kk * stride_ak,
                mask=a_mask & ((offs_ak[None, :] + kk) < k_len), other=0.0)
            smem_a.store(a0)
            gl.amd.cdna4.async_copy.wait_group(0)
            acc0 = gl.amd.cdna4.mfma(smem_a.load(dot_a), cur_b, acc0)

            a1 = gl.amd.cdna4.buffer_load(
                ptr=a_ptr,
                offsets=offs_a + kk * stride_ak + BLOCK_M * stride_am,
                mask=((offs_am[:, None] + BLOCK_M) < M_ROWS)
                & ((offs_ak[None, :] + kk) < k_len), other=0.0)
            smem_a.store(a1)
            gl.amd.cdna4.async_copy.wait_group(0)
            acc1 = gl.amd.cdna4.mfma(smem_a.load(dot_a), cur_b, acc1)

            if ROWS_PER_BLOCK > 2:
                a2 = gl.amd.cdna4.buffer_load(
                    ptr=a_ptr,
                    offsets=offs_a + kk * stride_ak + 2 * BLOCK_M * stride_am,
                    mask=((offs_am[:, None] + 2 * BLOCK_M) < M_ROWS)
                    & ((offs_ak[None, :] + kk) < k_len), other=0.0)
                smem_a.store(a2)
                gl.amd.cdna4.async_copy.wait_group(0)
                acc2 = gl.amd.cdna4.mfma(smem_a.load(dot_a), cur_b, acc2)

    offs_cm0 = pid_m * (BLOCK_M * ROWS_PER_BLOCK) + gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(1, mfma_layout))
    offs_cn = pid_n * BLOCK_N + gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout))

    _emit(acc0, 0, offs_cm0, offs_cn, bias_ptr, c_ptr, partial_ptr, M, N,
          stride_cm, stride_cn, stride_ps, pid_k, BLOCK_M, HAS_BIAS, SPLIT_K,
          ATOMIC)
    if ROWS_PER_BLOCK == 1 and NT_REPEAT > 1:
        _emit(acc1, 0, offs_cm0, offs_cn + BLOCK_N, bias_ptr, c_ptr, partial_ptr,
              M, N, stride_cm, stride_cn, stride_ps, pid_k, BLOCK_M, HAS_BIAS,
              SPLIT_K, ATOMIC)
    if ROWS_PER_BLOCK > 1:
        _emit(acc1, 1, offs_cm0, offs_cn, bias_ptr, c_ptr, partial_ptr, M, N,
              stride_cm, stride_cn, stride_ps, pid_k, BLOCK_M, HAS_BIAS, SPLIT_K,
              ATOMIC)
    if ROWS_PER_BLOCK > 2:
        _emit(acc2, 2, offs_cm0, offs_cn, bias_ptr, c_ptr, partial_ptr, M, N,
              stride_cm, stride_cn, stride_ps, pid_k, BLOCK_M, HAS_BIAS, SPLIT_K,
              ATOMIC)


@triton.jit
def _reduce_splits(partial_ptr, bias_ptr, out_ptr, M, N, stride_ps,
                   stride_om, stride_on, HAS_BIAS: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   SPLIT_K: tl.constexpr):
    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for s in tl.static_range(SPLIT_K):
        acc += tl.load(partial_ptr + s * stride_ps + offs_m[:, None] * N
                       + offs_n[None, :], mask=mask, other=0.0)
    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)[None, :].to(
            tl.float32)
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             acc.to(out_ptr.dtype.element_ty), mask=mask)


# Tuned on gfx950 against the FlyDSL kernel these shapes otherwise reach, with
# the weights rotating so each call pays the fetch a decode step pays. Keyed on
# (m, n, k) because the choice is not monotonic in any one of them: the same
# n x k wants split_k = 6 at m = 1 and rows_per_block = 3 at m = 84.
_TUNED: dict[tuple[int, int, int], dict] = {
    (1, 3072, 6144): dict(block_n=64, block_k=256, split_k=8, num_warps=4, xcd_band=1),
    (6, 3072, 6144): dict(block_n=64, block_k=256, split_k=8, num_warps=4, b_cpol=".cg", xcd_band=1),
    (48, 3072, 6144): dict(block_n=64, block_k=128, split_k=8, num_warps=4, tiles_per_warp=2, b_cpol=".cg", xcd_band=1),
    (84, 3072, 6144): dict(block_n=32, block_k=128, split_k=6, num_warps=1, tiles_per_warp=2, xcd_band=2),
    (1, 6144, 1536): dict(block_n=16, block_k=512, split_k=1, num_warps=2, reduce="atomic", xcd_band=1),
    (6, 6144, 1536): dict(block_n=16, block_k=512, split_k=1, num_warps=2, reduce="atomic", xcd_band=2),
    (48, 6144, 1536): dict(block_n=32, block_k=256, split_k=1, num_warps=4, reduce="atomic", xcd_band=2),
    (84, 6144, 1536): dict(block_n=32, block_k=256, split_k=1, num_warps=4, tiles_per_warp=2, reduce="atomic", xcd_band=2),
    (1, 3584, 512): dict(block_n=16, block_k=512, split_k=6, num_warps=4, xcd_band=1),
    (6, 3584, 512): dict(block_n=16, block_k=512, split_k=8, num_warps=4, reduce="atomic", xcd_band=1),
    (48, 3584, 512): dict(block_n=64, block_k=512, split_k=1, num_warps=4, reduce="atomic", xcd_band=1),
    (84, 3584, 512): dict(block_n=64, block_k=512, split_k=8, num_warps=4, tiles_per_warp=2, xcd_band=1),
}


# How many calls the kernel has taken, per shape. A server run is expected to
# hit specific shapes a specific number of times; reading this back is a check
# on whether the routing fired that does not depend on anything being logged.
CALL_COUNTS: dict[tuple[int, int, int], int] = {}


def is_tuned_shape(m: int, n: int, k: int) -> bool:
    """Whether this shape has a measured configuration.

    The caller uses it to keep untuned shapes on the incumbent path: the
    fallback is correct everywhere but was only ever compared against the
    incumbent on the shapes in the table.
    """
    return (m, n, k) in _TUNED


def default_config(m: int, n: int, k: int) -> dict:
    """The tuned configuration for a shape, or a conservative fallback.

    A shape with no row falls back to a narrow tile with no K split: that is
    never the fastest choice but it is the one that stays close to the
    incumbent across the space, and a decode step cannot afford a tuning probe.
    """
    cfg = _TUNED.get((m, n, k))
    if cfg is not None:
        return dict(cfg)
    return dict(block_n=32, block_k=256, split_k=1, num_warps=4)


def skinny_gemm_gluon(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    *,
    block_n: int = 64,
    block_k: int = 64,
    split_k: int = 1,
    xcd_band: int = 2,
    b_cpol: str = "",
    k_rot: int = 0,
    rows_per_block: int = 1,
    reduce: str = "pass",
    prefetch_b: bool = False,
    b_to_lds: bool = False,
    nt_repeat: int = 1,
    tiles_per_warp: int = 1,
    num_warps: int = 4,
) -> torch.Tensor:
    """out = a @ b.T (+ bias), for [m, k] @ [n, k].T with small m.

    Rows past the first MFMA tile go on the grid: a verify batch brings 48-84 of
    them, and one tile holds 16.
    """
    m, k = a.shape
    n = b.shape[0]
    CALL_COUNTS[(m, n, k)] = CALL_COUNTS.get((m, n, k), 0) + 1
    if out is None:
        out = torch.empty((m, n), dtype=a.dtype, device=a.device)

    grid_n = triton.cdiv(n, block_n * nt_repeat)
    rows = 16 * tiles_per_warp * rows_per_block

    # Rounding a split up to a whole BLOCK_K can leave the last splits starting
    # past K -- k = 6144 at split_k = 8 with BLOCK_K = 512 gives a 1024 slice,
    # and eight of those reach 8192. Those carry no work, so the grid gets the
    # count that does; the kernel also guards, since EVEN_K drops the masks
    # that would otherwise contain them.
    k_per_split = triton.cdiv(triton.cdiv(k, split_k), block_k) * block_k
    live_splits = triton.cdiv(k, k_per_split)
    # Every live split must cover whole BLOCK_K tiles for the masks to go, the
    # tail one included; k % k_per_split == 0 is what says the tail is full.
    even_k = k % k_per_split == 0 and k_per_split % block_k == 0
    even_n = n % (block_n * nt_repeat) == 0
    even_m = m % rows == 0

    atomic = live_splits > 1 and reduce == "atomic"
    if atomic:
        # One plane the splits add into, rather than live_splits planes and a
        # pass to sum them. Zeroing it costs one memset over m x n floats.
        partial = torch.zeros((m, n), dtype=torch.float32, device=a.device)
        ps = 0
    elif live_splits > 1:
        partial = torch.empty((live_splits, triton.cdiv(m, rows) * rows, n),
                              dtype=torch.float32, device=a.device)
        ps = partial.stride(0)
    else:
        partial, ps = a, 0

    _skinny_gemm_gluon_kernel[(grid_n, live_splits, triton.cdiv(m, rows))](
        a, b, bias, out,
        m, n, k,
        a.stride(0), a.stride(1),
        b.stride(1), b.stride(0),  # b is [n, k]; the kernel wants [k, n] strides
        out.stride(0), out.stride(1),
        ps, partial,
        HAS_BIAS=bias is not None,
        M_ROWS=m,
        BLOCK_N=block_n, BLOCK_K=block_k, K_PER_SPLIT=k_per_split,
        N_KT=(k_per_split // block_k) if even_k else 0,
        SPLIT_K=live_splits,
        NUM_XCDS=8, XCD_BAND=xcd_band, B_CPOL=b_cpol, K_ROT=k_rot,
        ROWS_PER_BLOCK=rows_per_block, ATOMIC=atomic,
        PREFETCH_B=prefetch_b, B_TO_LDS=b_to_lds,
        EVEN_K=even_k, EVEN_N=even_n, EVEN_M=even_m,
        NT_REPEAT=nt_repeat, TILES_PER_WARP=tiles_per_warp,
        NUM_WARPS=num_warps, num_warps=num_warps,
    )
    if atomic:
        if bias is not None:
            partial += bias.float()
        out.copy_(partial.to(out.dtype))
    elif live_splits > 1:
        # Its own grid: grid_n counts NT_REPEAT tiles per block, and the
        # reduction covers one tile at a time.
        _reduce_splits[(triton.cdiv(n, block_n),)](
            partial, bias, out, m, n, ps,
            out.stride(0), out.stride(1),
            HAS_BIAS=bias is not None,
            BLOCK_M=triton.next_power_of_2(m), BLOCK_N=block_n,
            SPLIT_K=live_splits)
    return out
