# SPDX-License-Identifier: Apache-2.0
"""W4A16 GEMM for a few activation rows (``M <= 16``: decode and speculative verify) on
gfx1151 (RDNA3.5) WMMA, in Gluon.

Weights use the HIP skinny int4 layout of ``w4a16_skinny_gemm``: ``w[n, k // 8]``
int32 with eight codes per word, zero point 8, lane ``j`` at bit ``(j // 2) * 4 +
(j % 2) * 16``; ``s[n, k // 128]`` bf16 scales, groups of 128 along K, symmetric.

The product is computed transposed, ``out.T[N, 16] = W[N, K] @ x.T[K, 16]``, with the
weights as the WMMA A operand. An RDNA3 A operand gives lane ``l`` the 16 consecutive K
values of row ``l % 16``, i.e. two consecutive words of one weight row, so each lane loads
its operand straight from global memory. ``((word >> 4 i) & 0x000F000F) | 0x43004300`` is
two exact bf16 ``128 + q`` already packed as the operand register wants them; per group
``acc += s[n] * (x . (128 + q) - 136 * sum(x))`` in fp32. Activations are staged in LDS,
weights and scales are prefetched ``DEPTH`` groups ahead, no load is masked (rows past M
repeat row M - 1), and the grid is persistent over (row block, K split) items.

On gfx1151 this reads the weights at about 165-190 GB/s for the Qwen3.5/3.8 projections at
M = 8, against 40-70 GB/s for the skinny Triton GEMM.
"""

import functools

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDWMMALayout, rdna3

from sglang.srt.utils import use_gfx1151_kernels

GROUP = gl.constexpr(128)


@gluon.jit
def _pair(w, I: gl.constexpr):
    """bf16 ``128 + q`` of K ``2 I`` and ``2 I + 1`` of every word, joined on a new axis."""
    p = ((w >> (4 * I)) & 0x000F000F) | 0x43004300
    lo = p.to(gl.int16).to(gl.bfloat16, bitcast=True)
    hi = (p >> 16).to(gl.int16).to(gl.bfloat16, bitcast=True)
    return gl.join(lo, hi)


@gluon.jit
def _a_tile(w, BN: gl.constexpr, A_L: gl.constexpr):
    """A operand ``[BN, 16]`` (bf16 ``128 + q``) from ``[BN, 2]`` consecutive words."""
    v = gl.join(gl.join(_pair(w, 0), _pair(w, 1)), gl.join(_pair(w, 2), _pair(w, 3)))
    # [BN, j, h, i0, i1] -> k = 8 j + 4 i1 + 2 i0 + h
    v = gl.reshape(gl.permute(v, [0, 1, 4, 3, 2]), [BN, 16])
    return gl.convert_layout(v, A_L, assert_trivial=True)


@gluon.jit
def _step(
    w,
    x_g,
    T: gl.constexpr,
    part,
    BN: gl.constexpr,
    A_L: gl.constexpr,
    B_L: gl.constexpr,
):
    """K slice ``T`` of a group: ``part += (128 + q) . x``."""
    b = x_g.slice(16 * T, 16, dim=1).permute([1, 0]).load(B_L)
    return rdna3.wmma(_a_tile(w, BN, A_L), b, part)


@gluon.jit
def _group(
    words,
    sc,
    x_g,
    xs,
    acc,
    BN: gl.constexpr,
    NCH: gl.constexpr,
    W: gl.constexpr,
    A_L: gl.constexpr,
    B_L: gl.constexpr,
):
    """``acc += s * (W_group @ x_group.T)`` for one 128-wide group; ``xs`` is
    ``136 * sum(x_group)`` per activation row. The eight K slices rotate over ``NCH``
    accumulators: dependent WMMAs issue about four times slower than independent ones,
    and other waves do not fill the gap."""
    # [BN, 16] -> eight [BN, 2] K slices (t = 4 t2 + 2 t1 + t0), all in registers
    w4 = gl.permute(gl.reshape(words, [BN, 2, 2, 2, 2]), [0, 4, 3, 2, 1])
    h0, h1 = gl.split(w4)
    q0, q1 = gl.split(h0)
    q2, q3 = gl.split(h1)
    s0, s1 = gl.split(q0)
    s2, s3 = gl.split(q1)
    s4, s5 = gl.split(q2)
    s6, s7 = gl.split(q3)
    p0 = gl.zeros([BN, 16], gl.float32, layout=W)
    p1 = gl.zeros([BN, 16], gl.float32, layout=W)
    p0 = _step(s0, x_g, 0, p0, BN, A_L, B_L)
    p1 = _step(s1, x_g, 1, p1, BN, A_L, B_L)
    if NCH == 4:
        p2 = gl.zeros([BN, 16], gl.float32, layout=W)
        p3 = gl.zeros([BN, 16], gl.float32, layout=W)
        p2 = _step(s2, x_g, 2, p2, BN, A_L, B_L)
        p3 = _step(s3, x_g, 3, p3, BN, A_L, B_L)
        p0 = _step(s4, x_g, 4, p0, BN, A_L, B_L)
        p1 = _step(s5, x_g, 5, p1, BN, A_L, B_L)
        p2 = _step(s6, x_g, 6, p2, BN, A_L, B_L)
        p3 = _step(s7, x_g, 7, p3, BN, A_L, B_L)
        part = (p0 + p1) + (p2 + p3)
    else:
        p0 = _step(s2, x_g, 2, p0, BN, A_L, B_L)
        p1 = _step(s3, x_g, 3, p1, BN, A_L, B_L)
        p0 = _step(s4, x_g, 4, p0, BN, A_L, B_L)
        p1 = _step(s5, x_g, 5, p1, BN, A_L, B_L)
        p0 = _step(s6, x_g, 6, p0, BN, A_L, B_L)
        p1 = _step(s7, x_g, 7, p1, BN, A_L, B_L)
        part = p0 + p1
    return acc + sc[:, None] * (part - xs[None, :])


@gluon.jit
def _fence(v):
    """``v`` through an empty side-effecting asm: memory operations cannot move across it,
    and whatever consumes the result cannot move above it."""
    return gl.inline_asm_elementwise(
        "", "=v,0", [v], dtype=v.dtype, is_pure=False, pack=1
    )


@gluon.jit
def _one(
    w,
    s,
    x_g,
    xs_g,
    J: gl.constexpr,
    gn,
    w_ptrs,
    s_ptrs,
    acc,
    BN: gl.constexpr,
    NCH: gl.constexpr,
    W: gl.constexpr,
    A_L: gl.constexpr,
    B_L: gl.constexpr,
):
    """Group ``J`` of the chunk from registers, then the load of group ``gn`` into the
    registers it just freed. ``s`` is the int32 word holding the bf16 scales of groups
    ``2 i`` (low half) and ``2 i + 1``: 16-bit loads get packed into register halves, and
    the packing move waits for the load (and so for the whole prefetch)."""
    w = _fence(w)
    s = _fence(s)
    if J % 2 == 0:
        sc = (s << 16).to(gl.float32, bitcast=True)
    else:
        sc = (s & -65536).to(gl.float32, bitcast=True)
    xs = xs_g.slice(16 * J, 16).load(gl.SliceLayout(0, W))
    acc = _group(w, sc, x_g.slice(16 * J, 16, dim=0), xs, acc, BN, NCH, W, A_L, B_L)
    return gl.load(w_ptrs + gn * 16), gl.load(s_ptrs + gn // 2), acc


@gluon.jit
def _quad(
    w0,
    w1,
    w2,
    w3,
    s0,
    s1,
    s2,
    s3,
    x_g,
    xs_g,
    c,
    g1,
    w_ptrs,
    s_ptrs,
    acc,
    BN: gl.constexpr,
    NCH: gl.constexpr,
    DEPTH: gl.constexpr,
    W: gl.constexpr,
    A_L: gl.constexpr,
    B_L: gl.constexpr,
):
    """Groups ``c .. c + 3``, each followed by the load of the group ``DEPTH`` ahead into
    the registers it freed (the tail reloads the last group). ``DEPTH == 2`` uses only
    ``w0, w1`` (and ``s0, s1``)."""
    if DEPTH == 4:
        w0, s0, acc = _one(
            w0,
            s0,
            x_g,
            xs_g,
            0,
            gl.minimum(c + 4, g1 - 1),
            w_ptrs,
            s_ptrs,
            acc,
            BN,
            NCH,
            W,
            A_L,
            B_L,
        )
        w1, s1, acc = _one(
            w1,
            s1,
            x_g,
            xs_g,
            1,
            gl.minimum(c + 5, g1 - 1),
            w_ptrs,
            s_ptrs,
            acc,
            BN,
            NCH,
            W,
            A_L,
            B_L,
        )
        w2, s2, acc = _one(
            w2,
            s2,
            x_g,
            xs_g,
            2,
            gl.minimum(c + 6, g1 - 1),
            w_ptrs,
            s_ptrs,
            acc,
            BN,
            NCH,
            W,
            A_L,
            B_L,
        )
        w3, s3, acc = _one(
            w3,
            s3,
            x_g,
            xs_g,
            3,
            gl.minimum(c + 7, g1 - 1),
            w_ptrs,
            s_ptrs,
            acc,
            BN,
            NCH,
            W,
            A_L,
            B_L,
        )
    else:
        w0, s0, acc = _one(
            w0,
            s0,
            x_g,
            xs_g,
            0,
            gl.minimum(c + 2, g1 - 1),
            w_ptrs,
            s_ptrs,
            acc,
            BN,
            NCH,
            W,
            A_L,
            B_L,
        )
        w1, s1, acc = _one(
            w1,
            s1,
            x_g,
            xs_g,
            1,
            gl.minimum(c + 3, g1 - 1),
            w_ptrs,
            s_ptrs,
            acc,
            BN,
            NCH,
            W,
            A_L,
            B_L,
        )
        w0, s0, acc = _one(
            w0,
            s0,
            x_g,
            xs_g,
            2,
            gl.minimum(c + 4, g1 - 1),
            w_ptrs,
            s_ptrs,
            acc,
            BN,
            NCH,
            W,
            A_L,
            B_L,
        )
        w1, s1, acc = _one(
            w1,
            s1,
            x_g,
            xs_g,
            3,
            gl.minimum(c + 5, g1 - 1),
            w_ptrs,
            s_ptrs,
            acc,
            BN,
            NCH,
            W,
            A_L,
            B_L,
        )
    return w0, w1, w2, w3, s0, s1, s2, s3, acc


@gluon.jit
def _stage(x, x_sm, xs_sm):
    """Store an activation chunk and ``136 * sum`` of each of its group rows."""
    x_sm.store(x)
    xs_sm.store(gl.sum(x.to(gl.float32), axis=1) * 136.0)


@gluon.jit
def _gl_w4a16_small_m_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    out_ptr,
    M,
    N,
    K: gl.constexpr,
    WS: gl.constexpr,
    BN: gl.constexpr,
    SPLIT: gl.constexpr,
    NCH: gl.constexpr,
    DEPTH: gl.constexpr,
    W: gl.constexpr,
    WL: gl.constexpr,
    XL: gl.constexpr,
    SMEM: gl.constexpr,
):
    A_L: gl.constexpr = gl.DotOperandLayout(0, W, 16)
    B_L: gl.constexpr = gl.DotOperandLayout(1, W, 16)
    G: gl.constexpr = K // GROUP
    CG: gl.constexpr = 4
    s32_ptr = s_ptr.to(gl.pointer_type(gl.int32), bitcast=True)
    rn = gl.arange(0, BN, layout=gl.SliceLayout(1, WL))
    wk = gl.arange(0, 16, layout=gl.SliceLayout(0, WL))
    sr = gl.arange(0, BN, layout=gl.SliceLayout(1, W))
    om = gl.arange(0, 16, layout=gl.SliceLayout(0, W))
    # activation chunk: CG groups x 16 rows, [CG * 16, 128]. No masks anywhere: a masked
    # load is a branch, and the wait pass then cannot count the loads in flight and
    # drains them all (weight prefetch included); rows past M repeat row M - 1
    xr = gl.arange(0, CG * 16, layout=gl.SliceLayout(1, XL))
    xk = gl.arange(0, GROUP, layout=gl.SliceLayout(0, XL))
    x_ptrs = (
        x_ptr
        + gl.minimum(xr % 16, M - 1)[:, None] * K
        + (xr // 16 * GROUP)[:, None]
        + xk[None, :]
    )
    x_sm = gl.allocate_shared_memory(gl.bfloat16, [2, CG * 16, GROUP], SMEM)
    xs_sm = gl.allocate_shared_memory(
        gl.float32, [2, CG * 16], gl.SwizzledSharedLayout(1, 1, 1, [0])
    )

    # persistent over (row block, K split) items; consecutive items share rows
    for item in range(gl.program_id(0), (N // BN) * SPLIT, gl.num_programs(0)):
        pn = item // SPLIT
        g0 = (item % SPLIT) * (G // SPLIT)
        g1 = g0 + G // SPLIT
        w_ptrs = w_ptr + (pn * BN + rn)[:, None] * WS + wk[None, :]
        sn = pn * BN + sr
        s_ptrs = s32_ptr + sn * (G // 2)

        # LDS hazards (this item's first chunk vs the previous item's reads, each chunk's
        # store vs the other buffer's reads) get their barriers from Triton's membar pass;
        # those wait for LDS only, unlike gl.barrier(), which also drains global loads
        _stage(gl.load(x_ptrs + g0 * GROUP), x_sm.index(0), xs_sm.index(0))
        w0 = gl.load(w_ptrs + g0 * 16)
        s0 = gl.load(s_ptrs + g0 // 2)
        w1 = gl.load(w_ptrs + (g0 + 1) * 16)
        s1 = gl.load(s_ptrs + g0 // 2)
        if DEPTH == 4:
            w2 = gl.load(w_ptrs + (g0 + 2) * 16)
            s2 = gl.load(s_ptrs + g0 // 2 + 1)
            w3 = gl.load(w_ptrs + (g0 + 3) * 16)
            s3 = gl.load(s_ptrs + g0 // 2 + 1)
        else:
            w2 = w0
            s2 = s0
            w3 = w1
            s3 = s1
        acc = gl.zeros([BN, 16], gl.float32, layout=W)
        for c in range(g0, g1, CG):
            bi = ((c - g0) // CG) % 2
            # the next chunk's activations go to registers first and to the other LDS
            # buffer after this chunk's math (the last chunk reloads itself)
            x_n = gl.load(x_ptrs + gl.minimum(c + CG, g1 - CG) * GROUP)
            w0, w1, w2, w3, s0, s1, s2, s3, acc = _quad(
                w0,
                w1,
                w2,
                w3,
                s0,
                s1,
                s2,
                s3,
                x_sm.index(bi),
                xs_sm.index(bi),
                c,
                g1,
                w_ptrs,
                s_ptrs,
                acc,
                BN,
                NCH,
                DEPTH,
                W,
                A_L,
                B_L,
            )
            _stage(x_n, x_sm.index(1 - bi), xs_sm.index(1 - bi))

        o_ptrs = out_ptr + om[None, :] * N + sn[:, None]
        o_mask = (om < M)[None, :]
        if SPLIT == 1:
            gl.store(o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=o_mask)
        else:
            gl.atomic_add(o_ptrs, acc, mask=o_mask, sem="relaxed")


def _bases(n: int, unit: int):
    out, b = [], unit
    while b < n * unit:
        out.append(b)
        b *= 2
    return out


@functools.lru_cache(maxsize=None)
def layouts(tn: int, num_warps: int):
    """``(W, WL, XL, SMEM)``: accumulator ``[16 * num_warps * tn, 16]`` (warps split the
    weight rows in 16-row tiles, each warp repeats ``tn`` tiles in registers), the word
    tile (lane ``l`` holds all 16 words of a group of row ``l % 16`` per tile), the
    ``[64, 128]`` activation chunk (16 bytes per thread) and its LDS layout: rows padded
    by 16 bytes, so the 16 rows a B-operand load touches land in different banks while
    the addresses stay affine (an XOR swizzle makes every K slice's address a separate
    loop-invariant register, about 70 VGPRs)."""
    bn = 16 * num_warps * tn
    W = AMDWMMALayout(
        1,
        True,
        [[b, 0] for b in _bases(num_warps, 1)],
        [[b, 0] for b in _bases(tn, num_warps)],
        [16, 16, 16],
    )
    WL = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8]]
        + [[b, 0] for b in _bases(tn, 16 * num_warps)],
        lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]],
        warp_bases=[[b, 0] for b in _bases(num_warps, 16)],
        block_bases=[],
        shape=[bn, 16],
    )
    XL = gl.BlockedLayout([1, 8], [2, 16], [num_warps, 1], [1, 0])
    SMEM = gl.PaddedSharedLayout.with_identity_for([[128, 8]], [64, 128], [1, 0])
    return W, WL, XL, SMEM


def pick_config(n: int, k: int):
    """``(tn, num_warps, split, programs per CU, accumulator chains, prefetch depth)`` by
    shape, from a config sweep on gfx1151 at M = 8. ``tn = 2`` and ``nch = 4`` spill;
    three 8-warp programs (35 KB LDS each) fill a WGP's 128 KB. Split-K stays at most 2: two
    fp32 atomic adds onto zero give the same sum in either order, so results stay
    deterministic."""
    if n >= 14336 and (k // 128) % 8 == 0:
        return 1, 8, 2, 2, 2, 4
    if k >= 16384:
        return 1, 8, 1, 2, 2, 4
    if n <= 8192 and (k // 128) % 8 == 0:
        return 1, 8, 2, 3, 2, 4
    return 1, 8, 1, 2, 2, 4


def supports(n: int, k: int) -> bool:
    """Whether ``w4a16_gemm_small_m`` takes an ``[n, k]`` weight with its default config."""
    tn, num_warps, split, _, _, _ = pick_config(n, k)
    return (
        k % 128 == 0
        and n % (16 * num_warps * tn) == 0
        and (k // 128) % (4 * split) == 0
    )


@functools.cache
def _cu_count(device: int) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


GROUP_SIZE = 128


def covered(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor) -> bool:
    """Whether ``w4a16_small_m_gemm`` takes these operands (gfx1151, 2D bf16 ``x`` with at
    most 16 rows, contiguous, a shape its config table supports)."""
    if not (use_gfx1151_kernels() and x.dim() == 2 and 0 < x.shape[0] <= 16):
        return False
    n, k8 = w.shape
    k = x.shape[1]
    return (
        x.dtype == torch.bfloat16
        and w.dtype == torch.int32
        and s.dtype == torch.bfloat16
        and k8 * 8 == k
        and s.shape == (n, k // GROUP_SIZE)
        and x.is_contiguous()
        and w.stride(1) == 1
        and s.is_contiguous()
        and supports(n, k)
    )


def w4a16_small_m_gemm(
    x, w, s, tn=None, num_warps=None, split=None, per_cu=None, nch=None, depth=None
):
    """``x @ dequant(w).T`` (bf16) for ``M <= 16`` rows; see ``covered``. The keyword
    arguments override the tuned config (for benchmarking)."""
    m, k = x.shape
    n = w.shape[0]
    d_tn, d_nw, d_split, d_per_cu, d_nch, d_depth = pick_config(n, k)
    tn, num_warps = tn or d_tn, num_warps or d_nw
    split, per_cu, nch, depth = (
        split or d_split,
        per_cu or d_per_cu,
        nch or d_nch,
        depth or d_depth,
    )
    bn = 16 * num_warps * tn
    assert m <= 16 and n % bn == 0 and (k // GROUP_SIZE) % (4 * split) == 0, (
        m,
        n,
        k,
        bn,
        split,
    )
    W, WL, XL, SMEM = layouts(tn, num_warps)
    if split == 1:
        out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    else:
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
    programs = min(n // bn * split, per_cu * _cu_count(x.device.index or 0))
    _gl_w4a16_small_m_kernel[(programs,)](
        x,
        w,
        s,
        out,
        m,
        n,
        k,
        w.stride(0),
        BN=bn,
        SPLIT=split,
        NCH=nch,
        DEPTH=depth,
        W=W,
        WL=WL,
        XL=XL,
        SMEM=SMEM,
        num_warps=num_warps,
    )
    return out if split == 1 else out.to(x.dtype)


# On gfx11, int4 weight rows read together at power-of-two byte strides land on the same
# cache channel; a 128 B row pad cuts one 8-row verify step's GEMMs by about 7% on gfx1151
# (512 B is slower). Same idea as the bf16 row pad of vLLM's cache_aliasing_pad_elems.
ROW_PAD_WORDS = 32


def pad_rows(w: torch.Tensor) -> torch.Tensor:
    """``w[n, k // 8]`` as a view of rows padded by ``ROW_PAD_WORDS`` int32 words (the pad
    holds zero-point codes and is never read). Every W4A16 kernel here and
    ``w4a16_skinny_gemm`` take the row stride from the tensor."""
    n, k8 = w.shape
    buf = torch.full(
        (n, k8 + ROW_PAD_WORDS), -2004318072, dtype=torch.int32, device=w.device
    )  # 0x88888888
    buf[:, :k8] = w
    return buf[:, :k8]


_CLIP_RATIOS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75)


def quantize_w4a16(w: torch.Tensor, rows_per_chunk: int = 8192):
    """``[N, K]`` float weight -> ``(w, s)`` in this kernel's layout: symmetric int4, group
    128, per-group clipping chosen by the reconstruction error. For load-time int4 copies of
    bf16 weights whose rounding cannot change the model's output (speculative drafters)."""
    n, k = w.shape
    assert k % GROUP_SIZE == 0
    words = torch.empty((n, k // 8), dtype=torch.int32, device=w.device)
    scales = torch.empty((n, k // GROUP_SIZE), dtype=torch.bfloat16, device=w.device)
    shifts = torch.tensor(
        [(j // 2) * 4 + (j % 2) * 16 for j in range(8)],
        dtype=torch.int64,
        device=w.device,
    )
    for r0 in range(0, n, rows_per_chunk):
        g = w[r0 : r0 + rows_per_chunk].float().view(-1, k // GROUP_SIZE, GROUP_SIZE)
        amax = g.abs().amax(-1, keepdim=True).clamp_min(1e-10)
        best_err = best_s = None
        for ratio in _CLIP_RATIOS:
            sc = (amax * ratio / 7.0).bfloat16().float()
            err = ((torch.clamp(torch.round(g / sc), -8, 7) * sc - g) ** 2).sum(
                -1, keepdim=True
            )
            if best_err is None:
                best_err, best_s = err, sc
            else:
                better = err < best_err
                best_err = torch.where(better, err, best_err)
                best_s = torch.where(better, sc, best_s)
        q = (torch.clamp(torch.round(g / best_s), -8, 7) + 8).to(torch.int64)
        packed = (q.view(q.shape[0], k // 8, 8) << shifts).sum(-1)
        packed = packed - (packed >= 2**31).to(torch.int64) * 2**32
        words[r0 : r0 + rows_per_chunk] = packed.to(torch.int32)
        scales[r0 : r0 + rows_per_chunk] = best_s.view(-1, k // GROUP_SIZE).bfloat16()
    return words, scales
