# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.
"""
MXFP8 dense GEMM for gfx950 (MI355X), written in FlyDSL.

    C[M, N] (bf16) = A[M, K] @ B[N, K]^T

A is row-major fp8 e4m3. B is fp8 e4m3, row-major or preshuffled by aiter's
``shuffle_weight(b, (16, 16))``. Each has one ue8m0 scale per 32 consecutive K elements,
passed in the layout of ``mxfp8_scale_layout``. The pipeline follows aiter's 8-wave a8w8
FP8 kernel.

Work split. A workgroup of 8 waves (2 rows x 4 columns) computes a block_m x 256 tile of
C. A is split in halves A0/A1 along M, B in halves B0/B1 along N, giving four quadrants
(A0,B0) (A0,B1) (A1,B0) (A1,B1). Each wave owns the same (block_m/4) x 32 piece of every
quadrant: (block_m/64) x 2 MFMA tiles of 16 x 16.

K loop. K advances 128 per step. Each half has two LDS buffers, K-tile t in buffer t % 2,
filled by a DMA (global -> LDS, no registers) that all 8 waves share.

LDS layout. A DMA instruction fills 1 KB of a buffer. A row-major operand is stored
[row][16-byte K-chunk], chunk c of row x at c ^ (x // 2) % 8 so the 16 rows of an MFMA
operand hit different banks.

Wave rows. Wave row 1 runs one barrier behind row 0, so one row's MFMAs overlap the
other row's loads. A barrier then only proves the other row reached its previous barrier.

Epilogue. B is the MFMA's first operand, so a lane accumulates 4 consecutive columns of
one row of C; the tile is staged through the idle LDS.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
import triton
import triton.language as tl
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec

BLOCK_N, BLOCK_K, NUM_WAVES = 256, 128, 8
GROUP_STEPS = 4  # K steps per scale word, one byte each


def _ceil_div(a, b):
    return (a + b - 1) // b


def _swizzle(row, col):
    """
    LDS byte offset of the 16-byte chunk at byte ``col`` of ``row``. Avoid an MFMA operand
    hit different banks.
    """
    return col ^ ((row // 2 % 8) * 16)


def _buffer(tensor):
    """
    Element-indexed buffer view of a flat tensor: loads past its end return 0 and stores past
    it are dropped.
    """
    return fx.logical_divide(
        fx.rocdl.make_buffer_tensor(tensor, max_size=False), fx.make_layout(1, 1)
    )


def _lds(addr, dtype, n):
    """
    View of ``n`` elements of ``dtype`` at LDS byte address ``addr``.
    """
    ptr = fx.inttoptr(fx.PointerType.get(dtype.ir_type, 2, n * dtype.width // 8), addr)
    return fx.make_view(ptr, fx.make_layout(n, 1))


def _opaque(x):
    """
    ``x``, hidden from constant folding: constants added later then fit ds_read's 16-bit
    offset instead of each (buffer, tile) costing a VGPR once a base exceeds 64 KB.
    """
    v = fx.Int32(x).ir_value()
    return fx.Int32(llvm.inline_asm(v.type, [v], "", "=v,0", has_side_effects=False))


def _barrier(vmcnt=None, lds=False):
    """
    s_barrier once this wave has at most ``vmcnt`` VMEM loads in flight and, with
    ``lds``, no LDS access in flight. Also a compiler fence for the async DMA.
    """
    waits = []
    if vmcnt is not None:
        waits.append(f"vmcnt({vmcnt})")
    if lds:
        waits.append("lgkmcnt(0)")
    asm = (f"s_waitcnt {' '.join(waits)}\n" if waits else "") + "s_barrier"
    llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string=asm,
        constraints="~{memory}",
        has_side_effects=True,
    )


def _workgroup_tile(n_tiles_m, n_tiles_n, band):
    """
    (tile_m, tile_n) of this workgroup. The hardware deals workgroups round-robin to
    the 8 XCDs; remap them so each XCD walks ``band`` tile rows and the band shares L2.
    """
    wgid = fx.block_idx.x
    if const_expr(band == 0):
        return wgid // n_tiles_n, wgid % n_tiles_n
    num_wg = n_tiles_m * n_tiles_n
    xcd, slot = wgid % 8, wgid // 8
    wgid = xcd * (num_wg // 8) + fx.min(xcd, num_wg % 8) + slot
    first_m, in_band = wgid // (band * n_tiles_n) * band, wgid % (band * n_tiles_n)
    band_rows = fx.min(n_tiles_m - first_m, band)
    return first_m + in_band % band_rows, in_band // band_rows


@functools.lru_cache
def _num_cus(device):
    return torch.cuda.get_device_properties(device).multi_processor_count


def _default_block_m(m, n, device):
    """
    The tile height needing fewer rounds of workgroups over the CUs; a 128-row tile
    costs ~0.6 of a 256-row one (fits every shape and M measured on MI355X).
    """
    rounds = {
        bm: _ceil_div(_ceil_div(m, bm) * _ceil_div(n, BLOCK_N), _num_cus(device))
        for bm in (128, 256)
    }
    return 128 if 0.6 * rounds[128] < rounds[256] else 256


def _build(K, block_m, xcd_swizzle, b_shuffled):
    k_tiles = K // BLOCK_K
    n_groups = _ceil_div(
        k_tiles, GROUP_STEPS
    )  # 512-K scale groups; one scale word per group per lane
    rows = {"A": block_m // 2, "B": BLOCK_N // 2}  # rows of a half
    tiles = {"A": block_m // 64, "B": 2}  # MFMA tiles per wave per half
    n_dma = {
        op: rows[op] // 64 for op in "AB"
    }  # DMA instructions per wave per half; 8 rows each
    # VMEM loads that may still be in flight at a step's wait: the DMAs of B0, A0 and B1
    # for two K-tiles ahead. Loads complete in order, so everything older has landed.
    in_flight = 2 * n_dma["B"] + n_dma["A"]

    # LDS: per operand a region (<= 64 KB) of 4 buffers, [half 0 | half 1] x [even | odd K-tile];
    # afterwards the epilogue reuses all of it for the C tile
    buf_bytes = {op: rows[op] * BLOCK_K for op in "AB"}
    region = {"A": 0, "B": 4 * buf_bytes["A"]}

    def buf(op, h, kt):
        return (2 * h + kt % 2) * buf_bytes[op]

    @fx.struct
    class Lds:
        data: fx.Array[
            fx.Int8,
            max(4 * (buf_bytes["A"] + buf_bytes["B"]), block_m * BLOCK_N * 2),
            16,
        ]

    # FlyDSL's compile cache keys on the scalar values the kernel closes over, so the kernel
    # must read the build options (K, block_m, xcd_swizzle, b_shuffled) themselves, not
    # containers derived from them.
    layout = "_bshuf" if b_shuffled else ""

    @flyc.kernel(
        name=f"flydsl_mxfp8_gemm{layout}_{block_m}x{BLOCK_N}_x{xcd_swizzle}_k{K}",
        known_block_size=[NUM_WAVES * 64, 1, 1],
    )
    def kernel(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        m: fx.Int32,
        n: fx.Int32,
    ):
        lane, wave = fx.thread_idx.x % 64, fx.thread_idx.x // 64
        wave_row, wave_col = wave // 4, wave % 4
        tile_m, tile_n = _workgroup_tile(
            _ceil_div(m, block_m), _ceil_div(n, BLOCK_N), xcd_swizzle
        )
        row0 = {"A": tile_m * block_m, "B": tile_n * BLOCK_N}
        wave_tile = {
            "A": wave_row * tiles["A"],
            "B": wave_col * tiles["B"],
        }  # first MFMA tile in a half
        lds = fx.Int32(fx.ptrtoint(fx.SharedAllocator().allocate(Lds).peek().data.ptr))

        # global -> LDS. Instruction i of wave w fills KB 8i + w of the buffer. Row-major:
        # lane fetches row 64i + 8w + lane // 8, the chunk the swizzle puts at position
        # lane % 8. Shuffled: the instruction copies half w % 2 of 16-row group 4i + w // 2;
        # a K-tile is 2 KB further.
        gmem = {"A": _buffer(A), "B": _buffer(B)}
        dma_atom = fx.make_copy_atom(fx.rocdl.cdna4.BufferLoadAsyncLDS128b(), 128)
        lds_ptr = fx.PointerType.get(fx.Int8.ir_type, 2, 512)
        k_tile_bytes = {"A": BLOCK_K, "B": BLOCK_K * 16 if b_shuffled else BLOCK_K}

        def src(op, i):
            if const_expr(op == "B" and b_shuffled):
                return (wave // 2 + 4 * i) * 16 * K + wave % 2 * 1024 + lane * 16
            x = wave * 8 + lane // 8
            return (x + 64 * i) * K + _swizzle(x, lane % 8 * 16)

        src_offsets = {op: [src(op, i) for i in range(n_dma[op])] for op in "AB"}

        def dma(op, h, kt):
            """Start the copy of K-tile ``kt`` of half ``h`` (nothing past the last K-tile)."""
            if const_expr(kt < k_tiles):
                for i in range_constexpr(n_dma[op]):
                    dst = fx.inttoptr(
                        lds_ptr,
                        lds
                        + region[op]
                        + buf(op, h, kt)
                        + (wave + NUM_WAVES * i) * 1024,
                    )
                    fx.copy(
                        dma_atom,
                        fx.slice(gmem[op], (None, fx.Int32(src_offsets[op][i]))),
                        fx.make_view(dst, fx.make_layout(1, 1)),
                        soffset=fx.Int32(
                            (row0[op] + h * rows[op]) * K + kt * k_tile_bytes[op]
                        ),
                    )

        # LDS -> registers. Lane (r, g) of an MFMA operand holds row r, K-chunks g and 4 + g
        # (K [16g, 16g+16) and [64+16g, 64+16g+16)): two 16-byte reads at a per-lane address
        # plus constants (the next 16-row tile is 2 KB further in both layouts).
        r, g = lane % 16, lane // 16

        def lane_offset(op, c):
            if const_expr(
                op == "B" and b_shuffled
            ):  # group = tile: [chunk][16 rows][16 bytes]
                return wave_tile["B"] * 16 * BLOCK_K + c * 256 + r * 16
            return (wave_tile[op] * 16 + r) * BLOCK_K + _swizzle(r, c * 16)

        lane_addr = {
            op: [_opaque(lds + region[op] + lane_offset(op, c)) for c in (g, 4 + g)]
            for op in "AB"
        }

        def read(op, h, kt):
            """This wave's MFMA operands (8 x i32 each) of half ``h``, K-tile ``kt``."""
            ops = []
            for i in range_constexpr(tiles[op]):
                lo, hi = (
                    _lds(addr + buf(op, h, kt) + i * 16 * BLOCK_K, fx.Int32, 4).load()
                    for addr in lane_addr[op]
                )
                ops.append(lo.shuffle(hi, list(range(8))))
            return ops

        # Scale words of this wave's MFMA tiles: word (tile T, group q, lane) at
        # (T * n_groups + q) * 64 + lane, a per-lane index plus a uniform offset.
        scale_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        scale_buf = {"A": _buffer(A_scale), "B": _buffer(B_scale)}
        word0 = {
            op: (row0[op] // 16 + wave_tile[op]) * (n_groups * 64) + lane for op in "AB"
        }

        def load_scales(q):
            """[op][half][tile] words of 512-K group q."""

            def word(op, tile):
                reg = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
                fx.copy(
                    scale_atom,
                    fx.slice(scale_buf[op], (None, fx.Int32(word0[op]))),
                    reg,
                    soffset=fx.Int32((tile * n_groups + q) * 64),
                )
                return Vec(fx.memref_load_vec(reg))[0]

            return {
                op: [
                    [word(op, h * rows[op] // 16 + i) for i in range(tiles[op])]
                    for h in range(2)
                ]
                for op in "AB"
            }

        # MFMAs. Lane (r, g) supplies the scale of row r, MX block g of the step; op_sel, an
        # instruction constant, picks the step's byte of the word: one atom per byte.
        atoms = [
            fx.make_mma_atom(
                fx.rocdl.cdna4.MFMA_Scale(
                    16, 16, 128, fx.Float8E4M3FN, opsel_a=j, opsel_b=j
                )
            )
            for j in range(GROUP_STEPS)
        ]

        def regs(values, n, dtype):
            """Register tensors holding ``values``: fx.gemm takes those, not SSA values."""
            out = []
            for v in values:
                reg = fx.make_rmem_tensor(n, dtype)
                reg.store(Vec(v))
                out.append(reg)
            return out

        def mma(acc, a_ops, b_ops, a_scales, b_scales, opsel):
            """
            acc[i][j] += A tile i x B tile j. B goes first, so tile (i, j) accumulates C^T
            and lane (r, g) gets row r, columns 4g .. 4g+3.
            """
            a, b = regs(a_ops, 8, fx.Int32), regs(b_ops, 8, fx.Int32)
            c = [regs(row, 4, fx.Float32) for row in acc]
            rocdl.s_setprio(1)
            for i in range_constexpr(len(a)):
                for j in range_constexpr(len(b)):
                    fx.gemm(
                        atoms[opsel],
                        c[i][j],
                        b[j],
                        a[i],
                        c[i][j],
                        scale_a=b_scales[j],
                        scale_b=a_scales[i],
                    )
            rocdl.s_setprio(0)
            return [[x.load().ir_value() for x in row] for row in c]

        # Prologue: K-tile 0 of every half and the two-ahead halves of K-tile 1, then wait
        # for K-tile 0 and put wave row 1 one barrier behind.
        scales = {0: load_scales(0)}
        for op, h in (("B", 0), ("A", 0), ("B", 1), ("A", 1)):
            dma(op, h, 0)
        for op, h in (("B", 0), ("A", 0), ("B", 1)):
            dma(op, h, 1)
        _barrier(vmcnt=in_flight if k_tiles > 1 else 0)
        if wave_row == 1:
            _barrier()

        zero = Vec.filled(4, 0.0, fx.Float32)
        acc = {
            (qa, qb): [[zero] * tiles["B"] for _ in range(tiles["A"])]
            for qa in (0, 1)
            for qb in (0, 1)
        }
        for k in range_constexpr(k_tiles):
            q, opsel = divmod(k, GROUP_STEPS)

            def quadrant(qa, qb, a_ops, b_ops):
                sc = scales[q]
                acc[qa, qb] = mma(
                    acc[qa, qb], a_ops, b_ops, sc["A"][qa], sc["B"][qb], opsel
                )
                _barrier()

            a0, b0 = read("A", 0, k), read("B", 0, k)
            dma("A", 1, k + 1)
            _barrier(
                lds=True
            )  # the leading wave row refills B0 right after the next barrier
            quadrant(0, 0, a0, b0)

            b1 = read("B", 1, k)
            dma("B", 0, k + 2)
            _barrier()
            quadrant(0, 1, a0, b1)

            a1 = read("A", 1, k)
            dma("A", 0, k + 2)
            _barrier()
            quadrant(1, 0, a1, b0)

            dma("B", 1, k + 2)
            _barrier(
                vmcnt=in_flight if k + 2 < k_tiles else 0
            )  # K-tile k + 1 has landed
            if const_expr((k + 2) % GROUP_STEPS == 0 and q + 1 < n_groups):
                scales[q + 1] = load_scales(q + 1)  # two steps before its first use
            quadrant(1, 1, a1, b1)

        # Epilogue: wave row 0 catches up, the bf16 tile goes to LDS (16-byte chunks XORed by
        # row % 16 against bank conflicts) and comes back as whole rows.
        if wave_row == 0:
            _barrier()

        def stage(row, chunk):
            return lds + row * (BLOCK_N * 2) + (chunk ^ (row % 16)) * 16

        for (qa, qb), tiles_ab in acc.items():
            for i in range_constexpr(tiles["A"]):
                for j in range_constexpr(tiles["B"]):
                    row = qa * rows["A"] + (wave_tile["A"] + i) * 16 + r
                    col = qb * rows["B"] + (wave_tile["B"] + j) * 16 + 4 * g
                    _lds(stage(row, col // 8) + col % 8 * 2, fx.BFloat16, 4).store(
                        Vec(tiles_ab[i][j]).to(fx.BFloat16)
                    )
        _barrier(lds=True)

        c_buf = _buffer(C)
        store_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        out = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
        chunk = (
            lane % 32
        )  # a wave instruction stores two rows of 32 chunks (8 columns each)
        col = row0["B"] + chunk * 8
        for s in range_constexpr(block_m // 16):
            row = wave * (block_m // NUM_WAVES) + 2 * s + lane // 32
            fx.memref_store_vec(_lds(stage(row, chunk), fx.BFloat16, 8).load(), out)
            # rows past M land past C's end and columns past N are sent there: both are dropped
            idx = (col < n).select((row0["A"] + row) * n + col, m * n)
            fx.copy(store_atom, out, fx.slice(c_buf, (None, fx.Int32(idx))))

    @flyc.jit
    def launch(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        m: fx.Int32,
        n: fx.Int32,
        stream: fx.Stream,
    ):
        # one 512-thread workgroup per CU (its LDS fills the CU): two waves per SIMD
        kernel(
            A,
            B,
            C,
            A_scale,
            B_scale,
            m,
            n,
            value_attrs={
                "rocdl.waves_per_eu": 2,
                "rocdl.flat_work_group_size": "512,512",
            },
        ).launch(
            grid=(_ceil_div(m, block_m) * _ceil_div(n, BLOCK_N), 1, 1),
            block=(NUM_WAVES * 64, 1, 1),
            stream=stream,
        )

    return launch


@triton.jit
def _scale_layout_kernel(src, dst, rows, blocks, stride_r, stride_b, n_groups):
    # one program per (16-row tile, 512-K group): 16 rows x 16 scales -> 256 bytes
    tile, group = tl.program_id(0), tl.program_id(1)
    r = tl.arange(0, 16)[:, None]
    b = tl.arange(0, 16)[None, :]
    row, blk = tile * 16 + r, group * 16 + b
    v = tl.load(
        src + row * stride_r + blk * stride_b,
        mask=(row < rows) & (blk < blocks),
        other=127,
    )
    # scale b = 4j + g of the group is byte j of word (tile, group, g, r)
    tl.store(dst + (tile * n_groups + group) * 256 + (b % 4) * 64 + r * 4 + b // 4, v)


def mxfp8_scale_layout(scale: torch.Tensor) -> torch.Tensor:
    """
    ue8m0 scales [rows, K/32] (uint8) -> int32 words [ceil(rows/16), ceil(K/512), 4, 16]; word
    (row, q, g) holds in byte j the scale of MX block 16q + 4j + g. Padding is 127 (x1).
    """
    scale = scale.view(torch.uint8)
    rows, blocks = scale.shape
    tiles, n_groups = _ceil_div(rows, 16), _ceil_div(blocks, 16)
    words = torch.empty(
        (tiles, n_groups, 4, 16), dtype=torch.int32, device=scale.device
    )
    if rows:
        _scale_layout_kernel[(tiles, n_groups)](
            scale,
            words.view(torch.uint8),
            rows,
            blocks,
            scale.stride(0),
            scale.stride(1),
            n_groups,
        )
    return words


_KERNELS = {}


def mxfp8_gemm(
    a, a_scale, b, b_scale, *, b_shuffled=False, out=None, block_m=None, xcd_swizzle=4
):
    """
    ``a @ b.T`` in bf16, with K % 128 == 0 and N % 8 == 0.

    a: [M, K] fp8 e4m3, row-major. b: [N, K] fp8 e4m3, row-major or, with ``b_shuffled``,
    aiter's shuffle_weight(b, (16, 16)). a_scale, b_scale: ``mxfp8_scale_layout`` of their
    [rows, K/32] ue8m0 scales. block_m (128 or 256) defaults to ``_default_block_m``.
    """
    (m, k), n = a.shape, b.shape[0]
    if b.shape[1] != k or k % BLOCK_K or n % 8:
        raise ValueError(
            f"need K % {BLOCK_K} == 0 and N % 8 == 0: a {tuple(a.shape)}, b {tuple(b.shape)}"
        )
    block_m = block_m or _default_block_m(m, n, a.device)
    if block_m not in (128, 256):
        raise ValueError(f"block_m must be 128 or 256, got {block_m}")
    n_groups = _ceil_div(k, GROUP_STEPS * BLOCK_K)
    for name, s, rows in (("a_scale", a_scale, m), ("b_scale", b_scale, n)):
        if s.dtype != torch.int32 or tuple(s.shape) != (
            _ceil_div(rows, 16),
            n_groups,
            4,
            16,
        ):
            raise ValueError(
                f"{name} must come from mxfp8_scale_layout, got {s.dtype} {tuple(s.shape)}"
            )
    if out is None:
        out = torch.empty(m, n, dtype=torch.bfloat16, device=a.device)
    args = (
        a.view(torch.int8).reshape(-1),
        b.view(torch.int8).reshape(-1),
        out.view(-1),
        a_scale.reshape(-1),
        b_scale.reshape(-1),
        m,
        n,
        fx.Stream(torch.cuda.current_stream()),
    )
    key = (k, block_m, xcd_swizzle, bool(b_shuffled))
    if key not in _KERNELS:
        _KERNELS[key] = flyc.compile(_build(*key), *args)  # compiling runs it once
    else:
        _KERNELS[key](*args)
    return out
