# SPDX-License-Identifier: Apache-2.0
"""Hopper (SM90) WGMMA kernel for per-token-block W4AFP8 grouped GEMM.

Drop-in numerical replacement for the Torch reference in
``cutedsl_w4afp8_gemm_per_token_block``. Implements the exact fp8 mixed-input
path:

  * INT4 weights (packed 2-per-byte int8) are unpacked to fp8_e4m3 in SMEM.
    INT4 ``[-8, 7]`` is exactly representable in e4m3, so the unpack folds NO
    scale and is lossless.
  * Per 128-element K-block ``b``:
        block_acc = wgmma(A_fp8_block, B_fp8_block)          # fp32 accumulation
        main_acc[m,n] += block_acc[m,n] * a_scale[m,b] * w_scale[n,b]
    i.e. DeepGEMM-style blockwise scaling with a rank-1 fp32 promotion, matching
    the block-wise fp32 reference up to fp32 associativity.

K is streamed in blocks of 128 through at most 16 KiB of FP8 shared memory
plus 512 bytes of scales. Payload transfers use contiguous vector segments
inside the WGMMA swizzle; scales are loaded once per row/column per CTA.

For M > 32, CTAs compute 64x64 output tiles. For M <= 32, swapping the WGMMA
operands places tokens on its variable-width N axis (16 or 32), reducing
padding without changing the blockwise scaling formula. Invalid activation
rows are never read, empty tiles skip the K loop, and invalid output rows are
explicitly zeroed. Three specializations per device cover all supported shapes.

This module is only importable on machines with the CuTe DSL toolchain; the
caller (``cutedsl_w4afp8_gemm_per_token_block``) guards the import.
"""

# Keep annotations eagerly evaluated: CuTe must recognize Constexpr parameters.
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cutlass.cute.nvgpu.warpgroup import Field, OperandMajorMode
from cutlass.cute.runtime import from_dlpack

TILE_M = 64
TILE_N = 64
GROUP = 128  # per-token-block / per-channel-block K group size
F8 = cutlass.Float8E4M3FN
BF16 = cutlass.BFloat16
ACC = cutlass.Float32


@cute.kernel
def _w4afp8_kernel(
    mA: cute.Tensor,  # [E, M, K]      fp8_e4m3
    mWq: cute.Tensor,  # [E, N, K//2]   int8 (packed int4)
    mAs: cute.Tensor,  # [E, M, K//128] f32
    mWs: cute.Tensor,  # [E, N, K//128] f32 (logical)
    mC: cute.Tensor,  # [E, M, N]      bf16 (written in place)
    mMasked: cute.Tensor,  # [E]          int32
    tiled_mma: cute.TiledMma,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
    tile_m: cutlass.Constexpr,
    transposed: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    nb, mb, eidx = cute.arch.block_idx()

    M_total = cute.size(mA, mode=[1])
    N_total = cute.size(mWq, mode=[1])
    NBLK = cute.size(mWs, mode=[2])  # K // 128 (dynamic)
    masked = mMasked[eidx]

    m_base = mb * tile_m
    n_base = nb * TILE_N
    # This condition is CTA-uniform: empty tiles skip all loads and WGMMA,
    # but still execute the epilogue to zero the caller's output buffer.
    if m_base >= masked:
        NBLK = 0

    # fp8 zero (via a 4-wide vector convert; scalar/1-elem fp->fp8 is unsupported).
    zf = cute.make_fragment(4, F8)
    ztmp = cute.make_fragment(4, ACC)
    for _j in cutlass.range_constexpr(4):
        ztmp[_j] = 0.0
    zf.store(ztmp.load().to(F8))
    z_f8 = zf[0]

    smem = utils.SmemAllocator()
    sA = smem.allocate_tensor(F8, a_smem_layout.outer, 128, a_smem_layout.inner)
    sB = smem.allocate_tensor(F8, b_smem_layout.outer, 128, b_smem_layout.inner)
    sAs = smem.allocate_tensor(ACC, cute.make_layout(tile_m), 16)
    sWs = smem.allocate_tensor(ACC, cute.make_layout(TILE_N), 16)

    thr_mma = tiled_mma.get_slice(tidx)
    if cutlass.const_expr(transposed):
        tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sB))
        tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sA))
        idC = thr_mma.partition_C(cute.make_identity_tensor((TILE_N, tile_m)))
    else:
        tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA))
        tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB))
        idC = thr_mma.partition_C(cute.make_identity_tensor((tile_m, TILE_N)))
    main_acc = cute.make_rmem_tensor(idC.shape, ACC)
    block_acc = cute.make_rmem_tensor(idC.shape, ACC)
    n_elems = cute.size(main_acc)
    n_ksub = cute.size(tCrA, mode=[2])  # WGMMA k-subblocks per 128-K block (=4)

    for i in cutlass.range_constexpr(n_elems):
        main_acc[i] = 0.0

    K_half = GROUP // 2  # packed bytes per 128-K block

    for blk in cutlass.range(NBLK, unroll=1):
        # A row segments are 16-byte aligned even after the WGMMA swizzle.
        # Copy a whole segment rather than issuing one byte store per element.
        for idx in cutlass.range(tidx, tile_m * (GROUP // 16), 128, unroll=1):
            m = idx // (GROUP // 16)
            segment = idx % (GROUP // 16)
            gm = m_base + m
            av = cute.make_fragment(16, F8)
            if gm < M_total and gm < masked:
                src = cute.zipped_divide(mA[eidx, gm, None], (16,))
                cute.autovec_copy(src[(None,), (blk * (GROUP // 16) + segment,)], av)
            else:
                for j in cutlass.range_constexpr(16):
                    av[j] = z_f8
            dst = cute.zipped_divide(sA[m, None], (16,))
            cute.autovec_copy(av, dst[(None,), (segment,)])

        # Eight packed bytes become one contiguous 16-byte fp8 SMEM store.
        VW = 8
        n_bytes = TILE_N * K_half
        for base in cutlass.range(tidx * VW, n_bytes, 128 * VW, unroll=1):
            nn = base // K_half
            segment = (base % K_half) // VW
            gnn = n_base + nn
            frag = cute.make_fragment(VW, cutlass.Int8)
            if gnn < N_total:
                src = cute.zipped_divide(mWq[eidx, gnn, None], (VW,))
                cute.autovec_copy(src[(None,), (blk * (K_half // VW) + segment,)], frag)
            else:
                frag.fill(0)
            unpacked = cute.make_fragment(2 * VW, F8)
            if cutlass.const_expr(transposed):
                # PRMT performs four parallel 8-entry byte lookups. Low three
                # bits index the E4M3 tables; bit three selects the signed table.
                # Positive bytes: 00 38 40 44 48 4a 4c 4e (0 through 7).
                # Negative bytes: d0 ce cc ca c8 c4 c0 b8 (-8 through -1).
                # Nibble ordering matches PRMT selectors: low0, high0, low1, high1.
                packed_words = cute.recast_tensor(frag, cutlass.Uint32)
                fp8_words = cute.recast_tensor(unpacked, cutlass.Uint32)
                for j in cutlass.range_constexpr(2 * VW // 4):
                    nibbles = packed_words[j // 2] // (1 << (16 * (j % 2)))
                    indices = nibbles & 0x7777
                    positive = cute.arch.prmt(0x44403800, 0x4E4C4A48, indices)
                    negative = cute.arch.prmt(
                        cutlass.Uint32(0xCACCCED0), cutlass.Uint32(0xB8C0C4C8), indices
                    )
                    sign_mask = cute.arch.prmt(0x0000FF00, 0, (nibbles // 8) & 0x1111)
                    fp8_words[j] = positive ^ ((positive ^ negative) & sign_mask)
            else:
                # Numeric conversion remains faster for the larger output tile.
                p = frag.load().to(cutlass.Int32)
                ub = p & 0xFF
                low = ub & 0xF
                high = (ub - low) // 16
                low = low - ((low & 8) * 2)
                high = high - ((high & 8) * 2)
                of_low = cute.make_fragment(VW, F8)
                of_high = cute.make_fragment(VW, F8)
                of_low.store(low.to(ACC).to(F8))
                of_high.store(high.to(ACC).to(F8))
                for j in cutlass.range_constexpr(VW):
                    unpacked[2 * j] = of_low[j]
                    unpacked[2 * j + 1] = of_high[j]
            dst = cute.zipped_divide(sB[nn, None], (2 * VW,))
            cute.autovec_copy(unpacked, dst[(None,), (segment,)])

        # Load each row/column scale once per CTA, then broadcast from SMEM.
        if tidx < tile_m:
            gm = m_base + tidx
            sAs[tidx] = 0.0
            if gm < M_total and gm < masked:
                sAs[tidx] = mAs[eidx, gm, blk]
        if tidx < TILE_N:
            gn = n_base + tidx
            sWs[tidx] = 0.0
            if gn < N_total:
                sWs[tidx] = mWs[eidx, gn, blk]
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.barrier()

        # --- WGMMA over the 128-K block into block_acc (fresh accumulate) ---
        tiled_mma.set(Field.ACCUMULATE, False)
        cute.nvgpu.warpgroup.fence()
        for kb in cutlass.range_constexpr(n_ksub):
            cute.gemm(
                tiled_mma,
                block_acc,
                tCrA[None, None, kb],
                tCrB[None, None, kb],
                block_acc,
            )
            tiled_mma.set(Field.ACCUMULATE, True)
        cute.nvgpu.warpgroup.commit_group()
        cute.nvgpu.warpgroup.wait_group(0)

        # --- fp32 rank-1 blockwise promotion ---
        for i in cutlass.range_constexpr(n_elems):
            if cutlass.const_expr(transposed):
                row = idC[i][1]
                col = idC[i][0]
            else:
                row = idC[i][0]
                col = idC[i][1]
            main_acc[i] = main_acc[i] + block_acc[i] * sAs[row] * sWs[col]

        cute.arch.barrier()  # protect sA/sB reuse before the next block's loads

    # --- epilogue: cast to bf16 (vector) and store predicated ---
    out_bf = cute.make_rmem_tensor(idC.shape, BF16)
    out_bf.store(main_acc.load().to(BF16))
    for i in cutlass.range_constexpr(n_elems):
        if cutlass.const_expr(transposed):
            gm = m_base + idC[i][1]
            gn = n_base + idC[i][0]
        else:
            gm = m_base + idC[i][0]
            gn = n_base + idC[i][1]
        if gm < M_total and gn < N_total:
            if gm < masked:
                mC[eidx, gm, gn] = out_bf[i]
            else:
                mC[eidx, gm, gn] = BF16(0.0)


@cute.jit
def _launch(
    mA: cute.Tensor,
    mWq: cute.Tensor,
    mAs: cute.Tensor,
    mWs: cute.Tensor,
    mC: cute.Tensor,
    mMasked: cute.Tensor,
    stream: cuda.CUstream,
    tile_m: cutlass.Constexpr,
    transposed: cutlass.Constexpr,
):
    k_major = utils.LayoutEnum.ROW_MAJOR
    if cutlass.const_expr(transposed):
        # W @ A.T puts the small token dimension on WGMMA's variable N axis.
        tiler = (TILE_N, tile_m, GROUP)
        a_lay = cute.slice_(
            sm90_utils.make_smem_layout_b(k_major, tiler, F8, 1), (None, None, 0)
        )
        b_lay = cute.slice_(
            sm90_utils.make_smem_layout_a(k_major, tiler, F8, 1), (None, None, 0)
        )
    else:
        tiler = (tile_m, TILE_N, GROUP)
        a_lay = cute.slice_(
            sm90_utils.make_smem_layout_a(k_major, tiler, F8, 1), (None, None, 0)
        )
        b_lay = cute.slice_(
            sm90_utils.make_smem_layout_b(k_major, tiler, F8, 1), (None, None, 0)
        )
    tiled_mma = sm90_utils.make_trivial_tiled_mma(
        F8,
        F8,
        OperandMajorMode.K,
        OperandMajorMode.K,
        ACC,
        (1, 1, 1),
        tiler_mn=tiler[:2],
    )

    E = cute.size(mA, mode=[0])
    M = cute.size(mA, mode=[1])
    N = cute.size(mWq, mode=[1])
    num_m = (M + tile_m - 1) // tile_m
    num_n = (N + TILE_N - 1) // TILE_N

    _w4afp8_kernel(
        mA, mWq, mAs, mWs, mC, mMasked, tiled_mma, a_lay, b_lay, tile_m, transposed
    ).launch(grid=(num_n, num_m, E), block=(128, 1, 1), stream=stream)


# CUDA modules belong to a device context, not to the process as a whole.
_COMPILED = {}


def _fp8_ct(t_fp8: torch.Tensor) -> cute.Tensor:
    """fp8 is not dlpack-able: back with int8, then override element_type."""
    ct = from_dlpack(t_fp8.view(torch.int8), assumed_align=16).mark_layout_dynamic(
        leading_dim=t_fp8.ndim - 1
    )
    ct.element_type = F8
    return ct


def _ct(t: torch.Tensor) -> cute.Tensor:
    # Only the payload copies need 16-byte alignment. Scales and output use
    # scalar accesses and may be views with a nonzero storage offset.
    align = 16 if t.dtype == torch.int8 else t.element_size()
    return from_dlpack(t, assumed_align=align).mark_layout_dynamic(
        leading_dim=t.ndim - 1
    )


def hopper_w4afp8_gemm_per_token_block(
    a: torch.Tensor,  # [E, M, K]      fp8_e4m3
    a_scale: torch.Tensor,  # [E, M, K//128] f32
    w: torch.Tensor,  # [E, N, K//2]   int8 (packed int4)
    w_scale: torch.Tensor,  # [E, N, K//128] f32 (logical)
    output: torch.Tensor,  # [E, M, N]      bf16, written in place
    masked_m: torch.Tensor,  # [E]            int32
) -> None:
    """Hopper WGMMA implementation of the per-token-block W4AFP8 grouped GEMM.

    Matches ``cutedsl_w4afp8_gemm_per_token_block`` numerically. Writes into
    ``output`` in place, including exact zeros for rows ``>= masked_m[e]``.
    The output need not be initialized. Warm up before CUDA graph capture.
    """
    from sglang.srt.layers.moe.cutedsl_w4afp8_gemm_per_token_block import (
        _validate_inputs,
    )

    _validate_inputs(a, a_scale, w, w_scale, output, masked_m)
    if masked_m is None:
        raise ValueError("the direct Hopper entry requires masked_m")
    if not a.is_cuda or torch.cuda.get_device_capability(a.device) != (9, 0):
        raise ValueError("Hopper W4AFP8 requires an SM90 CUDA device")
    if output.dtype != torch.bfloat16 or not output.is_contiguous():
        raise ValueError("Hopper W4AFP8 requires contiguous bfloat16 output")
    if output.numel() == 0:
        return
    if a.shape[-1] == 0:
        output.zero_()
        return

    a = a.contiguous()
    w = w.contiguous()
    # Contiguous slices can still start at an unaligned storage offset.
    if a.data_ptr() % 16:
        a = a.clone()
    if w.data_ptr() % 16:
        w = w.clone()
    a_scale = a_scale.contiguous().float()
    w_scale = w_scale.contiguous().float()
    masked_m = masked_m.to(torch.int32).contiguous()

    with torch.cuda.device(a.device):
        mA = _fp8_ct(a)
        mWq = _ct(w)
        mAs = _ct(a_scale)
        mWs = _ct(w_scale)
        mC = _ct(output)
        mMasked = _ct(masked_m)
        stream = cuda.CUstream(torch.cuda.current_stream(a.device).cuda_stream)

        # Layouts and K-block loop bounds remain dynamic within each tactic.
        # The supported FP8 WGMMA minimum N width in this toolchain is 16.
        transposed = a.shape[1] <= 32
        tile_m = max(16, 1 << (a.shape[1] - 1).bit_length()) if transposed else TILE_M
        key = (a.device.index, tile_m, transposed)
        if key not in _COMPILED:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm up Hopper W4AFP8 before CUDA graph capture")
            _COMPILED[key] = cute.compile(
                _launch, mA, mWq, mAs, mWs, mC, mMasked, stream, tile_m, transposed
            )
        _COMPILED[key](mA, mWq, mAs, mWs, mC, mMasked, stream)
