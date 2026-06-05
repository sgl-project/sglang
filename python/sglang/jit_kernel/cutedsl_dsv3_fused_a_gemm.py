# Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CuTe DSL DeepSeek-V3 fused-A GEMM (sm90+): out[M, N] = mat_a[M, K] @ weight,
N in {2112, 6144}, M = num_tokens in [1, 16], K any multiple of 1024, bf16.

Adapted from NVIDIA TensorRT-LLM dsv3FusedAGemm.cu
(cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3FusedAGemm.cu), reimplemented
in the CuTe DSL: AB-swap, warp-specialized 4-way split-K, cp.async + mbarrier
pipeline, ldmatrix + mma.sync.m16n8k16, 3-4-3 swizzle.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils
import torch
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack

from sglang.srt.utils import get_device_sm

TILE_M = 16
TILE_K = 256
SPLITK = 4
LOAD_WARPS = 4  # 4 loader + 4 compute warps -> 256-thread block (matches CUDA)
MAX_NSTAGE = 16  # upper bound on the pipeline depth
PWK = TILE_K // SPLITK
KSTEPS = PWK // 16
COMPUTE_THREADS = SPLITK * 32
LOADER_THREADS = LOAD_WARPS * 32
NTHREADS = COMPUTE_THREADS + LOADER_THREADS
KI = TILE_K // 2  # int32 cols per K-tile (a bf16 pair per int32)

_BAR_I32 = 2 * MAX_NSTAGE * 2  # full[] + empty[] mbarriers (int64 = 2 int32 each)


def _stage_i32(tile_n: int) -> int:
    return (TILE_M + tile_n) * KI


def _cp_async(smem_ptr, gmem_ptr):
    llvm.inline_asm(
        None,
        [smem_ptr.toint().ir_value(), gmem_ptr.toint().ir_value()],
        "{ .reg .u32 sa; cvt.u32.u64 sa, $0; cp.async.cg.shared.global.L2::128B [sa], [$1], 16; }",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=0,
    )


def _cp_async_pred(smem_ptr, gmem_ptr, pred_i32):
    llvm.inline_asm(
        None,
        [smem_ptr.toint().ir_value(), gmem_ptr.toint().ir_value(), pred_i32.ir_value()],
        "{ .reg .pred p; .reg .u32 sa; setp.ne.s32 p, $2, 0; cvt.u32.u64 sa, $0; "
        "@p cp.async.cg.shared.global.L2::128B [sa], [$1], 16; }",
        "l,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=0,
    )


def _ldmatrix(smem_ptr):
    i32 = ir.IntegerType.get_signless(32)
    res = llvm.inline_asm(
        llvm.StructType.get_literal([i32, i32, i32, i32]),
        [smem_ptr.toint().ir_value()],
        "{ .reg .u32 sa; cvt.u32.u64 sa, $4; "
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {$0,$1,$2,$3}, [sa]; }",
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=0,
    )
    return [llvm.extractvalue(i32, res, [i]) for i in range(4)]


def _ldmatrix_x2(smem_ptr):
    i32 = ir.IntegerType.get_signless(32)
    res = llvm.inline_asm(
        llvm.StructType.get_literal([i32, i32]),
        [smem_ptr.toint().ir_value()],
        "{ .reg .u32 sa; cvt.u32.u64 sa, $2; "
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {$0,$1}, [sa]; }",
        "=r,=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=0,
    )
    return [llvm.extractvalue(i32, res, [i]) for i in range(2)]


def _mma(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3):
    f32 = ir.F32Type.get()
    res = llvm.inline_asm(
        llvm.StructType.get_literal([f32, f32, f32, f32]),
        [a0, a1, a2, a3, b0, b1, c0, c1, c2, c3],
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=0,
    )
    return [llvm.extractvalue(f32, res, [i]) for i in range(4)]


def _swizzle(row, col):
    # 3-4-3 swizzle (bank-conflict-free ldmatrix; the XOR is a multiple of 4 so
    # each 16B group stays contiguous).
    return col ^ ((row % 8) * 4)


def _k_project(tile, col, kgi):
    # Within-tile int32 col -> global-K int32, so compute warp (col // kp_warp)
    # reduces a contiguous global-K region.
    kp_warp = KI // SPLITK
    kp_chunk = kgi // SPLITK
    return (col // kp_warp) * kp_chunk + tile * kp_warp + (col % kp_warp)


def _load_tile(ltid, feat0, sa, sb, mW, mA, tile, buf, M, kgi, tile_n):
    # All loader threads stream the weight tile first (the bandwidth-critical load, so
    # it gets the full loader-thread count for maximum cp.async parallelism), then the
    # activation tile (rows >= num_tokens predicated off). A 50/50 A/B warp split, as
    # in the CUDA reference, was measured to halve weight-load MLP and regress badly.
    for it in range(TILE_M * KI // (LOADER_THREADS * 4)):
        idx = (it * LOADER_THREADS + ltid) * 4
        row, col = idx // KI, idx % KI
        _cp_async(
            sa.iterator + (buf * TILE_M * KI + row * KI + _swizzle(row, col)),
            mW.iterator + ((feat0 + row) * kgi + _k_project(tile, col, kgi)),
        )
    for it in range(tile_n * KI // (LOADER_THREADS * 4)):
        idx = (it * LOADER_THREADS + ltid) * 4
        row, col = idx // KI, idx % KI
        pred = (row < M).to(cutlass.Int32)
        _cp_async_pred(
            sb.iterator + (buf * tile_n * KI + row * KI + _swizzle(row, col)),
            mA.iterator + (row * pred * kgi + _k_project(tile, col, kgi)),
            pred,
        )


@cute.kernel
def _kernel(
    mW: cute.Tensor,
    mA: cute.Tensor,
    mOut: cute.Tensor,
    M: cutlass.Int32,
    num_kt: cutlass.Constexpr,
    nstage: cutlass.Constexpr,
    tile_n: cutlass.Constexpr,
):
    NB = tile_n // 8
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    warp, lane = tid // 32, tid % 32
    r0, cc = lane // 4, lane % 4
    feat0 = bid * TILE_M
    kgi = num_kt * KI

    # Pipeline ring carved from raw dynamic smem. nstage is a compile-time constant
    # (chosen host-side from the device's opt-in smem) so the stage wrap / phase-flip
    # logic constant-folds, matching the CUDA reference's templated stage_cnt. The
    # epilogue accumulator sC aliases the start of the sA stage region (reused after
    # the k-loop, as in the CUDA reference) so it costs no extra smem -> +1 stage.
    base = cute.arch.get_dyn_smem(cutlass.Int32, alignment=16)
    bar = cute.recast_ptr(base, dtype=cutlass.Int64)
    full, empty = bar, bar + MAX_NSTAGE
    sa_off = _BAR_I32
    # Pad the split-K partial stride (as the CUDA reference does) so the four warps'
    # partials sit in different smem banks for the epilogue reduction.
    sc_warp_stride = TILE_M * tile_n + 2
    sC = cute.make_tensor(
        cute.recast_ptr(base + sa_off, dtype=cutlass.Float32),
        cute.make_layout((SPLITK, TILE_M, tile_n), stride=(sc_warp_stride, tile_n, 1)),
    )
    sA = cute.make_tensor(
        base + sa_off,
        cute.make_layout((nstage, TILE_M, KI), stride=(TILE_M * KI, KI, 1)),
    )
    sB = cute.make_tensor(
        base + sa_off + nstage * TILE_M * KI,
        cute.make_layout((nstage, tile_n, KI), stride=(tile_n * KI, KI, 1)),
    )

    if tid == 0:
        for s in range(nstage):
            cute.arch.mbarrier_init(full + s, LOADER_THREADS)
            cute.arch.mbarrier_init(empty + s, COMPUTE_THREADS)
    cute.arch.barrier()

    if warp >= SPLITK:
        # Loaders read global memory, so they (not the compute warps) gate on the
        # programmatic-dependent launch barrier.
        cute.arch.griddepcontrol_wait()
        ltid = tid - COMPUTE_THREADS
        # The k-loop is fully unrolled (range_constexpr) so the compiler can
        # software-pipeline cp.async across iterations, and uses the CUDA reference's
        # try_wait look-ahead: peek the next stage's empty barrier to skip the blocking
        # wait whenever the consumer is keeping up. lph(kt) = ((kt//nstage)&1)^1.
        need_wait = cutlass.Boolean(True)
        for kt in cutlass.range_constexpr(num_kt):
            st = kt % nstage
            if kt >= nstage:
                if need_wait:
                    cute.arch.mbarrier_wait(empty + st, ((kt // nstage) & 1) ^ 1)
            if kt + 1 < num_kt and kt + 1 >= nstage:
                nst = (kt + 1) % nstage
                nph = (((kt + 1) // nstage) & 1) ^ 1
                need_wait = not cute.arch.mbarrier_try_wait(empty + nst, nph)
            _load_tile(ltid, feat0, sA, sB, mW, mA, kt, st, M, kgi, tile_n)
            cute.arch.cp_async_mbarrier_arrive_noinc(full + st)
    else:
        acc = [[cutlass.Float32(0.0) for _ in range(4)] for _ in range(NB)]
        ph = 0
        for kt in cutlass.range_constexpr(num_kt):
            buf = kt % nstage
            cute.arch.mbarrier_wait(full + buf, ph)
            for step in cutlass.range_constexpr(KSTEPS):
                kbh = warp * (PWK // 2) + step * 8
                arow, aoff = lane % 16, (lane // 16) * 4
                a = _ldmatrix(
                    sA.iterator
                    + (buf * TILE_M * KI + arow * KI + _swizzle(arow, kbh + aoff))
                )
                if cutlass.const_expr(NB == 2):
                    # One ldmatrix.x4 loads both 8-wide n-blocks for this k-step
                    # (B is n-major with a multiple-of-4 swizzle, same as A). The
                    # four regs map to mma operands as nb0=(0,2), nb1=(1,3).
                    brow, boff = lane % 16, (lane // 16) * 4
                    bb = _ldmatrix(
                        sB.iterator
                        + (buf * tile_n * KI + brow * KI + _swizzle(brow, kbh + boff))
                    )
                    b01 = [(bb[0], bb[2]), (bb[1], bb[3])]
                else:
                    # 8 n-rows -> ldmatrix.x2 (two 8x8 tiles = the two B regs b0, b1).
                    brow8 = lane % 8
                    boff8 = ((lane // 8) & 1) * 4
                    bb = _ldmatrix_x2(
                        sB.iterator
                        + (
                            buf * tile_n * KI
                            + brow8 * KI
                            + _swizzle(brow8, kbh + boff8)
                        )
                    )
                    b01 = [(bb[0], bb[1])]
                for nb in cutlass.range_constexpr(NB):
                    b0, b1 = b01[nb]
                    d = _mma(
                        a[0],
                        a[1],
                        a[2],
                        a[3],
                        b0,
                        b1,
                        acc[nb][0].ir_value(),
                        acc[nb][1].ir_value(),
                        acc[nb][2].ir_value(),
                        acc[nb][3].ir_value(),
                    )
                    for i in cutlass.range_constexpr(4):
                        acc[nb][i] = cutlass.Float32(d[i])
            cute.arch.mbarrier_arrive(empty + buf)
            ph = (ph ^ 1) if buf == nstage - 1 else ph
        for nb in cutlass.range_constexpr(NB):
            for i in cutlass.range_constexpr(4):
                m = r0 + (8 if i >= 2 else 0)
                n = nb * 8 + cc * 2 + (i % 2)
                sC[warp, m, n] = acc[nb][i]

        # Epilogue runs entirely within the compute warps behind a named barrier
        # (bar.sync 1, COMPUTE_THREADS), matching the CUDA reference; the loader
        # warps do not participate, so no full-block sync is needed here. The
        # split-K reduction uses all COMPUTE_THREADS (not the CUDA reference's
        # warp-0-only path, which serializes the output stores and was measured
        # slower) over the bank-padded sC.
        cute.arch.barrier(barrier_id=1, number_of_threads=COMPUTE_THREADS)
        nred = TILE_M * tile_n
        for it in cutlass.range_constexpr(
            (nred + COMPUTE_THREADS - 1) // COMPUTE_THREADS
        ):
            e = it * COMPUTE_THREADS + tid
            if e < nred:
                m, n = e // tile_n, e % tile_n
                s = sC[0, m, n]
                for w in cutlass.range_constexpr(1, SPLITK):
                    s = s + sC[w, m, n]
                if n < M:
                    mOut[n, feat0 + m] = s.to(cutlass.BFloat16)

    cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def _launch(
    mW: cute.Tensor,
    mA: cute.Tensor,
    mOut: cute.Tensor,
    M: cutlass.Int32,
    stream: cuda.CUstream,
    num_kt: cutlass.Constexpr,
    gemm_m: cutlass.Constexpr,
    smem_bytes: cutlass.Constexpr,
    nstage: cutlass.Constexpr,
    tile_n: cutlass.Constexpr,
):
    _kernel(mW, mA, mOut, M, num_kt, nstage, tile_n).launch(
        grid=[gemm_m // TILE_M, 1, 1],
        block=[NTHREADS, 1, 1],
        smem=smem_bytes,
        use_pdl=True,
        stream=stream,
    )


_compiled: dict[tuple[int, int, int], object] = {}


def _pick_nstage(num_kt: int, smem_optin_bytes: int, tile_n: int) -> int:
    # sC aliases the sA region, so it does not consume budget beyond one stage.
    nstage = (smem_optin_bytes // 4 - _BAR_I32) // _stage_i32(tile_n)
    return min(nstage, MAX_NSTAGE, num_kt)


def _pick_tile_n(num_tokens: int) -> int:
    # Match the CUDA reference: a narrow 8-wide token tile (1 MMA n-block) for the
    # small-batch case, the 16-wide tile (2 n-blocks) otherwise.
    return 8 if num_tokens <= 8 else 16


def _compiled_kernel(num_kt: int, gemm_m: int, tile_n: int):
    if get_device_sm() < 90:
        raise RuntimeError("dsv3_fused_a_gemm requires SM90 (Hopper) or later")
    if (num_kt, gemm_m, tile_n) not in _compiled:
        smem_optin = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
        nstage = _pick_nstage(num_kt, smem_optin, tile_n)
        smem_bytes = (_BAR_I32 + nstage * _stage_i32(tile_n)) * 4
        k = num_kt * TILE_K
        w = torch.empty(gemm_m, k, dtype=torch.bfloat16, device="cuda")
        a = torch.empty(16, k, dtype=torch.bfloat16, device="cuda")
        o = torch.empty(16, gemm_m, dtype=torch.bfloat16, device="cuda")
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        _compiled[(num_kt, gemm_m, tile_n)] = cute.compile(
            _launch,
            from_dlpack(w.view(torch.int32)),
            from_dlpack(a.view(torch.int32)),
            from_dlpack(o),
            cutlass.Int32(16),
            stream,
            num_kt,
            gemm_m,
            smem_bytes,
            nstage,
            tile_n,
        )
    return _compiled[(num_kt, gemm_m, tile_n)]


def dsv3_fused_a_gemm(
    mat_a: torch.Tensor, mat_b: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """out[M, N] = mat_a[M, K] @ mat_b, with mat_a row-major [M, K] (M in [1, 16]),
    mat_b column-major [K, N] (the weight, stride(0) == 1), N a multiple of 16
    (e.g. 2112, 6144), K a multiple of 1024."""
    M, K = mat_a.shape
    N = mat_b.shape[1]
    assert mat_a.dtype == torch.bfloat16 and mat_b.dtype == torch.bfloat16
    assert K % 1024 == 0, f"K must be a multiple of 1024, got {K}"
    assert N % TILE_M == 0, f"N must be a multiple of {TILE_M}, got {N}"
    assert (
        tuple(mat_b.shape) == (K, N) and mat_b.stride(0) == 1
    ), "mat_b must be [K, N] column-major"
    assert 1 <= M <= 16, "num_tokens must be in [1, 16]"

    mat_a = mat_a.contiguous()
    weight = mat_b.t()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if out is None:
        out = torch.empty(M, N, dtype=torch.bfloat16, device=mat_a.device)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    _compiled_kernel(K // TILE_K, N, _pick_tile_n(M))(
        from_dlpack(weight.view(torch.int32)),
        from_dlpack(mat_a.view(torch.int32)),
        from_dlpack(out),
        M,
        stream,
    )
    return out
