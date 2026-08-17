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
import torch
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import get_smem_capacity_in_bytes

from sglang.kernel_api_logging import debug_kernel_api
from sglang.srt.utils import get_device_sm
from sglang.srt.utils.common import direct_register_custom_op

TILE_M = 16
TILE_K = 256
SPLITK = 4
LOAD_WARPS = 4
MAX_NSTAGE = 16
PWK = TILE_K // SPLITK
KSTEPS = PWK // 16
COMPUTE_THREADS = SPLITK * 32
LOADER_THREADS = LOAD_WARPS * 32
NTHREADS = COMPUTE_THREADS + LOADER_THREADS
KI = TILE_K // 2

_BAR_I32 = 2 * MAX_NSTAGE * 2


def _stage_i32(tile_n: int) -> int:
    return (TILE_M + tile_n) * KI


def _cp_async_16b(smem_ptr, gmem_ptr):
    llvm.inline_asm(
        None,
        [smem_ptr.toint().ir_value(), gmem_ptr.toint().ir_value()],
        "{ .reg .u32 sa; cvt.u32.u64 sa, $0; cp.async.cg.shared.global.L2::128B [sa], [$1], 16; }",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=0,
    )


def _cp_async_16b_pred(smem_ptr, gmem_ptr, pred_i32):
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


def _ldmatrix_x4(smem_ptr):
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


def _mma_m16n8k16(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3):
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


def _swizzle_343(row, col):
    return col ^ ((row % 8) * 4)


def _global_k_col(tile, col, kgi):
    kp_warp = KI // SPLITK
    kp_chunk = kgi // SPLITK
    return (col // kp_warp) * kp_chunk + tile * kp_warp + (col % kp_warp)


def _load_stage(ltid, feat0, sa, sb, mW, mA, tile, buf, M, kgi, tile_n):
    for it in range(TILE_M * KI // (LOADER_THREADS * 4)):
        idx = (it * LOADER_THREADS + ltid) * 4
        row, col = idx // KI, idx % KI
        _cp_async_16b(
            sa.iterator + (buf * TILE_M * KI + row * KI + _swizzle_343(row, col)),
            mW.iterator + ((feat0 + row) * kgi + _global_k_col(tile, col, kgi)),
        )
    for it in range(tile_n * KI // (LOADER_THREADS * 4)):
        idx = (it * LOADER_THREADS + ltid) * 4
        row, col = idx // KI, idx % KI
        pred = (row < M).to(cutlass.Int32)
        _cp_async_16b_pred(
            sb.iterator + (buf * tile_n * KI + row * KI + _swizzle_343(row, col)),
            mA.iterator + (row * pred * kgi + _global_k_col(tile, col, kgi)),
            pred,
        )


@cute.kernel
def _dsv3_fused_a_gemm_kernel(
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

    base = cute.arch.get_dyn_smem(cutlass.Int32, alignment=16)
    bar = cute.recast_ptr(base, dtype=cutlass.Int64)
    full, empty = bar, bar + MAX_NSTAGE
    sa_off = _BAR_I32
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
        cute.arch.griddepcontrol_wait()
        ltid = tid - COMPUTE_THREADS
        for kt in cutlass.range_constexpr(num_kt):
            st = kt % nstage
            if kt >= nstage:
                cute.arch.mbarrier_wait(empty + st, ((kt // nstage) & 1) ^ 1)
            _load_stage(ltid, feat0, sA, sB, mW, mA, kt, st, M, kgi, tile_n)
            cute.arch.cp_async_mbarrier_arrive_noinc(full + st)
    else:
        acc = [[cutlass.Float32(0.0) for _ in range(4)] for _ in range(NB)]
        ph = 0
        for kt in cutlass.range_constexpr(num_kt):
            buf = kt % nstage
            cute.arch.mbarrier_wait(full + buf, ph)
            brow_lo = lane % 8
            boff = ((lane // 8) & 1) * 4
            for step in cutlass.range_constexpr(KSTEPS):
                kbh = warp * (PWK // 2) + step * 8
                arow, aoff = lane % 16, (lane // 16) * 4
                a = _ldmatrix_x4(
                    sA.iterator
                    + (buf * TILE_M * KI + arow * KI + _swizzle_343(arow, kbh + aoff))
                )
                for nb in cutlass.range_constexpr(NB):
                    brow = nb * 8 + brow_lo
                    bb = _ldmatrix_x2(
                        sB.iterator
                        + (
                            buf * tile_n * KI
                            + brow * KI
                            + _swizzle_343(brow, kbh + boff)
                        )
                    )
                    d = _mma_m16n8k16(
                        a[0],
                        a[1],
                        a[2],
                        a[3],
                        bb[0],
                        bb[1],
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
def _dsv3_fused_a_gemm_host(
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
    _dsv3_fused_a_gemm_kernel(mW, mA, mOut, M, num_kt, nstage, tile_n).launch(
        grid=[gemm_m // TILE_M, 1, 1],
        block=[NTHREADS, 1, 1],
        max_number_threads=[NTHREADS, 1, 1],
        min_blocks_per_mp=1,
        smem=smem_bytes,
        use_pdl=True,
        stream=stream,
    )


_compiled: dict[tuple[int, int, int], object] = {}


def _pick_nstage(num_kt: int, tile_n: int) -> int:
    nstage = (get_smem_capacity_in_bytes() // 4 - _BAR_I32) // _stage_i32(tile_n)
    return min(nstage, MAX_NSTAGE, num_kt)


def _pick_tile_n(num_tokens: int) -> int:
    return 8 if num_tokens <= 8 else 16


def _compiled_kernel(num_kt: int, gemm_m: int, tile_n: int):
    if get_device_sm() < 90:
        raise RuntimeError("dsv3_fused_a_gemm requires SM90 (Hopper) or later")
    if (num_kt, gemm_m, tile_n) not in _compiled:
        nstage = _pick_nstage(num_kt, tile_n)
        smem_bytes = (_BAR_I32 + nstage * _stage_i32(tile_n)) * 4
        k = num_kt * TILE_K
        w = torch.empty(gemm_m, k, dtype=torch.bfloat16, device="cuda")
        a = torch.empty(16, k, dtype=torch.bfloat16, device="cuda")
        o = torch.empty(16, gemm_m, dtype=torch.bfloat16, device="cuda")
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        _compiled[(num_kt, gemm_m, tile_n)] = cute.compile(
            _dsv3_fused_a_gemm_host,
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


def _dsv3_fused_a_gemm_run(mat_a: torch.Tensor, mat_b: torch.Tensor) -> torch.Tensor:
    M, K = mat_a.shape
    N = mat_b.shape[1]
    assert mat_a.dtype == torch.bfloat16 and mat_b.dtype == torch.bfloat16
    assert K % 1024 == 0, f"K must be a multiple of 1024, got {K}"
    assert N % TILE_M == 0, f"N must be a multiple of {TILE_M}, got {N}"
    assert (
        tuple(mat_b.shape) == (K, N) and mat_b.stride(0) == 1
    ), "mat_b must be [K, N] column-major"
    assert 1 <= M <= 16, "num_tokens must be in [1, 16]"
    assert mat_a.stride(1) == 1, "mat_a must be row-major [M, K]"

    weight = mat_b.t()
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


def _dsv3_fused_a_gemm_fake(mat_a: torch.Tensor, mat_b: torch.Tensor) -> torch.Tensor:
    return mat_a.new_empty((mat_a.shape[0], mat_b.shape[1]), dtype=torch.bfloat16)


direct_register_custom_op(
    op_name="cutedsl_dsv3_fused_a_gemm",
    op_func=_dsv3_fused_a_gemm_run,
    mutates_args=[],
    fake_impl=_dsv3_fused_a_gemm_fake,
)


@debug_kernel_api
def dsv3_fused_a_gemm(
    mat_a: torch.Tensor, mat_b: torch.Tensor, output: torch.Tensor | None = None
) -> torch.Tensor:
    """out[M, N] = mat_a[M, K] @ mat_b, with mat_a row-major [M, K] (M in [1, 16]),
    mat_b column-major [K, N] (the weight, stride(0) == 1), N a multiple of 16
    (e.g. 2112, 6144), K a multiple of 1024."""
    result = torch.ops.sglang.cutedsl_dsv3_fused_a_gemm(mat_a, mat_b)
    if output is not None:
        output.copy_(result)
        return output
    return result
