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
"""CuTe DSL DeepSeek-V3 fused-A GEMM (sm90+): out[M, 2112] = mat_a[M, K] @ weight,
M = num_tokens in [1, 16], K any multiple of 1024, bf16.

Adapted from NVIDIA TensorRT-LLM dsv3FusedAGemm.cu
(cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3FusedAGemm.cu), reimplemented
in the CuTe DSL: AB-swap, warp-specialized 4-way split-K, cp.async + mbarrier
pipeline, ldmatrix + mma.sync.m16n8k16, 3-4-3 swizzle.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass.cute.runtime import from_dlpack
from cutlass._mlir.dialects import llvm
from cutlass._mlir import ir

from sglang.srt.utils import get_device_sm

GEMM_M = 2112
TILE_M = 16
TILE_N = 16
NB = TILE_N // 8
TILE_K = 256
SPLITK = 4
LOAD_WARPS = 8
MAX_NSTAGE = 16                 # upper bound on runtime-chosen pipeline depth
PWK = TILE_K // SPLITK
KSTEPS = PWK // 16
COMPUTE_THREADS = SPLITK * 32
LOADER_THREADS = LOAD_WARPS * 32
NTHREADS = COMPUTE_THREADS + LOADER_THREADS
KI = TILE_K // 2                # int32 cols per K-tile (a bf16 pair per int32)

_BAR_I32 = 2 * MAX_NSTAGE * 2   # full[] + empty[] mbarriers (int64 = 2 int32 each)
_SC_I32 = SPLITK * TILE_M * TILE_N
_STAGE_I32 = (TILE_M + TILE_N) * KI


def _cp_async(smem_ptr, gmem_ptr):
    llvm.inline_asm(
        None, [smem_ptr.toint().ir_value(), gmem_ptr.toint().ir_value()],
        "{ .reg .u32 sa; cvt.u32.u64 sa, $0; cp.async.cg.shared.global [sa], [$1], 16; }",
        "l,l", has_side_effects=True, is_align_stack=False, asm_dialect=0,
    )


def _cp_async_pred(smem_ptr, gmem_ptr, pred_i32):
    llvm.inline_asm(
        None, [smem_ptr.toint().ir_value(), gmem_ptr.toint().ir_value(), pred_i32.ir_value()],
        "{ .reg .pred p; .reg .u32 sa; setp.ne.s32 p, $2, 0; cvt.u32.u64 sa, $0; "
        "@p cp.async.cg.shared.global [sa], [$1], 16; }",
        "l,l,r", has_side_effects=True, is_align_stack=False, asm_dialect=0,
    )


def _ldmatrix(smem_ptr):
    i32 = ir.IntegerType.get_signless(32)
    res = llvm.inline_asm(
        llvm.StructType.get_literal([i32, i32, i32, i32]), [smem_ptr.toint().ir_value()],
        "{ .reg .u32 sa; cvt.u32.u64 sa, $4; "
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {$0,$1,$2,$3}, [sa]; }",
        "=r,=r,=r,=r,l", has_side_effects=True, is_align_stack=False, asm_dialect=0,
    )
    return [llvm.extractvalue(i32, res, [i]) for i in range(4)]


def _mma(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3):
    f32 = ir.F32Type.get()
    res = llvm.inline_asm(
        llvm.StructType.get_literal([f32, f32, f32, f32]),
        [a0, a1, a2, a3, b0, b1, c0, c1, c2, c3],
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=0,
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


def _load_tile(ltid, feat0, sa, sb, mW, mA, tile, buf, M, kgi):
    for it in range(TILE_M * KI // (LOADER_THREADS * 4)):
        idx = (it * LOADER_THREADS + ltid) * 4
        row, col = idx // KI, idx % KI
        _cp_async(sa.iterator + (buf * TILE_M * KI + row * KI + _swizzle(row, col)),
                  mW.iterator + ((feat0 + row) * kgi + _k_project(tile, col, kgi)))
    for it in range(TILE_N * KI // (LOADER_THREADS * 4)):
        idx = (it * LOADER_THREADS + ltid) * 4
        row, col = idx // KI, idx % KI
        pred = (row < M).to(cutlass.Int32)
        _cp_async_pred(sb.iterator + (buf * TILE_N * KI + row * KI + _swizzle(row, col)),
                       mA.iterator + (row * pred * kgi + _k_project(tile, col, kgi)), pred)


@cute.kernel
def _kernel(mW: cute.Tensor, mA: cute.Tensor, mOut: cute.Tensor, M: cutlass.Int32,
            num_kt: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    warp, lane = tid // 32, tid % 32
    r0, cc = lane // 4, lane % 4
    feat0 = bid * TILE_M
    kgi = num_kt * KI

    # Carve the pipeline ring from raw dynamic smem; the stage count adapts to the
    # device's available shared memory (queried in bytes via get_dyn_smem_size).
    smem_i32 = cute.arch.get_dyn_smem_size() // 4
    nstage = (smem_i32 - _BAR_I32 - _SC_I32) // _STAGE_I32
    nstage = nstage if nstage < MAX_NSTAGE else MAX_NSTAGE
    nstage = nstage if nstage < num_kt else num_kt

    base = cute.arch.get_dyn_smem(cutlass.Int32, alignment=16)
    bar = cute.recast_ptr(base, dtype=cutlass.Int64)
    full, empty = bar, bar + MAX_NSTAGE
    sC = cute.make_tensor(cute.recast_ptr(base + _BAR_I32, dtype=cutlass.Float32),
                          cute.make_layout((SPLITK, TILE_M, TILE_N), stride=(TILE_M * TILE_N, TILE_N, 1)))
    sa_off = _BAR_I32 + _SC_I32
    sA = cute.make_tensor(base + sa_off,
                          cute.make_layout((nstage, TILE_M, KI), stride=(TILE_M * KI, KI, 1)))
    sB = cute.make_tensor(base + sa_off + nstage * TILE_M * KI,
                          cute.make_layout((nstage, TILE_N, KI), stride=(TILE_N * KI, KI, 1)))

    if tid == 0:
        for s in range(nstage):
            cute.arch.mbarrier_init(full + s, LOADER_THREADS)
            cute.arch.mbarrier_init(empty + s, COMPUTE_THREADS)
    cute.arch.barrier()

    cute.arch.griddepcontrol_wait()

    if warp >= SPLITK:
        ltid = tid - COMPUTE_THREADS
        ph = 1
        for kt in cutlass.range_constexpr(num_kt):
            st = kt % nstage
            if kt >= nstage:
                cute.arch.mbarrier_wait(empty + st, ph)
            _load_tile(ltid, feat0, sA, sB, mW, mA, kt, st, M, kgi)
            cute.arch.cp_async_mbarrier_arrive_noinc(full + st)
            ph = (ph ^ 1) if st == nstage - 1 else ph
    else:
        acc = [[cutlass.Float32(0.0) for _ in range(4)] for _ in range(NB)]
        ph = 0
        for kt in cutlass.range_constexpr(num_kt):
            buf = kt % nstage
            cute.arch.mbarrier_wait(full + buf, ph)
            for step in cutlass.range_constexpr(KSTEPS):
                kbh = warp * (PWK // 2) + step * 8
                arow, aoff = lane % 16, (lane // 16) * 4
                a = _ldmatrix(sA.iterator + (buf * TILE_M * KI + arow * KI + _swizzle(arow, kbh + aoff)))
                for nb in cutlass.range_constexpr(NB):
                    brow = nb * 8 + r0
                    b0 = sB[buf, brow, _swizzle(brow, kbh + cc)].ir_value()
                    b1 = sB[buf, brow, _swizzle(brow, kbh + 4 + cc)].ir_value()
                    d = _mma(a[0], a[1], a[2], a[3], b0, b1,
                             acc[nb][0].ir_value(), acc[nb][1].ir_value(),
                             acc[nb][2].ir_value(), acc[nb][3].ir_value())
                    for i in cutlass.range_constexpr(4):
                        acc[nb][i] = cutlass.Float32(d[i])
            cute.arch.mbarrier_arrive(empty + buf)
            ph = (ph ^ 1) if buf == nstage - 1 else ph
        for nb in cutlass.range_constexpr(NB):
            for i in cutlass.range_constexpr(4):
                m = r0 + (8 if i >= 2 else 0)
                n = nb * 8 + cc * 2 + (i % 2)
                sC[warp, m, n] = acc[nb][i]

    cute.arch.barrier()

    nred = TILE_M * TILE_N
    for it in cutlass.range_constexpr((nred + NTHREADS - 1) // NTHREADS):
        e = it * NTHREADS + tid
        if e < nred:
            m, n = e // TILE_N, e % TILE_N
            s = sC[0, m, n]
            for w in cutlass.range_constexpr(1, SPLITK):
                s = s + sC[w, m, n]
            if n < M:
                mOut[n, feat0 + m] = s.to(cutlass.BFloat16)

    cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def _launch(mW: cute.Tensor, mA: cute.Tensor, mOut: cute.Tensor, M: cutlass.Int32,
            stream: cuda.CUstream, num_kt: cutlass.Constexpr, smem_bytes: cutlass.Constexpr):
    _kernel(mW, mA, mOut, M, num_kt).launch(
        grid=[GEMM_M // TILE_M, 1, 1], block=[NTHREADS, 1, 1], smem=smem_bytes,
        use_pdl=True, stream=stream)


_compiled: dict[int, object] = {}


def _compiled_kernel(num_kt: int):
    if get_device_sm() < 90:
        raise RuntimeError("dsv3_fused_a_gemm requires SM90 (Hopper) or later")
    if num_kt not in _compiled:
        smem_bytes = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
        k = num_kt * TILE_K
        w = torch.empty(GEMM_M, k, dtype=torch.bfloat16, device="cuda")
        a = torch.empty(16, k, dtype=torch.bfloat16, device="cuda")
        o = torch.empty(16, GEMM_M, dtype=torch.bfloat16, device="cuda")
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        _compiled[num_kt] = cute.compile(
            _launch, from_dlpack(w.view(torch.int32)), from_dlpack(a.view(torch.int32)),
            from_dlpack(o), cutlass.Int32(16), stream, num_kt, smem_bytes)
    return _compiled[num_kt]


def dsv3_fused_a_gemm(mat_a: torch.Tensor, mat_b: torch.Tensor,
                      out: torch.Tensor | None = None) -> torch.Tensor:
    """out[M, 2112] = mat_a[M, K] @ mat_b, with mat_a row-major [M, K] (M in [1, 16]),
    mat_b column-major [K, 2112] (the weight, stride(0) == 1), K a multiple of 1024."""
    M, K = mat_a.shape
    assert mat_a.dtype == torch.bfloat16 and mat_b.dtype == torch.bfloat16
    assert K % 1024 == 0, f"K must be a multiple of 1024, got {K}"
    assert tuple(mat_b.shape) == (K, GEMM_M) and mat_b.stride(0) == 1, "mat_b must be [K, 2112] column-major"
    assert 1 <= M <= 16, "num_tokens must be in [1, 16]"

    mat_a = mat_a.contiguous()
    weight = mat_b.t()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if out is None:
        out = torch.empty(M, GEMM_M, dtype=torch.bfloat16, device=mat_a.device)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    _compiled_kernel(K // TILE_K)(
        from_dlpack(weight.view(torch.int32)), from_dlpack(mat_a.view(torch.int32)),
        from_dlpack(out), M, stream)
    return out
