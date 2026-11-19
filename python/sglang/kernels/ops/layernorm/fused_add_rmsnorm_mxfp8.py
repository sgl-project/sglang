"""FlashInfer's CuTe-DSL fused add + RMSNorm with an MXFP8 output.

The add, the reduction and ``y = h * rstd * (w + weight_bias)`` are FlashInfer's
``FusedAddRMSNormKernel`` (same tiling, same reduction helpers), so the bf16 ``y``
it would store is bitwise flashinfer's. Instead of storing ``y`` the epilogue
quantizes it with ``flashinfer.mxfp8_quantize``'s own helpers: E4M3 data and
128x4-swizzled UE8M0 scales, bitwise that kernel's output for ``y``.
"""

from __future__ import annotations

import functools
import math

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint8, Uint32
from flashinfer.cute_dsl.fp4_common import get_ptr_as_int64, st_global_u64
from flashinfer.norm.kernels.fused_add_rmsnorm import FusedAddRMSNormKernel
from flashinfer.norm.kernels.rmsnorm import RMSNormKernel
from flashinfer.norm.utils import get_sm_version, predicate_k, row_reduce_sum_multirow
from flashinfer.quantization.quantization_cute_dsl_utils import (
    INV_FLOAT8_E4M3_MAX,
    bfloat2_hmax_reduce_to_f32,
    bfloat2_max_abs_4,
    bfloat2x4_to_fp8x8_packed,
    compute_sf_index_swizzled_128x4_gpu,
    float_to_ue8m0_fast,
    reduce_max_4threads,
    ue8m0_to_inv_scale_fast,
)


class _FusedAddRMSNormMxfp8Kernel(FusedAddRMSNormKernel):
    def __init__(self, H: int, weight_bias: float, sm_version: int):
        super().__init__(cutlass.BFloat16, H, weight_bias, sm_version=sm_version)
        assert self.cluster_n == 1 and self.vec_size == 8
        assert self.threads_per_row % 4 == 0
        assert self.H % (self.vec_size * self.threads_per_row) == 0

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        mQ: cute.Tensor,
        mS: cute.Tensor,
        M: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        tv_shape, tv_stride = RMSNormKernel._make_tv_layout(
            self.threads_per_row,
            self.rows_per_block,
            self.vec_size,
            self.num_vec_blocks,
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (self.rows_per_block, self.cols_per_tile)
        self.kernel(mX, mR, mW, mQ, mS, M, eps, enable_pdl, tv_layout, tiler_mn).launch(
            grid=[cute.ceil_div(M, self.rows_per_block), 1, 1],
            block=[self.num_threads, 1, 1],
            smem=self._smem_size_in_bytes(),
            stream=stream,
            use_pdl=enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        mQ: cute.Tensor,
        mS: cute.Tensor,
        M: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        if enable_pdl:
            cute.arch.griddepcontrol_wait()

        H = self.H
        weight_bias = self.weight_bias
        copy_bits = self.copy_bits
        threads_per_row = tv_layout.shape[0][0]
        rows_per_block = tiler_mn[0]
        warps_per_row = max(threads_per_row // 32, 1)

        # ===== Pass 1: FusedAddRMSNormKernel.kernel, cluster_n == 1 =====
        smem = cutlass.utils.SmemAllocator()
        if cutlass.const_expr(self.use_async_copy):
            sX = smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )
            sR = smem.allocate_tensor(
                mR.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )
        reduction_buffer = smem.allocate_tensor(
            Float32,
            cute.make_layout((rows_per_block, warps_per_row)),
            byte_alignment=4,
        )

        idX = cute.make_identity_tensor(mX.shape)
        gX = cute.local_tile(mX, tiler_mn, (bidx, 0))
        gR = cute.local_tile(mR, tiler_mn, (bidx, 0))
        cX = cute.local_tile(idX, tiler_mn, (bidx, 0))
        mW_expanded_layout = cute.prepend(
            mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))
        )
        mW_2d = cute.make_tensor(mW.iterator, mW_expanded_layout)
        gW = cute.local_tile(mW_2d, tiler_mn, (0, 0))

        copy_atom_sync = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mX.element_type, num_bits_per_copy=copy_bits
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mX.element_type, num_bits_per_copy=copy_bits
        )
        if cutlass.const_expr(self.use_async_copy):
            copy_atom_async = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mX.element_type,
                num_bits_per_copy=copy_bits,
            )
            tiled_copy_load = cute.make_tiled_copy(copy_atom_async, tv_layout, tiler_mn)
        else:
            tiled_copy_load = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)
        tiled_copy_W = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)
        tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)

        thr_copy_X = tiled_copy_load.get_slice(tidx)
        thr_copy_W = tiled_copy_W.get_slice(tidx)
        thr_copy_O = tiled_copy_store.get_slice(tidx)

        tXgX = thr_copy_X.partition_S(gX)
        tXcX = thr_copy_X.partition_S(cX)
        tXrX = cute.make_fragment_like(tXgX)
        tRgR = thr_copy_X.partition_S(gR)
        tRrR = cute.make_fragment_like(tRgR)
        if cutlass.const_expr(self.use_async_copy):
            tXsX = thr_copy_X.partition_D(sX)
            tRsR = thr_copy_X.partition_D(sR)
        tWgW = thr_copy_W.partition_S(gW)
        tWrW = cute.make_fragment_like(tWgW)
        tXrW = thr_copy_X.retile(tWrW)
        tXgO = thr_copy_O.partition_D(gX)
        tRgO = thr_copy_O.partition_D(gR)
        tXrO = cute.make_fragment_like(tXgO)

        tXpX = predicate_k(tXcX, limit=H)
        tWpW = predicate_k(thr_copy_W.partition_S(cX), limit=H)
        row_coord = tXcX[(0, 0), 0, 0]
        row_in_bounds = row_coord[0] < M

        if cutlass.const_expr(self.use_async_copy):
            if row_in_bounds:
                cute.copy(copy_atom_async, tXgX, tXsX, pred=tXpX)
                cute.copy(copy_atom_async, tRgR, tRsR, pred=tXpX)
            cute.arch.cp_async_commit_group()
            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)
            cute.arch.cp_async_wait_group(0)
            cute.autovec_copy(tXsX, tXrX)
            cute.autovec_copy(tRsR, tRrR)
        else:
            tXrX.store(cute.zeros_like(tXrX, dtype=mX.element_type))
            tRrR.store(cute.zeros_like(tRrR, dtype=mR.element_type))
            if row_in_bounds:
                cute.copy(copy_atom_sync, tXgX, tXrX, pred=tXpX)
                cute.copy(copy_atom_sync, tRgR, tRrR, pred=tXpX)
            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

        x_in = tXrX.load().to(Float32)
        r_in = tRrR.load().to(Float32)
        h = x_in + r_in
        tXrO.store(h.to(mR.element_type))
        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tRgO, pred=tXpX)

        h_sq = h * h
        sum_sq = row_reduce_sum_multirow(
            h_sq, threads_per_row, reduction_buffer, None, 1
        )
        mean_sq = sum_sq / Float32(H)
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)
        cute.arch.barrier()

        w = tXrW.load().to(Float32)
        y = h * rstd * (w + Float32(weight_bias))
        tXrO.store(y.to(mX.element_type))

        # ===== Pass 2: flashinfer.mxfp8_quantize of the bf16 y =====
        # Thread t0 of a row holds columns (b * threads_per_row + t0) * 8 + [0, 8),
        # so four consecutive lanes hold one 32-wide scale group.
        y_u32 = cute.recast_tensor(tXrO, Uint32)
        t0 = tidx % threads_per_row
        row = Int32(row_coord[0])
        sf_cols = Int32(H // 32)
        for b in cutlass.range_constexpr(self.num_vec_blocks):
            v0 = y_u32[4 * b]
            v1 = y_u32[4 * b + 1]
            v2 = y_u32[4 * b + 2]
            v3 = y_u32[4 * b + 3]
            local_max = bfloat2_hmax_reduce_to_f32(bfloat2_max_abs_4(v0, v1, v2, v3))
            global_max = reduce_max_4threads(local_max)
            sf = float_to_ue8m0_fast(global_max * Float32(INV_FLOAT8_E4M3_MAX))
            packed = bfloat2x4_to_fp8x8_packed(
                v0, v1, v2, v3, ue8m0_to_inv_scale_fast(sf)
            )
            col = Int32((b * threads_per_row + t0) * 8)
            if row_in_bounds:
                st_global_u64(get_ptr_as_int64(mQ[row, None], col), packed)
                if t0 % 4 == 0:
                    sf_idx = compute_sf_index_swizzled_128x4_gpu(
                        row, col // Int32(32), sf_cols
                    )
                    mS[sf_idx] = sf.to(Uint8)

        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()


@functools.cache
def _compiled(H: int, weight_bias: float, enable_pdl: bool, sm_version: int):
    kernel = _FusedAddRMSNormMxfp8Kernel(H, weight_bias, sm_version)
    sym_m = cute.sym_int(64)
    sym_sf = cute.sym_int()
    align = math.gcd(128, H * 2)
    bf16 = cutlass.BFloat16
    x = cute.runtime.make_fake_compact_tensor(
        bf16, (sym_m, H), stride_order=(1, 0), assumed_align=align
    )
    r = cute.runtime.make_fake_compact_tensor(
        bf16, (sym_m, H), stride_order=(1, 0), assumed_align=align
    )
    w = cute.runtime.make_fake_compact_tensor(bf16, (H,), assumed_align=16)
    q = cute.runtime.make_fake_compact_tensor(
        cutlass.Float8E4M3FN, (sym_m, H), stride_order=(1, 0), assumed_align=16
    )
    s = cute.runtime.make_fake_compact_tensor(Uint8, (sym_sf,), assumed_align=16)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        x,
        r,
        w,
        q,
        s,
        Int64(1),
        Float32(1e-6),
        enable_pdl,
        stream,
        options="--enable-tvm-ffi",
    )


def _reference_runs_cute_norm() -> bool:
    """sgl_kernel.gemma_fused_add_rmsnorm, GemmaRMSNorm's kernel, forwards to
    flashinfer's CuTe-DSL kernel this one copies; its CUDA fallback reduces in
    another order, so the bytes would differ."""
    try:
        import flashinfer.norm as flashinfer_norm
        from sgl_kernel import elementwise

        return bool(elementwise._has_flashinfer) and not flashinfer_norm._use_cuda_norm(
            torch.device("cuda", torch.cuda.current_device())
        )
    except (ImportError, AttributeError):
        return False


def supports_fused_add_rmsnorm_mxfp8(hidden_size: int) -> bool:
    if hidden_size % 128 != 0 or not _reference_runs_cute_norm():
        return False
    try:
        _FusedAddRMSNormMxfp8Kernel(hidden_size, 1.0, get_sm_version())
    except AssertionError:
        return False
    return True


def gemma_fused_add_rmsnorm_mxfp8(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``residual += input`` in place (as ``gemma_fused_add_rmsnorm``), and the
    Gemma RMSNorm of it as MXFP8 ``([T, H] e4m3, 128x4 UE8M0 scales)``, bitwise
    ``mxfp8_quantize(is_sf_swizzled_layout=True)`` of the bf16 norm that
    ``flashinfer.gemma_fused_add_rmsnorm`` writes into ``input``; ``input`` is left as is."""
    m, h = input.shape
    assert input.dtype == residual.dtype == weight.dtype == torch.bfloat16
    assert input.is_contiguous() and residual.is_contiguous() and m * h < 2**31
    quantized = torch.empty(m, h, dtype=torch.float8_e4m3fn, device=input.device)
    padded = (m + 127) // 128 * 128
    scales = torch.empty(padded * (h // 32), dtype=torch.uint8, device=input.device)
    if padded != m:
        scales[(padded - 128) * (h // 32) :].zero_()
    from flashinfer.utils import device_support_pdl

    kernel = _compiled(
        h, 1.0, device_support_pdl(input.device), get_sm_version(input.device)
    )
    kernel(input, residual, weight, quantized, scales, m, eps)
    return quantized, scales
