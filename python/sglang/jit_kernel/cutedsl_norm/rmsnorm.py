"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

RMSNorm CuTe DSL Kernels
========================

Includes:
- RMSNormKernel: Basic RMSNorm (also handles Gemma variant with weight_bias=1.0)
- QKRMSNormKernel: RMSNorm for 3D tensors [batch, heads, head_dim]
- RMSNormQuantKernel: RMSNorm + FP8 quantization
"""

import functools
import math
import operator

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64

from .utils import (
    COPY_BITS,
    FLOAT8_E4M3_MAX,
    _torch_dtype_to_str,
    cvt_and_store_2xf32_to_e4m3_hw,
    cvt_and_store_4xf32_to_e4m3_hw,
    cvt_and_store_8xf32_to_e4m3_hw,
    cvt_and_store_f32_to_e4m3_hw,
    cvt_and_store_f32_to_e4m3_sw,
    get_cutlass_dtype,
    get_ptr_as_int64,
    get_sm_version,
    has_hw_fp8_cvt,
    make_tv_layout,
    predicate_k,
    rcp_approx_ftz,
    row_reduce_sum_multirow,
    warp_reduce,
)

# =============================================================================
# RMSNormKernel
# =============================================================================


class RMSNormKernel:
    """
    RMSNorm Kernel using CuTe-DSL.

    Computes: output = input / sqrt(mean(input^2) + eps) * (weight + weight_bias)
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        H: int,
        weight_bias: float = 0.0,
        sm_version: int | None = None,
    ):
        self.dtype = dtype
        self.H = H
        self.weight_bias = weight_bias
        self.sm_version = sm_version if sm_version is not None else get_sm_version()

        self.cluster_n = self._compute_cluster_n(H, dtype, self.sm_version)
        self.H_per_cta = H // self.cluster_n

        elem_bytes = dtype.width // 8
        max_vec_size = COPY_BITS // 8 // elem_bytes

        h_align = self.H_per_cta & (-self.H_per_cta)
        self.vec_size = min(h_align, max_vec_size)
        self.copy_bits = self.vec_size * dtype.width

        self.threads_per_row = self._compute_threads_per_row(self.H_per_cta)
        self.num_threads = self._compute_num_threads(self.H_per_cta)
        self.rows_per_block = self.num_threads // self.threads_per_row
        self.warps_per_row = max(self.threads_per_row // 32, 1)

        self.num_vec_blocks = max(
            1,
            (self.H_per_cta // self.vec_size + self.threads_per_row - 1)
            // self.threads_per_row,
        )
        self.cols_per_tile = self.vec_size * self.num_vec_blocks * self.threads_per_row

        if self.copy_bits >= 32:
            tile_bytes = self.rows_per_block * self.cols_per_tile * elem_bytes
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            self.use_async_copy = tile_bytes <= props.shared_memory_per_block_optin // 2
        else:
            self.use_async_copy = False

    @staticmethod
    def _compute_cluster_n(H: int, dtype: cutlass.Numeric, sm_version: int) -> int:
        """Compute optimal cluster size based on H and device shared memory."""
        if sm_version < 90:
            return 1

        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        max_smem_bytes = props.shared_memory_per_block_optin
        elem_size = dtype.width // 8

        for cluster_n in [1, 2, 4, 8, 16]:
            if H % cluster_n != 0:
                continue
            smem_needed = RMSNormKernel._estimate_smem_bytes(H, cluster_n, elem_size)
            if smem_needed <= max_smem_bytes:
                return cluster_n

        return 16

    @staticmethod
    def _estimate_smem_bytes(H: int, cluster_n: int, elem_size: int) -> int:
        """Estimate shared memory bytes for a given cluster configuration."""
        H_per_cta = H // cluster_n
        threads_per_row = RMSNormKernel._compute_threads_per_row(H_per_cta)
        num_threads = RMSNormKernel._compute_num_threads(H_per_cta)
        rows_per_block = num_threads // threads_per_row
        warps_per_row = max(threads_per_row // 32, 1)

        max_vec_size = COPY_BITS // 8 // elem_size
        h_align = H_per_cta & (-H_per_cta)
        vec_size = min(h_align, max_vec_size)
        num_vec_blocks = max(
            1, (H_per_cta // vec_size + threads_per_row - 1) // threads_per_row
        )
        cols_per_tile = vec_size * num_vec_blocks * threads_per_row

        tile_bytes = rows_per_block * cols_per_tile * elem_size

        if cluster_n == 1:
            return tile_bytes + rows_per_block * warps_per_row * 4
        else:
            return (
                tile_bytes
                + rows_per_block * warps_per_row * cluster_n * 4
                + 8  # mbarrier
            )

    @staticmethod
    def _compute_threads_per_row(H: int) -> int:
        if H <= 64:
            return 8
        elif H <= 128:
            return 16
        elif H <= 3072:
            return 32
        elif H <= 6144:
            return 64
        elif H <= 16384:
            return 128
        else:
            return 256

    @staticmethod
    def _compute_num_threads(H: int) -> int:
        return 128 if H <= 16384 else 256

    @staticmethod
    def _make_tv_layout(threads_per_row, rows_per_block, vec_size, num_vec_blocks):
        """Create Thread-Value layout for multi-row coalesced vectorized access."""
        shape = (
            (threads_per_row, rows_per_block),
            (vec_size, num_vec_blocks),
        )
        stride = (
            (vec_size * rows_per_block, 1),
            (rows_per_block, rows_per_block * vec_size * threads_per_row),
        )
        return shape, stride

    def _smem_size_in_bytes(self) -> int:
        if self.use_async_copy:
            tile_bytes = (
                self.rows_per_block * self.cols_per_tile * (self.dtype.width // 8)
            )
        else:
            tile_bytes = 0

        if self.cluster_n == 1:
            reduction_bytes = self.rows_per_block * self.warps_per_row * 4
        else:
            reduction_bytes = (
                self.rows_per_block * self.warps_per_row * self.cluster_n * 4
            )

        mbar_bytes = 8 if self.cluster_n > 1 else 0
        return tile_bytes + reduction_bytes + mbar_bytes

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        M: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        tv_shape, tv_stride = self._make_tv_layout(
            self.threads_per_row,
            self.rows_per_block,
            self.vec_size,
            self.num_vec_blocks,
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (self.rows_per_block, self.cols_per_tile)

        cluster_n = self.cluster_n

        self.kernel(mX, mW, mY, M, eps, enable_pdl, tv_layout, tiler_mn).launch(
            grid=[cute.ceil_div(M, self.rows_per_block), cluster_n, 1],
            block=[self.num_threads, 1, 1],
            cluster=[1, cluster_n, 1] if cutlass.const_expr(cluster_n > 1) else None,
            smem=self._smem_size_in_bytes(),
            stream=stream,
            use_pdl=enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        M: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        # PDL: Wait for previous kernel (SM90+ only)
        if enable_pdl:
            cute.arch.griddepcontrol_wait()

        H = self.H
        cluster_n = self.cluster_n
        weight_bias = self.weight_bias
        copy_bits = self.copy_bits
        threads_per_row = tv_layout.shape[0][0]
        rows_per_block = tiler_mn[0]
        warps_per_row = max(threads_per_row // 32, 1)

        if cutlass.const_expr(cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = cutlass.const_expr(0)

        # ===== Allocate shared memory =====
        smem = cutlass.utils.SmemAllocator()

        if cutlass.const_expr(self.use_async_copy):
            sX = smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )

        if cutlass.const_expr(cluster_n == 1):
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, warps_per_row)),
                byte_alignment=4,
            )
            mbar_ptr = None
        else:
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, (warps_per_row, cluster_n))),
                byte_alignment=4,
            )
            mbar_ptr = smem.allocate_array(cutlass.Int64, num_elems=1)

        # ===== Initialize cluster =====
        if cutlass.const_expr(cluster_n > 1):
            if tidx == 0:
                cute.arch.mbarrier_init(mbar_ptr, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

        # ===== Coordinate tracking and tiling =====
        idX = cute.make_identity_tensor(mX.shape)

        gX = cute.local_tile(mX, tiler_mn, (bidx, cluster_y))
        gY = cute.local_tile(mY, tiler_mn, (bidx, cluster_y))
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

        mW_expanded_layout = cute.prepend(
            mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))
        )
        mW_2d = cute.make_tensor(mW.iterator, mW_expanded_layout)
        gW = cute.local_tile(mW_2d, tiler_mn, (0, cluster_y))

        # ===== Create TiledCopy atoms =====
        copy_atom_sync = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=copy_bits,
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mY.element_type,
            num_bits_per_copy=copy_bits,
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

        # Partition input
        tXgX = thr_copy_X.partition_S(gX)
        tXcX = thr_copy_X.partition_S(cX)
        tXrX = cute.make_fragment_like(tXgX)

        if cutlass.const_expr(self.use_async_copy):
            tXsX = thr_copy_X.partition_D(sX)

        # Partition weight (sync, separate tiled copy)
        tWgW = thr_copy_W.partition_S(gW)
        tWrW = cute.make_fragment_like(tWgW)
        tXrW = thr_copy_X.retile(tWrW)

        # Partition output
        tXgO = thr_copy_O.partition_D(gY)
        tXrO = cute.make_fragment_like(tXgO)

        # ===== Bounds checking =====
        tXpX = predicate_k(tXcX, limit=H)
        tWpW = predicate_k(thr_copy_W.partition_S(cX), limit=H)
        row_coord = tXcX[(0, 0), 0, 0]
        row_in_bounds = row_coord[0] < M

        # ===== Pass 1: Load input + compute sum of squares =====
        if cutlass.const_expr(self.use_async_copy):
            if row_in_bounds:
                cute.copy(copy_atom_async, tXgX, tXsX, pred=tXpX)
            cute.arch.cp_async_commit_group()

            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

            cute.arch.cp_async_wait_group(0)

            cute.autovec_copy(tXsX, tXrX)
        else:
            tXrX.store(cute.zeros_like(tXrX, dtype=mX.element_type))
            if row_in_bounds:
                cute.copy(copy_atom_sync, tXgX, tXrX, pred=tXpX)

            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

        x = tXrX.load().to(Float32)
        x_sq = x * x
        sum_sq = row_reduce_sum_multirow(
            x_sq, threads_per_row, reduction_buffer, mbar_ptr, cluster_n
        )

        mean_sq = sum_sq / Float32(H)
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        if cutlass.const_expr(cluster_n > 1):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier()

        # ===== Pass 2: Normalize and store output =====
        # Re-load x from shared memory to relieve register pressure.
        # Without this, x (up to 128 FP32 values/thread at large H) must
        # survive across the reduction + barrier, causing spills to local mem.
        if cutlass.const_expr(self.use_async_copy):
            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(Float32)

        w = tXrW.load().to(Float32)
        y = x * rstd * (w + Float32(weight_bias))

        tXrO.store(y.to(mY.element_type))

        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tXgO, pred=tXpX)

        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()


# =============================================================================
# QKRMSNormKernel
# =============================================================================


class QKRMSNormKernel:
    """
    QK RMSNorm Kernel using CuTe-DSL for 3D tensors [batch, heads, head_dim].

    Supports arbitrary stride (only stride[-1] == 1 required). Each block
    processes rows_per_block rows, where each row is a (batch, head) pair
    handled by threads_per_row threads.

    Architecture mirrors RMSNormKernel but uses per-row 3D->2D tiles instead
    of a single multi-row 2D tile, since 3D strides may be non-uniform across
    the batch/head dimensions.
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        head_dim: int,
        weight_bias: float = 0.0,
    ):
        self.dtype = dtype
        self.head_dim = head_dim
        self.weight_bias = weight_bias

        elem_bytes = dtype.width // 8
        max_vec_size = COPY_BITS // 8 // elem_bytes

        h_align = head_dim & (-head_dim)
        self.vec_size = min(h_align, max_vec_size)
        self.copy_bits = self.vec_size * dtype.width

        self.threads_per_row = RMSNormKernel._compute_threads_per_row(head_dim)
        self.num_threads = RMSNormKernel._compute_num_threads(head_dim)
        self.rows_per_block = self.num_threads // self.threads_per_row
        self.warps_per_row = max(self.threads_per_row // 32, 1)

        self.num_vec_blocks = max(
            1,
            (head_dim // self.vec_size + self.threads_per_row - 1)
            // self.threads_per_row,
        )
        self.cols_per_tile = self.vec_size * self.num_vec_blocks * self.threads_per_row

        if self.copy_bits >= 32:
            tile_bytes = self.rows_per_block * self.cols_per_tile * elem_bytes
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            self.use_async_copy = tile_bytes <= props.shared_memory_per_block_optin // 2
        else:
            self.use_async_copy = False

    def _smem_size_in_bytes(self) -> int:
        if self.use_async_copy:
            tile_bytes = (
                self.rows_per_block * self.cols_per_tile * (self.dtype.width // 8)
            )
        else:
            tile_bytes = 0
        reduction_bytes = self.rows_per_block * self.warps_per_row * 4
        return tile_bytes + reduction_bytes

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        B: Int64,
        N: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        tv_shape, tv_stride = make_tv_layout(
            self.threads_per_row, self.vec_size, self.num_vec_blocks
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_1d = (1, self.cols_per_tile)

        M = B * N

        self.kernel(mX, mW, mY, N, M, eps, enable_pdl, tv_layout, tiler_1d).launch(
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
        mW: cute.Tensor,
        mY: cute.Tensor,
        N: Int64,
        M: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        tv_layout: cute.Layout,
        tiler_1d: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        # PDL: Wait for previous kernel (SM90+ only)
        if enable_pdl:
            cute.arch.griddepcontrol_wait()

        head_dim = self.head_dim
        weight_bias = self.weight_bias
        copy_bits = self.copy_bits
        threads_per_row = self.threads_per_row
        rows_per_block = self.rows_per_block
        warps_per_row = self.warps_per_row

        # Each group of threads_per_row threads handles one row
        lane_in_row = tidx % threads_per_row
        row_in_block = tidx // threads_per_row
        actual_row = bidx * rows_per_block + row_in_block

        batch_idx = actual_row // N
        head_idx = actual_row % N
        row_in_bounds = actual_row < M

        # ===== Allocate shared memory =====
        smem = cutlass.utils.SmemAllocator()

        if cutlass.const_expr(self.use_async_copy):
            sX = smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(
                    (rows_per_block, self.cols_per_tile), order=(1, 0)
                ),
                byte_alignment=16,
            )

        reduction_buffer = smem.allocate_tensor(
            Float32,
            cute.make_layout((rows_per_block, warps_per_row)),
            byte_alignment=4,
        )
        mbar_ptr = None
        cluster_n = 1

        # ===== Per-row 3D -> 2D tiles =====
        gX_3d = cute.local_tile(
            mX, (1, 1, self.cols_per_tile), (batch_idx, head_idx, 0)
        )
        gX = cute.group_modes(gX_3d, 0, 2)

        gY_3d = cute.local_tile(
            mY, (1, 1, self.cols_per_tile), (batch_idx, head_idx, 0)
        )
        gY = cute.group_modes(gY_3d, 0, 2)

        mW_2d = cute.prepend_ones(mW, up_to_rank=2)

        # ===== Create TiledCopy atoms =====
        copy_atom_sync = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=copy_bits,
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mY.element_type,
            num_bits_per_copy=copy_bits,
        )

        if cutlass.const_expr(self.use_async_copy):
            copy_atom_async = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mX.element_type,
                num_bits_per_copy=copy_bits,
            )
            tiled_copy_load = cute.make_tiled_copy(copy_atom_async, tv_layout, tiler_1d)
        else:
            tiled_copy_load = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_1d)

        tiled_copy_W = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_1d)
        tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_1d)

        thr_copy_X = tiled_copy_load.get_slice(lane_in_row)
        thr_copy_W = tiled_copy_W.get_slice(lane_in_row)
        thr_copy_O = tiled_copy_store.get_slice(lane_in_row)

        # ===== Partition input =====
        tXgX = thr_copy_X.partition_S(gX)
        tXrX = cute.make_fragment_like(tXgX)

        if cutlass.const_expr(self.use_async_copy):
            sX_row = cute.local_tile(sX, tiler_1d, (row_in_block, 0))
            tXsX = thr_copy_X.partition_D(sX_row)

        # ===== Partition weight (sync, separate tiled copy) =====
        tWgW = thr_copy_W.partition_S(mW_2d)
        tWrW = cute.make_fragment_like(tWgW)
        tXrW = thr_copy_X.retile(tWrW)

        # ===== Partition output =====
        tXgO = thr_copy_O.partition_D(gY)
        tXrO = cute.make_fragment_like(tXgO)

        # ===== Bounds checking =====
        id1d = cute.make_identity_tensor(tiler_1d)
        tXpX = predicate_k(thr_copy_X.partition_S(id1d), limit=head_dim)
        tWpW = predicate_k(thr_copy_W.partition_S(id1d), limit=head_dim)

        # ===== Pass 1: Load input + compute sum of squares =====
        if cutlass.const_expr(self.use_async_copy):
            if row_in_bounds:
                cute.copy(copy_atom_async, tXgX, tXsX, pred=tXpX)
            cute.arch.cp_async_commit_group()

            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

            cute.arch.cp_async_wait_group(0)

            cute.autovec_copy(tXsX, tXrX)
        else:
            tXrX.store(cute.zeros_like(tXrX, dtype=mX.element_type))
            if row_in_bounds:
                cute.copy(copy_atom_sync, tXgX, tXrX, pred=tXpX)

            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

        x = tXrX.load().to(Float32)
        x_sq = x * x
        sum_sq = row_reduce_sum_multirow(
            x_sq, threads_per_row, reduction_buffer, mbar_ptr, cluster_n
        )

        mean_sq = sum_sq / Float32(head_dim)
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        cute.arch.barrier()

        # ===== Pass 2: Normalize and store output =====
        # Re-load x from shared memory to relieve register pressure.
        # Without this, x (up to 128 FP32 values/thread at large H) must
        # survive across the reduction + barrier, causing spills to local mem.
        if cutlass.const_expr(self.use_async_copy):
            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(Float32)

        w = tXrW.load().to(Float32)
        y = x * rstd * (w + Float32(weight_bias))

        tXrO.store(y.to(mY.element_type))

        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tXgO, pred=tXpX)

        # PDL: Signal dependent kernels (SM90+ only)
        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()


# =============================================================================
# RMSNormFusedParallelKernel
# =============================================================================


class RMSNormFusedParallelKernel:
    """Fused parallel RMSNorm over two independent 2D inputs sharing a row count.

    A single kernel normalizes ``input1`` ([M, d1], weight1) and ``input2``
    ([M, d2], weight2) in one launch: each block owns one row and splits its
    warps into two cooperative groups (``nw1`` warps for input1, ``nw2`` for
    input2). This replaces the dual-CUDA-stream q/k-norm overlap with a single
    launch whose two workloads stream concurrently on the same SM.

    Ported from flashinfer/norm.cuh ``RMSNormFusedParallelKernel``. Pure RMSNorm
    (no weight_bias). Supports arbitrary row strides (stride[-1] must be 1).
    """

    def __init__(self, dtype: cutlass.Numeric, d1: int, d2: int):
        self.dtype = dtype
        self.d1 = d1
        self.d2 = d2

        elem_bytes = dtype.width // 8
        max_vec_size = COPY_BITS // 8 // elem_bytes
        # One vec_size for both inputs so a single copy atom serves both groups;
        # gcd with both d's guarantees no partial vectors.
        self.vec_size = math.gcd(max_vec_size, math.gcd(d1, d2))
        self.copy_bits = self.vec_size * dtype.width
        self.align_bytes = self.copy_bits // 8

        self.nt1, self.nt2 = self._alloc_threads(d1, d2, self.vec_size)
        self.num_threads = self.nt1 + self.nt2
        self.nw1 = self.nt1 // 32
        self.nw2 = self.nt2 // 32
        self.num_warps = self.nw1 + self.nw2
        self.rounds1 = (d1 + self.vec_size * self.nt1 - 1) // (self.vec_size * self.nt1)
        self.rounds2 = (d2 + self.vec_size * self.nt2 - 1) // (self.vec_size * self.nt2)

    @staticmethod
    def _alloc_threads(d1: int, d2: int, vec: int) -> tuple[int, int]:
        """Split <=1024 threads between the two inputs proportionally to their
        per-input thread demand, rounded to whole warps. Mirrors norm.cuh."""
        min1 = (d1 + vec - 1) // vec
        min2 = (d2 + vec - 1) // vec
        total = min1 + min2
        if total <= 1024:
            nt1, nt2 = min1, min2
        else:
            ratio = min1 / total
            nt1 = max(32, int(1024 * ratio))
            nt2 = max(32, 1024 - nt1)
        nt1 = (nt1 + 31) // 32 * 32
        nt2 = (nt2 + 31) // 32 * 32
        while nt1 + nt2 > 1024:
            if nt1 > nt2:
                nt1 -= 32
            else:
                nt2 -= 32
        return nt1, nt2

    def _smem_size_in_bytes(self) -> int:
        return 2 * self.num_warps * 4

    @cute.jit
    def _load_sum_sq(
        self,
        mX: cute.Tensor,
        bidx: Int32,
        local_tid: Int32,
        d: cutlass.Constexpr[int],
        nt: cutlass.Constexpr[int],
        rounds: cutlass.Constexpr[int],
        copy_atom,
    ) -> Float32:
        vec = self.vec_size
        sum_sq = Float32(0.0)
        for i in cutlass.range_constexpr(rounds):
            col = (i * nt + local_tid) * vec
            if col < d:
                ptr = (mX.iterator + cute.crd2idx((bidx, col), mX.layout)).align(
                    self.align_bytes
                )
                vsrc = cute.make_tensor(ptr, cute.make_layout((vec,), stride=(1,)))
                frag = cute.make_rmem_tensor((vec,), mX.element_type)
                cute.copy(copy_atom, vsrc, frag)
                x = frag.load().to(Float32)
                sum_sq += (x * x).reduce(
                    cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0
                )
        return sum_sq

    @cute.jit
    def _normalize_store(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        bidx: Int32,
        local_tid: Int32,
        rstd: Float32,
        d: cutlass.Constexpr[int],
        nt: cutlass.Constexpr[int],
        rounds: cutlass.Constexpr[int],
        copy_atom,
    ):
        vec = self.vec_size
        for i in cutlass.range_constexpr(rounds):
            col = (i * nt + local_tid) * vec
            if col < d:
                xptr = (mX.iterator + cute.crd2idx((bidx, col), mX.layout)).align(
                    self.align_bytes
                )
                fx = cute.make_rmem_tensor((vec,), mX.element_type)
                cute.copy(
                    copy_atom,
                    cute.make_tensor(xptr, cute.make_layout((vec,), stride=(1,))),
                    fx,
                )
                wptr = (mW.iterator + cute.crd2idx((col,), mW.layout)).align(
                    self.align_bytes
                )
                fw = cute.make_rmem_tensor((vec,), mW.element_type)
                cute.copy(
                    copy_atom,
                    cute.make_tensor(wptr, cute.make_layout((vec,), stride=(1,))),
                    fw,
                )
                y = fx.load().to(Float32) * rstd * fw.load().to(Float32)
                fo = cute.make_rmem_tensor((vec,), mY.element_type)
                fo.store(y.to(mY.element_type))
                optr = (mY.iterator + cute.crd2idx((bidx, col), mY.layout)).align(
                    self.align_bytes
                )
                cute.copy(
                    copy_atom,
                    fo,
                    cute.make_tensor(optr, cute.make_layout((vec,), stride=(1,))),
                )

    @cute.jit
    def __call__(
        self,
        mX1: cute.Tensor,
        mW1: cute.Tensor,
        mY1: cute.Tensor,
        mX2: cute.Tensor,
        mW2: cute.Tensor,
        mY2: cute.Tensor,
        M: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        self.kernel(mX1, mW1, mY1, mX2, mW2, mY2, M, eps, enable_pdl).launch(
            grid=[M, 1, 1],
            block=[self.num_threads, 1, 1],
            smem=self._smem_size_in_bytes(),
            stream=stream,
            use_pdl=enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mX1: cute.Tensor,
        mW1: cute.Tensor,
        mY1: cute.Tensor,
        mX2: cute.Tensor,
        mW2: cute.Tensor,
        mY2: cute.Tensor,
        M: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        if enable_pdl:
            cute.arch.griddepcontrol_wait()

        nt1 = self.nt1
        nw1 = self.nw1
        nw2 = self.nw2
        d1 = self.d1
        d2 = self.d2

        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX1.element_type,
            num_bits_per_copy=self.copy_bits,
        )

        smem = cutlass.utils.SmemAllocator()
        red = smem.allocate_tensor(
            Float32, cute.make_layout((2, self.num_warps)), byte_alignment=4
        )

        warp_idx = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()

        # ===== Phase 1: per-thread sum of squares, split by warp group =====
        sum_sq = Float32(0.0)
        if warp_idx < nw1:
            sum_sq = self._load_sum_sq(
                mX1, bidx, tidx, d1, nt1, self.rounds1, copy_atom
            )
        else:
            sum_sq = self._load_sum_sq(
                mX2, bidx, tidx - nt1, d2, self.nt2, self.rounds2, copy_atom
            )

        # ===== Phase 2: warp reduce, then stash per-warp partials by group =====
        warp_val = warp_reduce(sum_sq, operator.add, width=32)
        if lane == 0:
            if warp_idx < nw1:
                red[0, warp_idx] = warp_val
            else:
                red[1, warp_idx - nw1] = warp_val
        cute.arch.barrier()

        # ===== Phase 3: warp 0 finishes both groups' cross-warp reductions =====
        if warp_idx == 0:
            v1 = Float32(0.0)
            if lane < nw1:
                v1 = red[0, lane]
            v1 = warp_reduce(v1, operator.add, width=32)

            v2 = Float32(0.0)
            if lane < nw2:
                v2 = red[1, lane]
            v2 = warp_reduce(v2, operator.add, width=32)

            if lane == 0:
                red[0, 0] = cute.math.rsqrt(v1 / Float32(d1) + eps, fastmath=True)
                red[1, 0] = cute.math.rsqrt(v2 / Float32(d2) + eps, fastmath=True)
        cute.arch.barrier()

        # ===== Phase 4: normalize + store, split by warp group =====
        if warp_idx < nw1:
            self._normalize_store(
                mX1, mW1, mY1, bidx, tidx, red[0, 0], d1, nt1, self.rounds1, copy_atom
            )
        else:
            self._normalize_store(
                mX2,
                mW2,
                mY2,
                bidx,
                tidx - nt1,
                red[1, 0],
                d2,
                self.nt2,
                self.rounds2,
                copy_atom,
            )

        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()


# =============================================================================
# RMSNormQuantKernel
# =============================================================================


class RMSNormQuantKernel:
    """
    RMSNorm + FP8 Quantization Kernel using CuTe-DSL.

    Computes: output = clamp(input / sqrt(mean(input^2) + eps) * weight / scale, -448, 448)
    Then quantizes to FP8 E4M3.
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        H: int,
        weight_bias: float = 0.0,
        use_hw_fp8: bool = True,
        sm_version: int | None = None,
    ):
        self.dtype = dtype
        self.H = H
        self.weight_bias = weight_bias
        self.use_hw_fp8 = use_hw_fp8
        self.sm_version = sm_version if sm_version is not None else get_sm_version()

        self.cluster_n = RMSNormKernel._compute_cluster_n(H, dtype, self.sm_version)
        self.H_per_cta = H // self.cluster_n

        elem_bytes = dtype.width // 8
        max_vec_size = COPY_BITS // 8 // elem_bytes

        h_align = self.H_per_cta & (-self.H_per_cta)
        self.vec_size = min(h_align, max_vec_size)
        self.copy_bits = self.vec_size * dtype.width

        self.threads_per_row = RMSNormKernel._compute_threads_per_row(self.H_per_cta)
        self.num_threads = RMSNormKernel._compute_num_threads(self.H_per_cta)
        if self.H_per_cta > 8192 and self.num_threads < 256:
            self.num_threads = 256
        self.rows_per_block = self.num_threads // self.threads_per_row
        self.warps_per_row = max(self.threads_per_row // 32, 1)

        self.num_vec_blocks = max(
            1,
            (self.H_per_cta // self.vec_size + self.threads_per_row - 1)
            // self.threads_per_row,
        )
        self.cols_per_tile = self.vec_size * self.num_vec_blocks * self.threads_per_row

        if self.copy_bits >= 32:
            tile_bytes = self.rows_per_block * self.cols_per_tile * elem_bytes
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            self.use_async_copy = tile_bytes <= props.shared_memory_per_block_optin // 2
        else:
            self.use_async_copy = False

    def _smem_size_in_bytes(self) -> int:
        if self.use_async_copy:
            tile_bytes = (
                self.rows_per_block * self.cols_per_tile * (self.dtype.width // 8)
            )
        else:
            tile_bytes = 0

        if self.cluster_n == 1:
            reduction_bytes = self.rows_per_block * self.warps_per_row * 4
        else:
            reduction_bytes = (
                self.rows_per_block * self.warps_per_row * self.cluster_n * 4
            )

        mbar_bytes = 8 if self.cluster_n > 1 else 0
        return tile_bytes + reduction_bytes + mbar_bytes

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        M: Int64,
        mS: cute.Tensor,
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

        cluster_n = self.cluster_n

        self.kernel(mX, mW, mY, M, mS, eps, enable_pdl, tv_layout, tiler_mn).launch(
            grid=[cute.ceil_div(M, self.rows_per_block), cluster_n, 1],
            block=[self.num_threads, 1, 1],
            cluster=[1, cluster_n, 1] if cutlass.const_expr(cluster_n > 1) else None,
            smem=self._smem_size_in_bytes(),
            stream=stream,
            use_pdl=enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        M: Int64,
        mS: cute.Tensor,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        # PDL: Wait for previous kernel (SM90+ only)
        if enable_pdl:
            cute.arch.griddepcontrol_wait()

        H = self.H
        cluster_n = self.cluster_n
        cols_per_tile = self.cols_per_tile
        weight_bias = self.weight_bias
        copy_bits = self.copy_bits
        vec_size = self.vec_size
        num_vec_blocks = self.num_vec_blocks
        threads_per_row = tv_layout.shape[0][0]
        rows_per_block = tiler_mn[0]
        warps_per_row = max(threads_per_row // 32, 1)

        if cutlass.const_expr(cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = cutlass.const_expr(0)

        inv_scale = rcp_approx_ftz(mS[0])

        # ===== Allocate shared memory =====
        smem = cutlass.utils.SmemAllocator()

        if cutlass.const_expr(self.use_async_copy):
            sX = smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )

        if cutlass.const_expr(cluster_n == 1):
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, warps_per_row)),
                byte_alignment=4,
            )
            mbar_ptr = None
        else:
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, (warps_per_row, cluster_n))),
                byte_alignment=4,
            )
            mbar_ptr = smem.allocate_array(cutlass.Int64, num_elems=1)

        # ===== Initialize cluster =====
        if cutlass.const_expr(cluster_n > 1):
            if tidx == 0:
                cute.arch.mbarrier_init(mbar_ptr, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

        # ===== Coordinate tracking and tiling =====
        idX = cute.make_identity_tensor(mX.shape)

        gX = cute.local_tile(mX, tiler_mn, (bidx, cluster_y))
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

        mW_expanded_layout = cute.prepend(
            mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))
        )
        mW_2d = cute.make_tensor(mW.iterator, mW_expanded_layout)
        gW = cute.local_tile(mW_2d, tiler_mn, (0, cluster_y))

        # ===== Create TiledCopy atoms =====
        copy_atom_sync = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=copy_bits,
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

        thr_copy_X = tiled_copy_load.get_slice(tidx)
        thr_copy_W = tiled_copy_W.get_slice(tidx)

        # Partition input
        tXgX = thr_copy_X.partition_S(gX)
        tXcX = thr_copy_X.partition_S(cX)
        tXrX = cute.make_fragment_like(tXgX)

        if cutlass.const_expr(self.use_async_copy):
            tXsX = thr_copy_X.partition_D(sX)

        # Partition weight
        tWgW = thr_copy_W.partition_S(gW)
        tWrW = cute.make_fragment_like(tWgW)
        tXrW = thr_copy_X.retile(tWrW)

        # ===== Bounds checking =====
        tXpX = predicate_k(tXcX, limit=H)
        tWpW = predicate_k(thr_copy_W.partition_S(cX), limit=H)
        row_coord = tXcX[(0, 0), 0, 0]
        row_in_bounds = row_coord[0] < M

        # ===== Pass 1: Load input + compute sum of squares =====
        if cutlass.const_expr(self.use_async_copy):
            if row_in_bounds:
                cute.copy(copy_atom_async, tXgX, tXsX, pred=tXpX)
            cute.arch.cp_async_commit_group()

            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

            cute.arch.cp_async_wait_group(0)

            cute.autovec_copy(tXsX, tXrX)
        else:
            tXrX.store(cute.zeros_like(tXrX, dtype=mX.element_type))
            if row_in_bounds:
                cute.copy(copy_atom_sync, tXgX, tXrX, pred=tXpX)

            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

        x = tXrX.load().to(Float32)
        x_sq = x * x
        sum_sq = row_reduce_sum_multirow(
            x_sq, threads_per_row, reduction_buffer, mbar_ptr, cluster_n
        )

        mean_sq = sum_sq / Float32(H)
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        if cutlass.const_expr(cluster_n > 1):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier()

        # ===== Pass 2: Normalize, quantize, and store FP8 output =====
        # Re-load x from shared memory to relieve register pressure.
        # Without this, x (up to 128 FP32 values/thread at large H) must
        # survive across the reduction + barrier, causing spills to local mem.
        if cutlass.const_expr(self.use_async_copy):
            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(Float32)

        w = tXrW.load().to(Float32)
        y = x * rstd * (w + Float32(weight_bias)) * inv_scale

        tYrY_f32 = cute.make_rmem_tensor(tXrX.shape, Float32)
        tYrY_f32.store(y)

        lane_in_row = tidx % threads_per_row
        row_in_block = tidx // threads_per_row
        # Compute actual_row in int64 so that, with M now widened to Int64,
        # bidx * rows_per_block does not overflow int32 before being compared
        # against M or used in the address arithmetic below.
        actual_row = Int64(bidx) * rows_per_block + row_in_block
        col_offset = lane_in_row * vec_size

        if cutlass.const_expr(self.use_hw_fp8 and vec_size == 8):
            for v in cutlass.range_constexpr(num_vec_blocks):
                local_col = col_offset + v * threads_per_row * vec_size
                abs_col = cluster_y * cols_per_tile + local_col
                if abs_col + 8 <= H and actual_row < M:
                    base = v * 8
                    cvt_and_store_8xf32_to_e4m3_hw(
                        tYrY_f32[base],
                        tYrY_f32[base + 1],
                        tYrY_f32[base + 2],
                        tYrY_f32[base + 3],
                        tYrY_f32[base + 4],
                        tYrY_f32[base + 5],
                        tYrY_f32[base + 6],
                        tYrY_f32[base + 7],
                        get_ptr_as_int64(
                            mY,
                            cute.crd2idx(
                                (Int64(actual_row), Int32(abs_col)), mY.layout
                            ),
                        ),
                    )
                else:
                    for e in cutlass.range_constexpr(vec_size):
                        abs_col_e = cluster_y * cols_per_tile + local_col + e
                        if abs_col_e < H and actual_row < M:
                            flat_idx = v * vec_size + e
                            clamped = max(tYrY_f32[flat_idx], Float32(-FLOAT8_E4M3_MAX))
                            clamped = min(clamped, Float32(FLOAT8_E4M3_MAX))
                            cvt_and_store_f32_to_e4m3_hw(
                                clamped,
                                get_ptr_as_int64(
                                    mY,
                                    cute.crd2idx(
                                        (Int64(actual_row), Int32(abs_col_e)),
                                        mY.layout,
                                    ),
                                ),
                            )
        elif cutlass.const_expr(self.use_hw_fp8 and vec_size == 4):
            for v in cutlass.range_constexpr(num_vec_blocks):
                local_col = col_offset + v * threads_per_row * vec_size
                abs_col = cluster_y * cols_per_tile + local_col
                if abs_col + 4 <= H and actual_row < M:
                    base = v * 4
                    cvt_and_store_4xf32_to_e4m3_hw(
                        tYrY_f32[base],
                        tYrY_f32[base + 1],
                        tYrY_f32[base + 2],
                        tYrY_f32[base + 3],
                        get_ptr_as_int64(
                            mY,
                            cute.crd2idx(
                                (Int64(actual_row), Int32(abs_col)), mY.layout
                            ),
                        ),
                    )
                else:
                    for e in cutlass.range_constexpr(vec_size):
                        abs_col_e = cluster_y * cols_per_tile + local_col + e
                        if abs_col_e < H and actual_row < M:
                            flat_idx = v * vec_size + e
                            clamped = max(tYrY_f32[flat_idx], Float32(-FLOAT8_E4M3_MAX))
                            clamped = min(clamped, Float32(FLOAT8_E4M3_MAX))
                            cvt_and_store_f32_to_e4m3_hw(
                                clamped,
                                get_ptr_as_int64(
                                    mY,
                                    cute.crd2idx(
                                        (Int64(actual_row), Int32(abs_col_e)),
                                        mY.layout,
                                    ),
                                ),
                            )
        elif cutlass.const_expr(self.use_hw_fp8 and vec_size == 2):
            for v in cutlass.range_constexpr(num_vec_blocks):
                local_col = col_offset + v * threads_per_row * vec_size
                abs_col = cluster_y * cols_per_tile + local_col
                if abs_col + 2 <= H and actual_row < M:
                    base = v * 2
                    cvt_and_store_2xf32_to_e4m3_hw(
                        tYrY_f32[base],
                        tYrY_f32[base + 1],
                        get_ptr_as_int64(
                            mY,
                            cute.crd2idx(
                                (Int64(actual_row), Int32(abs_col)), mY.layout
                            ),
                        ),
                    )
                else:
                    for e in cutlass.range_constexpr(vec_size):
                        abs_col_e = cluster_y * cols_per_tile + local_col + e
                        if abs_col_e < H and actual_row < M:
                            flat_idx = v * vec_size + e
                            clamped = max(tYrY_f32[flat_idx], Float32(-FLOAT8_E4M3_MAX))
                            clamped = min(clamped, Float32(FLOAT8_E4M3_MAX))
                            cvt_and_store_f32_to_e4m3_hw(
                                clamped,
                                get_ptr_as_int64(
                                    mY,
                                    cute.crd2idx(
                                        (Int64(actual_row), Int32(abs_col_e)),
                                        mY.layout,
                                    ),
                                ),
                            )
        else:
            for v in cutlass.range_constexpr(num_vec_blocks):
                for e in cutlass.range_constexpr(vec_size):
                    local_col = col_offset + v * threads_per_row * vec_size + e
                    abs_col = cluster_y * cols_per_tile + local_col
                    if abs_col < H and actual_row < M:
                        flat_idx = v * vec_size + e
                        clamped = max(tYrY_f32[flat_idx], Float32(-FLOAT8_E4M3_MAX))
                        clamped = min(clamped, Float32(FLOAT8_E4M3_MAX))
                        out_ptr = get_ptr_as_int64(
                            mY,
                            cute.crd2idx(
                                (Int64(actual_row), Int32(abs_col)), mY.layout
                            ),
                        )
                        if self.use_hw_fp8:
                            cvt_and_store_f32_to_e4m3_hw(clamped, out_ptr)
                        else:
                            cvt_and_store_f32_to_e4m3_sw(clamped, out_ptr)

        # PDL: Signal dependent kernels (SM90+ only)
        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()


# =============================================================================
# Compiled Kernel Getters
# =============================================================================


@functools.cache
def _get_compiled_rmsnorm_kernel(
    dtype_str: str,
    H: int,
    weight_bias: float,
    enable_pdl: bool,
    sm_version: int,
    contiguous: bool = True,
):
    """Get a compiled RMSNorm kernel using TVM-FFI.

    When contiguous=True, tensors are compiled with compact (dense) layouts for
    optimal codegen. When False, symbolic row strides are used to support
    arbitrary row strides at the cost of some performance.
    """
    dtype = get_cutlass_dtype(dtype_str)
    kernel_obj = RMSNormKernel(dtype, H, weight_bias, sm_version=sm_version)

    # 64-bit M so row-index arithmetic (row * H) does not overflow when
    # M * H exceeds INT32_MAX.
    sym_m = cute.sym_int(64)

    if contiguous:
        elem_bytes = dtype.width // 8
        tensor_align = math.gcd(128, H * elem_bytes)
        x_fake = cute.runtime.make_fake_compact_tensor(
            dtype, (sym_m, H), stride_order=(1, 0), assumed_align=tensor_align
        )
        y_fake = cute.runtime.make_fake_compact_tensor(
            dtype, (sym_m, H), stride_order=(1, 0), assumed_align=tensor_align
        )
    else:
        sym_row_stride_x = cute.sym_int64(divisibility=kernel_obj.vec_size)
        sym_row_stride_y = cute.sym_int64(divisibility=kernel_obj.vec_size)
        x_fake = cute.runtime.make_fake_tensor(
            dtype, (sym_m, H), (sym_row_stride_x, 1), assumed_align=16
        )
        y_fake = cute.runtime.make_fake_tensor(
            dtype, (sym_m, H), (sym_row_stride_y, 1), assumed_align=16
        )

    w_fake = cute.runtime.make_fake_compact_tensor(dtype, (H,), assumed_align=16)

    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_kernel = cute.compile(
        kernel_obj,
        x_fake,
        w_fake,
        y_fake,
        Int64(1),
        Float32(1e-6),
        enable_pdl,
        stream_fake,
        options="--enable-tvm-ffi",
    )

    return compiled_kernel


@functools.cache
def _get_compiled_qk_rmsnorm_kernel(
    dtype_str: str, head_dim: int, weight_bias: float, enable_pdl: bool
):
    """Get a compiled QKRMSNorm kernel for 3D tensors with arbitrary stride."""
    dtype = get_cutlass_dtype(dtype_str)
    kernel_obj = QKRMSNormKernel(dtype, head_dim, weight_bias)

    # 64-bit B and N so the flattened row index B*N is not truncated.
    sym_b = cute.sym_int(64)
    sym_n = cute.sym_int(64)

    # Stride divisibility = vec_size guarantees each row start is aligned
    # for the chosen copy_bits (e.g. vec_size=8 for fp16 → 16-byte aligned).
    sym_batch_stride_x = cute.sym_int64(divisibility=kernel_obj.vec_size)
    sym_head_stride_x = cute.sym_int64(divisibility=kernel_obj.vec_size)
    sym_batch_stride_y = cute.sym_int64(divisibility=kernel_obj.vec_size)
    sym_head_stride_y = cute.sym_int64(divisibility=kernel_obj.vec_size)

    x_fake = cute.runtime.make_fake_tensor(
        dtype,
        (sym_b, sym_n, head_dim),
        (sym_batch_stride_x, sym_head_stride_x, 1),
        assumed_align=16,
    )
    y_fake = cute.runtime.make_fake_tensor(
        dtype,
        (sym_b, sym_n, head_dim),
        (sym_batch_stride_y, sym_head_stride_y, 1),
        assumed_align=16,
    )
    w_fake = cute.runtime.make_fake_compact_tensor(dtype, (head_dim,), assumed_align=16)

    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_kernel = cute.compile(
        kernel_obj,
        x_fake,
        w_fake,
        y_fake,
        Int64(1),  # Dummy B
        Int64(1),  # Dummy N
        Float32(1e-6),  # Dummy eps
        enable_pdl,
        stream_fake,
        options="--enable-tvm-ffi",
    )

    return compiled_kernel


@functools.cache
def _get_compiled_rmsnorm_fused_parallel_kernel(
    dtype_str: str,
    d1: int,
    d2: int,
    enable_pdl: bool,
    contiguous: bool = False,
):
    """Get a compiled fused-parallel RMSNorm kernel for two 2D inputs.

    Inputs are row-strided by default (q/k_nope are slices of a packed latent).
    Outputs are freshly allocated and dense.
    """
    dtype = get_cutlass_dtype(dtype_str)
    kernel_obj = RMSNormFusedParallelKernel(dtype, d1, d2)

    sym_m = cute.sym_int(64)
    vec = kernel_obj.vec_size

    def _x_fake(d):
        if contiguous:
            elem_bytes = dtype.width // 8
            align = math.gcd(128, d * elem_bytes)
            return cute.runtime.make_fake_compact_tensor(
                dtype, (sym_m, d), stride_order=(1, 0), assumed_align=align
            )
        # Stride divisibility = vec_size keeps every row start aligned for the
        # chosen copy_bits, matching the .align() hint used in the kernel.
        sym_row_stride = cute.sym_int64(divisibility=vec)
        return cute.runtime.make_fake_tensor(
            dtype, (sym_m, d), (sym_row_stride, 1), assumed_align=16
        )

    x1_fake = _x_fake(d1)
    x2_fake = _x_fake(d2)
    # Outputs are always dense (caller allocates them).
    y1_fake = cute.runtime.make_fake_compact_tensor(
        dtype,
        (sym_m, d1),
        stride_order=(1, 0),
        assumed_align=math.gcd(128, d1 * (dtype.width // 8)),
    )
    y2_fake = cute.runtime.make_fake_compact_tensor(
        dtype,
        (sym_m, d2),
        stride_order=(1, 0),
        assumed_align=math.gcd(128, d2 * (dtype.width // 8)),
    )
    w1_fake = cute.runtime.make_fake_compact_tensor(dtype, (d1,), assumed_align=16)
    w2_fake = cute.runtime.make_fake_compact_tensor(dtype, (d2,), assumed_align=16)

    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_kernel = cute.compile(
        kernel_obj,
        x1_fake,
        w1_fake,
        y1_fake,
        x2_fake,
        w2_fake,
        y2_fake,
        Int64(1),
        Float32(1e-6),
        enable_pdl,
        stream_fake,
        options="--enable-tvm-ffi",
    )

    return compiled_kernel


@functools.cache
def _get_compiled_rmsnorm_quant_kernel(
    dtype_str: str,
    out_dtype_str: str,
    H: int,
    weight_bias: float,
    enable_pdl: bool,
    use_hw_fp8: bool = True,
    sm_version: int = 80,
    contiguous: bool = True,
):
    """Get a compiled RMSNorm + Quant kernel using TVM-FFI.

    See _get_compiled_rmsnorm_kernel for contiguous parameter semantics.
    """
    dtype = get_cutlass_dtype(dtype_str)
    out_dtype = get_cutlass_dtype(out_dtype_str)
    kernel_obj = RMSNormQuantKernel(
        dtype, H, weight_bias, use_hw_fp8=use_hw_fp8, sm_version=sm_version
    )

    # 64-bit M so row-index arithmetic (row * H) does not overflow.
    sym_m = cute.sym_int(64)

    if contiguous:
        in_align = math.gcd(128, H * (dtype.width // 8))
        out_align = math.gcd(128, H * (out_dtype.width // 8))
        x_fake = cute.runtime.make_fake_compact_tensor(
            dtype, (sym_m, H), stride_order=(1, 0), assumed_align=in_align
        )
        y_fake = cute.runtime.make_fake_compact_tensor(
            out_dtype, (sym_m, H), stride_order=(1, 0), assumed_align=out_align
        )
    else:
        sym_row_stride_x = cute.sym_int64(divisibility=kernel_obj.vec_size)
        sym_row_stride_y = cute.sym_int64(divisibility=kernel_obj.vec_size)
        x_fake = cute.runtime.make_fake_tensor(
            dtype, (sym_m, H), (sym_row_stride_x, 1), assumed_align=16
        )
        y_fake = cute.runtime.make_fake_tensor(
            out_dtype, (sym_m, H), (sym_row_stride_y, 1), assumed_align=16
        )

    w_fake = cute.runtime.make_fake_compact_tensor(dtype, (H,), assumed_align=16)
    s_fake = cute.runtime.make_fake_compact_tensor(Float32, (1,), assumed_align=4)

    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_kernel = cute.compile(
        kernel_obj,
        x_fake,
        w_fake,
        y_fake,
        Int64(1),
        s_fake,
        Float32(1e-6),
        enable_pdl,
        stream_fake,
        options="--enable-tvm-ffi",
    )

    return compiled_kernel


# =============================================================================
# CuTe DSL API Functions
# =============================================================================


def rmsnorm_cute(
    input: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    eps: float = 1e-6,
    weight_bias: float = 0.0,
    enable_pdl: bool = False,
) -> None:
    """CuTe DSL RMSNorm implementation.

    Supports non-contiguous tensors (stride[-1] must be 1). Uses an optimized
    compact kernel for contiguous inputs and a general strided kernel otherwise.
    """
    shape = input.shape
    H = shape[-1]

    if len(shape) == 3:
        M = shape[0] * shape[1]
        input_2d = input.reshape(M, H)
        out_2d = out.reshape(M, H)
    else:
        M = shape[0]
        input_2d = input
        out_2d = out

    is_contiguous = input_2d.is_contiguous() and out_2d.is_contiguous()
    # When M*H exceeds INT32_MAX, use the strided compile path: its row stride
    # is a dynamic int64 so the offset arithmetic widens to int64. The compact
    # path bakes the row stride in as a constexpr int and computes row*H in
    # int32, which overflows.
    if is_contiguous and M * H > 2**31 - 1:
        is_contiguous = False
    kernel = _get_compiled_rmsnorm_kernel(
        _torch_dtype_to_str(input.dtype),
        H,
        weight_bias,
        enable_pdl,
        get_sm_version(input.device),
        contiguous=is_contiguous,
    )
    kernel(input_2d, weight, out_2d, M, eps)


def qk_rmsnorm_cute(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    eps: float = 1e-6,
    weight_bias: float = 0.0,
    enable_pdl: bool = False,
) -> None:
    """CuTe DSL QKRMSNorm for 3D tensors [batch, heads, head_dim].

    Supports arbitrary stride. Uses multi-row blocks with async/sync copy
    depending on head_dim alignment. Each block processes multiple (batch, head)
    rows independently.
    """
    shape = input.shape
    assert len(shape) == 3, "QKRMSNorm expects 3D input [batch, heads, head_dim]"

    batch_size, num_heads, head_dim = shape

    dtype_str = _torch_dtype_to_str(input.dtype)
    kernel = _get_compiled_qk_rmsnorm_kernel(
        dtype_str, head_dim, weight_bias, enable_pdl
    )

    kernel(input, weight, output, batch_size, num_heads, eps)


def rmsnorm_fused_parallel_cute(
    input1: torch.Tensor,
    weight1: torch.Tensor,
    out1: torch.Tensor,
    input2: torch.Tensor,
    weight2: torch.Tensor,
    out2: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: bool = False,
) -> None:
    """CuTe DSL fused-parallel RMSNorm over two 2D inputs in a single launch.

    Normalizes ``input1`` ([M, d1], weight1) -> ``out1`` and ``input2``
    ([M, d2], weight2) -> ``out2`` together, splitting each block's warps
    between the two. Inputs share the row count M and may be row-strided
    (stride[-1] must be 1); outputs are dense. Pure RMSNorm (no weight_bias).
    """
    assert input1.shape[0] == input2.shape[0], "inputs must share row count M"
    assert input1.dtype == input2.dtype == out1.dtype == out2.dtype
    M = input1.shape[0]
    d1 = input1.shape[-1]
    d2 = input2.shape[-1]

    # The kernel's .align() hint assumes aligned row starts; require the
    # vec_size stride divisibility that guarantees it (dense rows always pass).
    vec = math.gcd(COPY_BITS // 8 // (input1.element_size()), math.gcd(d1, d2))
    contiguous = (
        input1.is_contiguous()
        and input2.is_contiguous()
        and input1.stride(0) % vec == 0
        and input2.stride(0) % vec == 0
    )
    if not contiguous:
        assert (
            input1.stride(0) % vec == 0 and input2.stride(0) % vec == 0
        ), "input row strides must be divisible by vec_size for aligned access"

    kernel = _get_compiled_rmsnorm_fused_parallel_kernel(
        _torch_dtype_to_str(input1.dtype),
        d1,
        d2,
        enable_pdl,
        contiguous=contiguous,
    )
    kernel(input1, weight1, out1, input2, weight2, out2, M, eps)


def rmsnorm_quant_cute(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float = 1e-6,
    weight_bias: float = 0.0,
    enable_pdl: bool = False,
) -> None:
    """CuTe DSL RMSNorm + FP8 quantization implementation.

    Supports non-contiguous tensors (stride[-1] must be 1). Uses an optimized
    compact kernel for contiguous inputs and a general strided kernel otherwise.
    """
    shape = input.shape
    H = shape[-1]
    M = shape[0]

    is_contiguous = input.is_contiguous() and out.is_contiguous()
    # When M*H exceeds INT32_MAX, fall back to the strided path so its dynamic
    # int64 row stride widens the offset arithmetic to int64.
    if is_contiguous and M * H > 2**31 - 1:
        is_contiguous = False
    dtype_str = _torch_dtype_to_str(input.dtype)
    out_dtype_str = _torch_dtype_to_str(out.dtype)
    kernel = _get_compiled_rmsnorm_quant_kernel(
        dtype_str,
        out_dtype_str,
        H,
        weight_bias,
        enable_pdl,
        use_hw_fp8=has_hw_fp8_cvt(input.device),
        sm_version=get_sm_version(input.device),
        contiguous=is_contiguous,
    )
    kernel(input, weight, out, M, scale, eps)


__all__ = [
    # Kernel classes
    "RMSNormKernel",
    "QKRMSNormKernel",
    "RMSNormFusedParallelKernel",
    "RMSNormQuantKernel",
    # Compiled kernel getters
    "_get_compiled_rmsnorm_kernel",
    "_get_compiled_qk_rmsnorm_kernel",
    "_get_compiled_rmsnorm_fused_parallel_kernel",
    "_get_compiled_rmsnorm_quant_kernel",
    # CuTe DSL APIs
    "rmsnorm_cute",
    "qk_rmsnorm_cute",
    "rmsnorm_fused_parallel_cute",
    "rmsnorm_quant_cute",
]
