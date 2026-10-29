# Copyright 2026 SGLang Team
# Copyright 2025 FlashInfer team
# Licensed under the Apache License, Version 2.0.
"""Three RMSNorm outputs sharing FlashInfer's input load and reduction.

Uses the CuTe RMSNormKernel layout/reduction interface from FlashInfer 0.6.18.
The row-count dispatch covers the measured decode buckets on Blackwell.
"""

from functools import cache

import torch

_FANOUT_MAX_ROWS = 2048


def can_use_rmsnorm_fanout(x):
    return (
        0 < x.shape[0] <= _FANOUT_MAX_ROWS
        and _fanout_kernel(x.dtype, x.shape[1]) is not None
    )


@cache
def _fanout_kernel(dtype, width):
    if dtype not in (torch.bfloat16, torch.float16):
        return None
    kernel = _fanout_kernel_type()(_cutlass_dtype(dtype), width)
    return kernel if kernel.cluster_n == 1 else None


def _cutlass_dtype(dtype):
    import cutlass

    return cutlass.BFloat16 if dtype == torch.bfloat16 else cutlass.Float16


def rmsnorm_fanout(x, weight0, weight1, weight2, eps):
    """Normalize contiguous FP16/BF16 rows with three matching [H] weights."""
    outputs = tuple(torch.empty_like(x) for _ in range(3))
    compiled = _compile_fanout(x.device, x.dtype, x.shape[1])
    compiled(x, weight0, weight1, weight2, *outputs, x.shape[0], eps)
    return outputs


@cache
def _compile_fanout(device, torch_dtype, width):
    import cutlass
    import cutlass.cute as cute

    dtype = _cutlass_dtype(torch_dtype)
    kernel = _fanout_kernel(torch_dtype, width)
    rows = cute.sym_int(64)
    matrix = cute.runtime.make_fake_compact_tensor(
        dtype, (rows, width), stride_order=(1, 0), assumed_align=16
    )
    weight = cute.runtime.make_fake_compact_tensor(dtype, (width,), assumed_align=16)
    return cute.compile(
        kernel,
        matrix,
        weight,
        weight,
        weight,
        matrix,
        matrix,
        matrix,
        cutlass.Int64(1),
        cutlass.Float32(1e-6),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@cache
def _fanout_kernel_type():
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32, Int64
    from flashinfer.norm.kernels.rmsnorm import RMSNormKernel
    from flashinfer.norm.utils import predicate_k, row_reduce_sum_multirow

    class RMSNormFanout(RMSNormKernel):
        @cute.jit
        def __call__(
            self,
            x: cute.Tensor,
            w0: cute.Tensor,
            w1: cute.Tensor,
            w2: cute.Tensor,
            y0: cute.Tensor,
            y1: cute.Tensor,
            y2: cute.Tensor,
            m: Int64,
            eps: Float32,
            stream,
        ):
            shape, stride = self._make_tv_layout(
                self.threads_per_row,
                self.rows_per_block,
                self.vec_size,
                self.num_vec_blocks,
            )
            layout = cute.make_layout(shape, stride=stride)
            tile = (self.rows_per_block, self.cols_per_tile)
            self.kernel(x, w0, w1, w2, y0, y1, y2, m, eps, layout, tile).launch(
                grid=[cute.ceil_div(m, self.rows_per_block), 1, 1],
                block=[self.num_threads, 1, 1],
                smem=self._smem_size_in_bytes(),
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            x: cute.Tensor,
            w0: cute.Tensor,
            w1: cute.Tensor,
            w2: cute.Tensor,
            y0: cute.Tensor,
            y1: cute.Tensor,
            y2: cute.Tensor,
            m: Int64,
            eps: Float32,
            layout: cute.Layout,
            tile: cute.Shape,
        ):
            tid, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            smem = cutlass.utils.SmemAllocator()
            if cutlass.const_expr(self.use_async_copy):
                sx = smem.allocate_tensor(
                    x.element_type,
                    cute.make_ordered_layout(tile, order=(1, 0)),
                    byte_alignment=16,
                )
            reduction = smem.allocate_tensor(
                Float32,
                cute.make_layout((self.rows_per_block, self.warps_per_row)),
                byte_alignment=4,
            )
            gx = cute.local_tile(x, tile, (block, 0))
            coords = cute.local_tile(
                cute.make_identity_tensor(x.shape), tile, (block, 0)
            )
            sync = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                x.element_type,
                num_bits_per_copy=self.copy_bits,
            )
            if cutlass.const_expr(self.use_async_copy):
                async_copy = cute.make_copy_atom(
                    cute.nvgpu.cpasync.CopyG2SOp(),
                    x.element_type,
                    num_bits_per_copy=self.copy_bits,
                )
                loader = cute.make_tiled_copy(async_copy, layout, tile)
            else:
                loader = cute.make_tiled_copy(sync, layout, tile)
            copier = cute.make_tiled_copy(sync, layout, tile)
            thread_x = loader.get_slice(tid)
            thread_o = copier.get_slice(tid)
            txg = thread_x.partition_S(gx)
            txc = thread_x.partition_S(coords)
            txr = cute.make_fragment_like(txg)
            x_pred = predicate_k(txc, limit=self.H)
            row_valid = txc[(0, 0), 0, 0][0] < m
            if cutlass.const_expr(self.use_async_copy):
                txs = thread_x.partition_D(sx)
                if row_valid:
                    cute.copy(async_copy, txg, txs, pred=x_pred)
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.autovec_copy(txs, txr)
            else:
                txr.store(cute.zeros_like(txr, dtype=x.element_type))
                if row_valid:
                    cute.copy(sync, txg, txr, pred=x_pred)
            value = txr.load().to(Float32)
            summed = row_reduce_sum_multirow(
                value * value, self.threads_per_row, reduction, None, 1
            )
            rstd = cute.math.rsqrt(summed / Float32(self.H) + eps, fastmath=True)
            cute.arch.barrier()
            if cutlass.const_expr(self.use_async_copy):
                cute.autovec_copy(txs, txr)
                value = txr.load().to(Float32)
            weights = (w0, w1, w2)
            outputs = (y0, y1, y2)
            wpred = predicate_k(thread_o.partition_S(coords), limit=self.H)
            for index in cutlass.range_constexpr(3):
                weight = weights[index]
                expanded = cute.make_tensor(
                    weight.iterator,
                    cute.prepend(
                        weight.layout, cute.make_layout((tile[0],), stride=(0,))
                    ),
                )
                gw = cute.local_tile(expanded, tile, (0, 0))
                twg = thread_o.partition_S(gw)
                twr = cute.make_fragment_like(twg)
                cute.copy(sync, twg, twr, pred=wpred)
                weight_value = thread_x.retile(twr).load().to(Float32)
                gy = cute.local_tile(outputs[index], tile, (block, 0))
                tog = thread_o.partition_D(gy)
                tor = cute.make_fragment_like(tog)
                result = value * rstd * (weight_value + Float32(0.0))
                tor.store(result.to(outputs[index].element_type))
                if row_valid:
                    cute.copy(sync, tor, tog, pred=x_pred)

    return RMSNormFanout
