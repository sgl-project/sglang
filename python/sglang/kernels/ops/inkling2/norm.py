import math
from functools import cache, partial
from typing import Type

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, const_expr

from sglang.kernels.ops.inkling2.cute_utils import (
    make_fake_2d_row_major,
    torch_to_cute_dtype,
)
from sglang.kernels.ops.inkling2.norm_common import (
    as_norm_rows,
    ceil_div,
    copy_partition,
    expand_dim,
    load_norm_rows,
    make_fake_norm_rows,
    make_tiled_copy_2d,
    norm_num_rows,
    norm_row_alignment,
    predicate_n,
    rmsnorm_fwd_layout,
    rmsnorm_row_rstd,
)


class RMSNormFwd:
    def __init__(
        self,
        N: int,
        *,
        has_pre_affine: bool = False,
        preload_weight: bool = False,
        x_alignment: int = 16,
    ):
        self.x_alignment = x_alignment
        self.N = N
        self.has_pre_affine = has_pre_affine
        self.preload_weight = preload_weight
        self.use_x_g2r_load = N <= 128
        self.reload_x_from_smem = N > 8192 and (not self.use_x_g2r_load)
        assert N > 0, f"N={N!r} must be positive"

    def _num_threads(self, dtype: Type[cutlass.Numeric]) -> int:
        if self.N > 8192 and dtype.width >= 32:
            return 256
        return 128

    def _make_tiled_copy(
        self,
        dtype: Type[cutlass.Numeric],
        vecsize: int,
        num_threads: int,
        threads_per_row: int,
    ) -> tuple:
        assert num_threads % threads_per_row == 0
        num_n_blocks = ceil_div(self.N // vecsize, threads_per_row)
        tile_mn = (
            num_threads // threads_per_row,
            vecsize * num_n_blocks * threads_per_row,
        )
        tiled_copy = make_tiled_copy_2d(
            dtype,
            threads_per_row=threads_per_row,
            num_threads=num_threads,
            vecsize=vecsize,
        )
        return (tiled_copy, tile_mn, threads_per_row)

    def _reduction_buffer_layout(self, tv_layout: cute.Layout) -> cute.Layout:
        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        warps_per_row = (
            num_warps
            if cute.rank(tv_layout.shape[0]) == 1
            else max(tv_layout.shape[0][0] // cute.arch.WARP_SIZE, 1)
        )
        return cute.make_ordered_layout(
            (num_warps // warps_per_row, warps_per_row), order=(1, 0)
        )

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor | None,
        mY: cute.Tensor,
        mPW: cute.Tensor | None,
        eps: Float32,
        stream,
    ) -> None:
        weight_bits = const_expr(
            mW.element_type.width
            if mW is not None
            else mPW.element_type.width
            if self.has_pre_affine
            else 32
        )
        widest_dtype_bits = const_expr(
            max(mX.element_type.width, weight_bits, mY.element_type.width)
        )
        num_threads = self._num_threads(mX.element_type)
        threads_per_row, vecsize = rmsnorm_fwd_layout(
            self.N, widest_dtype_bits, num_threads
        )
        tiled_copy, tile_mn, threads_per_row = self._make_tiled_copy(
            mX.element_type, vecsize, num_threads, threads_per_row
        )
        mW_expanded = (
            expand_dim(mW, dim=0, size=tile_mn[0])
            if const_expr(mW is not None)
            else None
        )
        mPW_expanded = (
            expand_dim(mPW, dim=0, size=tile_mn[0])
            if const_expr(self.has_pre_affine)
            else None
        )
        self.kernel(
            mX,
            mW_expanded,
            mY,
            mPW_expanded,
            eps,
            tile_mn,
            tiled_copy,
            threads_per_row,
        ).launch(
            grid=[cute.ceil_div(norm_num_rows(mX), tile_mn[0]), 1, 1],
            block=[cute.size(tiled_copy), 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor | None,
        mY: cute.Tensor,
        mPW: cute.Tensor | None,
        eps: Float32,
        tile_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bid_m, _, _ = cute.arch.block_idx()
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        tv_layout = tiled_copy.layout_tv_tiled
        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(tile_mn, order=(1, 0)),
            byte_alignment=BASE_ALIGN,
        )
        reduction_buffer = smem.allocate_tensor(
            cute.Float32, self._reduction_buffer_layout(tv_layout), byte_alignment=8
        )
        shape = (norm_num_rows(mX), self.N)
        cX_full = cute.make_identity_tensor(shape)
        if const_expr(cute.rank(mX) == 2):
            gX = cute.local_tile(mX, tile_mn, (bid_m, 0))
        if const_expr(mW is not None):
            gW = cute.local_tile(mW, tile_mn, (0, 0))
        gY = cute.local_tile(mY, tile_mn, (bid_m, 0))
        cX = cute.local_tile(cX_full, tile_mn, (bid_m, 0))
        if const_expr(self.has_pre_affine):
            gPW = cute.local_tile(mPW, tile_mn, (0, 0))
        thr_copy = tiled_copy.get_slice(tidx)
        if const_expr(cute.rank(mX) == 2):
            tXgX = thr_copy.partition_S(gX)
        tXsX = thr_copy.partition_D(sX)
        if const_expr(mW is not None):
            tXgW = thr_copy.partition_S(gW)
        tXgY = thr_copy.partition_D(gY)
        tXcX = thr_copy.partition_S(cX)[(0, None), None, None]
        tXrX = cute.make_rmem_tensor_like(tXsX)
        if const_expr(mW is not None):
            tXrW = cute.make_rmem_tensor_like(tXgW)
        tXrY = cute.make_rmem_tensor_like(tXgY)
        is_even_n = const_expr(shape[1] == tile_mn[1])
        pred_n = (
            None if is_even_n else predicate_n(thr_copy.partition_S(cX), limit=shape[1])
        )
        copy_n = partial(copy_partition, pred=pred_n)
        if const_expr(self.has_pre_affine):
            tXgPW = thr_copy.partition_S(gPW)
            tXrPW = cute.make_rmem_tensor_like(tXgPW)
            if const_expr(not is_even_n):
                tXrPW.fill(0.0)
            copy_n(tXgPW, tXrPW)
            pw = tXrPW.load().to(cute.Float32)
        if const_expr(mW is not None and self.preload_weight):
            if const_expr(not is_even_n):
                tXrW.fill(0.0)
            copy_n(tXgW, tXrW)
        row = tXcX[0][0]
        if const_expr(self.use_x_g2r_load):
            tXrX.fill(0.0)
            if row < shape[0]:
                if const_expr(cute.rank(mX) == 2 and self.x_alignment == 16):
                    copy_n(tXgX, tXrX)
                else:
                    load_norm_rows(mX, tXrX, thr_copy.partition_S(cX), self.x_alignment)
        else:
            if row < shape[0]:
                if const_expr(cute.rank(mX) == 2 and self.x_alignment == 16):
                    copy_n(tXgX, tXsX, is_async=True)
                else:
                    load_norm_rows(
                        mX,
                        tXsX,
                        thr_copy.partition_S(cX),
                        self.x_alignment,
                        is_async=True,
                    )
            cute.arch.cp_async_commit_group()
        if const_expr(not self.use_x_g2r_load):
            cute.arch.cp_async_wait_group(0)
            cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(cute.Float32)
        if const_expr(self.has_pre_affine):
            x = x * pw
        rstd = rmsnorm_row_rstd(x, self.N, eps, threads_per_row, reduction_buffer)
        if const_expr(self.reload_x_from_smem):
            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(cute.Float32)
            if const_expr(self.has_pre_affine):
                x = x * pw
        y = x * rstd
        if const_expr(mW is not None):
            if const_expr(not self.preload_weight):
                if const_expr(not is_even_n):
                    tXrW.fill(0.0)
                copy_n(tXgW, tXrW)
            w = tXrW.load().to(cute.Float32)
            y = y * w
        y = y.to(tXrY.element_type)
        tXrY.store(y)
        if row < shape[0]:
            copy_n(tXrY, tXgY)


BASE_ALIGN = 16


@cache
def _compile_norm(
    x_dtype, weight_dtype, output_dtype, pre_weight_dtype, width, x_layout, preload
):
    kernel = RMSNormFwd(
        width,
        has_pre_affine=pre_weight_dtype is not None,
        preload_weight=preload,
        x_alignment=x_layout[1],
    )
    tokens = cute.sym_int64()
    x = make_fake_norm_rows(x_dtype, width, *x_layout)

    def weight(dtype):
        return (
            cute.runtime.make_fake_tensor(
                dtype, (width,), stride=(1,), assumed_align=16
            )
            if dtype is not None
            else None
        )

    y = make_fake_2d_row_major(
        output_dtype,
        (tokens, width),
        16,
        math.gcd(width, 16 // (output_dtype.width // 8)),
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        x,
        weight(weight_dtype),
        y,
        weight(pre_weight_dtype),
        Float32(0.0),
        stream,
        options="--enable-tvm-ffi",
    )


def rms_norm(x, weight, eps, *, pre_weight=None, output_dtype=None, out=None):
    assert x.is_cuda and x.shape[-1] > 0 and x.shape[-1] % 4 == 0
    shape = x.shape
    layout = (as_norm_rows(x).ndim, norm_row_alignment(x))
    x = as_norm_rows(x)
    for parameter in (weight, pre_weight):
        if parameter is not None:
            assert parameter.is_contiguous() and parameter.shape == (x.shape[-1],)
            assert parameter.device == x.device
    if weight is not None and pre_weight is not None:
        assert weight.dtype == pre_weight.dtype
    output_dtype = output_dtype or x.dtype
    if out is not None:
        assert out.shape == shape and out.dtype == output_dtype and out.is_contiguous()
    y = (
        out
        if out is not None
        else torch.empty(shape, dtype=output_dtype, device=x.device)
    )
    if not x.numel():
        return y
    width = x.shape[-1]
    preload = (
        weight is not None
        and pre_weight is None
        and 1536 <= width <= 8192
        and x.numel() // width <= 256
    )
    compiled = _compile_norm(
        torch_to_cute_dtype[x.dtype],
        torch_to_cute_dtype[weight.dtype] if weight is not None else None,
        torch_to_cute_dtype[output_dtype],
        torch_to_cute_dtype[pre_weight.dtype] if pre_weight is not None else None,
        width,
        layout,
        preload,
    )
    compiled(x, weight, y.view(-1, width), pre_weight, float(eps))
    return y
