# type: ignore

import math
from functools import cache, partial

import cutlass
import torch
from cutlass import Float32, Int32, const_expr, cute, range_constexpr

from sglang.kernels.ops.inkling2.cute_utils import (
    assert_cuda_contig,
    empty_cuda,
    make_fake_2d_row_major,
    torch_to_cute_dtype,
)
from sglang.kernels.ops.inkling2.norm_common import (
    ceil_div,
    copy_partition,
    load_norm_rows,
    make_fake_norm_rows,
    make_tiled_copy_2d,
    norm_num_rows,
    norm_row_alignment,
    predicate_n,
    qk_l2_clip_fwd_layout,
    qk_l2_clip_row_stats,
)

BASE_ALIGN = 16


class QKL2ClipFwd:
    NUM_THREADS = 128

    def __init__(
        self,
        N: int,
        scale_rank: int = 0,
        rows_per_output_scale: int = 0,
        x_alignment: int = 16,
    ):
        assert N > 0, f"{N=} must be positive"
        assert scale_rank in {0, 1, 2}, f"{scale_rank=} must be 0, 1, or 2"
        self.x_alignment = x_alignment
        self.N = N
        self.scale_rank = scale_rank
        self.rows_per_output_scale = rows_per_output_scale

    def _make_tiled_copy(
        self, dtype: type[cutlass.Numeric], vecsize: int, threads_per_row: int
    ) -> tuple[cute.TiledCopy, tuple[int, int], int]:
        num_n_blocks = ceil_div(self.N // vecsize, threads_per_row)
        tile_mn = (
            self.NUM_THREADS // threads_per_row,
            vecsize * num_n_blocks * threads_per_row,
        )
        tiled_copy = make_tiled_copy_2d(
            dtype,
            threads_per_row=threads_per_row,
            num_threads=self.NUM_THREADS,
            vecsize=vecsize,
        )
        return tiled_copy, tile_mn, threads_per_row

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
        mScale: cute.Tensor,
        mY: cute.Tensor,
        mOutputScale: cute.Tensor | None,
        threshold: Float32,
        eps: Float32,
        stream,
    ) -> None:
        widest_dtype_bits = const_expr(
            max(mX.element_type.width, mY.element_type.width)
        )
        threads_per_row, vecsize = qk_l2_clip_fwd_layout(self.N, widest_dtype_bits)
        tiled_copy, tile_mn, threads_per_row = self._make_tiled_copy(
            mX.element_type, vecsize, threads_per_row
        )
        self.kernel(
            mX,
            mScale,
            mY,
            mOutputScale,
            threshold,
            eps,
            tile_mn,
            tiled_copy,
            threads_per_row,
        ).launch(
            grid=[cute.ceil_div(norm_num_rows(mX), tile_mn[0]), 1, 1],
            block=[cute.size(tiled_copy), 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mScale: cute.Tensor,
        mY: cute.Tensor,
        mOutputScale: cute.Tensor | None,
        threshold: Float32,
        eps: Float32,
        tile_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bid_m, _, _ = cute.arch.block_idx()
        tv_layout = tiled_copy.layout_tv_tiled

        smem = cutlass.utils.SmemAllocator()
        reduction_buffer = smem.allocate_tensor(
            cute.Float32,
            self._reduction_buffer_layout(tv_layout),
            byte_alignment=8,
        )

        shape = (norm_num_rows(mX), self.N)
        cX_full = cute.make_identity_tensor(shape)
        gY = cute.local_tile(mY, tile_mn, (bid_m, 0))
        cX = cute.local_tile(cX_full, tile_mn, (bid_m, 0))

        thr_copy = tiled_copy.get_slice(tidx)
        tXgY = thr_copy.partition_D(gY)
        tXcX_full = thr_copy.partition_S(cX)
        tXcX = tXcX_full[(0, None), None, None]

        tXrX = cute.make_rmem_tensor(tXgY.layout, mX.element_type)
        tXrY = cute.make_rmem_tensor_like(tXgY)

        is_even_n = const_expr(shape[1] == tile_mn[1])
        pred_n = (
            None if is_even_n else predicate_n(thr_copy.partition_S(cX), limit=shape[1])
        )
        copy_n = partial(copy_partition, pred=pred_n)

        row = tXcX[0][0]
        tXrX.fill(0.0)
        if row < shape[0]:
            if const_expr(cute.rank(mX) == 2 and self.x_alignment == 16):
                gX = cute.local_tile(mX, tile_mn, (bid_m, 0))
                copy_n(thr_copy.partition_S(gX), tXrX)
            else:
                load_norm_rows(mX, tXrX, tXcX_full, self.x_alignment)

        x = tXrX.load().to(cute.Float32)
        if const_expr(self.scale_rank != 0):
            tXrScale = cute.make_rmem_tensor(tXrX.layout, mScale.element_type)
            tXrScale.fill(0.0)
            scale_head = (
                row % Int32(mScale.shape[0]) if const_expr(self.scale_rank == 2) else 0
            )
            for rest_v in range_constexpr(cute.size(tXrScale.shape[0])):
                for rest_m in range_constexpr(cute.size(tXrScale, mode=[1])):
                    for rest_n in range_constexpr(cute.size(tXrScale, mode=[2])):
                        row_i = tXcX_full[(rest_v, 0), rest_m, rest_n][0]
                        col_i = tXcX_full[(rest_v, 0), rest_m, rest_n][1]
                        if row_i < shape[0] and col_i < shape[1]:
                            if const_expr(self.scale_rank == 1):
                                tXrScale[(rest_v, 0), rest_m, rest_n] = mScale[col_i]
                            else:
                                tXrScale[(rest_v, 0), rest_m, rest_n] = mScale[
                                    scale_head, col_i
                                ]
            x = x * tXrScale.load().to(cute.Float32)

        coeff, _ = qk_l2_clip_row_stats(
            x, threshold, eps, threads_per_row, reduction_buffer
        )

        y = x * coeff
        if const_expr(mOutputScale is not None):
            output_scale = Float32(0)
            if row < shape[0]:
                output_scale = mOutputScale[row // self.rows_per_output_scale]
            y = y.to(mY.element_type).to(Float32) * output_scale
        tXrY.store(y.to(tXrY.element_type))
        if row < shape[0]:
            copy_n(tXrY, tXgY)


def _make_fake_scale(
    scale_dtype: type[cutlass.Numeric],
    N: int,
    scale_rank: int,
) -> cute.Tensor:
    if scale_rank == 1:
        return cute.runtime.make_fake_tensor(
            scale_dtype,
            (N,),
            stride=(1,),
            assumed_align=BASE_ALIGN,
        )
    if scale_rank == 2:
        H_sym = cute.sym_int64()
        return cute.runtime.make_fake_tensor(
            scale_dtype,
            (H_sym, N),
            stride=(N, 1),
            assumed_align=BASE_ALIGN,
        )
    raise ValueError(f"{scale_rank=} must be 1 or 2")


@cache
def compile_qk_l2_clip_fwd(
    x_dtype: type[cutlass.Numeric],
    output_dtype: type[cutlass.Numeric],
    N: int,
    scale_rank: int = 0,
    scale_dtype: type[cutlass.Numeric] | None = None,
    rows_per_output_scale: int = 0,
    x_layout: tuple[int, int] = (2, 16),
):
    kernel = QKL2ClipFwd(N, scale_rank, rows_per_output_scale, x_layout[1])
    M_sym = cute.sym_int64()
    mX = make_fake_norm_rows(x_dtype, N, *x_layout)
    mScale = mX if scale_rank == 0 else _make_fake_scale(scale_dtype, N, scale_rank)
    y_stride_div = math.gcd(N, BASE_ALIGN // (output_dtype.width // 8))
    mY = make_fake_2d_row_major(
        output_dtype,
        (M_sym, N),
        align=BASE_ALIGN,
        stride_divisibility=y_stride_div,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        mX,
        mScale,
        mY,
        cute.runtime.make_fake_tensor(
            Float32, (cute.sym_int64(),), stride=(1,), assumed_align=4
        )
        if rows_per_output_scale
        else None,
        Float32(0.0),
        Float32(0.0),
        stream,
        options="--enable-tvm-ffi",
    )


def qk_l2_clip_fwd_cutedsl(
    x: torch.Tensor,
    threshold: float,
    eps: float = 1e-6,
    *,
    output_dtype: torch.dtype | None = None,
    scale: torch.Tensor | None = None,
    output_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    x_alignment = norm_row_alignment(x)
    M, N = math.prod(x.shape[:-1]), x.shape[-1]
    rows_per_output_scale = 0
    if output_scale is not None:
        assert not torch.is_grad_enabled(), "output_scale requires no_grad"
        assert_cuda_contig(output_scale, name="output_scale", ndim=1)
        assert output_scale.device == x.device, f"{output_scale.device=} {x.device=}"
        assert output_scale.dtype == torch.float32, (
            f"{output_scale.dtype=} must be FP32"
        )
        if M == 0:
            assert output_scale.numel() == 0, "Empty input requires empty output_scale"
        else:
            assert output_scale.numel() > 0 and M % output_scale.numel() == 0, (
                f"Output scale groups must divide the rows: {output_scale.shape=} {x.shape=}"
            )
            rows_per_output_scale = M // output_scale.numel()
    if scale is not None:
        assert scale.is_cuda, f"{scale.device=}"
        assert scale.device == x.device, f"{scale.device=} {x.device=}"
        assert scale.ndim in {1, 2}, f"{scale.ndim=}"
        if scale.ndim == 1:
            assert scale.shape == (N,), f"{scale.shape=} {x.shape=}"
        else:
            assert scale.shape[1] == N, f"{scale.shape=} {x.shape=}"
            assert M % scale.shape[0] == 0, f"{x.shape=} {scale.shape=}"
        scale = scale.contiguous()
    output_dtype = output_dtype or x.dtype
    assert M < 2**31, f"{M=} must fit in Int32"
    assert N > 0, f"{N=} must be positive"
    y = empty_cuda(M, N, dtype=output_dtype)
    if x.numel() == 0:
        return y

    compiled = compile_qk_l2_clip_fwd(
        torch_to_cute_dtype[x.dtype],
        torch_to_cute_dtype[output_dtype],
        N,
        0 if scale is None else scale.ndim,
        None if scale is None else torch_to_cute_dtype[scale.dtype],
        rows_per_output_scale,
        x_layout=(x.ndim, x_alignment),
    )
    compiled(
        x,
        x if scale is None else scale,
        y,
        output_scale,
        float(threshold),
        float(eps),
    )
    return y
