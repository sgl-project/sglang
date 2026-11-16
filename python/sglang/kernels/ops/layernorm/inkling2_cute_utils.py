# type: ignore

import math
import operator
from typing import Type

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Boolean, Float32, Int64, const_expr, range_constexpr
from cutlass._mlir.dialects import arith as mlir_arith
from cutlass._mlir.dialects import llvm
from cutlass._mlir.extras import types as mlir_types
from cutlass.cutlass_dsl import dsl_user_op

BASE_ALIGN = 16

torch_to_cute_dtype = {
    torch.float16: cute.Float16,
    torch.bfloat16: cute.BFloat16,
    torch.float32: cute.Float32,
    torch.float64: cute.Float64,
    torch.int32: cutlass.Int32,
    torch.uint32: cutlass.Uint32,
    torch.int64: cutlass.Int64,
    torch.uint64: cutlass.Uint64,
}


def make_fake_2d_row_major(
    elem_dtype: Type[cutlass.Numeric],
    shape: tuple,
    align: int,
    stride_divisibility: int,
) -> cute.Tensor:
    stride = (cute.sym_int64(divisibility=stride_divisibility), 1)
    return cute.runtime.make_fake_tensor(
        elem_dtype,
        shape,
        stride=stride,
        assumed_align=align,
    )


def make_fake_dense_mn(
    elem_dtype: Type[cutlass.Numeric],
    shape: tuple,
    align: int,
    stride_divisibility: int,
) -> cute.Tensor:
    if len(shape) == 2:
        return make_fake_2d_row_major(elem_dtype, shape, align, stride_divisibility)

    assert len(shape) == 3, f"{shape=}"
    stride = (
        cute.sym_int64(divisibility=stride_divisibility),
        cute.sym_int64(divisibility=stride_divisibility),
        1,
    )
    return cute.runtime.make_fake_tensor(
        elem_dtype,
        shape,
        stride=stride,
        assumed_align=align,
    )


@dsl_user_op
def domain_offset_raw_aligned(
    offset: cutlass.Int64,
    tensor: cute.Tensor,
    assumed_align: cutlass.Constexpr[int],
    *,
    loc=None,
    ip=None,
) -> cute.Tensor:
    assert isinstance(tensor.iterator, cute.Pointer)
    offset_ptr = tensor.iterator + offset
    new_ptr = cute.make_ptr(
        tensor.element_type,
        offset_ptr.toint(),
        tensor.memspace,
        assumed_align=assumed_align,
        loc=loc,
        ip=ip,
    )
    return cute.make_tensor(new_ptr, tensor.layout)


@dsl_user_op
def fma_rn_f32(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            mlir_types.f32(),
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                Float32(c).ir_value(loc=loc, ip=ip),
            ],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fmul_rn_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            mlir_types.f32(),
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
            ],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fadd_rn_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            mlir_types.f32(),
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
            ],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def min_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        mlir_arith.minimumf(
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def copy_partition(
    src: cute.Tensor,
    dst: cute.Tensor,
    *,
    pred: cute.Tensor | None = None,
    is_async: bool = False,
    loc=None,
    ip=None,
) -> None:
    copy_elems = src.shape[0][0]
    copy_bits = const_expr(min(128, copy_elems * src.element_type.width))
    use_async = const_expr(is_async and copy_bits >= 32)
    copy_op = (
        cute.nvgpu.cpasync.CopyG2SOp() if use_async else cute.nvgpu.CopyUniversalOp()
    )
    copy_atom = cute.make_copy_atom(
        copy_op, src.element_type, num_bits_per_copy=copy_bits
    )
    cute.copy(copy_atom, src, dst, pred=pred, loc=loc, ip=ip)


@cute.jit
def row_sum(
    x: cute.TensorSSA,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer: cute.Tensor,
) -> Float32:
    val = x.reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0)
    val = cute.arch.warp_reduction(
        val,
        operator.add,
        threads_in_group=min(threads_per_row, cute.arch.WARP_SIZE),
    )
    if const_expr(cute.size(reduction_buffer.shape[1]) > 1):
        val = block_row_sum(val, reduction_buffer)
    return val


@cute.jit
def block_row_sum(
    val: Float32,
    reduction_buffer: cute.Tensor,
) -> Float32:
    lane_idx = cute.arch.lane_idx()
    warp_idx = cute.arch.warp_idx()
    warps_per_row = cute.size(reduction_buffer.shape[1])
    row_idx = warp_idx // warps_per_row
    col_idx = warp_idx % warps_per_row

    cute.arch.barrier()
    if lane_idx == 0:
        reduction_buffer[row_idx, col_idx] = val
    cute.arch.barrier()

    reduced = Float32(0.0)
    if lane_idx < warps_per_row:
        reduced = reduction_buffer[row_idx, lane_idx]
    return cute.arch.warp_reduction(reduced, operator.add)


def reduction_buffer_layout(tv_layout: cute.Layout) -> cute.Layout:
    num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
    warps_per_row = (
        num_warps
        if cute.rank(tv_layout.shape[0]) == 1
        else max(tv_layout.shape[0][0] // cute.arch.WARP_SIZE, 1)
    )
    return cute.make_ordered_layout(
        (num_warps // warps_per_row, warps_per_row), order=(1, 0)
    )


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def expand_dim(a: cute.Tensor, dim: int, size: int) -> cute.Tensor:
    shape = (*a.shape[:dim], size, *a.shape[dim:])
    stride = (*a.layout.stride[:dim], 0, *a.layout.stride[dim:])
    return cute.make_tensor(a.iterator, cute.make_layout(shape, stride=stride))


def make_tiled_copy_2d(
    dtype: Type[cutlass.Numeric],
    *,
    threads_per_row: int,
    num_threads: int,
    vecsize: int,
) -> cute.TiledCopy:
    copy_bits = vecsize * dtype.width
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        dtype,
        num_bits_per_copy=copy_bits,
    )
    thread_layout = cute.make_ordered_layout(
        (num_threads // threads_per_row, threads_per_row),
        order=(1, 0),
    )
    value_layout = cute.make_layout((1, vecsize))
    return cute.make_tiled_copy_tv(copy_atom, thread_layout, value_layout)


def rmsnorm_fwd_layout(
    N: int, widest_dtype_bits: int, num_threads: int
) -> tuple[int, int]:
    for limit, threads in ((128, 16), (3072, 32), (8192, 128), (16384, num_threads)):
        if limit >= N:
            return threads, math.gcd(N, 128 // widest_dtype_bits)
    raise NotImplementedError(f"RMSNorm forward only supports N <= 16384: {N=}")


def qk_l2_clip_fwd_layout(N: int, widest_dtype_bits: int) -> tuple[int, int]:
    for limit, threads in ((128, 8), (3072, 32), (8192, 128)):
        if limit >= N:
            return threads, math.gcd(N, 128 // widest_dtype_bits)
    raise NotImplementedError(f"QK L2 clip forward only supports N <= 8192: {N=}")


@cute.jit
def rmsnorm_row_rstd(
    x: cute.TensorSSA,
    N: cutlass.Constexpr[int],
    eps: Float32,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer: cute.Tensor,
) -> Float32:
    sum_sq = row_sum(x * x, threads_per_row, reduction_buffer)
    return cute.math.rsqrt(sum_sq / N + eps)


@cute.jit
def qk_l2_clip_row_stats(
    x: cute.TensorSSA,
    threshold: Float32,
    eps: Float32,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer: cute.Tensor,
) -> Float32:
    sum_sq = row_sum(x * x, threads_per_row, reduction_buffer)
    norm = Float32(0.0)
    if sum_sq > Float32(0.0):
        norm = sum_sq * cute.math.rsqrt(sum_sq)
    return min_f32(Float32(1.0), threshold / (norm + eps))


def _norm_rows_overlap(x: torch.Tensor) -> bool:
    span = x.shape[-1]
    for axis in sorted(range(x.ndim - 1), key=x.stride):
        if x.shape[axis] <= 1:
            continue
        if x.stride(axis) < span:
            return True
        span += (x.shape[axis] - 1) * x.stride(axis)
    return False


def as_norm_rows(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError(f"Expected at least two dimensions, got {x.shape=}")
    if x.shape[-1] == 1 and x.stride(-1) != 1:
        x = x.as_strided(x.shape, (*x.stride()[:-1], 1))
    if x.stride(-1) != 1 or x.ndim > 3 or _norm_rows_overlap(x):
        x = x.contiguous()
    if x.is_contiguous():
        return x.view(-1, x.shape[-1])
    return x


def norm_row_alignment(x: torch.Tensor) -> int:
    assert x.is_cuda and x.ndim in (2, 3), (x.shape, x.stride())
    assert x.shape[-1] > 0, x.shape
    if x.numel() == 0:
        return 16
    assert x.stride(-1) == 1, (x.shape, x.stride())
    assert not _norm_rows_overlap(x), (x.shape, x.stride())
    return math.gcd(16, x.data_ptr(), *(s * x.element_size() for s in x.stride()[:-1]))


def make_fake_norm_rows(
    dtype: Type[cutlass.Numeric], width: int, rank: int, alignment: int
):
    shape = (*tuple(cute.sym_int64() for _ in range(rank - 1)), width)
    return make_fake_dense_mn(
        dtype, shape, alignment, math.gcd(width, alignment // (dtype.width // 8))
    )


@cute.jit
def norm_num_rows(x: cute.Tensor):
    return cute.size(x.shape) // x.shape[-1]


@cute.jit
def load_norm_rows(
    src: cute.Tensor,
    dst: cute.Tensor,
    coords: cute.Tensor,
    alignment: cutlass.Constexpr[int],
    *,
    is_async: cutlass.Constexpr[bool] = False,
):
    vec = const_expr(cute.size(dst.shape[0]))
    bits = const_expr(min(128, alignment * 8, vec * src.element_type.width))
    use_async = const_expr(is_async and bits >= 32)
    op = cute.nvgpu.cpasync.CopyG2SOp() if use_async else cute.nvgpu.CopyUniversalOp()
    atom = cute.make_copy_atom(op, src.element_type, num_bits_per_copy=bits)
    atom_elems = const_expr(bits // src.element_type.width)
    layout = cute.make_layout(
        ((atom_elems, vec // atom_elems),), stride=((1, atom_elems),)
    )
    pred = cute.make_rmem_tensor(
        cute.make_layout((vec // atom_elems,), stride=(0,)), Boolean
    )
    for m in range_constexpr(dst.shape[1]):
        for n in range_constexpr(dst.shape[2]):
            row, col = coords[(0, 0), m, n]
            row, col = Int64(row), Int64(col)
            if const_expr(cute.rank(src) == 3):
                offset = (
                    row // src.shape[1] * src.stride[0]
                    + row % src.shape[1] * src.stride[1]
                )
            else:
                offset = row * src.stride[0]
            ptr = cute.make_ptr(
                src.element_type,
                (src.iterator + offset + col).toint(),
                cute.AddressSpace.gmem,
                assumed_align=bits // 8,
            )
            g = cute.make_tensor(ptr, layout)
            d = cute.make_tensor(dst[(None, 0), m, n].iterator, layout)
            valid = row < norm_num_rows(src) and col < src.shape[-1]
            pred[0] = valid
            if const_expr(not use_async):
                d.fill(0)
            cute.copy(atom, g, d, pred=pred)
