# Copyright (c) 2026 StepFun Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack as cute_from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

from sglang.kernels.ops.attention.step5.cute.sm100_union.sparse_gqa.kernel_utils.fp_compat import (
    fmax_f32,
)
from sglang.kernels.ops.attention.step5.cute.sm100_union.utils import (
    torch2cute_dtype_map,
    warp_reduce,
)

_SPARSE_GQA_METADATA_DTYPES = {
    torch.uint8: cutlass.Uint8,
    torch.int32: cutlass.Int32,
    torch.int64: cutlass.Int64,
}


def _cutlass_dtype_from_torch_with_metadata(dtype: torch.dtype):
    try:
        return torch2cute_dtype_map[dtype]
    except KeyError:
        try:
            return _SPARSE_GQA_METADATA_DTYPES[dtype]
        except KeyError as err:
            raise TypeError(f"unsupported sparse_gqa tensor dtype: {dtype}") from err


def _shape_with_dynamic_dims(
    shape,
    dynamic_shape_dim: int | None,
    dynamic_shape_dims: tuple[int, ...] | None,
    divisibility: int,
):
    modes = (
        tuple(int(v) for v in dynamic_shape_dims)
        if dynamic_shape_dims is not None
        else (() if dynamic_shape_dim is None else (int(dynamic_shape_dim),))
    )
    if not modes:
        return tuple(int(v) for v in shape)
    return tuple(
        cute.sym_int(divisibility=divisibility) if idx in modes else int(v)
        for idx, v in enumerate(shape)
    )


def make_fake_tensor_like_with_dynamic_dim(
    tensor: torch.Tensor,
    *,
    alignment: int,
    dtype=None,
    dynamic_layout_dim: int | None = None,
    dynamic_shape_dim: int | None = None,
    dynamic_shape_dims: tuple[int, ...] | None = None,
    dynamic_stride_dims: tuple[int, ...] = (),
    divisibility: int = 1,
) -> cute.Tensor:
    if dynamic_shape_dim is not None and dynamic_shape_dims is not None:
        raise ValueError("pass dynamic_shape_dim or dynamic_shape_dims, not both")
    shape_modes = (
        tuple(int(v) for v in dynamic_shape_dims)
        if dynamic_shape_dims is not None
        else (() if dynamic_shape_dim is None else (int(dynamic_shape_dim),))
    )
    stride = [
        cute.sym_int(divisibility=divisibility)
        if idx in dynamic_stride_dims
        else int(v)
        for idx, v in enumerate(tensor.stride())
    ]
    # Equivalent to the fork's descriptor shim, without mutating CUTLASS classes.
    if dynamic_layout_dim is not None and tensor.stride()[dynamic_layout_dim] != 1:
        raise ValueError("Dynamic layout leading dimension must have stride 1")
    order = tensor.dim_order()
    from cutlass.cute.typing import SymInt

    for mode in shape_modes:
        for dim in order[: order.index(mode)]:
            if not isinstance(stride[dim], SymInt):
                stride[dim] = cute.sym_int(divisibility=divisibility)
    return cute.runtime.make_fake_tensor(
        _cutlass_dtype_from_torch_with_metadata(tensor.dtype)
        if dtype is None
        else dtype,
        _shape_with_dynamic_dims(
            tensor.shape, dynamic_shape_dim, dynamic_shape_dims, divisibility
        ),
        tuple(stride),
        assumed_align=int(alignment),
    )


def shape_key_with_dynamic_dim(x, dynamic_dim: int):
    shape = [int(dim) for dim in x.shape]
    shape[dynamic_dim] = None
    return tuple(shape)


def convert_from_dlpack(
    tensor,
    leading_dim: int,
    alignment: int = 16,
    *,
    dynamic_shape_mode: int | None = None,
    divisibility: int = 1,
    enable_tvm_ffi: bool = True,
) -> cute.Tensor:
    dynamic_shape_mode = (
        leading_dim if dynamic_shape_mode is None else dynamic_shape_mode
    )
    desc = cute_from_dlpack(
        tensor,
        assumed_align=int(alignment),
        enable_tvm_ffi=bool(enable_tvm_ffi),
    ).mark_layout_dynamic(leading_dim=int(leading_dim))
    return desc.mark_compact_shape_dynamic(
        mode=int(dynamic_shape_mode),
        stride_order=tensor.dim_order(),
        divisibility=int(divisibility),
    )


def device_cache_key(device: torch.device) -> tuple[str, int | None]:
    if device.type == "cuda":
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        return (device.type, int(index))
    return (device.type, None)


def device_from_cache_key(device_key: tuple[str, int | None]) -> torch.device:
    device_type, device_index = device_key
    return torch.device(device_type, device_index)


def tensor_signature_dynamic(
    tensor: torch.Tensor,
    *,
    dynamic_shape_dims: tuple[int, ...] = (),
    dynamic_stride_dims: tuple[int, ...] = (),
) -> tuple[torch.dtype, tuple[int | None, ...], tuple[int | None, ...]]:
    shape: list[int | None] = [int(v) for v in tensor.shape]
    stride: list[int | None] = [int(v) for v in tensor.stride()]
    for dim in dynamic_shape_dims:
        shape[int(dim)] = None
    for dim in dynamic_stride_dims:
        stride[int(dim)] = None
    return tensor.dtype, tuple(shape), tuple(stride)


def placeholder_from_signature(
    signature: tuple[torch.dtype, tuple[int | None, ...], tuple[int | None, ...]],
    *,
    device: torch.device,
    dynamic_shape_fill: int,
    dynamic_stride_fill: int = 1,
) -> torch.Tensor:
    dtype, shape, stride = signature
    concrete_shape = tuple(
        int(dynamic_shape_fill) if dim is None else int(dim) for dim in shape
    )
    concrete_stride = tuple(
        int(dynamic_stride_fill) if dim is None else int(dim) for dim in stride
    )
    return torch.empty_strided(
        concrete_shape,
        concrete_stride,
        device=device,
        dtype=dtype,
    )


def select(a: cute.Tensor, mode: list[int]) -> cute.Tensor:
    return cute.make_tensor(a.iterator, cute.select(a.layout, mode))


def make_tiled_copy_A(
    copy_atom: cute.CopyAtom,
    tiled_mma: cute.TiledMma,
    swapAB: cutlass.Constexpr[bool] = False,
) -> cute.TiledCopy:
    if const_expr(swapAB):
        return cute.make_tiled_copy_B(copy_atom, tiled_mma)
    return cute.make_tiled_copy_A(copy_atom, tiled_mma)


def make_tiled_copy_B(
    copy_atom: cute.CopyAtom,
    tiled_mma: cute.TiledMma,
    swapAB: cutlass.Constexpr[bool] = False,
) -> cute.TiledCopy:
    if const_expr(swapAB):
        return cute.make_tiled_copy_A(copy_atom, tiled_mma)
    return cute.make_tiled_copy_B(copy_atom, tiled_mma)


def _acc_layout_to_mn_view_layout(
    acc_layout: cute.Layout, transpose: bool = False
) -> cute.Layout:
    acc_layout_col_major = cute.make_layout(acc_layout.shape)
    shape = (
        (acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),
        (
            acc_layout_col_major.shape[0][0],
            *acc_layout_col_major.shape[0][2:],
            acc_layout_col_major.shape[2],
        ),
        *acc_layout_col_major.shape[3:],
    )
    stride = (
        (acc_layout_col_major.stride[0][1], acc_layout_col_major.stride[1]),
        (
            acc_layout_col_major.stride[0][0],
            *acc_layout_col_major.stride[0][2:],
            acc_layout_col_major.stride[2],
        ),
        *acc_layout_col_major.stride[3:],
    )
    if const_expr(transpose):
        shape = (shape[1], shape[0], *shape[2:])
        stride = (stride[1], stride[0], *stride[2:])
    return cute.composition(acc_layout, cute.make_layout(shape, stride=stride))


def make_acc_tensor_mn_view_from_mma(
    acc: cute.Tensor, transpose: bool = False
) -> cute.Tensor:
    return cute.make_tensor(
        acc.iterator, _acc_layout_to_mn_view_layout(acc.layout, transpose=transpose)
    )


@cute.jit
def acc_layout_to_frgA_split2(acc_layout: cute.Layout) -> cute.Layout:
    if const_expr(cute.rank(acc_layout.shape[0]) == 3):
        layout = cute.logical_divide(acc_layout, ((None, None, 2), None, None))
        return cute.make_layout(
            (
                (layout.shape[0][0], layout.shape[0][1], layout.shape[0][2][0]),
                layout.shape[1],
                (layout.shape[0][2][1], layout.shape[2]),
            ),
            stride=(
                (layout.stride[0][0], layout.stride[0][1], layout.stride[0][2][0]),
                layout.stride[1],
                (layout.stride[0][2][1], layout.stride[2]),
            ),
        )
    layout = cute.logical_divide(acc_layout, (None, None, 2))
    return cute.make_layout(
        (
            (layout.shape[0], layout.shape[2][0]),
            layout.shape[1],
            layout.shape[2][1],
        ),
        stride=(
            (layout.stride[0], layout.stride[2][0]),
            layout.stride[1],
            layout.stride[2][1],
        ),
    )


def transpose_first_two_modes_view(a: cute.Tensor) -> cute.Tensor:
    shape = (a.shape[1], a.shape[0], *a.shape[2:])
    order = (1, 0, *range(2, cute.rank(a)))
    return cute.composition(a, cute.make_ordered_layout(shape, order=order))


@dsl_user_op
def elem_pointer_i64_offset(
    x: cute.Tensor,
    coord: cute.Coord,
    *,
    loc=None,
    ip=None,  # noqa: ARG001
) -> cute.Pointer:
    flat_coord_i64 = tuple(cutlass.Int64(c) for c in cute.flatten(coord))
    flat_stride = cute.flatten_to_tuple(x.stride)
    if len(flat_coord_i64) != len(flat_stride):
        raise ValueError("Coordinate and stride must have the same length")
    offset = sum(c * s for c, s in zip(flat_coord_i64, flat_stride, strict=True))
    byte_offset = offset * x.element_type.width // 8
    return cute.make_ptr(
        x.element_type,
        x.iterator.toint() + byte_offset,
        x.memspace,
        assumed_align=x.iterator.alignment,
    )


@cute.jit
def shuffle_sync(
    value: cute.Numeric,
    offset: cute.typing.Int,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.Numeric:
    if const_expr(value.width % 32 != 0):
        raise ValueError("value type must be a multiple of 32 bits")
    mask = cute.arch.WARP_SIZE - width
    clamp = cute.arch.WARP_SIZE - 1
    mask_and_clamp = mask << 8 | clamp
    val = cute.make_fragment(1, type(value))
    val[0] = value
    val_i32 = cute.recast_tensor(val, cutlass.Int32)
    for i in cutlass.range_constexpr(cute.size(val_i32)):
        val_i32[i] = cute.arch.shuffle_sync(
            val_i32[i], offset, mask_and_clamp=mask_and_clamp
        )
    return val[0]


@cute.jit
def exp2f(x: cute.TensorSSA | Float32) -> cute.TensorSSA | Float32:
    if const_expr(isinstance(x, cute.TensorSSA)):
        res = cute.make_fragment(x.shape, Float32)
        res.store(x)
        for i in cutlass.range_constexpr(cute.size(x.shape)):
            res[i] = cute.arch.exp2(res[i])
        return res.load()
    return cute.arch.exp2(x)


@dsl_user_op
def log2f(a: float | Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "lg2.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def fmax_reduce(
    x: cute.TensorSSA,
    init_val: float | Float32 | None = None,
) -> Float32:
    res = cute.make_fragment(x.shape, Float32)
    res.store(x)
    local_max = [res[0], res[1], res[2], res[3]]
    for i in cutlass.range_constexpr(4, cute.size(x.shape), 4):
        local_max[0] = fmax_f32(local_max[0], res[i + 0])
        local_max[1] = fmax_f32(local_max[1], res[i + 1])
        local_max[2] = fmax_f32(local_max[2], res[i + 2])
        local_max[3] = fmax_f32(local_max[3], res[i + 3])
    local_max[0] = fmax_f32(local_max[0], local_max[1])
    local_max[2] = fmax_f32(local_max[2], local_max[3])
    local_max[0] = fmax_f32(local_max[0], local_max[2])
    return (
        local_max[0]
        if const_expr(init_val is None)
        else fmax_f32(local_max[0], init_val)
    )


@cute.jit
def fadd_reduce(
    x: cute.TensorSSA,
    init_val: float | Float32 | None = None,
) -> Float32:
    if const_expr(init_val is None):
        init_val = Float32.zero
    return x.reduce(cute.ReductionOp.ADD, init_val, 0)


@dsl_user_op
def _vector_copy_atom_with_explicit_width(
    dtype: type[cutlass.Numeric],
    num_copy_elems: int,
    is_async: bool = False,
    *,
    loc=None,
    ip=None,  # noqa: ARG001
) -> cute.CopyAtom:
    num_copy_bits = const_expr(min(128, num_copy_elems * dtype.width))
    copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    return cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)


@dsl_user_op
def vector_copy_with_explicit_width(
    src: cute.Tensor,
    dst: cute.Tensor,
    *,
    pred: Optional[cute.Tensor] = None,
    num_copy_elems: int = 1,
    is_async: bool = False,
    loc=None,
    ip=None,
    **kwargs,
) -> None:
    copy_atom = _vector_copy_atom_with_explicit_width(
        src.element_type, num_copy_elems, is_async
    )
    cute.copy(copy_atom, src, dst, pred=pred, loc=loc, ip=ip, **kwargs)


@dsl_user_op
def async_copy_hi_lo_pair_with_explicit_width(
    src_ptr_lo: cute.Pointer,
    dst_ptr_lo: cute.Pointer,
    src_ptr_hi: cute.Pointer,
    dst_ptr_hi: cute.Pointer,
    *,
    num_copy_elems: int,
    loc=None,
    ip=None,
) -> None:
    if const_expr(num_copy_elems == 8):
        vec_layout = cute.make_layout((8,), stride=(1,))
    elif const_expr(num_copy_elems == 4):
        vec_layout = cute.make_layout((4,), stride=(1,))
    elif const_expr(num_copy_elems == 2):
        vec_layout = cute.make_layout((2,), stride=(1,))
    else:
        raise ValueError(
            f"sparse_gqa hi/lo async copy supports only 2, 4, or 8 elems, got {num_copy_elems}"
        )
    src_vec_lo = cute.make_tensor(src_ptr_lo, vec_layout)
    dst_vec_lo = cute.make_tensor(dst_ptr_lo, vec_layout)
    vector_copy_with_explicit_width(
        src_vec_lo,
        dst_vec_lo,
        num_copy_elems=num_copy_elems,
        is_async=True,
        loc=loc,
        ip=ip,
    )
    src_vec_hi = cute.make_tensor(src_ptr_hi, vec_layout)
    dst_vec_hi = cute.make_tensor(dst_ptr_hi, vec_layout)
    vector_copy_with_explicit_width(
        src_vec_hi,
        dst_vec_hi,
        num_copy_elems=num_copy_elems,
        is_async=True,
        loc=loc,
        ip=ip,
    )


__all__ = [
    "make_fake_tensor_like_with_dynamic_dim",
    "shape_key_with_dynamic_dim",
    "convert_from_dlpack",
    "device_cache_key",
    "device_from_cache_key",
    "tensor_signature_dynamic",
    "placeholder_from_signature",
    "select",
    "make_tiled_copy_A",
    "make_tiled_copy_B",
    "make_acc_tensor_mn_view_from_mma",
    "acc_layout_to_frgA_split2",
    "transpose_first_two_modes_view",
    "elem_pointer_i64_offset",
    "shuffle_sync",
    "exp2f",
    "log2f",
    "fmax_reduce",
    "fadd_reduce",
    "warp_reduce",
    "vector_copy_with_explicit_width",
    "async_copy_hi_lo_pair_with_explicit_width",
]
