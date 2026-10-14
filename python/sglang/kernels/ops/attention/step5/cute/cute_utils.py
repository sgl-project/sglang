# Copyright (c) 2026 StepFun Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from .device_utils import torch2cute_dtype_map

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
    stride = [
        (
            cute.sym_int(divisibility=divisibility)
            if idx in dynamic_stride_dims
            else int(v)
        )
        for idx, v in enumerate(tensor.stride())
    ]
    fake = cute.runtime.make_fake_tensor(
        (
            _cutlass_dtype_from_torch_with_metadata(tensor.dtype)
            if dtype is None
            else dtype
        ),
        _shape_with_dynamic_dims(
            tensor.shape, dynamic_shape_dim, dynamic_shape_dims, divisibility
        ),
        tuple(stride),
        assumed_align=int(alignment),
    )
    # CUTLASS DSL 4.6 fake tensors encode dynamic shape/stride entries
    # directly through SymInt. The legacy mark_* APIs are implemented only
    # by live DLPack descriptors and intentionally raise NotImplementedError
    # on _FakeTensor. dynamic_layout_dim is retained in this compatibility
    # helper for older callers; explicit fake descriptors already carry the
    # static stride contract supplied above.
    del dynamic_layout_dim
    return fake


@functools.cache
def _device_arch(index: int) -> tuple[int, int]:
    return torch.cuda.get_device_capability(index)


def device_cache_key(device: torch.device) -> tuple[str, int, tuple[int, int]]:
    # Persistent artifacts must not collide when CUDA ordinal zero refers to
    # different architectures across deployments sharing a cache directory.
    index = device.index if device.index is not None else torch.cuda.current_device()
    return device.type, int(index), _device_arch(int(index))


def device_from_cache_key(device_key: tuple[str, int, tuple[int, int]]) -> torch.device:
    device_type, device_index, _ = device_key
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
