from collections.abc import Sequence
from typing import Type

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cutlass_dsl import dsl_user_op

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


def assert_cuda_contig(
    x: torch.Tensor,
    name: str = "",
    *,
    ndim: int | None = None,
    allow_meta: bool = False,
) -> None:
    if not name:
        name = "tensor"
    assert x.is_cuda or (allow_meta and x.is_meta), (
        f"{name} must be on CUDA: device={x.device}"
    )
    assert x.is_contiguous(), (
        f"{name} must be contiguous: shape={x.shape} stride={x.stride()}"
    )
    if ndim is not None:
        assert x.ndim == ndim, f"{name} must be {ndim}D: ndim={x.ndim}"


def empty_cuda(
    *size: int | Sequence[int],
    dtype: torch.dtype,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    if len(size) == 1 and not isinstance(size[0], int):
        normalized_size = tuple(size[0])
    else:
        normalized_size = tuple(size)
    return torch.empty(normalized_size, device=device, dtype=dtype)


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
