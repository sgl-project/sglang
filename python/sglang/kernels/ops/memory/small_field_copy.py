"""One-launch copy of several small fields (JIT CUDA).

Same contract as ``small_copy.try_small_copy``: every ``dsts[i]`` receives
``srcs[i]`` (same shape; equal dtypes, or an int32/int64 width cast), and the
call returns False without touching anything when a field is not supported or
two different fields overlap. Validation is a handful of attribute reads per
field plus a C++ range check, so the per-call host cost stays small.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_MAX_FIELDS = 16
_RAW_KIND = {1: 0, 2: 1, 4: 2, 8: 3}
# (dst dtype, src dtype) -> width-cast kind
_CAST_KIND = {(torch.int64, torch.int32): 4, (torch.int32, torch.int64): 5}


@cache_once
def _jit_small_field_copy_module() -> Module:
    return load_jit(
        "small_field_copy",
        cuda_files=["elementwise/small_field_copy.cuh"],
        cuda_wrappers=[("small_field_copy", "SmallFieldCopy::run")],
    )


def small_field_copy(dsts: List[torch.Tensor], srcs: List[torch.Tensor]) -> bool:
    n = len(dsts)
    if n == 0 or n != len(srcs) or n > _MAX_FIELDS:
        return False
    device = dsts[0].get_device()
    if device < 0:
        return False
    desc: List[int] = []
    for dst, src in zip(dsts, srcs):
        shape = dst.shape
        if src.shape != shape or src.get_device() != device or dst.get_device() != device:
            return False
        ndim = len(shape)
        if ndim == 1:
            n = shape[0]
            if n <= 1 or (dst.stride(0) == 1 and src.stride(0) == 1):
                rows, cols, src_stride, dst_stride = 1, n, n, n
            else:
                rows, cols, src_stride, dst_stride = n, 1, src.stride(0), dst.stride(0)
        elif ndim == 2:
            rows, cols = shape
            if cols > 1 and (dst.stride(1) != 1 or src.stride(1) != 1):
                return False
            src_stride, dst_stride = src.stride(0), dst.stride(0)
            if rows > 1 and dst_stride < cols:
                return False
        else:
            return False
        if dst.dtype == src.dtype:
            kind = _RAW_KIND.get(dst.element_size())
            if kind is None or dst.is_neg() or src.is_neg() or dst.is_conj():
                return False
        else:
            kind = _CAST_KIND.get((dst.dtype, src.dtype))
            if kind is None:
                return False
        if dst.requires_grad or src.requires_grad:
            return False
        desc += (src.data_ptr(), dst.data_ptr(), rows, cols, src_stride, dst_stride, kind)
    return _jit_small_field_copy_module().small_field_copy(dsts[0], desc) == 0


__all__ = ["small_field_copy"]
