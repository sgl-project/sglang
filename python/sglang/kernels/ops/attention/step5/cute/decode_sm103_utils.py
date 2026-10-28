# Copyright (c) 2026 StepFun Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source vector copies and SM103 TMEM synchronization for sparse decode."""

from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import const_expr
from cutlass._mlir.dialects import nvvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import dsl_user_op


@dsl_user_op
def tcgen_before_sync(*, loc=None, ip=None):
    nvvm.tcgen05_fence(nvvm.Tcgen05FenceKind.BEFORE_THREAD_SYNC, loc=loc, ip=ip)


@dsl_user_op
def tcgen_after_sync(*, loc=None, ip=None):
    nvvm.tcgen05_fence(nvvm.Tcgen05FenceKind.AFTER_THREAD_SYNC, loc=loc, ip=ip)


@cute.jit
def elem_pointer(tensor: cute.Tensor, coord: cute.Coord):
    return tensor.iterator + tensor.layout(coord)


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
