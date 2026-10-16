# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/op.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import inspect
import os

import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice

from sglang.kernels.ops.attention.fla.utils import is_gather_supported

if os.environ.get("FLA_USE_FAST_OPS", "0") == "1":
    exp = tldevice.fast_expf
    exp2 = tldevice.exp2
    log = tldevice.fast_logf
    log2 = tldevice.fast_log2f
else:
    exp = tl.exp
    exp2 = tl.math.exp2
    log = tl.log
    log2 = tl.log2


@triton.jit
def safe_exp(x):
    return exp(tl.where(x <= 0, x, float("-inf")))


if not is_gather_supported:

    @triton.jit
    def gather(src, index, axis, _builder=None):
        """
        Gather operation that works when tl.gather is not supported.
        This is a fallback implementation that returns None.
        Just to make triton compiler happy.
        """
        return None

else:
    gather = tl.gather


# Triton main removed block pointers, and with them tl.load/tl.store's boundary_check
# (triton-lang/triton#10833). Where they are gone, a "block pointer" is a
# (pointers, mask) pair, which covers the 1D/2D boundary-checked blocks used here.
if "boundary_check" in inspect.signature(tl.load).parameters:
    make_block_ptr = tl.make_block_ptr

    @triton.jit
    def load_block(p, boundary_check: tl.constexpr):
        return tl.load(p, boundary_check=boundary_check)

    @triton.jit
    def store_block(p, value, boundary_check: tl.constexpr):
        tl.store(p, value.to(p.dtype.element_ty), boundary_check=boundary_check)

else:

    @triton.jit
    def make_block_ptr(
        base, shape, strides, offsets, block_shape: tl.constexpr, order: tl.constexpr
    ):
        o0 = offsets[0] + tl.arange(0, block_shape[0]).to(tl.int64)
        if len(block_shape) == 1:
            return base + o0 * strides[0], (o0 >= 0) & (o0 < shape[0])
        else:
            o1 = offsets[1] + tl.arange(0, block_shape[1]).to(tl.int64)
            ptrs = base + o0[:, None] * strides[0] + o1[None, :] * strides[1]
            mask = ((o0 >= 0) & (o0 < shape[0]))[:, None] & (
                (o1 >= 0) & (o1 < shape[1])
            )[None, :]
            return ptrs, mask

    @triton.jit
    def load_block(p, boundary_check: tl.constexpr):
        return tl.load(p[0], mask=p[1], other=0)

    @triton.jit
    def store_block(p, value, boundary_check: tl.constexpr):
        tl.store(p[0], value.to(p[0].dtype.element_ty), mask=p[1])


if hasattr(triton.language, "_experimental_make_tensor_descriptor"):
    # For Triton 3.3.x
    make_tensor_descriptor = triton.language._experimental_make_tensor_descriptor
elif hasattr(triton.language, "make_tensor_descriptor"):
    # For Triton 3.4.x and later
    make_tensor_descriptor = triton.language.make_tensor_descriptor
else:
    """
    Fallback implementation when TMA is not supported.
    Returns None to indicate TMA descriptors are unavailable.
    Just make triton compiler happy.
    """

    @triton.jit
    def make_tensor_descriptor(
        base,
        shape,
        strides,
        block_shape,
        _builder=None,
    ):
        return None
