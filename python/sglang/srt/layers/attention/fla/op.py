# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/op.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import os

import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice

from sglang.srt.layers.attention.fla.utils import is_gather_supported

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
