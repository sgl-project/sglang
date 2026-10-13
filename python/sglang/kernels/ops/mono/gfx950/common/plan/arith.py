# SPDX-License-Identifier: MIT
# Vendored from ROCm/ATOM @ 454d0e3023: atom/mono/plan/arith.py
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Arithmetic on Python ints and traced values alike: a kernel calls it on its
traced operands, a CPU test on ints, so the test checks what the kernel runs.
No FlyDSL import, so those tests collect where FlyDSL is not installed."""


def sel(pred, a, b):
    """a if pred else b: a conditional on Python values, ``select`` on traced
    ones (rewrapped in the traced operand's type)."""
    if isinstance(pred, bool):
        return a if pred else b
    out = pred.select(a, b)
    traced = [x for x in (a, b) if not isinstance(x, int)]
    return type(traced[0])(out) if traced else out


def pick(values, i):
    """values[i] (a select chain when ``i`` is traced)."""
    out = values[0]
    for j in range(1, len(values)):
        out = sel(i == j, values[j], out)
    return out


def imin(a, b):
    return sel(a < b, a, b)


def imax(a, b):
    return sel(a > b, a, b)


def i32(x):
    """``x``, asserted inside the device's Int32 range when it is a Python int."""
    if isinstance(x, int):
        assert -(2**31) <= x < 2**31, x
    return x


def cdiv(a, b):
    """ceil(a / b) for a >= 0, b > 0."""
    return i32((a + b - 1) // b)
