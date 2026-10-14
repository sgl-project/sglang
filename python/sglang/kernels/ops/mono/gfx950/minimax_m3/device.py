# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3's device helpers that ``gfx950/common`` does not hold."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import math as fmath
from flydsl.expr import rocdl
from flydsl.expr.typing import T

from sglang.kernels.ops.mono.gfx950.common.device.ops import rsrc, uniform
from sglang.kernels.ops.mono.gfx950.minimax_m3.config import LAYER_SLOTS


def step_tag(step_ptr, layer):
    """The layer's mailbox tag this step, ``step * LAYER_SLOTS + layer + 1`` with
    ``step`` the runner's device counter: graph-safe, and no mailbox is cleared
    between steps."""
    step = uniform(bo.buffer_load(rsrc(step_ptr), 0, vec_width=1, dtype=T.i32))
    return step * LAYER_SLOTS + layer + 1


def release_stores():
    """Order this wave's earlier device-scope (SC1) stores before a flag that
    other CTAs poll. A workgroup-scope release fence does not wait for global
    stores, so the flag could land first and a consumer read stale data; the
    stores write through to L2, so their completion is all it must wait for."""
    rocdl.s_waitcnt(vmcnt=0)
    fx.memory_fence(
        syncscope=rocdl.SyncScope.Workgroup, ordering=fx.AtomicOrdering.Release
    )


def ar_pack_sumsq(vals):
    """One logical thread of the custom all-reduce epilogue: its 8 elements'
    squares accumulated in order (hipcc contracts ``acc += v * v`` into fma)."""
    acc = fx.Float32(0.0)
    for v in vals:
        acc = fmath.fma(v, v, acc)
    return acc
