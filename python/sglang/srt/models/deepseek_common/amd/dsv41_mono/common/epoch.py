# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The launch's epoch: every CTA marks the epoch it read; CTA 0 moves it on once
every CTA marked it. Also the device helpers the layer's launch uses with it:
its kernel symbol and a store at a 64-bit global address. Adapted from ATOM's
mono decode framework (``atom/mono/device``) and V4.1 ``attn_post``."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.kernels_common import kernel_signature
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu
from flydsl.expr.typing import T, as_ir_value

from sglang.srt.models.deepseek_common.amd.dsv41_mono.common.ops import (
    CM_DEV,
    rsrc,
    traced,
)
from sglang.srt.models.deepseek_common.amd.dsv41_mono.common.plan import BLOCKS
from sglang.srt.models.deepseek_common.amd.dsv41_mono.common.sync import _spin


def kernel_symbol(stem, **params):
    """Kernel symbol that profilers demangle to ``aiter::<stem>_<params>``."""
    name = f"{stem}_{kernel_signature(**params)}"
    return f"_ZN5aiter{len(name)}{name}E"


def _gptr(addr):
    return llvm.IntToPtrOp(
        ir.Type.parse("!llvm.ptr<1>"), as_ir_value(fx.Int64(addr))
    ).result


def gstore(addr, value, words=4):
    """Store ``words`` i32 (an Int32 or a Vector) at a 64-bit global address."""
    v = as_ir_value(fx.Int32(value) if words == 1 else value)
    llvm.StoreOp(v, _gptr(addr), alignment=4 * words)


def spin_until(load, pending):
    """``load()`` again (s_sleep between) while ``pending(value)``; the value."""
    return _spin(load, pending)


EPOCH_MARKS = 4  # word offset of the per-CTA marks in the epoch buffer


@traced
def epoch_begin(c, epoch, ep):
    """This CTA has read the launch pair's epoch: its own mark word (no
    contended atomic) := ep + 1, which ``epoch_end`` checks."""
    if c["tid"] == 0:
        bo.buffer_store(
            ep + 1, rsrc(epoch), EPOCH_MARKS + c["bid"], cache_modifier=CM_DEV
        )


@traced
def epoch_end(c, epoch, ep, reset=None):
    """CTA 0, once every CTA marked this epoch (thread u checks CTA u's mark):
    ``reset()`` (state later pairs must find cleared), then the epoch moved on.
    The next pair's hand-offs carry a tag no earlier pair wrote; later launches'
    kernels start after this one ends."""
    if c["bid"] == 0:
        tid = c["tid"]
        if tid < BLOCKS:
            r = rsrc(epoch)

            def load_mark():
                return fx.Int32(
                    bo.buffer_load(
                        r,
                        EPOCH_MARKS + tid,
                        vec_width=1,
                        dtype=T.i32,
                        cache_modifier=CM_DEV,
                    )
                )

            _ = spin_until(load_mark, lambda v: v != ep + 1)
        gpu.barrier()
        if tid == 0:
            if const_expr(reset is not None):
                reset()
            gstore(epoch, ep + 1, words=1)
