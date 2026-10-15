# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The launch epoch: every CTA marks it, and CTA 0 moves it on once all have."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.kernels_common import kernel_signature
from flydsl.expr import const_expr, gpu
from flydsl.expr.typing import T

from sglang.experimental.monokernels.hip.common.ops import (
    CM_DEV,
    gstore,
    rsrc,
    traced,
)
from sglang.experimental.monokernels.hip.common.plan import BLOCKS
from sglang.experimental.monokernels.hip.common.sync import _spin

__all__ = [
    "EPOCH_MARKS",
    "epoch_begin",
    "epoch_end",
    "gstore",
    "kernel_symbol",
    "spin_until",
]


def kernel_symbol(stem, **params):
    """Kernel symbol that profilers demangle to ``aiter::<stem>_<params>``."""
    name = f"{stem}_{kernel_signature(**params)}"
    return f"_ZN5aiter{len(name)}{name}E"


def spin_until(load, pending, guard, site):
    """``load()`` again (s_sleep between) while ``pending(value)``; the value. A guarded wait named ``site``."""
    return _spin(load, pending, guard, site)


EPOCH_MARKS = 4  # word offset of the per-CTA marks in the epoch buffer


@traced
def epoch_begin(c, epoch, ep):
    """Mark that this CTA read epoch ``ep``."""
    if c["tid"] == 0:
        bo.buffer_store(
            ep + 1, rsrc(epoch), EPOCH_MARKS + c["bid"], cache_modifier=CM_DEV
        )


@traced
def epoch_end(c, epoch, ep, reset, guard):
    """CTA 0: once every CTA marked the epoch (a guarded wait), run ``reset()`` unless None and move the epoch
    on."""
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

            _ = spin_until(load_mark, lambda v: v != ep + 1, guard, "epoch")
        gpu.barrier()
        if tid == 0:
            if const_expr(reset is not None):
                reset()
            gstore(epoch, ep + 1, words=1)
