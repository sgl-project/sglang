"""Tell aiter which routing regime (prefill or decode) a MoE call belongs to.

Prefill and decode route tokens to very different expert sets: decode
concentrates on a few hot experts while prefill spreads nearly flat. The same
MoE shape therefore wants different kernels in the two regimes, but aiter's
tuned-config key cannot tell them apart -- it carries only the padded token
count, and both regimes land in the same power-of-two buckets. Whichever regime
resolves a bucket first fixes the kernel for both, and on a CUDA-graph deployment
that is always decode (every bucket is resolved during graph capture).

An aiter with the per-regime patch exposes ``set_moe_regime`` / ``moe_regime``
and reads a table per regime from ``AITER_CONFIG_FMOE_PREFILL`` /
``AITER_CONFIG_FMOE_DECODE``. This module is the sglang side of that contract:

    ModelRunner.forward           set_moe_regime_for_forward_mode() every pass
    DecodeCudaGraphRunner.capture pin_decode_moe_regime() around capture, since
                                  a replayed graph never reaches forward()

On a non-HIP build, or an aiter without the regime API, both are no-ops and
aiter keeps using its single ``AITER_CONFIG_FMOE`` table exactly as before.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, ContextManager

from sglang.srt.utils import is_hip

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

_aiter_set_moe_regime = None
_aiter_moe_regime = None
if is_hip():
    try:
        from aiter.fused_moe import moe_regime as _aiter_moe_regime
        from aiter.fused_moe import set_moe_regime as _aiter_set_moe_regime
    except ImportError:
        pass


def set_moe_regime_for_forward_mode(forward_mode: ForwardMode) -> None:
    """Select aiter's tuned MoE table matching this pass's routing.

    Target-verify is decode-regime: it scores a few draft tokens per request.
    MIXED and the draft-extend modes count as prefill: they carry extend tokens,
    which dominate the row count and spread routing the way prefill does.
    Set unconditionally rather than save/restore, since every pass calls this.
    """
    if _aiter_set_moe_regime is None:
        return
    is_decode = forward_mode.is_decode_or_idle() or forward_mode.is_target_verify()
    _aiter_set_moe_regime("decode" if is_decode else "prefill")


def pin_decode_moe_regime() -> ContextManager:
    """Pin the decode table for the duration of decode CUDA-graph capture.

    Capture builds its own ForwardBatch and calls the model directly, and a
    replayed graph runs no Python, so the kernel resolved during capture is the
    one every decode step uses for that batch size.
    """
    if _aiter_moe_regime is None:
        return contextlib.nullcontext()
    return _aiter_moe_regime("decode")
